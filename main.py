import asyncio
import logging
# import uuid # Will remove if not used elsewhere after pc_id removal # REMOVED
import cv2 # opencv-python for image processing if needed by model
import numpy as np # For array manipulation
import re # For regular expression matching of license plate format
import os # Moved to top-level
from datetime import datetime # Moved to top-level
from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel # Not strictly needed for this simplified version

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamError

from ultralytics import YOLO # Import YOLO
import torch # For device checking
# import pytesseract # REMOVED
from PIL import Image # For converting numpy array to PIL Image for pytesseract
# from paddleocr import PaddleOCR # REMOVED
import easyocr # ADDED

# Import the display utility # REMOVED
# from image_display_util import display_image_cv, close_all_cv_windows # REMOVED

# --- Configuration ---
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.7 # Overall confidence for "ok" status
YOLO_CONF_FOR_OCR = 0.4  # Minimum YOLO confidence to attempt OCR
# Tesseract OCR Configuration
# TESSERACT_LANG = 'eng' # REMOVED
# TESSERACT_CONFIG = r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-' # REMOVED
# Expected license plate format regex: 3 letters, a hyphen, 4 numbers
LP_FORMAT_REGEX = r"^[A-Z]{3}-[0-9]{4}$"

# Image Preprocessing for OCR
OCR_TARGET_PLATE_HEIGHT = 48 # Target height in pixels for the license plate crop before OCR (CURRENTLY UNUSED as perspective transform is off)
OCR_FIXED_WIDTH = 256  # ADDED: Target width for OCR input after resize
OCR_FIXED_HEIGHT = 64   # ADDED: Target height for OCR input after resize
DEBUG_SAVE_OCR_IMAGES = True# Set to True to save intermediate images for debugging OCR
# DEBUG_SHOW_OCR_IMAGE = False  # Set to True to display the final OCR image via cv2.imshow() # REMOVED

DEBUG_IMG_DIR = "debug_ocr_images" # For multi-step debug images
PROCESSED_GRAY_PLATES_DIR = "processed_gray_plates" # For unconditionally saved gray plates
PROCESSED_WARPED_COLOR_PLATES_DIR = "processed_warped_color_plates" # ADDED for PaddleOCR input


# --- Global Variables and Application Setup ---
app = FastAPI()

# Create directories
if DEBUG_SAVE_OCR_IMAGES:
    if not os.path.exists(DEBUG_IMG_DIR):
        os.makedirs(DEBUG_IMG_DIR)
if not os.path.exists(PROCESSED_GRAY_PLATES_DIR):
    os.makedirs(PROCESSED_GRAY_PLATES_DIR)
if not os.path.exists(PROCESSED_WARPED_COLOR_PLATES_DIR):
    os.makedirs(PROCESSED_WARPED_COLOR_PLATES_DIR)


app.mount("/static", StaticFiles(directory="static"), name="static")

# Logging
# logging.basicConfig(level=logging.INFO) # Already configured by uvicorn if run via uvicorn.run
logger = logging.getLogger("app") # Use "app" or specific module name

# Determine device for models (YOLO and EasyOCR)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device for deep learning models: {device}") # Generalizing the log message

# Initialize EasyOCR (do this once on startup)
easyocr_reader = None
try:
    logger.info("Initializing EasyOCR reader...")
    easyocr_reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))
    logger.info("EasyOCR reader initialized successfully.")
except Exception as e_easyocr_init:
    logger.error(f"Failed to initialize EasyOCR reader: {e_easyocr_init}", exc_info=True)
    # App can run, but /capture will fail if easyocr_reader is None

# Simplified global state for a single stream
# Stores the current RTCPeerConnection and its latest frame
current_stream_data = {
    "pc": None,  # type: RTCPeerConnection | None
    "latest_frame": None,  # type: aiortc.VideoFrame | None
    "video_track": None # type: aiortc.MediaStreamTrack | None
}

# Load YOLO model
# device = 'cuda' if torch.cuda.is_available() else 'cpu' # MOVED EARLIER
# logger.info(f"Using device for YOLO model: {device}") # MOVED EARLIER
try:
    model = YOLO(MODEL_PATH)
    model.to(device) # Use the globally defined device
    logger.info(f"Successfully loaded YOLO model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    model = None # Set model to None if loading fails

async def close_current_connection():
    """Helper to close the existing WebRTC connection if any."""
    if current_stream_data["pc"] and current_stream_data["pc"].signalingState != "closed":
        logger.info("Closing existing peer connection.")
        await current_stream_data["pc"].close()
    current_stream_data["pc"] = None
    current_stream_data["latest_frame"] = None
    current_stream_data["video_track"] = None

# --- WebRTC Signaling ---
@app.post("/offer")
async def offer(params: dict = Body(...)):
    await close_current_connection() # Close any existing connection first

    offer_sdp = params.get("sdp")
    offer_type = params.get("type")
    if not offer_sdp or not offer_type:
        raise HTTPException(status_code=400, detail="Missing sdp or type in offer")

    offer_desc = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    pc = RTCPeerConnection()
    current_stream_data["pc"] = pc

    logger.info("New PeerConnection created for single stream.")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")
        if pc.iceConnectionState in ["failed", "disconnected", "closed"]:
            logger.info("Peer connection closed/failed. Cleaning up.")
            # Check if this is still the current PC before clearing, 
            # to avoid race conditions if a new connection was made quickly.
            if current_stream_data["pc"] is pc:
                await close_current_connection() # Use the helper to also nullify frame/track
    
    @pc.on("track")
    async def on_track(track):
        logger.info(f"Track {track.kind} received")
        if track.kind == "video":
            current_stream_data["video_track"] = track
            while current_stream_data["pc"] is pc and pc.signalingState != "closed": 
                try:
                    frame = await track.recv()
                    current_stream_data["latest_frame"] = frame
                except MediaStreamError:
                    logger.info("Video track ended or media stream error.")
                    if current_stream_data["pc"] is pc: # ensure it's still the active pc
                        current_stream_data["latest_frame"] = None
                    break
                except Exception as e_recv:
                    logger.error(f"Error receiving frame: {e_recv}")
                    if current_stream_data["pc"] is pc:
                        current_stream_data["latest_frame"] = None
                    break
            logger.info("Stopped receiving frames from video track.")

        @track.on("ended")
        async def on_ended():
            logger.info(f"Track {track.kind} ended")
            if track.kind == "video" and current_stream_data["pc"] is pc:
                 current_stream_data["latest_frame"] = None

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type} # No pc_id


# --- Capture and Predict API (with MASK-BASED Cropping & Preprocessing) ---
@app.get("/capture") 
async def capture_and_predict():
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO Model not loaded")
    if easyocr_reader is None: # ADDED
        logger.error("Capture endpoint called but EasyOCR reader is not available.") # ADDED
        raise HTTPException(status_code=500, detail="EasyOCR reader not initialized or failed to initialize.") # ADDED

    pc_ref = current_stream_data["pc"]
    latest_frame = current_stream_data["latest_frame"]

    if pc_ref is None or pc_ref.signalingState == "closed" or latest_frame is None:
        raise HTTPException(status_code=404, detail="No active video stream or frame found.")
    
    logger.info("--- /capture endpoint called ---") # Added to confirm entry

    raw_ocr_text_for_response = "N/A"
    ts_debug = None
    if DEBUG_SAVE_OCR_IMAGES:
        ts_debug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    try:
        logger.info("Starting try block in /capture")
        img_np = latest_frame.to_ndarray(format="bgr24")
        frame_height, frame_width = img_np.shape[:2]
        logger.info(f"Frame received: {{frame_width}}x{{frame_height}}")
        
        if DEBUG_SAVE_OCR_IMAGES and ts_debug:
            cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_00_original_frame.png"), img_np)

        logger.info("Performing YOLO prediction...")
        results_yolo = model.predict(img_np, conf=YOLO_CONF_FOR_OCR, device=device, verbose=False)
        logger.info("YOLO prediction done.")

        ocr_text = "N/A"
        yolo_confidence = 0.0
        format_valid = False
        best_status = "error"
        message = "Initial error: No license plate by YOLO with sufficient confidence."

        if results_yolo and results_yolo[0].masks is not None and len(results_yolo[0].masks.xy) > 0:
            masks_polygons = results_yolo[0].masks.xy
            confs = results_yolo[0].boxes.conf.cpu().numpy()
            best_idx = np.argmax(confs)
            yolo_confidence = float(confs[best_idx])
            polygon = masks_polygons[best_idx].astype(np.int32)
            logger.info(f"Highest YOLO confidence: {yolo_confidence:.4f} for a mask.")

            if yolo_confidence >= YOLO_CONF_FOR_OCR:
                logger.info("YOLO confidence above threshold, proceeding with plate processing.")
                cropped_color_plate = None # RENAMED from final_cropped_plate_for_paddle

                # USER REQUEST: Bypass perspective transform, use simple bounding box crop from YOLO mask.
                logger.info("USER REQUEST: Bypassing perspective transform, using simple bounding box crop from YOLO mask.")
                
                # Create mask from polygon
                mask_for_crop = np.zeros((frame_height, frame_width), dtype=np.uint8)
                cv2.fillPoly(mask_for_crop, [polygon], 255)
                if DEBUG_SAVE_OCR_IMAGES and ts_debug: # Save the mask used for segmentation
                    cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_01a_simple_crop_mask.png"), mask_for_crop)

                # Apply mask to get segmented plate (consistent with previous fallback)
                segmented_plate_for_crop = cv2.bitwise_and(img_np, img_np, mask=mask_for_crop)
                if DEBUG_SAVE_OCR_IMAGES and ts_debug: # Save the segmented part
                     cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_01b_simple_segmented_for_crop.png"), segmented_plate_for_crop)
                
                # Get bounding box of the polygon
                x_poly, y_poly, w_poly, h_poly = cv2.boundingRect(polygon)
                
                if w_poly > 0 and h_poly > 0:
                    # Crop the segmented image using the bounding box
                    cropped_color_plate = segmented_plate_for_crop[y_poly:y_poly+h_poly, x_poly:x_poly+w_poly] # RENAMED
                    logger.info(f"Using simple bounding box crop. Dimensions: w={w_poly}, h={h_poly}")
                    if DEBUG_SAVE_OCR_IMAGES and ts_debug and cropped_color_plate.size > 0:
                         cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_01c_simple_color_cropped.png"), cropped_color_plate)
                else:
                    logger.warning("Simple bounding box crop has zero width or height. Plate might be too small or polygon invalid.")
                    # cropped_color_plate remains None
                
                # Original perspective transform try-except block is now replaced by the logic above.

                if cropped_color_plate is None or cropped_color_plate.size == 0:
                    message = "Cropped plate area for OCR is empty."
                    logger.warning(message)
                else:
                    logger.info(f"Cropped color plate for OCR shape: {cropped_color_plate.shape}, dtype: {cropped_color_plate.dtype}")
                    
                    unconditional_ts_color = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    # Save the original color crop (before grayscale, resize etc.)
                    color_plate_save_path = os.path.join(PROCESSED_WARPED_COLOR_PLATES_DIR, f"color_lp_{unconditional_ts_color}.png")
                    cv2.imwrite(color_plate_save_path, cropped_color_plate)
                    logger.info(f"Saved original color cropped plate to {color_plate_save_path}")

                    # Preprocessing for OCR: Grayscale -> CLAHE -> Resize -> Binarize
                    logger.info("Preprocessing image for OCR: Grayscale -> CLAHE -> Resize -> Binarize.")
                    gray_plate = cv2.cvtColor(cropped_color_plate, cv2.COLOR_BGR2GRAY)
                    
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                    enhanced_gray_plate = clahe.apply(gray_plate)

                    # Resize the enhanced grayscale image to a fixed size
                    logger.info(f"Resizing enhanced gray plate to {OCR_FIXED_WIDTH}x{OCR_FIXED_HEIGHT}.")
                    resized_enhanced_gray_plate = cv2.resize(enhanced_gray_plate, (OCR_FIXED_WIDTH, OCR_FIXED_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                    
                    logger.info("Applying Otsu's binarization to the resized enhanced grayscale image.")
                    _, binarized_plate = cv2.threshold(resized_enhanced_gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    if DEBUG_SAVE_OCR_IMAGES and ts_debug: # ts_debug is defined if this flag is true
                        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_01_cropped_color.png"), cropped_color_plate) # Renamed from _01c_ for sequence
                        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_02_gray.png"), gray_plate)
                        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_03_clahe_enhanced_gray.png"), enhanced_gray_plate)
                        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_04_resized_enhanced_gray.png"), resized_enhanced_gray_plate)
                        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_05_binarized_fixed_size.png"), binarized_plate)
                        logger.info(f"Saved intermediate OCR preprocessing images to debug dir.")

                        debug_final_ocr_input_path = os.path.join(DEBUG_IMG_DIR, "_99_final_input_for_ocr.png") 
                        cv2.imwrite(debug_final_ocr_input_path, binarized_plate) 
                        logger.info(f"Updated final input for OCR in debug dir (fixed size binarized): {debug_final_ocr_input_path}")

                    logger.info("Performing EasyOCR...")
                    try:
                        ocr_results_easyocr = easyocr_reader.readtext(binarized_plate, detail=1, paragraph=False)
                        logger.info(f"EasyOCR raw results: {ocr_results_easyocr}")

                        detected_texts = []
                        # if ocr_results_paddle and isinstance(ocr_results_paddle, list) and len(ocr_results_paddle) > 0: # REMOVED
                        #     first_item = ocr_results_paddle[0] # REMOVED
                        #     if isinstance(first_item, list) and len(first_item) > 0 and isinstance(first_item[0], tuple) and len(first_item[0]) == 2: # REMOVED
                        #          logger.warning("PaddleOCR output format might be unexpected (direct list of text/score tuples).") # REMOVED
                        #          for text_info_tuple in first_item: # REMOVED
                        #              if isinstance(text_info_tuple, tuple) and len(text_info_tuple) == 2: # REMOVED
                        #                 detected_texts.append(text_info_tuple) # REMOVED
                        #     elif isinstance(first_item, list): # REMOVED
                        #         for line_result in first_item: # REMOVED
                        #             if isinstance(line_result, list) and len(line_result) == 2: # REMOVED
                        #                 text_info = line_result[1] # REMOVED
                        #                 if isinstance(text_info, tuple) and len(text_info) == 2: # REMOVED
                        #                     detected_texts.append(text_info) # REMOVED
                        
                        # EasyOCR returns a list of (bbox, text, confidence)
                        if ocr_results_easyocr:
                            for (bbox, text, prob) in ocr_results_easyocr:
                                detected_texts.append((text, prob)) # Storing as (text, confidence)

                        if detected_texts:
                            detected_texts.sort(key=lambda x: x[1], reverse=True) # Sort by confidence
                            raw_ocr_text_for_response = detected_texts[0][0]
                            ocr_confidence_easyocr = detected_texts[0][1] # UPDATED
                            ocr_text = "".join(c for c in raw_ocr_text_for_response.strip().upper().replace(" ", "") if c.isalnum() or c == '-')
                            logger.info(f"EasyOCR Raw: '{raw_ocr_text_for_response}', Cleaned: '{ocr_text}', EasyOCR Conf: {ocr_confidence_easyocr:.4f}") # UPDATED
                            if re.match(LP_FORMAT_REGEX, ocr_text):
                                format_valid = True
                                logger.info(f"EasyOCR text '{ocr_text}' matches format.") # UPDATED
                            else:
                                logger.info(f"EasyOCR text '{ocr_text}' does NOT match format.") # UPDATED
                                message = f"EasyOCR text '{ocr_text}' (raw: '{raw_ocr_text_for_response}') no format match." # UPDATED
                        else:
                            raw_ocr_text_for_response = "NO_TEXT_DETECTED_EASYOCR" # UPDATED
                            ocr_text = "NO_TEXT_DETECTED_EASYOCR" # UPDATED
                            message = "EasyOCR detected no text on the plate." # UPDATED
                            logger.info(message)
                    except Exception as ocr_e_easyocr: # UPDATED
                        error_type = type(ocr_e_easyocr).__name__
                        error_msg = str(ocr_e_easyocr)
                        logger.error(f"Error during EasyOCR processing ({error_type}): {error_msg}", exc_info=True) # UPDATED
                        message = f"EasyOCR processing error: {error_type} - {error_msg}" # UPDATED
                        ocr_text = "EASYOCR_ERROR" # UPDATED
            else:
                message = f"YOLO confidence ({yolo_confidence:.2f}) too low for OCR."
                logger.info(message)
        else:
            if not (results_yolo and results_yolo[0].masks is not None and len(results_yolo[0].masks.xy) > 0):
                 message = "No objects (masks) detected by YOLO model."
            logger.info(f"YOLO processing outcome: {message}")

        logger.info(f"Message before final status check: {message}")
        if yolo_confidence >= CONFIDENCE_THRESHOLD and format_valid:
            best_status = "ok"
            if not message.startswith("EasyOCR text") and not message.startswith("License plate detected:"): # UPDATED
                 message = f"License plate detected: {ocr_text}"
        elif yolo_confidence >= YOLO_CONF_FOR_OCR and ocr_text not in ["N/A", "EASYOCR_ERROR", "NO_TEXT_DETECTED_EASYOCR"] and not format_valid: # UPDATED
             if not message.startswith("EasyOCR text") : # UPDATED
                message = f"Detected, but OCR invalid. YOLO:{yolo_confidence:.2f}, OCR:'{ocr_text}'(Raw:'{raw_ocr_text_for_response}')"
        
        logger.info(f"Final status: {best_status}, Final message: {message}")
        return JSONResponse(content={
            "status": best_status, "message": message,
            "yolo_confidence": round(yolo_confidence, 4),
            "ocr_text_raw": raw_ocr_text_for_response.strip() if isinstance(raw_ocr_text_for_response, str) else raw_ocr_text_for_response,
            "ocr_text_cleaned": ocr_text, "ocr_format_valid": format_valid
        }, status_code=200)

    except Exception as e_capture:
        import traceback # Add this import locally for debugging
        logger.error(f"!!! RAW EXCEPTION IN CAPTURE: {type(e_capture).__name__} - {str(e_capture)} !!!")
        logger.error("!!! TRACEBACK START !!!")
        traceback.print_exc() # This will print the full traceback to stderr
        logger.error("!!! TRACEBACK END !!!")
        
        logger.error(f"Capture/prediction error (logged before raising HTTPException): {e_capture}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Capture error: {type(e_capture).__name__} - {str(e_capture)}")


# --- Serve the main HTML page ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return FileResponse("static/index.html", media_type="text/html")

# --- Lifespan for startup and shutdown ---
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code here will run before the application starts
    logger.info("Application startup: Initializing resources...")
    # Potentially move model loading or other initializations here if desired,
    # but current global initialization outside lifespan is also fine for FastAPI.
    yield
    # Code here will run on shutdown
    logger.info("Shutting down. Closing peer connection.")
    await close_current_connection()
    logger.info("Peer connection closed during shutdown.")

app.router.lifespan_context = lifespan # Assign lifespan context manager


# --- Cleanup on shutdown --- # REMOVED OLD SHUTDOWN EVENT
# @app.on_event("shutdown") # REMOVED
# async def on_shutdown(): # REMOVED
#     logger.info("Shutting down. Closing peer connection.") # REMOVED
#     await close_current_connection() # REMOVED
#     logger.info("Peer connection closed.") # REMOVED


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", # CHANGED to import string
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    ) 