import cv2
import numpy as np
from ultralytics import YOLO
import torch # 用於檢查 CUDA

def run_webcam_segmentation():
    # --- 設定區 ---
    MODEL_PATH = "best.pt"  # *** 請修改為您最佳模型的路徑 ***
    CONFIDENCE_THRESHOLD = 0.25  # 物件偵測的信心度閾值
    OVERLAY_COLOR = (0, 255, 0)  # BGR 格式的顏色，例如綠色
    ALPHA = 0.4  # 疊加顏色的透明度 (0.0 完全透明, 1.0 完全不透明)
    # ---------------

    # 檢查 CUDA 是否可用，並相應地設定裝置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 載入 YOLOv8 分割模型
    try:
        model = YOLO(MODEL_PATH)
        model.to(device) # 將模型移動到指定裝置
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model path '{MODEL_PATH}' is correct and the model file is valid.")
        return

    # 開啟網路攝影機
    cap = cv2.VideoCapture(1)  # 0 代表預設攝影機，如果有多個攝影機，可以嘗試 1, 2, ...

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # 對當前幀執行預測
        # verbose=False 避免在控制台打印過多預測訊息
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)

        overlay_frame = frame.copy() # 建立一個副本用於繪製疊加效果

        if results and results[0].masks is not None:
            # results[0].masks.data 包含了所有偵測到的遮罩張量 (torch.Tensor)
            # results[0].masks.xy 包含了每個遮罩的正規化多邊形頂點 (list of numpy arrays)
            # results[0].boxes.cls 包含了每個偵測到的物件的類別索引
            # results[0].boxes.conf 包含了每個偵測到的物件的信心度

            for i, mask_data in enumerate(results[0].masks.data):
                # 目前我們只有一個類別 "license_plate" (索引為 0)
                class_id = int(results[0].boxes.cls[i]) 
                confidence = float(results[0].boxes.conf[i])

                # 將正規化的遮罩轉換為原始影像尺寸的二值遮罩
                mask_resized = cv2.resize(mask_data.cpu().numpy(), (frame.shape[1], frame.shape[0]))
                binary_mask = (mask_resized > 0.5).astype(np.uint8) # 二值化，大於0.5的部分視為遮罩

                # 建立顏色疊加層
                color_overlay = np.zeros_like(frame, dtype=np.uint8)
                color_overlay[binary_mask == 1] = OVERLAY_COLOR

                # 將顏色疊加層與原始幀混合
                cv2.addWeighted(color_overlay, ALPHA, overlay_frame, 1 - ALPHA, 0, overlay_frame)

                # 繪製邊界框和類別標籤 
                boxes = results[0].boxes.xyxy.cpu().numpy()
                x1, y1, x2, y2 = map(int, boxes[i])
                # cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), OVERLAY_COLOR, 2) 
                
                label_text = f"{model.names[class_id]}: {confidence:.2f}"
                
                # 計算文字放置位置
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20 

                cv2.putText(overlay_frame, label_text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, OVERLAY_COLOR, 2)

        # 顯示結果幀
        cv2.imshow("Webcam Segmentation Overlay", overlay_frame)

        # 按 'q' 鍵退出迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed and resources released.")

if __name__ == "__main__":
    run_webcam_segmentation() 