import os
import shutil

def organize_labels():
    """
    Organizes label files from a central 'dataset/labels/' directory into
    'dataset/labels/train/', 'dataset/labels/val/', and 'dataset/labels/test/'
    based on corresponding image files in 'dataset/train/', 'dataset/val/',
    and 'dataset/test/' respectively.

    Assumes label files are already named correctly (e.g., 'image_name.txt').
    """
    base_dir = "dataset"
    source_labels_main_dir = os.path.join(base_dir, "labels")

    image_sets = ["train", "val", "test"]
    image_extensions = {".jpg", ".jpeg", ".png"} # Common image extensions

    moved_count = 0
    missing_labels_count = 0
    already_in_place_count = 0
    
    print("Starting label organization...")

    if not os.path.isdir(source_labels_main_dir):
        print(f"Error: Main labels directory '{source_labels_main_dir}' not found. Nothing to organize.")
        return

    for img_set in image_sets:
        image_dir = os.path.join(base_dir, img_set)
        target_label_dir = os.path.join(source_labels_main_dir, img_set)

        if not os.path.isdir(image_dir):
            print(f"Info: Image directory '{image_dir}' for set '{img_set}' not found. Skipping this set.")
            continue

        os.makedirs(target_label_dir, exist_ok=True)
        print(f"Processing set: '{img_set}'. Image source: '{image_dir}', Label target: '{target_label_dir}'")

        for image_filename in os.listdir(image_dir):
            image_name, image_ext = os.path.splitext(image_filename)
            if image_ext.lower() in image_extensions:
                label_filename = image_name + ".txt"
                source_label_path = os.path.join(source_labels_main_dir, label_filename)
                target_label_path = os.path.join(target_label_dir, label_filename)

                if os.path.exists(target_label_path):
                    # print(f"Label file '{label_filename}' already in target '{target_label_dir}'. Skipping.")
                    already_in_place_count += 1
                    # If it's already in the target, check if a stray one exists in the main dir and remove it
                    if os.path.exists(source_label_path) and source_label_path != target_label_path :
                         try:
                            os.remove(source_label_path)
                            print(f"Removed duplicate label from main labels dir: '{source_label_path}' as it's already in '{target_label_dir}'.")
                         except OSError as e:
                            print(f"Error removing duplicate label '{source_label_path}': {e}")
                    continue


                if os.path.exists(source_label_path):
                    try:
                        shutil.move(source_label_path, target_label_path)
                        print(f"Moved: '{source_label_path}' -> '{target_label_path}'")
                        moved_count += 1
                    except Exception as e:
                        print(f"Error moving '{source_label_path}' to '{target_label_path}': {e}")
                else:
                    # It's possible the label is already in its correct subfolder from a previous run,
                    # and no longer in the root 'dataset/labels/' folder.
                    # This specific message is for when it's not in source_labels_main_dir AND not in target_label_path yet.
                    # The already_in_place_count handles if it's already in target_label_path.
                    if not os.path.exists(target_label_path): # Re-check to avoid false alarms if moved by another process or already handled.
                        print(f"Warning: Label file '{label_filename}' (expected at '{source_label_path}') not found for image '{image_filename}'.")
                        missing_labels_count += 1
            
    print("\n--- Organization Summary ---")
    print(f"Successfully moved: {moved_count} label file(s).")
    print(f"Label files already in correct subdirectories: {already_in_place_count} file(s).")
    print(f"Missing label files (source .txt not found for an image): {missing_labels_count} file(s).")
    
    # Optional: Check if source_labels_main_dir still contains any .txt files (excluding subdirs)
    remaining_files_in_source = [f for f in os.listdir(source_labels_main_dir) if os.path.isfile(os.path.join(source_labels_main_dir, f)) and f.endswith(".txt")]
    if remaining_files_in_source:
        print(f"Info: The main label directory '{source_labels_main_dir}' still contains {len(remaining_files_in_source)} .txt files.")
        print("These might be unmatched labels or require manual review:")
        for f_rem in remaining_files_in_source[:5]: # Print a few examples
            print(f"  - {f_rem}")
        if len(remaining_files_in_source) > 5:
            print(f"  ... and {len(remaining_files_in_source) - 5} more.")
    else:
        print(f"Info: The main label directory '{source_labels_main_dir}' (excluding subdirectories) is now clear of .txt files as expected.")


if __name__ == "__main__":
    organize_labels() 