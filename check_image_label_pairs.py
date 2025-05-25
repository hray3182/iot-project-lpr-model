import os

def check_pairs():
    base_dir = "dataset"
    image_sets = ["train", "val", "test"]
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"} # Common image extensions
    label_extension = ".txt"

    print("Starting image-label pair check...")
    all_missing_labels_for_images = []
    all_orphan_labels = []

    for img_set in image_sets:
        image_dir = os.path.join(base_dir, img_set)
        # Corresponding label dir is dataset/labels/[train|val|test]
        label_dir = os.path.join(base_dir, "labels", img_set)

        print(f"\nChecking set: '{img_set}'")
        print(f"Image directory: '{image_dir}'")
        print(f"Label directory: '{label_dir}'")

        if not os.path.isdir(image_dir):
            print(f"Warning: Image directory '{image_dir}' not found. Skipping this set.")
            continue
        
        if not os.path.isdir(label_dir):
            print(f"Warning: Label directory '{label_dir}' not found. Cannot check labels for this set.")
            # We can still check for images in image_dir if label_dir is missing, but it means all labels are missing
            # Or we can just skip. For now, let's report all images in image_dir as missing labels.
            if os.path.isdir(image_dir):
                for image_filename in os.listdir(image_dir):
                    _, img_ext = os.path.splitext(image_filename)
                    if img_ext.lower() in image_extensions:
                        all_missing_labels_for_images.append(os.path.join(image_dir, image_filename))
                        # print(f"  - Missing label for image: {os.path.join(image_dir, image_filename)} (label directory missing)")
            continue # Move to next set

        image_filenames_processed = set()

        # 1. Check if every image has a label
        missing_labels_current_set = []
        for image_filename in os.listdir(image_dir):
            img_name_no_ext, img_ext = os.path.splitext(image_filename)
            if img_ext.lower() in image_extensions:
                image_filenames_processed.add(img_name_no_ext)
                expected_label_filename = img_name_no_ext + label_extension
                expected_label_path = os.path.join(label_dir, expected_label_filename)
                if not os.path.exists(expected_label_path):
                    missing_path = os.path.join(image_dir, image_filename) # Keep full path for clarity
                    missing_labels_current_set.append(missing_path)
                    all_missing_labels_for_images.append(missing_path)
        
        if missing_labels_current_set:
            print(f"Images in '{image_dir}' missing corresponding labels in '{label_dir}':")
            for item in missing_labels_current_set:
                print(f"  - {item}")
        else:
            print(f"All images in '{image_dir}' have corresponding label files in '{label_dir}'.")

        # 2. Check if every label has an image (orphan labels)
        orphan_labels_current_set = []
        if os.path.isdir(label_dir): # Re-check as it might have been created if not initially present
            for label_filename in os.listdir(label_dir):
                if label_filename.endswith(label_extension):
                    label_name_no_ext, _ = os.path.splitext(label_filename)
                    # Check if this label_name_no_ext corresponds to any processed image file (ignoring extension)
                    found_image_for_label = False
                    for img_ext_check in image_extensions:
                        expected_image_filename = label_name_no_ext + img_ext_check
                        if os.path.exists(os.path.join(image_dir,expected_image_filename)):
                            found_image_for_label = True
                            break
                    if not found_image_for_label:
                        orphan_path = os.path.join(label_dir, label_filename)
                        orphan_labels_current_set.append(orphan_path)
                        all_orphan_labels.append(orphan_path)

        if orphan_labels_current_set:
            print(f"Orphan labels in '{label_dir}' missing corresponding images in '{image_dir}':")
            for item in orphan_labels_current_set:
                print(f"  - {item}")
        else:
            print(f"No orphan labels found in '{label_dir}'.")

    print("\n--- Overall Summary ---")
    if all_missing_labels_for_images:
        print(f"Total images missing labels: {len(all_missing_labels_for_images)}")
        print("List of images missing labels:")
        for item in all_missing_labels_for_images:
            print(f"  - {item}")
    else:
        print("All images across all sets have corresponding label files.")

    if all_orphan_labels:
        print(f"Total orphan labels: {len(all_orphan_labels)}")
        print("List of orphan labels:")
        for item in all_orphan_labels:
            print(f"  - {item}")
    else:
        print("No orphan labels found across all sets.")

if __name__ == "__main__":
    check_pairs() 