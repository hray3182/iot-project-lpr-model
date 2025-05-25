import os

def rename_label_files():
    """
    Renames label files in the 'dataset/labels/' directory.
    It changes filenames from '[random_string]__[plate_number].txt'
    to '[plate_number].txt'.
    """
    labels_dir = os.path.join("dataset", "labels")

    if not os.path.isdir(labels_dir):
        print(f"Error: Directory '{labels_dir}' not found.")
        return

    renamed_count = 0
    skipped_count = 0

    print(f"Scanning files in '{labels_dir}'...")

    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt") and "__" in filename:
            parts = filename.split("__")
            if len(parts) > 1:
                new_name_base = parts[-1].replace(".txt", "")
                # Handle cases like 'ALH-0100.2.txt' -> 'ALH-0100.2.txt' (no change if already correct)
                # or 'some_prefix__ACTUAL-NAME.txt' -> 'ACTUAL-NAME.txt'
                # If the part after __ is already the full filename base, it might indicate an issue or already processed.
                # However, the primary goal is to remove the prefix.
                
                # Special handling for names that might end with .<number>.txt like 'ALH-0100.2.txt'
                # The goal is to get 'ALH-0100.2' from '270b12db__ALH-0100.2.txt'
                # and 'BPQ-7254' from '7ef3bf02__BPQ-7254.txt'
                
                # The extraction logic should be robust. The last part after __ is the target.
                # parts[-1] gives 'ALH-0100.2.txt' or 'BPQ-7254.txt'
                # We just need to ensure we keep the .txt extension for the new name.
                new_filename = parts[-1]


                old_filepath = os.path.join(labels_dir, filename)
                new_filepath = os.path.join(labels_dir, new_filename)

                if old_filepath == new_filepath:
                    # This case might happen if the filename was, for example, "__actualname.txt"
                    # or if somehow a file is named like "prefix__actualname.txt" and "actualname.txt" is the part after __
                    # print(f"Skipping '{filename}' as the new name is the same (after potential processing).")
                    skipped_count +=1
                    continue

                if os.path.exists(new_filepath):
                    print(f"Warning: Target file '{new_filepath}' already exists. Skipping rename of '{filename}'.")
                    skipped_count += 1
                else:
                    try:
                        os.rename(old_filepath, new_filepath)
                        print(f"Renamed: '{filename}' -> '{new_filename}'")
                        renamed_count += 1
                    except OSError as e:
                        print(f"Error renaming file '{filename}': {e}")
                        skipped_count += 1
            else:
                # This case should ideally not happen if "__" is in filename
                print(f"Skipping '{filename}': Could not extract new name.")
                skipped_count += 1
        elif filename.endswith(".txt"):
            # Files that are .txt but don't contain "__" are assumed to be correctly named or irrelevant
            # print(f"Skipping '{filename}': Does not contain '__' or not a .txt file.")
            skipped_count +=1


    print("\n--- Summary ---")
    print(f"Successfully renamed: {renamed_count} file(s).")
    print(f"Skipped or already correct: {skipped_count} file(s).")

if __name__ == "__main__":
    rename_label_files() 