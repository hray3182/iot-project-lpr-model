import os

def validate_labels():
    base_label_dir = os.path.join("dataset", "labels")
    label_sets_to_check = ["train", "val"] # Add "test" if you have it and want to check

    print("Starting detailed label content validation...")
    total_files_checked = 0
    total_errors_found = 0
    error_details = []

    for label_set in label_sets_to_check:
        current_label_dir = os.path.join(base_label_dir, label_set)
        print(f"\nValidating labels in: '{current_label_dir}'")

        if not os.path.isdir(current_label_dir):
            print(f"Warning: Label directory '{current_label_dir}' not found. Skipping.")
            continue

        for filename in os.listdir(current_label_dir):
            if filename.endswith(".txt"):
                total_files_checked += 1
                filepath = os.path.join(current_label_dir, filename)
                # print(f"Checking file: {filepath}")
                try:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                        if not lines: # Check for empty files
                            error_msg = f"Empty label file."
                            error_details.append(f"ERROR: File '{filepath}' - {error_msg}")
                            total_errors_found +=1
                            continue
                            
                        for i, line in enumerate(lines):
                            line_number = i + 1
                            stripped_line = line.strip()

                            if not stripped_line: # Skip empty lines within a file silently, or report
                                # error_msg = f"Line {line_number}: Empty line."
                                # error_details.append(f"ERROR: File '{filepath}' - {error_msg}")
                                # total_errors_found +=1
                                continue 

                            parts = stripped_line.split()
                            
                            # Check class index
                            try:
                                class_idx = int(parts[0])
                                if class_idx != 0:
                                    error_msg = f"Line {line_number}: Class index '{class_idx}' is not 0."
                                    error_details.append(f"ERROR: File '{filepath}' - {error_msg}")
                                    total_errors_found +=1
                            except (ValueError, IndexError) as e:
                                error_msg = f"Line {line_number}: Invalid class index or malformed line. Details: {e}"
                                error_details.append(f"ERROR: File '{filepath}' - {error_msg}")
                                total_errors_found +=1
                                continue # Stop checking this line if class index is bad

                            # Check number of coordinates
                            if len(parts) < 1 + 6: # class_idx + at least 3 points (6 coords)
                                error_msg = f"Line {line_number}: Insufficient coordinates. Expected at least 6 (for 3 points), got {len(parts) - 1}."
                                error_details.append(f"ERROR: File '{filepath}' - {error_msg}")
                                total_errors_found +=1
                                continue
                            
                            if (len(parts) - 1) % 2 != 0:
                                error_msg = f"Line {line_number}: Odd number of coordinates ({len(parts) - 1}). Each point needs an x and y."
                                error_details.append(f"ERROR: File '{filepath}' - {error_msg}")
                                total_errors_found +=1
                                continue

                            # Check coordinate values (range and type)
                            coords_valid = True
                            for j in range(1, len(parts)):
                                try:
                                    coord_val = float(parts[j])
                                    if not (0.0 <= coord_val <= 1.0):
                                        error_msg = f"Line {line_number}: Coordinate '{coord_val}' (at index {j-1}) is out of [0.0, 1.0] range."
                                        error_details.append(f"ERROR: File '{filepath}' - {error_msg}")
                                        total_errors_found +=1
                                        coords_valid = False
                                        break # Stop checking coords for this line
                                except ValueError:
                                    error_msg = f"Line {line_number}: Coordinate '{parts[j]}' (at index {j-1}) is not a valid float."
                                    error_details.append(f"ERROR: File '{filepath}' - {error_msg}")
                                    total_errors_found +=1
                                    coords_valid = False
                                    break # Stop checking coords for this line
                except Exception as e:
                    error_msg = f"Could not process file. Details: {e}"
                    error_details.append(f"ERROR: File '{filepath}' - {error_msg}")
                    total_errors_found +=1
    
    print("\n--- Validation Summary ---")
    print(f"Total .txt files checked: {total_files_checked}")
    print(f"Total errors found: {total_errors_found}")
    if error_details:
        print("\nError Details (first 50 errors):")
        for i, detail in enumerate(error_details):
            if i < 50:
                print(detail)
            else:
                print(f"... and {len(error_details) - 50} more error(s).")
                break
    elif total_files_checked > 0:
        print("All checked label files appear to be valid based on the rules applied!")
    elif total_files_checked == 0:
        print("No label files were found to validate in the specified directories.")

if __name__ == "__main__":
    validate_labels() 