import os
import math

def get_polygon_area(vertices):
    """Calculates the area of a polygon using the Shoelace formula.
    Vertices is a list of (x, y) tuples.
    Returns the absolute area.
    """
    if len(vertices) < 3:
        return 0.0  # Not a polygon
    area = 0.0
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0

def check_duplicate_consecutive_vertices(vertices):
    duplicates = []
    if len(vertices) < 2:
        return duplicates
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)] # Compare with the next, wrap around for the last one
        # Using a small epsilon for float comparison, though exact match is often what LabelStudio might output if duplicated
        if math.isclose(p1[0], p2[0]) and math.isclose(p1[1], p2[1]):
            duplicates.append((p1, p2, i, (i+1)%len(vertices)))
    return duplicates

def inspect_labels():
    base_label_dir = os.path.join("dataset", "labels")
    label_sets_to_check = ["train", "val"]
    area_threshold = 1e-5 # Threshold for considering an area as "very small"

    print("Starting polygon properties inspection...")
    total_polygons_inspected = 0
    potential_issues = []

    for label_set in label_sets_to_check:
        current_label_dir = os.path.join(base_label_dir, label_set)
        print(f"\nInspecting labels in: '{current_label_dir}'")

        if not os.path.isdir(current_label_dir):
            print(f"Warning: Label directory '{current_label_dir}' not found. Skipping.")
            continue

        for filename in os.listdir(current_label_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(current_label_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                        if not lines:
                            potential_issues.append(f"INFO: File '{filepath}' is empty.")
                            continue
                        
                        for i, line in enumerate(lines):
                            line_number = i + 1
                            stripped_line = line.strip()
                            if not stripped_line:
                                continue

                            parts = stripped_line.split()
                            class_idx_str = parts[0]
                            coords_str = parts[1:]

                            if not coords_str or len(coords_str) < 6 or len(coords_str) % 2 != 0:
                                # This should have been caught by validate_segmentation_labels.py
                                # but good to have a basic check here too.
                                potential_issues.append(f"WARN: File '{filepath}', Line {line_number}: Malformed coordinates (num: {len(coords_str)}). Skipping line.")
                                continue
                            
                            total_polygons_inspected += 1
                            vertices = []
                            valid_coords = True
                            for k in range(0, len(coords_str), 2):
                                try:
                                    x = float(coords_str[k])
                                    y = float(coords_str[k+1])
                                    # Basic validation, though validate_segmentation_labels.py is more thorough
                                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                                        potential_issues.append(f"WARN: File '{filepath}', Line {line_number}: Coord out of [0,1] range. ({x},{y}). Skipping line for property check.")
                                        valid_coords = False
                                        break
                                    vertices.append((x, y))
                                except ValueError:
                                    potential_issues.append(f"WARN: File '{filepath}', Line {line_number}: Non-float coordinate. Skipping line for property check.")
                                    valid_coords = False
                                    break
                            
                            if not valid_coords:
                                continue

                            num_vertices = len(vertices)
                            # print(f"  File: {filename}, Line: {line_number}, Vertices: {num_vertices}") # Verbose

                            # 1. Check for duplicate consecutive vertices
                            duplicate_verts = check_duplicate_consecutive_vertices(vertices)
                            if duplicate_verts:
                                issue_detail = f"Duplicate consecutive vertices: {len(duplicate_verts)} pair(s). Example: {duplicate_verts[0][0]} and {duplicate_verts[0][1]} at indices {duplicate_verts[0][2]}-{duplicate_verts[0][3]}"
                                potential_issues.append(f"ISSUE: File '{filepath}', Line {line_number} (Class {class_idx_str}, {num_vertices} verts) - {issue_detail}")

                            # 2. Calculate normalized area
                            area = get_polygon_area(vertices)
                            if area < area_threshold:
                                issue_detail = f"Area is very small: {area:.8f}. (Threshold: {area_threshold})"
                                potential_issues.append(f"ISSUE: File '{filepath}', Line {line_number} (Class {class_idx_str}, {num_vertices} verts) - {issue_detail}")
                            
                            # 3. (Optional) Basic Co-linearity check for 3 points (triangles with near-zero area)
                            if num_vertices == 3 and area < area_threshold:
                                issue_detail = f"Triangle with very small area ({area:.8f}), points might be co-linear."
                                potential_issues.append(f"INFO: File '{filepath}', Line {line_number} (Class {class_idx_str}, {num_vertices} verts) - {issue_detail}")

                except Exception as e:
                    potential_issues.append(f"CRITICAL_ERROR: Could not process file '{filepath}'. Details: {e}")

    print("\n--- Polygon Inspection Summary ---")
    print(f"Total polygons inspected (from valid lines): {total_polygons_inspected}")
    if potential_issues:
        print(f"Found {len(potential_issues)} potential issues/observations:")
        for i, detail in enumerate(potential_issues):
            if i < 100: # Print more issues if needed
                print(detail)
            else:
                print(f"... and {len(potential_issues) - 100} more issue(s).")
                break
    else:
        print("No significant issues (like very small area or duplicate consecutive vertices) found in inspected polygons.")

if __name__ == "__main__":
    inspect_labels() 