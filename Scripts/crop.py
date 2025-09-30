import cv2
import os
from pathlib import Path

input_folder = "../Sub_Results/4paper/toys"    
output_folder = "../Sub_Results/4paper/toys/out"

# Coordinates of ROI (x, y, w, h)
roi = (1950, 950, 300, 300)  # example: rectangle at (100,150), width=200, height=200

# Create output folders
Path(output_folder).mkdir(parents=True, exist_ok=True)
(Path(output_folder) / "annotated").mkdir(exist_ok=True)
(Path(output_folder) / "crops").mkdir(exist_ok=True)

# Process each image
for file in Path(input_folder).glob("*.png"):  # change to *.png or *.* if needed
    img = cv2.imread(str(file))
    if img is None:
        continue

    # Unpack ROI
    x, y, w, h = roi

    # Draw red rectangle on a copy
    annotated = img.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Crop ROI
    crop = img[y:y + h, x:x + w]

    # Save outputs
    base = file.stem
    cv2.imwrite(str(Path(output_folder) / "annotated" / f"{base}_annotated.jpg"), annotated)
    cv2.imwrite(str(Path(output_folder) / "crops" / f"{base}_crop.jpg"), crop)

print("Processing complete!")
