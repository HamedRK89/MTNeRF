import cv2
import numpy as np
import sys
import os

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Images are identical
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def main(img_path1, img_path2, output_file):
    # Read images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None:
        print(f"Error: Could not read {img_path1}")
        return
    if img2 is None:
        print(f"Error: Could not read {img_path2}")
        return
    if img1.shape != img2.shape:
        print(f"Error: Images {img_path1} and {img_path2} must have the same dimensions")
        return

    psnr_value = calculate_psnr(img1.astype(np.float64), img2.astype(np.float64))

    # Append to results file
    with open(output_file, "a") as f:
        f.write(f"{os.path.basename(img_path1)} vs {os.path.basename(img_path2)}: {psnr_value:.2f} dB\n")

    print(f"Result appended to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <image1.png> <image2.png> <results.txt>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
