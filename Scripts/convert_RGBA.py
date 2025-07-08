from PIL import Image
import os

path = "./RGB"
destination_path = "./RGBA"

os.makedirs(destination_path, exist_ok=True)

for image in os.listdir(path):
    image_path = os.path.join(path, image)
    out_path = os.path.join(destination_path, image)
    img = Image.open(image_path).convert("RGBA")
    img.save(out_path)
    print(f"{image} Converted and saved")