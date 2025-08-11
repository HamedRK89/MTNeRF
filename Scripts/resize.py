import os
from PIL import Image

input_folder = '.'
output_folder = '..\images_4'

os.makedirs(output_folder, exist_ok=True)

scale_factor = 4

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)

    if not filename.lower().endswith('.png', 'jpg', 'jpeg, .tiff'):
        continue

    try:
        with open(input_path) as img:
            width, height = img.size

            new_size = (width * scale_factor, height * scale_factor)

            resized_img = img.resize(new_size, filename)

            output_path = os.path.join(output_folder, filename)

            resized_img.save(output_path)

            print(f"Processed: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("All images processed successfully!")