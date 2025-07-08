import os
import subprocess
from pathlib import Path
import argparse

def convert_yuv_to_png(input_folder, output_folder, width, height):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    yuv_files = sorted(input_path.glob("*.yuv"))

    for yuv_file in yuv_files:
        output_file = output_path / (yuv_file.stem + ".jpg")
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'gray16le',  # 4:0:0 16-bit (Y only)
            '-s', f'{width}x{height}',
            '-i', str(yuv_file),
            '-frames:v', '1',
            str(output_file)
        ]

        print(f"Converting {yuv_file.name} -> {output_file.name}")
        subprocess.run(ffmpeg_cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YUV 400 16bit depth maps to PNG.")
    parser.add_argument('--input', required=True, help='Input folder with .yuv files')
    parser.add_argument('--output', required=True, help='Output folder for .png files')
    parser.add_argument('--width', type=int, default=1920, required=False, help='Width of the images')
    parser.add_argument('--height', type=int, default=1080, required=False, help='Height of the images')

    args = parser.parse_args()
    convert_yuv_to_png(args.input, args.output, args.width, args.height)
