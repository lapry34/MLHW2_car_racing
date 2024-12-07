import os
import sys
from PIL import Image
import numpy as np

verbose = True

def unify_shade_near_target(image, target_shade=(99, 202, 100), target_green=(0, 255, 0), tolerance=50):
    img_array = np.array(image)
    r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
    target_r, target_g, target_b = target_shade
    mask = (
        (abs(r - target_r) <= tolerance) &
        (abs(g - target_g) <= tolerance) &
        (abs(b - target_b) <= tolerance)
    )
    img_array[mask] = target_green
    return Image.fromarray(img_array)

def convert_IMG_numpy(img_array, target_shade=(99, 202, 100), target_green=(0, 255, 0), tolerance=50):
    r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
    target_r, target_g, target_b = target_shade
    mask = (
        (abs(r - target_r) <= tolerance) &
        (abs(g - target_g) <= tolerance) &
        (abs(b - target_b) <= tolerance)
    )
    img_array[mask] = target_green
    return img_array

def process_images_in_directory(input_path, output_path, target_shade, target_green, tolerance):
    if not os.path.exists(input_path):
        if verbose:
            print(f"Directory does not exist: {input_path}")
        return
    os.makedirs(output_path, exist_ok=True)
    for image_name in os.listdir(input_path):
        if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            input_image_path = os.path.join(input_path, image_name)
            output_image_path = os.path.join(output_path, image_name)
            img = Image.open(input_image_path).convert("RGB")
            processed_img = unify_shade_near_target(img, target_shade=target_shade, target_green=target_green, tolerance=tolerance)
            processed_img.save(output_image_path)
            if verbose:
                print(f"Processed: {output_image_path}")

def process_images(raw_dir, output_dir, target_shade, target_green, tolerance):
    for split in ["train", "test"]:
        for label in range(5):
            input_path = os.path.join(raw_dir, split, str(label))
            output_path = os.path.join(output_dir, split, str(label))
            if verbose:
                print(f"Processing: {input_path}")
            process_images_in_directory(input_path, output_path, target_shade, target_green, tolerance)

if __name__ == "__main__":
    raw_dataset_dir = "raw_dataset"
    output_dataset_dir = "dataset"
    target_shade_to_match = (99, 202, 100)
    unified_green = (0, 255, 0)
    color_tolerance = 50
    process_images(raw_dataset_dir, output_dataset_dir, target_shade_to_match, unified_green, color_tolerance)

    img = Image.open("dataset/train/0/0000.png")
    print(f"Sample Image Size: {img.size}")
    sys.exit(0)