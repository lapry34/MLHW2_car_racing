import os
import sys
from PIL import Image

def unify_green_shade(image, target_green=(0, 255, 0)):
    """
    Unify all green shades in an image to a single shade.

    Args:
        image (PIL.Image): Input image.
        target_green (tuple): RGB values for the target green shade.

    Returns:
        PIL.Image: Processed image with unified green shades.
    """
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            r, g, b = pixels[x, y]
            if g > r and g > b:
                pixels[x, y] = target_green
    return image

def process_images(raw_dir, output_dir):
    """
    Process images in the raw dataset directory and save them in the output directory.

    Args:
        raw_dir (str): Path to the raw dataset directory.
        output_dir (str): Path to the output dataset directory.
    """
    for split in ["train", "test"]:
        for y in range(5):  # y goes from 0 to 4
            input_path = os.path.join(raw_dir, split, str(y))
            output_path = os.path.join(output_dir, split, str(y))

            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)

            # Process each image in the directory
            for image_name in os.listdir(input_path):
                if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    input_image_path = os.path.join(input_path, image_name)
                    output_image_path = os.path.join(output_path, image_name)

                    # Load and process the image
                    img = Image.open(input_image_path).convert("RGB")
                    processed_img = unify_green_shade(img)

                    # Save the processed image
                    processed_img.save(output_image_path)
                    print(f"Processed and saved: {output_image_path}")

# Example usage
raw_dataset_dir = "raw_dataset"
output_dataset_dir = "dataset"

if __name__ == "__main__":
    process_images(raw_dataset_dir, output_dataset_dir)
    sys.exit(0)