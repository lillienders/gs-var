import os
from PIL import Image

def make_gif_from_folder(folder_path, output_path='output.gif', duration=500, loop=0):
    # Get list of image files (sorted)
    image_files = sorted([
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        print("No images found in the folder.")
        return

    # Open images and convert to RGB (to ensure consistency)
    images = [Image.open(img).convert('RGB') for img in image_files]

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF saved as {output_path}")
