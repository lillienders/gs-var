import os
from PIL import Image
import netCDF4 as nc
import numpy as np
import xarray as xr
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd 
import seaborn as sns
import cmocean as cmo
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as ticker

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
