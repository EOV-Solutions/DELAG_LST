import os
import argparse
import glob
import shutil
import re
from datetime import datetime
from collections import defaultdict
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm

def count_nan_in_image(file_path: str) -> int:
    """Counts the number of NaN values in a GeoTIFF file."""
    try:
        with rasterio.open(file_path) as src:
            # Assuming LST data is in the first band
            band1 = src.read(1)
            nan_count = np.isnan(band1).sum()
            return nan_count
    except RasterioIOError as e:
        print(f"Error reading {os.path.basename(file_path)}: {e}. Skipping file.")
        return float('inf') # Return infinity to ensure this file is not chosen
    except Exception as e:
        print(f"An unexpected error occurred with {os.path.basename(file_path)}: {e}. Skipping file.")
        return float('inf')

def find_best_image_for_day(file_paths: list[str]) -> str | None:
    """
    Given a list of image paths for the same day, chooses the one with the fewest NaN values.

    Args:
        file_paths: A list of file paths for images from the same day.

    Returns:
        The path to the best image, or None if no valid image is found.
    """
    if not file_paths:
        return None
    if len(file_paths) == 1:
        return file_paths[0]

    min_nan_count = float('inf')
    best_file = None

    for f_path in file_paths:
        nan_count = count_nan_in_image(f_path)
        if nan_count < min_nan_count:
            min_nan_count = nan_count
            best_file = f_path
            
    if best_file:
        print(f"  > Selected '{os.path.basename(best_file)}' (NaNs: {min_nan_count}) from {len(file_paths)} candidates.")
    else:
        print(f"  > No valid image found among candidates.")

    return best_file

def process_lst_folder(input_folder: str, output_folder: str):
    """
    Processes LST files from an input folder, renames them, and saves them to an output folder.
    For days with multiple images, it selects the one with the fewest NaN values.
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"Scanning for .tif files in: {input_folder}")
    
    # Regex to find YYYYMMDD date in the specified filename format
    # Example: 1_LC09_124051_20211211_LST.tif
    date_pattern = re.compile(r'_(\d{8})_LST\.tif$')
    
    tif_files = glob.glob(os.path.join(input_folder, '*.tif'))
    if not tif_files:
        print("No .tif files found in the input folder.")
        return

    # Group files by date
    images_by_date = defaultdict(list)
    for file_path in tif_files:
        match = date_pattern.search(os.path.basename(file_path))
        if match:
            date_str = match.group(1)
            try:
                file_date = datetime.strptime(date_str, "%Y%m%d")
                images_by_date[file_date].append(file_path)
            except ValueError:
                print(f"Could not parse date from filename: {os.path.basename(file_path)}. Skipping.")
        else:
            print(f"Filename format not recognized for date extraction: {os.path.basename(file_path)}. Skipping.")

    print(f"Found {len(images_by_date)} unique dates to process.")
    
    # Process each day's images
    for date_obj, file_list in tqdm(sorted(images_by_date.items()), desc="Processing Dates"):
        # Find the best image for the day (least NaNs)
        best_image_path = find_best_image_for_day(file_list)
        
        if best_image_path:
            # Create the new filename
            new_date_str = date_obj.strftime('%Y-%m-%d')
            new_filename = f"lst16days_{new_date_str}.tif"
            output_path = os.path.join(output_folder, new_filename)
            
            # Copy and rename the file
            try:
                shutil.copy(best_image_path, output_path)
            except Exception as e:
                print(f"  > Error copying file {best_image_path} to {output_path}: {e}")

    print("\nProcessing complete.")


def main():
    """Main function to parse arguments and start processing."""
    parser = argparse.ArgumentParser(
        description="Process LST image files. This script finds the best image for each day "
                    "(based on the lowest NaN count), renames it to 'lst16days_YYYY-MM-DD.tif', "
                    "and saves it to a new folder."
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help="Path to the folder containing the original LST .tif files."
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help="Path to the folder where the processed images will be saved."
    )
    
    args = parser.parse_args()
    
    process_lst_folder(args.input_folder, args.output_folder)

if __name__ == '__main__':
    main() 