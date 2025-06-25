import os
import re
import argparse
from datetime import datetime
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def find_image_files(folder_path: str, pattern: str):
    """
    Finds image files in a folder and extracts dates based on a regex pattern.
    
    Args:
        folder_path (str): Path to the folder containing images.
        pattern (str): Regex pattern to match filenames and extract date parts.
                       It should have named groups 'year', 'month', and 'day'.
                       
    Returns:
        dict: A dictionary mapping datetime.date objects to file paths.
    """
    date_to_file_map = {}
    date_pattern = re.compile(pattern)
    
    print(f"Scanning folder: {folder_path}")
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return date_to_file_map
        
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            match = date_pattern.match(filename)
            if match:
                parts = match.groupdict()
                try:
                    year = int(parts['year'])
                    month = int(parts['month'])
                    day = int(parts['day'])
                    date_obj = datetime(year, month, day).date()
                    date_to_file_map[date_obj] = os.path.join(folder_path, filename)
                except (ValueError, KeyError) as e:
                    print(f"Warning: Could not parse date from '{filename}'. Error: {e}")
    print(f"Found {len(date_to_file_map)} valid image files.")
    return date_to_file_map

def plot_comparison(date, original_path, reconstructed_path, output_dir):
    """
    Reads two raster images and plots them side-by-side.
    
    Args:
        date (datetime.date): The date of the images.
        original_path (str): Path to the original LST image.
        reconstructed_path (str): Path to the reconstructed LST image.
        output_dir (str): Directory to save the output plot.
    """
    try:
        with rasterio.open(original_path) as src_orig:
            original_data = src_orig.read(1)
            # Use the nodata value from the file, if it exists
            nodata_val_orig = src_orig.nodata
            if nodata_val_orig is not None:
                original_data[original_data == nodata_val_orig] = np.nan
            
        with rasterio.open(reconstructed_path) as src_recon:
            reconstructed_data = src_recon.read(1)
            nodata_val_recon = src_recon.nodata
            if nodata_val_recon is not None:
                 reconstructed_data[reconstructed_data == nodata_val_recon] = np.nan

        # Determine a common color scale, ignoring NaNs
        valid_values = np.concatenate([
            original_data[~np.isnan(original_data)].flatten(),
            reconstructed_data[~np.isnan(reconstructed_data)].flatten()
        ])
        
        if len(valid_values) == 0:
            print(f"Skipping {date.strftime('%Y-%m-%d')}: Both images are empty or all NaNs.")
            return

        vmin = np.percentile(valid_values, 2)
        vmax = np.percentile(valid_values, 98)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        # fig.suptitle(f"LST Comparison for {date.strftime('%Y-%m-%d')}", fontsize=16)
        
        # Plot Original LST
        im1 = axes[0].imshow(original_data, cmap='YlOrRd', vmin=vmin, vmax=vmax)
        # axes[0].set_title("Original LST")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Plot Reconstructed LST
        im2 = axes[1].imshow(reconstructed_data, cmap='YlOrRd', vmin=vmin, vmax=vmax)
        # axes[1].set_title("Reconstructed LST")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        # Add a shared colorbar
        # fig.colorbar(im2, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.04)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
        
        output_filename = os.path.join(output_dir, f"comparison_{date.strftime('%Y%m%d')}.png")
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"Error processing date {date.strftime('%Y-%m-%d')}: {e}")

def main():
    """Main function to orchestrate the comparison plotting."""
    parser = argparse.ArgumentParser(description="Plot side-by-side comparisons of original and reconstructed LST images.")
    parser.add_argument("--original_folder", required=True, help="Path to the folder with original LST images (e.g., 'lst16days_YYYY-MM-DD.tif').")
    parser.add_argument("--reconstructed_folder", required=True, help="Path to the folder with reconstructed LST images (e.g., 'LST_RECON_YYYYMMDD.tif').")
    parser.add_argument("--output_folder", required=True, help="Path to the folder where comparison plots will be saved.")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # --- Define regex patterns for date extraction ---
    # Original LST folder pattern
    original_pattern = r"lst16days_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\.tif"
    
    # Reconstructed LST folder patterns (try a few common ones)
    recon_patterns = [
        r"LST_RECON_TEST_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\.tif", # From the latest script change
        r"LST_RECON_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\.tif",      # As mentioned by user
    ]

    # Find files and their dates
    original_files = find_image_files(args.original_folder, original_pattern)
    
    reconstructed_files = {}
    for pattern in recon_patterns:
        reconstructed_files = find_image_files(args.reconstructed_folder, pattern)
        if reconstructed_files:
            print(f"Used pattern '{pattern}' for reconstructed files.")
            break
            
    if not reconstructed_files:
        print(f"Could not find any reconstructed files in '{args.reconstructed_folder}' with the expected patterns.")
        return

    # Find common dates
    common_dates = sorted(list(set(original_files.keys()) & set(reconstructed_files.keys())))
    
    if not common_dates:
        print("\nNo common dates found between the two folders. Please check filenames and paths.")
        return

    print(f"\nFound {len(common_dates)} common dates. Starting plot generation...")
    
    # Process each common date
    for date in tqdm(common_dates, desc="Generating Comparison Plots"):
        plot_comparison(
            date,
            original_files[date],
            reconstructed_files[date],
            args.output_folder
        )
        
    print(f"\nProcessing complete. Plots are saved in: {args.output_folder}")

if __name__ == "__main__":
    main() 