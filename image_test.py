import os
import rasterio
import numpy as np

def print_unique_values(image_path):
    """
    Print all unique values in a TIF image.
    
    Args:
        image_path (str): Path to the TIF image file
    """
    try:
        with rasterio.open(image_path) as src:
            # Get number of bands
            num_bands = src.count
            
            print(f"\nAnalyzing image: {os.path.basename(image_path)}")
            print(f"Number of bands: {num_bands}")
            
            # Process each band
            for band_idx in range(1, num_bands + 1):
                data = src.read(band_idx)
                
                # Get unique values, excluding NaN
                unique_vals = np.unique(data[~np.isnan(data)])
                
                print(f"\nBand {band_idx} unique values:")
                print(f"Count: {len(unique_vals)}")
                print(f"Values: {unique_vals}")
                
                # Print some basic statistics
                if len(unique_vals) > 0:
                    print(f"Min: {np.nanmin(data):.2f}")
                    print(f"Max: {np.nanmax(data):.2f}")
                    print(f"Mean: {np.nanmean(data):.2f}")
                    print(f"Std: {np.nanstd(data):.2f}")
                
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Example usage
if __name__ == "__main__":
    # Define the path to your data folder
    base_path = "/mnt/ssd1tb/code/nhatvm/DELAG/DELAG_LST/KhanhXuan_BuonMaThuot_DakLak/s2_images/"
    
    # Process all TIF files in the folder
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".tif"):
                image_path = os.path.join(root, file)
                print_unique_values(image_path)
