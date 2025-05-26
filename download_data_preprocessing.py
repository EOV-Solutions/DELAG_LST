import os
import rasterio
import numpy as np

def process_lst_images(lst_folder_path):
    """
    Processes LST images in the given folder.
    Replaces all 0 values in the images with np.nan and overwrites the original files.
    Assumes LST images are single-band.

    Args:
        lst_folder_path (str): The path to the folder containing LST .tif images.
    """
    print(f"Processing LST images in: {lst_folder_path}")
    processed_files = 0
    failed_files = []

    for filename in os.listdir(lst_folder_path):
        if filename.endswith(".tif"):
            file_path = os.path.join(lst_folder_path, filename)
            try:
                with rasterio.open(file_path, 'r+') as src:
                    data = src.read(1)
                    
                    current_dtype = data.dtype
                    if not np.issubdtype(current_dtype, np.floating):
                        current_dtype = np.float32 
                        data = data.astype(current_dtype)

                    data[data == 0] = np.nan
                    
                    profile = src.profile
                    profile.update(nodata=np.nan, dtype=current_dtype)
                    
                    src.write(data, 1)
                
                processed_files += 1
                print(f"  Successfully processed LST image: {filename}")

            except Exception as e:
                print(f"  Failed to process LST image {filename}: {e}")
                failed_files.append(filename)

    print(f"\nLST processing complete.")
    print(f"Total LST files processed: {processed_files}")
    if failed_files:
        print(f"Failed to process {len(failed_files)} LST files: {', '.join(failed_files)}")

def process_era5_images(era5_folder_path):
    """
    Processes ERA5 images in the given folder.
    Replaces all 0 values in each band of the images with np.nan and overwrites the original files.

    Args:
        era5_folder_path (str): The path to the folder containing ERA5 .tif images.
    """
    print(f"\nProcessing ERA5 images in: {era5_folder_path}")
    processed_files = 0
    failed_files = []

    for filename in os.listdir(era5_folder_path):
        if filename.endswith(".tif"):
            file_path = os.path.join(era5_folder_path, filename)
            try:
                with rasterio.open(file_path, 'r+') as src:
                    profile = src.profile
                    num_bands = src.count
                    
                    output_dtype = np.float32 
                    
                    temp_check_data = src.read(1) # Check dtype of first band
                    if not np.issubdtype(temp_check_data.dtype, np.floating):
                        print(f"  Note: Data in ERA5 image {filename} will be converted to {output_dtype} to support np.nan.")
                    
                    profile.update(nodata=np.nan, dtype=output_dtype)
                    
                    for bidx in range(1, num_bands + 1):
                        data = src.read(bidx)
                        
                        if not np.issubdtype(data.dtype, np.floating):
                            data = data.astype(output_dtype)
                        
                        data[data == 0] = np.nan # ERA5 specific value to replace
                        src.write(data, bidx) 
                
                processed_files += 1
                print(f"  Successfully processed ERA5 image: {filename} ({num_bands} bands)")

            except Exception as e:
                print(f"  Failed to process ERA5 image {filename}: {e}")
                failed_files.append(filename)

    print(f"\nERA5 processing complete.")
    print(f"Total ERA5 files processed: {processed_files}")
    if failed_files:
        print(f"Failed to process {len(failed_files)} ERA5 files: {', '.join(failed_files)}")

def process_s2_images(s2_folder_path):
    """
    Processes Sentinel-2 (S2) 4-band images in the given folder.
    Replaces all -9999 and -np.inf values in each band with np.nan 
    and overwrites the original files.

    Args:
        s2_folder_path (str): The path to the folder containing S2 .tif images.
    """
    print(f"\nProcessing S2 images in: {s2_folder_path}")
    processed_files = 0
    failed_files = []

    for filename in os.listdir(s2_folder_path):
        if filename.endswith(".tif") and "s2_4bands" in filename: # Target S2 specific files if needed
            file_path = os.path.join(s2_folder_path, filename)
            try:
                with rasterio.open(file_path, 'r+') as src:
                    profile = src.profile
                    num_bands = src.count # Should be 4 for your S2 data

                    if num_bands != 4:
                        print(f"  Warning: Expected 4 bands for S2 image {filename}, but found {num_bands}. Processing based on found band count.")
                        # Decide how to handle or skip if band count is not 4

                    output_dtype = np.float32  # Ensure float type for np.nan

                    # Check if existing data is float, if not, note conversion
                    temp_check_data = src.read(1)
                    if not np.issubdtype(temp_check_data.dtype, np.floating):
                        print(f"  Note: Data in S2 image {filename} will be converted to {output_dtype} to support np.nan and -np.inf processing.")
                    
                    # Update profile for all bands to reflect nodata as np.nan and float dtype
                    profile.update(nodata=np.nan, dtype=output_dtype)

                    for bidx in range(1, num_bands + 1):
                        data = src.read(bidx)
                        
                        # Convert to float to handle np.nan and np.inf if necessary
                        if not np.issubdtype(data.dtype, np.floating):
                            data = data.astype(output_dtype)
                        
                        # Replace -9999 with np.nan
                        data[data == -9999] = np.nan
                        # Replace negative infinity with np.nan
                        data[data == -np.inf] = np.nan # Or use np.isneginf(data) for safety
                        
                        src.write(data, bidx)
                
                processed_files += 1
                print(f"  Successfully processed S2 image: {filename} ({num_bands} bands)")

            except Exception as e:
                print(f"  Failed to process S2 image {filename}: {e}")
                failed_files.append(filename)

    print(f"\nS2 processing complete.")
    print(f"Total S2 files processed: {processed_files}")
    if failed_files:
        print(f"Failed to process {len(failed_files)} S2 files: {', '.join(failed_files)}")


if __name__ == "__main__":
    # Define the path to the ROI folder
    roi_base_path = "DELAG_LST/KhanhXuan_BuonMaThuot_DakLak/" 
    
    # ---- Process LST Data ----
    lst_data_folder = os.path.join(roi_base_path, "lst")
    if not os.path.isdir(lst_data_folder):
        print(f"Error: LST folder not found at {lst_data_folder}")
    else:
        process_lst_images(lst_data_folder)
        
    # ---- Process ERA5 Data ----
    era5_data_folder = os.path.join(roi_base_path, "era5")
    if not os.path.isdir(era5_data_folder):
        print(f"Error: ERA5 folder not found at {era5_data_folder}")
    else:
        process_era5_images(era5_data_folder)

    # ---- Process S2 Data ----
    s2_data_folder = os.path.join(roi_base_path, "s2_images")
    if not os.path.isdir(s2_data_folder):
        print(f"Error: S2 folder not found at {s2_data_folder}")
    else:
        process_s2_images(s2_data_folder)