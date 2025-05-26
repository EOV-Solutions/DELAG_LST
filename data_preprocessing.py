"""
Data preprocessing for the DELAG project.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling as RasterioResampling
from rasterio.windows import Window
from sklearn.impute import KNNImputer
from tqdm import tqdm
import os
import glob

import config # Updated config with ROI structure
import utils

def load_landsat_lst(
    landsat_lst_dir: str,
    # start_date: str, # No longer used to define the primary iteration range
    # end_date: str,   # No longer used to define the primary iteration range
    # target_resolution_val: int, # Keep if used for something else, or remove if only for ref grid from LST
    lst_nodata_val: float,
    app_config: 'config', # For OUTPUT_DIR
    reference_grid_path: str = None
) -> tuple[np.ndarray, list, dict, str]: # Removed valid_pixel_counts for now, can be re-added if needed for this sparse approach
    """
    Loads, preprocesses, and aligns Landsat LST data for a given ROI
    ONLY for dates where LST files are found.
    Cloud mask is derived from LST_NODATA_VALUE.

    Args:
        landsat_lst_dir (str): Directory containing Landsat LST files for the ROI.
        lst_nodata_val (float): NoData value in LST files.
        app_config: Configuration object.
        reference_grid_path (str, optional): Path to a GeoTIFF defining the reference grid.
                                          If None, the first valid LST image is used as reference.

    Returns:
        tuple[np.ndarray, list, dict, str]:
            - lst_stack_nan (np.ndarray): Time-series stack of LST data (time, height, width)
                                         for dates with actual files, with np.nan for nodata pixels.
            - loaded_dates (list): List of datetime objects for which LST files were found and loaded.
            - geo_profile (dict): Georeferencing profile from the reference raster.
            - actual_reference_grid_path (str): Path of the raster used as the reference grid.
    """
    print(f"Loading and preprocessing Landsat LST data from: {landsat_lst_dir} (sparse approach)")
    lst_data_list = []
    # cloud_mask_list = [] # Can be re-added if a separate 0/1 mask is needed downstream
    loaded_dates = []
    actual_reference_grid_path = reference_grid_path

    all_lst_files_in_dir = sorted(glob.glob(os.path.join(landsat_lst_dir, "*.tif")) + glob.glob(os.path.join(landsat_lst_dir, "*.img")))

    if not all_lst_files_in_dir:
        # If no files at all, return empty structures that subsequent functions can handle
        print(f"No LST .tif or .img files found in {landsat_lst_dir}. Returning empty data.")
        # It's crucial that the return types match for downstream consistency, even if empty.
        # Determine a default geo_profile structure or handle this upstream if truly no reference can be established.
        # For now, this scenario will likely cause issues if no reference grid can be determined.
        # A better approach might be to raise FileNotFoundError here if no LST files are found,
        # as a reference grid is essential.
        raise FileNotFoundError(f"No LST .tif or .img files found in {landsat_lst_dir}. Cannot establish reference grid.")

    file_date_mapping = {} # Maps datetime object to list of file paths
    for f_path in all_lst_files_in_dir:
        fname = os.path.basename(f_path)
        import re
        match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', fname)
        if match:
            date_str_from_fname = match.group(1).replace('-', '').replace('_', '')
            try:
                file_dt = pd.to_datetime(date_str_from_fname, format='%Y%m%d')
                if file_dt not in file_date_mapping:
                    file_date_mapping[file_dt] = []
                file_date_mapping[file_dt].append(f_path)
            except ValueError:
                print(f"Could not parse date from LST filename: {fname}, skipping file.")
        else:
            print(f"Could not find date pattern in LST filename: {fname}, skipping file.")

    if not file_date_mapping:
        raise FileNotFoundError(f"No LST files with parseable dates found in {landsat_lst_dir}.")

    # Determine reference grid: use provided one or the first chronological LST file
    if not actual_reference_grid_path:
        first_available_date = sorted(file_date_mapping.keys())[0]
        actual_reference_grid_path = file_date_mapping[first_available_date][0]
        print(f"Using {actual_reference_grid_path} as reference grid for LST processing.")

    with rasterio.open(actual_reference_grid_path) as ref_src:
        geo_profile = ref_src.profile.copy()
        target_height, target_width = ref_src.height, ref_src.width
        geo_profile.update({
            'width': target_width,
            'height': target_height,
            'dtype': 'float32',
            'nodata': lst_nodata_val # LST files' nodata, will be converted to np.nan in stack
        })

    # Iterate through the sorted dates for which LST files were actually found and parsed
    sorted_file_dates = sorted(file_date_mapping.keys())
    
    # Filter dates based on config.START_DATE and config.END_DATE if they are set
    # This allows user to still constrain the overall period even with sparse loading
    if hasattr(app_config, 'START_DATE') and app_config.START_DATE and \
       hasattr(app_config, 'END_DATE') and app_config.END_DATE:
        start_dt_config = pd.to_datetime(app_config.START_DATE)
        end_dt_config = pd.to_datetime(app_config.END_DATE)
        dates_to_process = [
            dt for dt in sorted_file_dates if start_dt_config <= dt <= end_dt_config
        ]
        print(f"Filtered LST dates based on config START/END_DATE. Processing {len(dates_to_process)} dates.")
    else:
        dates_to_process = sorted_file_dates
        print(f"Processing all {len(dates_to_process)} found LST dates (no START/END_DATE filter from config).")


    for current_day_dt in tqdm(dates_to_process, desc="Processing Found Landsat LST Files"):
        lst_files_for_day = file_date_mapping.get(current_day_dt, []) # Should always find files here

        # Averaging logic for multiple files on the same day remains
        daily_lst_sum = np.zeros((target_height, target_width), dtype=np.float64)
        daily_valid_pixel_count = np.zeros((target_height, target_width), dtype=np.int16)
        # daily_final_mask = np.ones((target_height, target_width), dtype=np.uint8) # if re-adding cloud_mask_list

        processed_at_least_one_file_for_day = False
        for i, lst_fpath in enumerate(lst_files_for_day):
            temp_aligned_lst_path = os.path.join(app_config.OUTPUT_DIR, f"temp_aligned_lst_{current_day_dt.strftime('%Y%m%d')}_{i}.tif")
            try:
                utils.align_rasters(actual_reference_grid_path, lst_fpath, temp_aligned_lst_path,
                                    resampling_method=RasterioResampling.bilinear)
                with rasterio.open(temp_aligned_lst_path) as lsrc:
                    current_lst_data = lsrc.read(1, out_dtype=np.float32)
                    # Identify nodata pixels from the source (which match lst_nodata_val or are NaN from resampling)
                    nodata_mask_this_obs = ((current_lst_data == lst_nodata_val) | np.isnan(current_lst_data))
                    
                    # For averaging, use only valid data
                    valid_pixels_this_obs = ~nodata_mask_this_obs
                    
                    daily_lst_sum[valid_pixels_this_obs] += current_lst_data[valid_pixels_this_obs]
                    daily_valid_pixel_count[valid_pixels_this_obs] += 1
                    # daily_final_mask[valid_pixels_this_obs] = 0 # if using cloud_mask_list
                os.remove(temp_aligned_lst_path)
                processed_at_least_one_file_for_day = True
            except Exception as e:
                print(f"Error processing or aligning LST file {lst_fpath} for date {current_day_dt}: {e}. Skipping this file.")
                continue
        
        if not processed_at_least_one_file_for_day and not lst_files_for_day:
            # This should not be reached if dates_to_process is derived from file_date_mapping keys
            # and file_date_mapping is not empty. But as a safeguard:
            print(f"Warning: No LST files were successfully processed for date {current_day_dt}, though it was in the list. Skipping.")
            continue

        avg_daily_lst = np.full((target_height, target_width), np.nan, dtype=np.float32) # Default to NaN
        valid_for_avg = daily_valid_pixel_count > 0
        avg_daily_lst[valid_for_avg] = (daily_lst_sum[valid_for_avg] / daily_valid_pixel_count[valid_for_avg]).astype(np.float32)
        
        # Where original LST file might have had lst_nodata_val, and averaging didn't occur, it remains NaN.
        # If all observations for a pixel were nodata, it also remains NaN.
        
        lst_data_list.append(avg_daily_lst)
        # cloud_mask_list.append(np.where(np.isnan(avg_daily_lst), 1, 0).astype(np.uint8)) # if separate mask needed
        loaded_dates.append(current_day_dt)

    if not lst_data_list:
        # This implies dates_to_process was empty or all processing failed.
        raise ValueError("No Landsat LST data could be loaded based on available files and date filters.")

    lst_stack_nan = np.stack(lst_data_list, axis=0)
    # valid_pixel_counts_sparse = np.sum(~np.isnan(lst_stack_nan), axis=0) # Recalculate if needed

    print(f"Loaded {len(loaded_dates)} LST scenes. Stack shape: {lst_stack_nan.shape}")
    # Return LST stack (with NaNs for nodata pixels) and the list of dates for which data was loaded.
    # Cloud mask stack is implicitly handled by NaNs in lst_stack_nan.
    return lst_stack_nan, loaded_dates, geo_profile, actual_reference_grid_path

def load_era5_skin_temp(
    era5_skin_temp_dir: str, 
    lst_dates_with_data: list, # list of datetime objects from LST loading (actual LST dates)
    reference_grid_path: str, 
    # target_resolution_val: int, # Not directly used if aligning to reference_grid_path
    app_config: 'config' # For OUTPUT_DIR
) -> tuple[np.ndarray, list]:
    """
    Loads, preprocesses, and aligns ERA5 skin temperature data for the ROI,
    ONLY for dates present in lst_dates_with_data for which ERA5 files are also found.

    Args:
        era5_skin_temp_dir (str): Directory containing ERA5 files.
        lst_dates_with_data (list): List of datetime objects for which LST data was successfully loaded.
        reference_grid_path (str): Path to the reference raster for alignment.
        app_config: Configuration object for accessing OUTPUT_DIR.

    Returns:
        tuple[np.ndarray, list]:
            - era5_stack (np.ndarray): Time-series stack of ERA5 data (time, height, width)
                                     for dates where both LST and ERA5 data were found.
            - common_dates_era5 (list): List of datetime objects for which both LST and ERA5 data were found.
    """
    print(f"Loading and preprocessing ERA5 skin temperature from: {era5_skin_temp_dir} (sparse approach)")
    era5_data_list_for_common_dates = []
    common_dates_era5 = [] # Dates for which we actually found and processed ERA5 data
    
    all_era5_files_in_dir = sorted(glob.glob(os.path.join(era5_skin_temp_dir, "*.tif")) + glob.glob(os.path.join(era5_skin_temp_dir, "*.img")))
    file_date_mapping_era5 = {}
    for f_path in all_era5_files_in_dir:
        fname = os.path.basename(f_path)
        import re
        match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', fname)
        if match:
            date_str_from_fname = match.group(1).replace('-','').replace('_','')
            try:
                file_dt = pd.to_datetime(date_str_from_fname, format='%Y%m%d')
                if file_dt not in file_date_mapping_era5:
                    file_date_mapping_era5[file_dt] = []
                file_date_mapping_era5[file_dt].append(f_path)
            except ValueError:
                print(f"Could not parse date from ERA5 filename: {fname}")
        else:
            print(f"Could not find date pattern in ERA5 filename: {fname}")

    if not lst_dates_with_data:
        print("Warning: load_era5_skin_temp received an empty list of LST dates. Returning empty ERA5 data.")
        return np.empty((0,0,0), dtype=np.float32), [] # Shape needs to be compatible or handled

    # Get target H, W from ref grid, assume it's valid by now
    with rasterio.open(reference_grid_path) as ref_src:
        target_height, target_width = ref_src.height, ref_src.width

    for date_dt in tqdm(lst_dates_with_data, desc="Processing ERA5 Data for LST Dates"):
        era5_files_for_day = file_date_mapping_era5.get(date_dt, [])
        
        if not era5_files_for_day:
            # print(f"No ERA5 file found for LST date {date_dt.strftime('%Y-%m-%d')}. Skipping this date.")
            continue # Skip this date, as we only want common dates
        
        era5_fpath = era5_files_for_day[0] # Take the first file if multiple for the same date
        temp_aligned_era5_path = os.path.join(app_config.OUTPUT_DIR, f"temp_aligned_era5_{date_dt.strftime('%Y%m%d')}.tif")
        try:
            utils.align_rasters(reference_grid_path, era5_fpath, temp_aligned_era5_path, 
                                resampling_method=RasterioResampling.nearest)
            with rasterio.open(temp_aligned_era5_path) as src:
                era5_data = src.read(2).astype(np.float32)
                # Here, ERA5 data could also have its own NoData. If so, convert to NaN.
                # For now, assume align_rasters handles or it's not an issue for ERA5.
                # If ERA5 has a specific nodata, it should be converted to np.nan here.
                # For example: if era5_nodata_value is known: era5_data[era5_data == era5_nodata_value] = np.nan
                era5_data_list_for_common_dates.append(era5_data)
            os.remove(temp_aligned_era5_path)
            common_dates_era5.append(date_dt) # Add date only if processed successfully
        except Exception as e:
            print(f"Error aligning ERA5 file {era5_fpath} for LST date {date_dt.strftime('%Y-%m-%d')}: {e}. Skipping this date.")
            continue

    if not era5_data_list_for_common_dates:
        print("Warning: No common dates found with valid ERA5 data. Returning empty ERA5 stack.")
        # Decide on shape for empty array, e.g., (0, target_height, target_width)
        # This requires target_height/width to be known even if no data. Ref grid path must be valid.
        return np.empty((0, target_height, target_width), dtype=np.float32), []

    era5_stack = np.stack(era5_data_list_for_common_dates, axis=0)
    
    # Impute NaNs in ERA5 stack if any (this step might need reconsideration in full sparse approach)
    # If an ERA5 file was found but contained NaNs, they are imputed here.
    # If an ERA5 file was missing, the whole date is skipped *before* this point.
    h, w = era5_stack.shape[1], era5_stack.shape[2]
    era5_reshaped = era5_stack.transpose(1, 2, 0).reshape(-1, era5_stack.shape[0])
    if np.isnan(era5_reshaped).any() and era5_reshaped.shape[1] > 0: # Check if imputation is possible
        print("Found NaNs in constructed ERA5 stack (from existing files), attempting KNN imputation...")
        # Ensure n_neighbors is less than or equal to the number of samples if time dim is small
        max_neighbors = era5_stack.shape[0]
        if np.any(np.all(np.isnan(era5_reshaped), axis=1)):
             print("Warning: Some ERA5 pixels have all NaNs across time, KNN imputation might be problematic or fail.")
        
        # Calculate how many non-NaN values exist for each pixel across time
        non_nan_counts_per_pixel_timeseries = np.sum(~np.isnan(era5_reshaped), axis=1)
        min_valid_obs_for_imputation = np.min(non_nan_counts_per_pixel_timeseries[np.any(~np.isnan(era5_reshaped), axis=1)])
        
        n_neighbors_val = min(5, max_neighbors) 
        if min_valid_obs_for_imputation < n_neighbors_val and min_valid_obs_for_imputation > 0:
            n_neighbors_val = min_valid_obs_for_imputation
        
        if n_neighbors_val > 0:
            imputer = KNNImputer(n_neighbors=n_neighbors_val, weights='distance')
            try:
                era5_imputed_reshaped = imputer.fit_transform(era5_reshaped)
                era5_stack = era5_imputed_reshaped.reshape(h, w, -1).transpose(2, 0, 1)
                print("KNN imputation for ERA5 complete.")
            except ValueError as e:
                print(f"KNN imputation failed for ERA5: {e}. Proceeding with existing NaNs in ERA5.")
        else:
            print("Skipping KNN imputation for ERA5 as n_neighbors would be 0.")

    print(f"Loaded ERA5 data for {len(common_dates_era5)} common dates. Stack shape: {era5_stack.shape}")
    return era5_stack, common_dates_era5

def load_sentinel2_reflectance(
    s2_dir: str, 
    # unique_dates_str: list[str], # No longer takes unique_dates_str directly
    common_dates_lst_era5: list, # List of datetime objects for which LST & ERA5 data exist
    actual_ref_grid_path: str, 
    target_height: int, 
    target_width: int,
    s2_nodata_value: float,
    app_config: 'config'
) -> tuple[np.ndarray, list]: # Returns stack and final common dates
    """
    Loads, preprocesses, and temporally aligns Sentinel-2 reflectance data,
    ONLY for dates in common_dates_lst_era5 for which a usable S2 image is also found.
    Selects the least cloudy S2 image if multiple exist for a date.

    Args:
        s2_dir (str): Directory containing Sentinel-2 TIFF files.
        common_dates_lst_era5 (list): list of datetime objects for which LST and ERA5 data exist.
        actual_ref_grid_path (str): Path to the reference raster file used for alignment.
        target_height (int): Target height for aligned rasters.
        target_width (int): Target width for aligned rasters.
        s2_nodata_value (float): NoData value in S2 files.
        app_config: Configuration object.

    Returns:
        tuple[np.ndarray, list]:
            - s2_stack (np.ndarray): Stack of S2 reflectance data (time, bands, height, width)
                                    for dates where LST, ERA5, and S2 data were all found.
            - final_common_dates (list): List of datetime objects for these fully common dates.
    """
    print(f"Loading and preprocessing Sentinel-2 reflectance from: {s2_dir} (sparse approach)")
    s2_data_list_final_common = []
    final_common_dates = [] # Dates for which LST, ERA5, AND S2 are found
    num_expected_bands = 4

    all_s2_files = glob.glob(os.path.join(s2_dir, "s2_4bands_*.tif"))
    s2_files_by_date_str = {} # Map 'YYYY-MM-DD' string to list of file paths
    for f_path in all_s2_files:
        fname = os.path.basename(f_path)
        parts = fname.split('_')
        if len(parts) >= 3 and parts[0] == "s2" and parts[1] == "4bands":
            date_str_from_fname = parts[2]
            try:
                pd.to_datetime(date_str_from_fname, format='%Y-%m-%d') # Validate format
                if date_str_from_fname not in s2_files_by_date_str:
                    s2_files_by_date_str[date_str_from_fname] = []
                s2_files_by_date_str[date_str_from_fname].append(f_path)
            except ValueError:
                print(f"Warning: Could not parse date {date_str_from_fname} from S2 filename: {fname}")
        else:
            print(f"Warning: S2 filename {fname} does not match pattern 's2_4bands_YYYY-MM-DD_id.tif'")

    if not common_dates_lst_era5:
        print("Warning: load_sentinel2_reflectance received empty common_dates_lst_era5. Returning empty S2 data.")
        return np.empty((0, num_expected_bands, target_height, target_width), dtype=np.float32), []

    for date_dt in tqdm(common_dates_lst_era5, desc="Processing S2 Data for LST/ERA5 Dates"):
        target_date_str = date_dt.strftime('%Y-%m-%d')
        s2_files_for_this_date = s2_files_by_date_str.get(target_date_str, [])
        
        best_s2_image_for_date_data = None
        min_cloud_percentage = float('inf')

        if not s2_files_for_this_date:
            # print(f"No S2 file found for LST/ERA5 date {target_date_str}. Skipping this date.")
            continue # Skip this date, as we only want common dates with S2

        for s2_fpath in s2_files_for_this_date:
            try:
                with rasterio.open(s2_fpath) as src:
                    if src.count != num_expected_bands:
                        print(f"Warning: S2 file {s2_fpath} has {src.count} bands, expected {num_expected_bands}. Skipping.")
                        continue
                    s2_data_raw = src.read(out_dtype=np.float32)
                    nodata_mask_per_band = (s2_data_raw == s2_nodata_value)
                    pixel_is_nodata = np.any(nodata_mask_per_band, axis=0)
                    cloud_pixels = np.sum(pixel_is_nodata)
                    total_pixels = pixel_is_nodata.size
                    cloud_percentage = (cloud_pixels / total_pixels) * 100 if total_pixels > 0 else 100

                    if cloud_percentage < min_cloud_percentage:
                        min_cloud_percentage = cloud_percentage
                        temp_aligned_s2_path = os.path.join(app_config.OUTPUT_DIR, f"temp_aligned_s2_{target_date_str}_{os.path.basename(s2_fpath)}")
                        utils.align_rasters(actual_ref_grid_path, s2_fpath, temp_aligned_s2_path,
                                            resampling_method=RasterioResampling.bilinear)
                        with rasterio.open(temp_aligned_s2_path) as aligned_src:
                            aligned_s2_data = aligned_src.read(out_dtype=np.float32)
                            aligned_s2_data[aligned_s2_data == s2_nodata_value] = np.nan
                            # Check if all bands are NaN after masking (e.g. fully clouded after alignment)
                            if np.all(np.isnan(aligned_s2_data)):
                                print(f"Warning: S2 image {s2_fpath} for date {target_date_str} resulted in all NaNs after alignment and masking. Not using.")
                                # This specific image is bad, but loop might find a better one for the date
                                best_s2_image_for_date_data = None # Ensure this doesn't carry over if it was the only one
                            else:
                                best_s2_image_for_date_data = aligned_s2_data
                        os.remove(temp_aligned_s2_path)
                        if cloud_percentage == 0: break # Found a perfectly clear image
            except Exception as e:
                print(f"Error processing S2 file {s2_fpath} for date {target_date_str}: {e}. Skipping this file.")
                continue
        
        if best_s2_image_for_date_data is not None:
            if best_s2_image_for_date_data.shape == (num_expected_bands, target_height, target_width):
                s2_data_list_final_common.append(best_s2_image_for_date_data)
                final_common_dates.append(date_dt)
            else:
                print(f"Warning: Shape mismatch for selected S2 data on {target_date_str}. Skipping this date.")
        # If best_s2_image_for_date_data is None (no suitable S2 found), this date is skipped by not appending.

    if not s2_data_list_final_common:
        print("Warning: No common dates found with valid LST, ERA5, and S2 data. Returning empty S2 stack.")
        return np.empty((0, num_expected_bands, target_height, target_width), dtype=np.float32), []

    s2_stack = np.stack(s2_data_list_final_common, axis=0)
    print(f"Loaded S2 data for {len(final_common_dates)} common dates (LST, ERA5, S2). Stack shape: {s2_stack.shape}")
    return s2_stack, final_common_dates

def load_coordinates(reference_grid_path: str, normalize: bool = True) -> tuple[np.ndarray, np.ndarray, object, object]: # Updated Scaler type
    """
    Extracts x and y coordinates for each pixel from a reference grid.
    Optionally normalizes coordinates.
    """
    print("Loading and normalizing coordinates...")
    with rasterio.open(reference_grid_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        x_coords, y_coords = rasterio.transform.xy(transform, rows, cols, offset='center')
        x_coords = np.array(x_coords, dtype=np.float32)
        y_coords = np.array(y_coords, dtype=np.float32)

    x_scaler, y_scaler = None, None
    if normalize:
        # Ensure no NaNs/Infs if coords came from somewhere weird, though unlikely from rasterio.transform.xy
        x_coords_flat = x_coords.flatten()
        y_coords_flat = y_coords.flatten()
        x_coords_norm, y_coords_norm, x_scaler, y_scaler = utils.normalize_coordinates(x_coords_flat, y_coords_flat)
        x_coords = x_coords_norm.reshape(height, width)
        y_coords = y_coords_norm.reshape(height, width)
        print("Coordinates normalized.")
    else:
        print("Coordinates loaded without normalization.")
        
    return x_coords, y_coords, x_scaler, y_scaler

def preprocess_all_data(app_config) -> dict:
    """
    Main function to preprocess all data for a given ROI based on app_config.
    Implements a sparse data loading strategy: only dates where LST, ERA5, and S2
    data are all available and valid will be included in the final stacks.
    Raises ValueError if essential data cannot be loaded or aligned.
    """
    print(f"Starting preprocessing for ROI: {app_config.ROI_NAME}")
    os.makedirs(app_config.OUTPUT_DIR, exist_ok=True)

    # 1. Load Landsat LST data
    try:
        lst_stack_initial, loaded_lst_dates, geo_profile, actual_reference_grid_path = load_landsat_lst(
            landsat_lst_dir=app_config.LANDSAT_LST_PATH,
            lst_nodata_val=app_config.LST_NODATA_VALUE,
            app_config=app_config,
            reference_grid_path=None 
        )
    except FileNotFoundError as e:
        raise ValueError(f"Critical error during LST loading for ROI {app_config.ROI_NAME}: {e}") from e
    
    if not loaded_lst_dates:
        raise ValueError(f"No LST data found for ROI {app_config.ROI_NAME} after initial load. Cannot proceed.")

    target_height = geo_profile['height']
    target_width = geo_profile['width']

    # 2. Load ERA5 skin temperature data
    era5_stack_initial, common_dates_lst_era5 = load_era5_skin_temp(
        era5_skin_temp_dir=app_config.ERA5_SKIN_TEMP_PATH,
        lst_dates_with_data=loaded_lst_dates,
        reference_grid_path=actual_reference_grid_path,
        app_config=app_config
    )

    if not common_dates_lst_era5:
        raise ValueError(f"No common dates found after attempting to load ERA5 data for ROI {app_config.ROI_NAME}. Cannot proceed.")

    # 3. Filter initial LST stack to align with common_dates_lst_era5
    lst_date_to_index = {date: i for i, date in enumerate(loaded_lst_dates)}
    indices_for_lst_common_with_era5 = [lst_date_to_index[date] for date in common_dates_lst_era5 if date in lst_date_to_index]
    
    if not indices_for_lst_common_with_era5:
         raise ValueError(f"LST date filtering resulted in no common dates with ERA5 for ROI {app_config.ROI_NAME}. This is unexpected. Check date matching.")

    lst_stack_common_with_era5 = lst_stack_initial[indices_for_lst_common_with_era5, :, :]
    print(f"LST stack filtered to {len(common_dates_lst_era5)} dates common with ERA5. Shape: {lst_stack_common_with_era5.shape}")

    # 4. Load Sentinel-2 reflectance data
    s2_reflectance_stack, final_common_dates = load_sentinel2_reflectance(
        s2_dir=app_config.SENTINEL2_REFLECTANCE_PATH,
        common_dates_lst_era5=common_dates_lst_era5,
        actual_ref_grid_path=actual_reference_grid_path,
        target_height=target_height,
        target_width=target_width,
        s2_nodata_value=app_config.S2_NODATA_VALUE,
        app_config=app_config
    )

    if not final_common_dates:
        raise ValueError(f"No common dates found after attempting to load Sentinel-2 data for ROI {app_config.ROI_NAME}. Cannot proceed.")
    
    print(f"Final common dates after S2 processing: {len(final_common_dates)}. S2 stack shape: {s2_reflectance_stack.shape if s2_reflectance_stack is not None else 'None'}")

    # 5. Filter LST and ERA5 stacks to align with final_common_dates
    lst_era5_date_to_index = {date: i for i, date in enumerate(common_dates_lst_era5)}
    indices_for_lst_final = [lst_era5_date_to_index[date] for date in final_common_dates if date in lst_era5_date_to_index]
    if not indices_for_lst_final and final_common_dates: # Check if filtering wiped out all dates
        raise ValueError(f"Final LST date filtering resulted in no dates for ROI {app_config.ROI_NAME}, but final_common_dates was not empty.")
    lst_stack_final = lst_stack_common_with_era5[indices_for_lst_final, :, :]

    era5_date_to_index = {date: i for i, date in enumerate(common_dates_lst_era5)}
    indices_for_era5_final = [era5_date_to_index[date] for date in final_common_dates if date in era5_date_to_index]
    if not indices_for_era5_final and final_common_dates: # Check if filtering wiped out all dates
        raise ValueError(f"Final ERA5 date filtering resulted in no dates for ROI {app_config.ROI_NAME}, but final_common_dates was not empty.")
    era5_stack_final = era5_stack_initial[indices_for_era5_final, :, :]
    
    print(f"Final LST stack shape: {lst_stack_final.shape} for {len(final_common_dates)} dates.")
    print(f"Final ERA5 stack shape: {era5_stack_final.shape} for {len(final_common_dates)} dates.")

    if not (lst_stack_final.shape[0] == era5_stack_final.shape[0] == s2_reflectance_stack.shape[0] == len(final_common_dates)):
        error_msg = (
            f"Time dimension mismatch after final filtering for ROI {app_config.ROI_NAME}:\\n"
            f"LST: {lst_stack_final.shape[0]}, ERA5: {era5_stack_final.shape[0]}, "
            f"S2: {s2_reflectance_stack.shape[0]}, Dates: {len(final_common_dates)}"
        )
        raise ValueError(error_msg)

    # 6. Create Day of Year (DOY) stack
    doy_stack_1d = np.array([date.timetuple().tm_yday for date in final_common_dates], dtype=np.int16)
    doy_stack = np.tile(doy_stack_1d[:, np.newaxis, np.newaxis], (1, target_height, target_width))
    print(f"DOY stack created. Shape: {doy_stack.shape}")

    # 7. Load Coordinate data
    lon_coords, lat_coords, lon_scaler, lat_scaler = load_coordinates(
        reference_grid_path=actual_reference_grid_path, 
        normalize=True
    )
    print(f"Coordinates loaded. Lon shape: {lon_coords.shape}, Lat shape: {lat_coords.shape}")
    
    # --- Perform KNN Imputation on S2 data ---
    num_final_dates, num_s2_bands, s2_h, s2_w = s2_reflectance_stack.shape
    s2_reshaped_for_imputation = s2_reflectance_stack.transpose(0, 2, 3, 1).reshape(-1, num_s2_bands)
    s2_final_stack = s2_reflectance_stack # Default to original

    if np.isnan(s2_reshaped_for_imputation).any():
        print(f"NaNs found in S2 stack for ROI {app_config.ROI_NAME}. Performing KNN imputation...")
        if hasattr(app_config, 'KNN_N_NEIGHBORS') and app_config.KNN_N_NEIGHBORS > 0:
            imputer = KNNImputer(n_neighbors=app_config.KNN_N_NEIGHBORS)
            s2_imputed_flat = imputer.fit_transform(s2_reshaped_for_imputation)
            s2_reflectance_stack_imputed = s2_imputed_flat.reshape(num_final_dates, s2_h, s2_w, num_s2_bands).transpose(0, 3, 1, 2)
            print(f"KNN imputation complete for S2 stack. Imputed shape: {s2_reflectance_stack_imputed.shape}")
            
            if np.isnan(s2_reflectance_stack_imputed).any():
                print(f"WARNING: NaNs still present in S2 stack for ROI {app_config.ROI_NAME} after KNN imputation.")
            s2_final_stack = s2_reflectance_stack_imputed
        else:
            print(f"KNN_N_NEIGHBORS not configured or is <= 0. Skipping KNN imputation for S2 stack for ROI {app_config.ROI_NAME}.")
    else:
        print(f"No NaNs found in S2 stack for ROI {app_config.ROI_NAME}. Skipping KNN imputation.")

    print(f"Preprocessing complete for ROI: {app_config.ROI_NAME}. Final number of aligned dates: {len(final_common_dates)}")
    
    return {
        "lst_stack": np.array(lst_stack_final, dtype=np.float32),
        "era5_stack": np.array(era5_stack_final, dtype=np.float32),
        "s2_reflectance_stack": np.array(s2_final_stack, dtype=np.float32),
        "doy_stack": np.array(doy_stack, dtype=np.int16),
        "lon_coords": np.array(lon_coords, dtype=np.float32),
        "lat_coords": np.array(lat_coords, dtype=np.float32),
        "geo_profile": geo_profile,
        "common_dates": final_common_dates,
        "lon_scaler": lon_scaler,
        "lat_scaler": lat_scaler,
        "reference_grid_path": actual_reference_grid_path,
        "roi_name": app_config.ROI_NAME
    }

# def generate_dummy_raster(output_path, height, width, num_bands, dtype, nodata_val, constant_val=None, profile_base=None):
#     # ... existing code ...

# --- Test block for data_preprocessing.py ---
if __name__ == '__main__':
    import shutil
    from sklearn.preprocessing import MinMaxScaler # Added for TestConfig scaler objects

    # Define a dummy config for testing
    class TestConfig:
        # --- Test ROI Setup ---
        BASE_DATA_DIR = "./dummy_delag_data/"
        ROI_NAME = "TestROI_S2_Temporal" # New name for this test
        
        ROI_BASE_PATH = os.path.join(BASE_DATA_DIR, ROI_NAME)
        LANDSAT_LST_SUBDIR = "lst"
        ERA5_SKIN_TEMP_SUBDIR = "era5"
        SENTINEL2_REFLECTANCE_SUBDIR = "s2_images"

        LANDSAT_LST_PATH = os.path.join(ROI_BASE_PATH, LANDSAT_LST_SUBDIR)
        ERA5_SKIN_TEMP_PATH = os.path.join(ROI_BASE_PATH, ERA5_SKIN_TEMP_SUBDIR)
        SENTINEL2_REFLECTANCE_PATH = os.path.join(ROI_BASE_PATH, SENTINEL2_REFLECTANCE_SUBDIR)

        OUTPUT_DIR_BASE = "./dummy_delag_output/"
        OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, ROI_NAME) # ROI-specific output for temp files

        TARGET_RESOLUTION = 30 # Dummy, not directly used by align_rasters if ref exists
        START_DATE = "2023-01-01"
        END_DATE = "2023-01-03" # Short period for testing
        LST_NODATA_VALUE = -9999.0
        S2_NODATA_VALUE = -9999.0 # Added for S2
        DAYS_OF_YEAR = 365 # Dummy
        RANDOM_SEED = 42
        # GP_RESIDUAL_FEATURES might be defined in the main config, not strictly needed here for preprocessing test itself
        # but band order B2,B3,B4,B8 is implicitly assumed for S2 dummy data.

    test_config = TestConfig()

    # Create dummy directories and files
    os.makedirs(test_config.LANDSAT_LST_PATH, exist_ok=True)
    os.makedirs(test_config.ERA5_SKIN_TEMP_PATH, exist_ok=True)
    os.makedirs(test_config.SENTINEL2_REFLECTANCE_PATH, exist_ok=True)
    os.makedirs(test_config.OUTPUT_DIR, exist_ok=True) # For temp aligned files

    dummy_height, dummy_width = 10, 10
    num_bands_s2 = 4 # B2, B3, B4, B8
    np.random.seed(test_config.RANDOM_SEED)

    # Create a reference profile (e.g., from the first LST image)
    ref_affine = rasterio.Affine(test_config.TARGET_RESOLUTION, 0.0, 500000.0, 
                                 0.0, -test_config.TARGET_RESOLUTION, 6000000.0)
    ref_profile_test = {
        'driver': 'GTiff', 'dtype': 'float32', 'nodata': test_config.LST_NODATA_VALUE,
        'width': dummy_width, 'height': dummy_height, 'count': 1,
        'crs': rasterio.CRS.from_epsg(32630), # Example CRS
        'transform': ref_affine
    }

    dates_to_create = pd.to_datetime([test_config.START_DATE, "2023-01-02", test_config.END_DATE])

    # Dummy LST files
    for i, date_obj in enumerate(dates_to_create):
        date_str = date_obj.strftime("%Y%m%d")
        lst_file = os.path.join(test_config.LANDSAT_LST_PATH, f"LST_dummy_{date_str}.tif")
        dummy_lst_data = np.random.rand(dummy_height, dummy_width).astype(np.float32) * 30 + 273.15
        if i % 2 == 0: # Make some LST data cloudy
            dummy_lst_data[0:dummy_height//2, :] = test_config.LST_NODATA_VALUE
        with rasterio.open(lst_file, 'w', **ref_profile_test) as dst:
            dst.write(dummy_lst_data, 1)
        if i == 0: # Save one as a potential reference for other loaders if needed
            shutil.copy(lst_file, os.path.join(test_config.ROI_BASE_PATH, "reference_grid_dummy.tif"))


    # Dummy ERA5 files
    for date_obj in dates_to_create:
        date_str = date_obj.strftime("%Y%m%d")
        era5_file = os.path.join(test_config.ERA5_SKIN_TEMP_PATH, f"ERA5_dummy_{date_str}.tif")
        dummy_era5_data = np.random.rand(dummy_height, dummy_width).astype(np.float32) * 20 + 280.0
        # ERA5 typically coarser, but here we make it same res for simplicity of dummy data
        with rasterio.open(era5_file, 'w', **ref_profile_test) as dst:
            dst.write(dummy_era5_data, 1)

    # Dummy Sentinel-2 files (multi-band, multiple per day with varying clouds)
    s2_profile_test = ref_profile_test.copy()
    s2_profile_test['count'] = num_bands_s2
    s2_profile_test['nodata'] = test_config.S2_NODATA_VALUE # S2 nodata

    for date_obj in dates_to_create:
        date_str_s2_fmt = date_obj.strftime("%Y-%m-%d") # Filename format YYYY-MM-DD
        for i in range(3): # Create 3 versions for each day
            s2_file = os.path.join(test_config.SENTINEL2_REFLECTANCE_PATH, f"s2_4bands_{date_str_s2_fmt}_id{i}.tif")
            dummy_s2_data_multiband = np.random.rand(num_bands_s2, dummy_height, dummy_width).astype(np.float32) * 0.3
            
            # Introduce nodata to simulate clouds, more in earlier IDs
            if i == 0: # Most cloudy
                dummy_s2_data_multiband[:, 0:dummy_height*3//4, :] = test_config.S2_NODATA_VALUE
            elif i == 1: # Moderately cloudy
                dummy_s2_data_multiband[:, 0:dummy_height//2, :] = test_config.S2_NODATA_VALUE
            # else: i == 2 is least cloudy / clear
            
            with rasterio.open(s2_file, 'w', **s2_profile_test) as dst:
                dst.write(dummy_s2_data_multiband)
    
    # Add an S2 file for a date that does NOT exist in LST/ERA5 to test filtering
    extra_s2_date = (dates_to_create[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    s2_file_extra = os.path.join(test_config.SENTINEL2_REFLECTANCE_PATH, f"s2_4bands_{extra_s2_date}_id0.tif")
    dummy_s2_data_extra = np.random.rand(num_bands_s2, dummy_height, dummy_width).astype(np.float32) * 0.3
    with rasterio.open(s2_file_extra, 'w', **s2_profile_test) as dst:
        dst.write(dummy_s2_data_extra)


    print(f"Dummy data generated in: {test_config.BASE_DATA_DIR}")
    print(f"Dummy output will be in: {test_config.OUTPUT_DIR_BASE}")

    try:
        print("\\n--- Running preprocess_all_data with TestConfig ---")
        preprocessed_output = preprocess_all_data(test_config)
        print("\\n--- Preprocessing Output Summary ---")
        for key, value in preprocessed_output.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}, NaNs: {np.isnan(value).sum()}")
            elif isinstance(value, list) and value and isinstance(value[0], pd.Timestamp):
                 print(f"  {key}: {len(value)} timestamps from {value[0]} to {value[-1]}")
            elif isinstance(value, dict) and key=="geo_profile":
                 print(f"  {key}: CRS {value.get('crs')}, Transform {value.get('transform')}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Basic checks
        assert preprocessed_output['lst_stack'].shape == (len(dates_to_create), dummy_height, dummy_width)
        assert preprocessed_output['era5_stack'].shape == (len(dates_to_create), dummy_height, dummy_width)
        assert preprocessed_output['s2_reflectance_stack'].shape == (len(dates_to_create), num_bands_s2, dummy_height, dummy_width)
        assert preprocessed_output['doy_stack'].shape == (len(dates_to_create), dummy_height, dummy_width)
        assert preprocessed_output['x_coords'].shape == (dummy_height, dummy_width)
        
        # Check if S2 for 2023-01-01 (most cloudy original was id0) has been replaced by id2 (least cloudy)
        # This requires inspecting the actual values or cloud counts, which is more involved here.
        # For now, we check that it's not all NaNs if a clear one existed.
        s2_day1_data = preprocessed_output['s2_reflectance_stack'][0] # First day
        # If the "least cloudy" version (id2) was chosen, it should have no NaNs from nodata if it was fully clear
        # The dummy data for id2 has no S2_NODATA_VALUE
        assert np.sum(np.isnan(s2_day1_data)) == 0, "S2 data for the first day should be the least cloudy version (no NaNs from nodata if id2 was clear)"

        print("\\n--- Test for preprocess_all_data PASSED ---")

    except Exception as e:
        print(f"Error during preprocessing test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy directories
        # shutil.rmtree(test_config.BASE_DATA_DIR, ignore_errors=True)
        # shutil.rmtree(test_config.OUTPUT_DIR_BASE, ignore_errors=True)
        print(f"Dummy data and output directories ({test_config.BASE_DATA_DIR}, {test_config.OUTPUT_DIR_BASE}) were NOT automatically cleaned up. Please remove them manually if desired.") 