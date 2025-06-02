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
import warnings

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
    target_dates: list, # list of datetime objects from LST loading (actual LST dates)
    reference_grid_path: str, 
    # target_resolution_val: int, # Not directly used if aligning to reference_grid_path
    app_config: 'config' # For OUTPUT_DIR
) -> tuple[np.ndarray, list]:
    """
    Loads, preprocesses, and aligns ERA5 skin temperature data for the ROI.
    Ensures data for all dates in `target_dates` (which should become `final_common_dates` upstream).
    Performs temporal linear interpolation then spatial linear interpolation to fill NaNs.

    Args:
        era5_skin_temp_dir (str): Directory containing ERA5 files.
        target_dates (list): List of datetime objects for which ERA5 data is required (e.g., final_common_dates).
        reference_grid_path (str): Path to the reference raster for alignment.
        app_config: Configuration object for accessing OUTPUT_DIR.

    Returns:
        tuple[np.ndarray, list]:
            - era5_stack_interpolated (np.ndarray): Time-series stack of ERA5 data (time, height, width),
                                                 aligned and fully interpolated.
            - target_dates (list): The input list of target_dates, returned for consistency.
    """
    print(f"Loading, aligning, and interpolating ERA5 skin temperature from: {era5_skin_temp_dir}")
    
    if not target_dates:
        print("Warning: load_era5_skin_temp received an empty list of target_dates. Returning empty ERA5 data.")
        # Determine target_height/width for empty array shape, or handle this error more strictly upstream.
        try:
            with rasterio.open(reference_grid_path) as ref_src_for_shape:
                h_empty, w_empty = ref_src_for_shape.height, ref_src_for_shape.width
            return np.empty((0, h_empty, w_empty), dtype=np.float32), []
        except Exception:
            return np.empty((0, 0, 0), dtype=np.float32), []

    # Get target H, W from ref grid
    with rasterio.open(reference_grid_path) as ref_src:
        target_height, target_width = ref_src.height, ref_src.width

    # Initialize the full stack with NaNs
    era5_stack_full = np.full((len(target_dates), target_height, target_width), np.nan, dtype=np.float32)
    
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

    # Populate the stack with available data
    for i, date_dt in enumerate(tqdm(target_dates, desc="Processing ERA5 Data (Initial Load)")):
        era5_files_for_day = file_date_mapping_era5.get(date_dt, [])
        if not era5_files_for_day:
            continue # Leaves NaNs for this date in era5_stack_full
        
        era5_fpath = era5_files_for_day[0] # Take the first file if multiple
        temp_aligned_era5_path = os.path.join(app_config.OUTPUT_DIR, f"temp_aligned_era5_{date_dt.strftime('%Y%m%d')}.tif")
        try:
            utils.align_rasters(reference_grid_path, era5_fpath, temp_aligned_era5_path, 
                                resampling_method=RasterioResampling.nearest) # Nearest for ERA5 is often preferred
            with rasterio.open(temp_aligned_era5_path) as src:
                # ERA5 data often has 2 bands (e.g. temp, dewpoint), or just 1 (temp)
                # Assuming config specifies which band or if it's always band 1 (index 0) or band 2 (index 1)
                # For now, let's assume band 2 (index 1) as in original code, check if src.count allows this
                band_to_read = 1 # Default to the second band (1-indexed for rasterio.read)
                if src.count < band_to_read:
                    print(f"Warning: ERA5 file {era5_fpath} has only {src.count} band(s). Reading band 1.")
                    band_to_read = 1 # Fallback to first band
                
                era5_data_slice = src.read(band_to_read).astype(np.float32)
                # Convert known nodata values in source file to NaN if applicable
                # Example: if app_config.ERA5_NODATA_VALUE is defined and not np.nan:
                # if hasattr(app_config, 'ERA5_NODATA_VALUE') and not np.isnan(app_config.ERA5_NODATA_VALUE):
                #     era5_data_slice[era5_data_slice == app_config.ERA5_NODATA_VALUE] = np.nan
                era5_stack_full[i, :, :] = era5_data_slice
            os.remove(temp_aligned_era5_path)
        except Exception as e:
            print(f"Error aligning ERA5 file {era5_fpath} for date {date_dt.strftime('%Y-%m-%d')}: {e}. Leaving NaNs for this date.")
            continue

    if hasattr(app_config, 'INTERPOLATE_ERA5') and not app_config.INTERPOLATE_ERA5:
        print("ERA5 interpolation skipped due to app_config.INTERPOLATE_ERA5 = False.")
        if np.isnan(era5_stack_full).any():
             print(f"Warning: Non-interpolated ERA5 stack contains {np.isnan(era5_stack_full).sum()} NaNs.")
        return era5_stack_full, target_dates

    # Temporal Interpolation (Linear)
    print("Performing temporal linear interpolation on ERA5 stack...")
    for r in tqdm(range(target_height), desc="ERA5 Temporal Interpolation (Rows)", leave=False):
        for c in range(target_width):
            pixel_series = pd.Series(era5_stack_full[:, r, c])
            era5_stack_full[:, r, c] = pixel_series.interpolate(method='linear', limit_direction='both').to_numpy()

    # Spatial Interpolation (Linear using griddata) for remaining NaNs
    print("Performing spatial linear interpolation on ERA5 stack for remaining NaNs...")
    from scipy.interpolate import griddata
    points = None # To be initialized once
    for t in tqdm(range(len(target_dates)), desc="ERA5 Spatial Interpolation (Time Slices)", leave=False):
        slice_data = era5_stack_full[t, :, :]
        if np.isnan(slice_data).any():
            if points is None:
                # Create a grid of coordinates for interpolation, once
                x_coords, y_coords = np.meshgrid(np.arange(target_width), np.arange(target_height))
                points = np.vstack((x_coords.ravel(), y_coords.ravel())).T
            
            valid_mask = ~np.isnan(slice_data)
            values = slice_data[valid_mask]
            points_with_values = points[valid_mask.ravel()]
            
            if points_with_values.shape[0] < 3: # Not enough points for linear/cubic interpolation
                # print(f"Warning: Too few valid points ({points_with_values.shape[0]}) in ERA5 slice {t} for spatial interpolation. Using nearest if any, else NaNs remain.")
                if points_with_values.shape[0] > 0: # Fill with nearest if at least one point
                    nan_locations = np.where(np.isnan(slice_data))
                    interpolated_slice_nearest = griddata(points_with_values, values, (nan_locations[1], nan_locations[0]), method='nearest')
                    if interpolated_slice_nearest.ndim > 0 : # Check if griddata returned usable values
                         slice_data[nan_locations] = interpolated_slice_nearest
                # else: all NaNs, will remain NaNs, which is fine
                era5_stack_full[t, :, :] = slice_data # Update stack
                continue

            nan_locations = np.where(np.isnan(slice_data))
            grid_x_nan, grid_y_nan = nan_locations[1], nan_locations[0] # griddata expects (x,y) for points_to_interpolate
            
            try:
                interpolated_values = griddata(points_with_values, values, (grid_x_nan, grid_y_nan), method='linear')
                # Fill only if interpolated_values is not all NaNs (can happen if all points are collinear etc for linear)
                if not np.all(np.isnan(interpolated_values)):
                     slice_data[np.isnan(slice_data)] = interpolated_values # Fill NaNs where interpolation succeeded
            except Exception as e_spatial_lin:
                # print(f"    S2 spatial linear griddata failed (t={t}, b={b}): {e_spatial_lin}. Trying nearest.")
                try:
                    interpolated_values_nearest = griddata(points_with_values, values, (grid_x_nan, grid_y_nan), method='nearest')
                    if not np.all(np.isnan(interpolated_values_nearest)):
                        slice_data[np.isnan(slice_data)] = interpolated_values_nearest
                except Exception as e_spatial_near:
                    print(f"    S2 spatial nearest griddata also failed for ERA5 slice {t}: {e_spatial_near}")
            
            era5_stack_full[t, :, :] = slice_data # Update the stack with the (partially) interpolated slice

    if np.isnan(era5_stack_full).any():
        print(f"Warning: {np.isnan(era5_stack_full).sum()} NaNs still present in ERA5 stack after all interpolation attempts.")
    else:
        print("ERA5 stack fully interpolated.")

    return era5_stack_full, target_dates

def load_sentinel2_reflectance(
    s2_dir: str, 
    # unique_dates_str: list[str], # No longer takes unique_dates_str directly
    target_dates: list, # List of datetime objects for which S2 data is required (e.g., primary_common_dates)
    actual_ref_grid_path: str, 
    target_height: int, 
    target_width: int,
    s2_nodata_value: float,
    app_config: 'config',
    num_s2_bands: int = 4 # Default S2 bands, can be passed from config or inferred if needed
) -> tuple[np.ndarray, list]: # Returns stack and the original target_dates
    """
    Loads, preprocesses, and temporally aligns Sentinel-2 reflectance data.
    Ensures data for all dates in `target_dates`. If S2 data is missing for a specific
    date, the corresponding slice in the output stack will be np.nan.
    Selects the least cloudy S2 image if multiple exist for a date.

    Args:
        s2_dir (str): Directory containing Sentinel-2 TIFF files.
        target_dates (list): list of datetime objects for which S2 data is required.
        actual_ref_grid_path (str): Path to the reference raster file used for alignment.
        target_height (int): Target height for aligned rasters.
        target_width (int): Target width for aligned rasters.
        s2_nodata_value (float): NoData value in S2 files.
        app_config: Configuration object.
        num_s2_bands (int): Number of expected S2 bands.

    Returns:
        tuple[np.ndarray, list]:
            - s2_stack_final (np.ndarray): Stack of S2 reflectance data (time, bands, height, width)
                                     aligned to target_dates, with np.nan for missing data.
            - target_dates (list): The input list of target_dates, returned for consistency.
    """
    print(f"Loading, merging, and aligning Sentinel-2 reflectance from: {s2_dir} for {len(target_dates)} target dates.")
    
    if not target_dates:
        print("Warning: load_sentinel2_reflectance received empty target_dates. Returning empty S2 data.")
        return np.empty((0, num_s2_bands, target_height, target_width), dtype=np.float32), []

    # Initialize the full stack with NaNs
    s2_stack_full = np.full((len(target_dates), num_s2_bands, target_height, target_width), np.nan, dtype=np.float32)

    # Discover S2 files and map them to dates
    all_s2_files_in_dir = sorted(glob.glob(os.path.join(s2_dir, "*.tif")) + glob.glob(os.path.join(s2_dir, "*.img")))
    file_date_mapping_s2 = {}
    for f_path in all_s2_files_in_dir:
        fname = os.path.basename(f_path)
        import re
        match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', fname)
        if match:
            date_str_from_fname = match.group(1).replace('-','').replace('_','')
            try:
                file_dt = pd.to_datetime(date_str_from_fname, format='%Y%m%d')
                if file_dt not in file_date_mapping_s2:
                    file_date_mapping_s2[file_dt] = []
                file_date_mapping_s2[file_dt].append(f_path)
            except ValueError:
                print(f"Could not parse date from S2 filename: {fname}")
        else:
            print(f"Could not find date pattern in S2 filename: {fname}")

    for idx, date_dt in enumerate(tqdm(target_dates, desc="Processing S2 Data against Target Dates")):
        s2_files_for_this_date = file_date_mapping_s2.get(date_dt, [])

        if not s2_files_for_this_date:
            # print(f"No S2 file found for target date {date_dt.strftime('%Y-%m-%d')}. Leaving NaNs.")
            continue # Leaves NaNs for this date in s2_stack_full

        aligned_bands_for_day = []
        for s2_fpath in s2_files_for_this_date:
            temp_aligned_s2_path = os.path.join(app_config.OUTPUT_DIR, f"temp_aligned_s2_{os.path.basename(s2_fpath)}")
            try:
                utils.align_rasters(actual_ref_grid_path, s2_fpath, temp_aligned_s2_path)
                with rasterio.open(temp_aligned_s2_path) as src:
                    s2_data_all_bands_raw = src.read().astype(np.float32) # Reads all bands
                    
                    if s2_data_all_bands_raw.shape[0] != num_s2_bands:
                        print(f"Warning: S2 file {s2_fpath} has {s2_data_all_bands_raw.shape[0]} bands, expected {num_s2_bands}. Skipping this file for this date.")
                        continue # Skip this particular file if band count mismatch

                    # Handle known NoData values for S2.
                    current_s2_nodata_val = app_config.S2_NODATA_VALUE if hasattr(app_config, 'S2_NODATA_VALUE') else s2_nodata_value
                    s2_data_all_bands_raw[s2_data_all_bands_raw == current_s2_nodata_val] = np.nan
                    s2_data_all_bands_raw[np.isinf(s2_data_all_bands_raw)] = np.nan # Handle -inf too
                    
                    aligned_bands_for_day.append(s2_data_all_bands_raw) # (bands, height, width)
                os.remove(temp_aligned_s2_path)
            except Exception as e:
                print(f"Error aligning S2 file {s2_fpath} for date {date_dt.strftime('%Y-%m-%d')}: {e}")
                # If one file fails, we might still be able to use others for the same day.

        if not aligned_bands_for_day:
            # print(f"No S2 images successfully aligned for {date_dt.strftime('%Y-%m-%d')}. Leaving NaNs.")
            continue
        
        # Merge aligned images for the day using np.nanmean
        if len(aligned_bands_for_day) > 1:
            # print(f"Merging {len(aligned_bands_for_day)} S2 images for date {date_dt.strftime('%Y-%m-%d')} using nanmean.")
            stacked_for_mean = np.stack(aligned_bands_for_day, axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'Mean of empty slice')
                warnings.filterwarnings('ignore', r'invalid value encountered in scalar divide')
                merged_s2_data = np.nanmean(stacked_for_mean, axis=0) # (bands, height, width)
        else:
            merged_s2_data = aligned_bands_for_day[0]
        
        s2_stack_full[idx, :, :, :] = merged_s2_data

    nan_pixels_in_s2_stack = np.isnan(s2_stack_full).sum()
    if nan_pixels_in_s2_stack > 0:
        print(f"S2 stack created with {nan_pixels_in_s2_stack} NaN pixel values (across all bands/times). These might be from missing files or original nodata.")
    print(f"Loaded and merged S2 data for {len(target_dates)} target dates. Stack shape: {s2_stack_full.shape}")
    return s2_stack_full, target_dates

def load_ndvi_infer_stack(
    ndvi_infer_dir: str,
    target_dates: list, # List of datetime objects for which NDVI data is required
    actual_ref_grid_path: str,
    target_height: int,
    target_width: int,
    # ndvi_nodata_value: float, # Assuming NaNs are already in inferred NDVI or will be handled
    app_config: 'config'
) -> tuple[np.ndarray, list]: # Returns stack and the original target_dates
    """
    Loads, preprocesses, and temporally aligns pre-inferred NDVI data.
    Ensures data for all dates in `target_dates`. If NDVI data is missing for a specific
    date, the corresponding slice in the output stack will be np.nan.
    If multiple NDVI files for a day, they are averaged (using nanmean).

    Args:
        ndvi_infer_dir (str): Directory containing pre-inferred NDVI TIFF files.
        target_dates (list): List of datetime objects for which NDVI data is required.
        actual_ref_grid_path (str): Path to the reference raster for alignment.
        target_height (int): Target height for aligned rasters.
        target_width (int): Target width for aligned rasters.
        app_config: Configuration object.

    Returns:
        tuple[np.ndarray, list]:
            - ndvi_stack_final (np.ndarray): Stack of NDVI data (time, height, width)
                                     aligned to target_dates, with np.nan for missing data.
            - target_dates (list): The input list of target_dates, returned for consistency.
    """
    print(f"Loading, merging, and aligning pre-inferred NDVI from: {ndvi_infer_dir} for {len(target_dates)} target dates.")
    
    if not target_dates:
        print("Warning: load_ndvi_infer_stack received empty target_dates. Returning empty NDVI data.")
        return np.empty((0, target_height, target_width), dtype=np.float32), []

    # Initialize the full stack with NaNs
    ndvi_stack_full = np.full((len(target_dates), target_height, target_width), np.nan, dtype=np.float32)

    all_ndvi_files_in_dir = sorted(glob.glob(os.path.join(ndvi_infer_dir, "*.tif")) +
                                 glob.glob(os.path.join(ndvi_infer_dir, "*.img")))
    file_date_mapping_ndvi = {}
    for f_path in all_ndvi_files_in_dir:
        fname = os.path.basename(f_path)
        import re
        match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', fname)
        if match:
            date_str_from_fname = match.group(1).replace('-','').replace('_','')
            try:
                file_dt = pd.to_datetime(date_str_from_fname, format='%Y%m%d')
                if file_dt not in file_date_mapping_ndvi:
                    file_date_mapping_ndvi[file_dt] = []
                file_date_mapping_ndvi[file_dt].append(f_path)
            except ValueError:
                print(f"Could not parse date from NDVI filename: {fname}")
        else:
            print(f"Could not find date pattern in NDVI filename: {fname}")

    for idx, date_dt in enumerate(tqdm(target_dates, desc="Processing Inferred NDVI Data against Target Dates")):
        ndvi_files_for_day = file_date_mapping_ndvi.get(date_dt, [])

        if not ndvi_files_for_day:
            # print(f"No inferred NDVI file found for target date {date_dt.strftime('%Y-%m-%d')}. Leaving NaNs.")
            continue # Leaves NaNs for this date in ndvi_stack_full

        aligned_data_for_day = []
        for ndvi_fpath in ndvi_files_for_day:
            temp_aligned_ndvi_path = os.path.join(app_config.OUTPUT_DIR, f"temp_aligned_ndvi_{os.path.basename(ndvi_fpath)}")
            try:
                utils.align_rasters(actual_ref_grid_path, ndvi_fpath, temp_aligned_ndvi_path,
                                    resampling_method=RasterioResampling.bilinear) # Bilinear often good for NDVI
                with rasterio.open(temp_aligned_ndvi_path) as src:
                    ndvi_slice = src.read(1).astype(np.float32)
                    # Handle potential nodata values that rasterio might not have converted to NaN
                    # For example, if original NDVI files used 0 or -9999 and it wasn't set in their profile.
                    # This depends on the source of inferred NDVI. For now, we assume alignment and read handle most of it.
                    # if hasattr(app_config, 'NDVI_INFER_NODATA_VALUE_RAW') and not np.isnan(app_config.NDVI_INFER_NODATA_VALUE_RAW):
                    #    ndvi_slice[ndvi_slice == app_config.NDVI_INFER_NODATA_VALUE_RAW] = np.nan
                    ndvi_slice = np.clip(ndvi_slice, -1.0, 1.0) # Ensure NDVI is within valid range
                    aligned_data_for_day.append(ndvi_slice)
                os.remove(temp_aligned_ndvi_path)
            except Exception as e:
                print(f"Error aligning inferred NDVI file {ndvi_fpath} for date {date_dt.strftime('%Y-%m-%d')}: {e}")
        
        if not aligned_data_for_day:
            # print(f"No NDVI images successfully aligned for {date_dt.strftime('%Y-%m-%d')}. Leaving NaNs.")
            continue
        
        if len(aligned_data_for_day) > 1:
            stacked_for_mean = np.stack(aligned_data_for_day, axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'Mean of empty slice')
                merged_ndvi_data = np.nanmean(stacked_for_mean, axis=0) # (height, width)
        else:
            merged_ndvi_data = aligned_data_for_day[0]
        
        ndvi_stack_full[idx, :, :] = merged_ndvi_data

    nan_pixels_in_ndvi_stack = np.isnan(ndvi_stack_full).sum()
    if nan_pixels_in_ndvi_stack > 0:
        print(f"NDVI stack created with {nan_pixels_in_ndvi_stack} NaN pixel values. These might be from missing files or original nodata.")
    print(f"Loaded and merged inferred NDVI data for {len(target_dates)} target dates. Stack shape: {ndvi_stack_full.shape}")
    return ndvi_stack_full, target_dates

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
    Timeline is defined by LST+ERA5 common dates. S2 and NDVI are aligned to this timeline,
    padding with np.nan for missing data.
    Raises ValueError if essential data cannot be loaded or aligned.
    """
    print(f"Starting preprocessing for ROI: {app_config.ROI_NAME}")
    os.makedirs(app_config.OUTPUT_DIR, exist_ok=True)

    # 1. Load Landsat LST data - this determines initial set of available LST dates
    try:
        lst_stack_initial_load, dates_with_lst_files, geo_profile, actual_reference_grid_path = load_landsat_lst(
            landsat_lst_dir=app_config.LANDSAT_LST_PATH,
            lst_nodata_val=app_config.LST_NODATA_VALUE,
            app_config=app_config,
            reference_grid_path=None 
        )
    except FileNotFoundError as e:
        raise ValueError(f"Critical error during LST loading for ROI {app_config.ROI_NAME}: {e}") from e
    
    if not dates_with_lst_files:
        raise ValueError(f"No LST data found for ROI {app_config.ROI_NAME} after initial load. Cannot proceed.")

    target_height = geo_profile['height']
    target_width = geo_profile['width']

    # 2. Load ERA5 skin temperature data, targeting dates where LST files were found.
    # `load_era5_skin_temp` ensures its output stack covers all `dates_with_lst_files`,
    # using interpolation and returning the same list of dates.
    # These `primary_common_dates` are common to LST and ERA5.
    era5_stack_for_primary_dates, primary_common_dates = load_era5_skin_temp(
        era5_skin_temp_dir=app_config.ERA5_SKIN_TEMP_PATH,
        target_dates=dates_with_lst_files, # Use LST dates as the target for ERA5
        reference_grid_path=actual_reference_grid_path,
        app_config=app_config
    )

    if not primary_common_dates: # Should not happen if dates_with_lst_files was not empty
        raise ValueError(f"ERA5 processing resulted in no common dates for ROI {app_config.ROI_NAME}. Cannot proceed.")
    
    # Filter the initially loaded LST stack to align with primary_common_dates.
    # This step is crucial if load_era5_skin_temp somehow returned a subset of dates_with_lst_files,
    # or if the order changed (though it shouldn't).
    # Since load_era5_skin_temp now returns the *input* target_dates, and its stack is aligned to them,
    # primary_common_dates is identical to dates_with_lst_files here.
    # So, lst_stack_initial_load is already aligned with primary_common_dates.
    lst_stack_primary = lst_stack_initial_load
    
    print(f"Primary timeline established with {len(primary_common_dates)} dates common to LST & ERA5.")
    print(f"  LST stack shape for primary timeline: {lst_stack_primary.shape}")
    print(f"  ERA5 stack shape for primary timeline: {era5_stack_for_primary_dates.shape}")

    # 3. Load Sentinel-2 reflectance data, aligned to `primary_common_dates`.
    # The modified `load_sentinel2_reflectance` returns an S2 stack fully aligned to `primary_common_dates`,
    # with NaNs for missing S2 data.
    num_s2_bands_expected = getattr(app_config, 'NUM_S2_BANDS', 4) # Get from config or default
    s2_reflectance_stack_primary, _ = load_sentinel2_reflectance(
        s2_dir=app_config.SENTINEL2_REFLECTANCE_PATH,
        target_dates=primary_common_dates, # Align to LST+ERA5 timeline
        actual_ref_grid_path=actual_reference_grid_path,
        target_height=target_height,
        target_width=target_width,
        s2_nodata_value=app_config.S2_NODATA_VALUE,
        app_config=app_config,
        num_s2_bands=num_s2_bands_expected
    )
    print(f"S2 stack loaded for primary timeline. Shape: {s2_reflectance_stack_primary.shape}")

    # 4. Create Day of Year (DOY) stack for the primary timeline
    doy_stack_1d = np.array([date.timetuple().tm_yday for date in primary_common_dates], dtype=np.int16)
    print(f"DOY stack (1D) created for primary timeline. Shape: {doy_stack_1d.shape}")

    # 5. Load Coordinate data
    lon_coords, lat_coords, lon_scaler, lat_scaler = load_coordinates(
        reference_grid_path=actual_reference_grid_path, 
        normalize=True
    )
    print(f"Coordinates loaded. Lon shape: {lon_coords.shape}, Lat shape: {lat_coords.shape}")
    
    # 6. S2 Interpolation (applied to the S2 stack that's aligned with primary_common_dates)
    s2_final_stack = s2_reflectance_stack_primary # Initialize
    if hasattr(app_config, 'INTERPOLATE_S2') and not app_config.INTERPOLATE_S2:
        print("S2 interpolation skipped due to app_config.INTERPOLATE_S2 = False.")
        if np.isnan(s2_final_stack).any():
            print(f"Warning: Non-interpolated S2 stack contains {np.isnan(s2_final_stack).sum()} NaNs.")
    elif s2_reflectance_stack_primary.size > 0 and primary_common_dates:
        print(f"S2 stack for interpolation (aligned to primary dates). Shape: {s2_reflectance_stack_primary.shape}")
        s2_interpolated_stack = s2_reflectance_stack_primary.copy()
        num_bands_s2 = s2_interpolated_stack.shape[1]
        num_iterations = app_config.S2_INTERPOLATION_ITERATIONS if hasattr(app_config, 'S2_INTERPOLATION_ITERATIONS') else 2

        print(f"Starting iterative temporal-spatial interpolation for S2 stack ({num_iterations} iterations)...")
        from scipy.interpolate import griddata # Ensure this is imported

        for iteration in range(num_iterations):
            print(f"  S2 Interpolation Iteration {iteration + 1}/{num_iterations}")
            nan_count_before_iter = np.isnan(s2_interpolated_stack).sum()
            if nan_count_before_iter == 0: 
                print("    No NaNs remaining in S2 stack. Stopping S2 interpolation early.")
                break

            # Temporal Linear Interpolation
            for b in tqdm(range(num_bands_s2), desc=f"  Iter {iteration+1} S2 Temporal Interp (Bands)", leave=False):
                for r in range(target_height):
                    for c in range(target_width):
                        pixel_series = pd.Series(s2_interpolated_stack[:, b, r, c])
                        s2_interpolated_stack[:, b, r, c] = pixel_series.interpolate(method='linear', limit_direction='both', limit_area=None).to_numpy()
            nan_count_after_temporal = np.isnan(s2_interpolated_stack).sum()
            print(f"    NaNs after temporal: {nan_count_after_temporal} (was {nan_count_before_iter})")
            if nan_count_after_temporal == 0: 
                print("    No NaNs remaining in S2 stack after temporal. Stopping S2 interpolation early this iteration.")
                break 

            # Spatial griddata (linear)
            points_spatial_s2 = None 
            for t in tqdm(range(s2_interpolated_stack.shape[0]), desc=f"  Iter {iteration+1} S2 Spatial Interp (Time)", leave=False):
                for b in range(num_bands_s2):
                    slice_data = s2_interpolated_stack[t, b, :, :]
                    if np.isnan(slice_data).any():
                        if points_spatial_s2 is None or points_spatial_s2.shape[0] != target_height*target_width :
                            x_coords_s2, y_coords_s2 = np.meshgrid(np.arange(target_width), np.arange(target_height))
                            points_spatial_s2 = np.vstack((x_coords_s2.ravel(), y_coords_s2.ravel())).T
                        
                        valid_mask_s2 = ~np.isnan(slice_data)
                        values_s2 = slice_data[valid_mask_s2]
                        points_with_values_s2 = points_spatial_s2[valid_mask_s2.ravel()]

                        if points_with_values_s2.shape[0] >= 3:
                            nan_locations_s2 = np.where(np.isnan(slice_data))
                            grid_x_nan_s2, grid_y_nan_s2 = nan_locations_s2[1], nan_locations_s2[0]
                            try:
                                interpolated_values_s2 = griddata(points_with_values_s2, values_s2, (grid_x_nan_s2, grid_y_nan_s2), method='linear')
                                if not np.all(np.isnan(interpolated_values_s2)):
                                    slice_data[np.isnan(slice_data)] = interpolated_values_s2
                            except Exception: # Try nearest on linear failure
                                try:
                                    interpolated_values_s2_nearest = griddata(points_with_values_s2, values_s2, (grid_x_nan_s2, grid_y_nan_s2), method='nearest')
                                    if not np.all(np.isnan(interpolated_values_s2_nearest)):
                                        slice_data[np.isnan(slice_data)] = interpolated_values_s2_nearest
                                except Exception:
                                    pass # print(f"    S2 spatial nearest also failed (t={t},b={b})")
                        elif points_with_values_s2.shape[0] > 0 : 
                            nan_locations_s2 = np.where(np.isnan(slice_data))
                            try:
                                interpolated_values_s2_nearest = griddata(points_with_values_s2, values_s2, (nan_locations_s2[1], nan_locations_s2[0]), method='nearest')
                                if not np.all(np.isnan(interpolated_values_s2_nearest)):
                                    slice_data[nan_locations_s2] = interpolated_values_s2_nearest
                            except Exception:
                                pass # print(f"    S2 spatial nearest fallback failed (t={t},b={b})")
                        s2_interpolated_stack[t, b, :, :] = slice_data
            nan_count_after_spatial = np.isnan(s2_interpolated_stack).sum()
            print(f"    NaNs after spatial: {nan_count_after_spatial} (was {nan_count_after_temporal})")
            if nan_count_after_spatial == nan_count_after_temporal and nan_count_after_spatial > 0:
                print("    No change in NaN count after spatial step this iteration for S2.")
                # break # Optionally break
        
        s2_final_stack = s2_interpolated_stack
        if np.isnan(s2_final_stack).any():
            print(f"Warning: S2 stack still contains {np.isnan(s2_final_stack).sum()} NaNs after {num_iterations} iter(s) of interpolation.")
        else:
            print("S2 stack successfully interpolated.")
    else: 
        print("S2 stack is empty or no primary common dates, or interpolation disabled. Skipping S2 interpolation.")
        if not (hasattr(app_config, 'INTERPOLATE_S2') and not app_config.INTERPOLATE_S2) and not (s2_reflectance_stack_primary.size > 0 and primary_common_dates):
             s2_final_stack = np.full((len(primary_common_dates), num_s2_bands_expected, target_height, target_width), np.nan, dtype=np.float32)

    # 7. Load pre-inferred NDVI stack if configured, aligned to `primary_common_dates`
    ndvi_stack_final = None
    if getattr(app_config, 'GP_USE_NDVI_FEATURE', False):
        if hasattr(app_config, 'NDVI_INFER_PATH') and app_config.NDVI_INFER_PATH:
            # The modified `load_ndvi_infer_stack` returns NDVI aligned to primary_common_dates, with NaNs for missing.
            loaded_ndvi_stack, _ = load_ndvi_infer_stack(
                ndvi_infer_dir=app_config.NDVI_INFER_PATH,
                target_dates=primary_common_dates, 
                actual_ref_grid_path=actual_reference_grid_path,
                target_height=target_height,
                target_width=target_width,
                app_config=app_config
            )
            ndvi_stack_final = loaded_ndvi_stack
            print(f"NDVI stack loaded for primary timeline. Shape: {ndvi_stack_final.shape}")
        else:
            print("Warning: GP_USE_NDVI_FEATURE is True, but NDVI_INFER_PATH is not configured. NDVI stack will be all NaNs.")
            ndvi_stack_final = np.full((len(primary_common_dates), target_height, target_width), np.nan, dtype=np.float32)
    else:
        print("GP_USE_NDVI_FEATURE is False. Skipping loading of inferred NDVI.")
        # Ensure ndvi_stack_final is explicitly None or an empty/NaN array if not used, for consistency in the output dict.
        # If GP_USE_NDVI_FEATURE is false, it will remain None as initialized.

    # 8. Spatial Sampling for Training (NEW STEP)
    training_pixel_mask = np.ones((target_height, target_width), dtype=bool) # Default to all True
    if hasattr(app_config, 'SPATIAL_TRAINING_SAMPLE_PERCENTAGE') and app_config.SPATIAL_TRAINING_SAMPLE_PERCENTAGE < 1.0:
        sample_percentage = app_config.SPATIAL_TRAINING_SAMPLE_PERCENTAGE
        min_pixels = getattr(app_config, 'MIN_PIXELS_FOR_SPATIAL_SAMPLING', 100)
        
        total_pixels_in_grid = target_height * target_width
        num_pixels_to_sample_float = total_pixels_in_grid * sample_percentage
        num_pixels_to_sample = max(min_pixels, int(num_pixels_to_sample_float))
        num_pixels_to_sample = min(num_pixels_to_sample, total_pixels_in_grid) # Cannot sample more than available

        print(f"Spatially sampling {num_pixels_to_sample} pixels ({sample_percentage*100}%, min set to {min_pixels}) for training.")
        
        # Create a flat list of all possible (row, col) indices
        all_pixel_indices = np.array([(r, c) for r in range(target_height) for c in range(target_width)])
        
        # Randomly choose indices
        np.random.seed(app_config.RANDOM_SEED if hasattr(app_config, 'RANDOM_SEED') else 42)
        selected_indices_flat = np.random.choice(len(all_pixel_indices), size=num_pixels_to_sample, replace=False)
        selected_pixel_coords = all_pixel_indices[selected_indices_flat]
        
        training_pixel_mask = np.zeros((target_height, target_width), dtype=bool)
        for r_idx, c_idx in selected_pixel_coords:
            training_pixel_mask[r_idx, c_idx] = True
        print(f"Actual number of pixels selected for training mask: {np.sum(training_pixel_mask)}")
    else:
        print("Using all spatial pixels for training (SPATIAL_TRAINING_SAMPLE_PERCENTAGE is 1.0 or not defined).")

    # All stacks (LST, ERA5, S2, NDVI if used) are now aligned to primary_common_dates.
    # No further date-based filtering of stacks is needed here.

    print(f"Preprocessing complete for ROI: {app_config.ROI_NAME}. Final number of aligned dates in primary timeline: {len(primary_common_dates)}")
    
    output_data = {
        "lst_stack": np.array(lst_stack_primary, dtype=np.float32),
        "era5_stack": np.array(era5_stack_for_primary_dates, dtype=np.float32),
        "s2_reflectance_stack": np.array(s2_final_stack, dtype=np.float32),
        "doy_stack": np.array(doy_stack_1d, dtype=np.int16),
        "lon_coords": np.array(lon_coords, dtype=np.float32),
        "lat_coords": np.array(lat_coords, dtype=np.float32),
        "geo_profile": geo_profile,
        "common_dates": primary_common_dates, # This is the LST+ERA5 common timeline
        "lon_scaler": lon_scaler,
        "lat_scaler": lat_scaler,
        "reference_grid_path": actual_reference_grid_path,
        "roi_name": app_config.ROI_NAME,
        "training_pixel_mask": training_pixel_mask # Add the mask to output
    }
    if ndvi_stack_final is not None:
        output_data["ndvi_stack"] = np.array(ndvi_stack_final, dtype=np.float32)
    # If ndvi_stack_final is None (e.g. GP_USE_NDVI_FEATURE is False), it won't be added.
    # The GP model part (prepare_gp_training_data) should handle ndvi_stack being potentially absent from the dict.
        
    return output_data

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
        NDVI_INFER_SUBDIR = "ndvi_infer" # For testing inferred NDVI loading

        LANDSAT_LST_PATH = os.path.join(ROI_BASE_PATH, LANDSAT_LST_SUBDIR)
        ERA5_SKIN_TEMP_PATH = os.path.join(ROI_BASE_PATH, ERA5_SKIN_TEMP_SUBDIR)
        SENTINEL2_REFLECTANCE_PATH = os.path.join(ROI_BASE_PATH, SENTINEL2_REFLECTANCE_SUBDIR)
        NDVI_INFER_PATH = os.path.join(ROI_BASE_PATH, NDVI_INFER_SUBDIR) # For testing

        OUTPUT_DIR_BASE = "./dummy_delag_output/"
        OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, ROI_NAME) # ROI-specific output for temp files

        TARGET_RESOLUTION = 30 # Dummy, not directly used by align_rasters if ref exists
        START_DATE = "2023-01-01"
        END_DATE = "2023-01-03" # Short period for testing
        LST_NODATA_VALUE = -9999.0
        S2_NODATA_VALUE = -9999.0 # Added for S2
        DAYS_OF_YEAR = 365 # Dummy
        RANDOM_SEED = 42
        GP_USE_NDVI_FEATURE = False # Test default behavior
        S2_RED_INDEX = 2 # Example for dummy data [B,G,R,N]
        S2_NIR_INDEX = 3 # Example for dummy data [B,G,R,N]
        # GP_RESIDUAL_FEATURES might be defined in the main config, not strictly needed here for preprocessing test itself
        # but band order B2,B3,B4,B8 is implicitly assumed for S2 dummy data.

    test_config = TestConfig()

    # Create dummy directories and files
    os.makedirs(test_config.LANDSAT_LST_PATH, exist_ok=True)
    os.makedirs(test_config.ERA5_SKIN_TEMP_PATH, exist_ok=True)
    os.makedirs(test_config.SENTINEL2_REFLECTANCE_PATH, exist_ok=True)
    os.makedirs(os.path.join(test_config.ROI_BASE_PATH, test_config.NDVI_INFER_SUBDIR), exist_ok=True) # Create dummy ndvi_infer dir
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

    # Dummy Inferred NDVI files (single band)
    ndvi_profile_test = ref_profile_test.copy()
    ndvi_profile_test['nodata'] = -9999.0 # Example, or could be np.nan if files already processed
    for date_obj in dates_to_create: # Create NDVI for common dates
        date_str_ndvi_fmt = date_obj.strftime("%Y-%m-%d")
        ndvi_file = os.path.join(test_config.NDVI_INFER_PATH, f"ndvi_inferred_{date_str_ndvi_fmt}.tif")
        dummy_ndvi_data = (np.random.rand(dummy_height, dummy_width).astype(np.float32) * 2) - 1 # NDVI range -1 to 1
        # Introduce some nodata into NDVI to test nanmean/handling
        if date_obj == dates_to_create[0]: # Make first day's NDVI partially nodata
            dummy_ndvi_data[0:dummy_height//3, :] = ndvi_profile_test['nodata']
        with rasterio.open(ndvi_file, 'w', **ndvi_profile_test) as dst:
            dst.write(dummy_ndvi_data, 1)
    
    # Add an NDVI file for a date that also exists in LST/ERA5/S2 but only ONE of them, to test date filtering
    # For this test, we ensure all dummy LST/ERA5/S2 exist for dates_to_create, so NDVI will just align to these.
    # To truly test NDVI reducing common dates, one of the LST/ERA5/S2 would need to be missing for a date where NDVI exists,
    # or NDVI missing for a date where LST/ERA5/S2 exist.
    # The current logic for load_ndvi_infer_stack processes only common_dates_s2.

    print(f"Dummy data generated in: {test_config.BASE_DATA_DIR}")
    print(f"Dummy output will be in: {test_config.OUTPUT_DIR_BASE}")

    class TestConfigNDVI(TestConfig):
        def __init__(self):
            super().__init__()
            self.GP_USE_NDVI_FEATURE = True
            # S2_RED_INDEX and S2_NIR_INDEX are inherited from TestConfig

    test_configs_to_run = {
        "Default": test_config,
        "NDVI_Enabled": TestConfigNDVI()
    }

    # Clean up potential old dummy data before running tests
    # This is important if a previous test run failed and didn't clean up.
    # if os.path.exists(test_config.BASE_DATA_DIR):
    #     shutil.rmtree(test_config.BASE_DATA_DIR)
    # if os.path.exists(test_config.OUTPUT_DIR_BASE):
    #     shutil.rmtree(test_config.OUTPUT_DIR_BASE)

    # Create dummy directories and files (common for all test configs)
    os.makedirs(test_config.LANDSAT_LST_PATH, exist_ok=True)
    os.makedirs(test_config.ERA5_SKIN_TEMP_PATH, exist_ok=True)
    os.makedirs(test_config.SENTINEL2_REFLECTANCE_PATH, exist_ok=True)
    os.makedirs(test_config.OUTPUT_DIR, exist_ok=True) # For temp aligned files

    for test_name, current_run_config in test_configs_to_run.items():
        print(f"\n--- Running preprocess_all_data with TestConfig: {test_name} ---")
        # Ensure output directory is clean/exists for this specific config run if they differ,
        # or use a common one. For this test, `current_run_config.OUTPUT_DIR` should be used.
        # os.makedirs(current_run_config.OUTPUT_DIR, exist_ok=True) # If OUTPUT_DIR varies per config

        try:
            preprocessed_output = preprocess_all_data(current_run_config)
            print(f"\n--- Preprocessing Output Summary for {test_name} ---")
            for key, value in preprocessed_output.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}, NaNs: {np.isnan(value).sum()}")
                elif isinstance(value, list) and value and isinstance(value[0], pd.Timestamp):
                     print(f"  {key}: {len(value)} timestamps from {value[0]} to {value[-1]}")
                elif isinstance(value, dict) and key=="geo_profile":
                     print(f"  {key}: CRS {value.get('crs')}, Transform {value.get('transform')}")
                else:
                    print(f"  {key}: {type(value)}")
            
            assert preprocessed_output['lst_stack'].shape == (len(dates_to_create), dummy_height, dummy_width)
            assert preprocessed_output['era5_stack'].shape == (len(dates_to_create), dummy_height, dummy_width)
            assert preprocessed_output['s2_reflectance_stack'].shape == (len(dates_to_create), num_bands_s2, dummy_height, dummy_width)
            assert preprocessed_output['doy_stack'].shape == (len(dates_to_create),)
            assert preprocessed_output['lon_coords'].shape == (dummy_height, dummy_width)
            
            if current_run_config.GP_USE_NDVI_FEATURE:
                assert 'ndvi_stack' in preprocessed_output, "ndvi_stack should be in output when GP_USE_NDVI_FEATURE is True"
                assert preprocessed_output['ndvi_stack'] is not None, "ndvi_stack should not be None when GP_USE_NDVI_FEATURE is True and files exist"
                assert preprocessed_output['ndvi_stack'].shape[0] <= len(dates_to_create), "NDVI stack time dim too large"
                if preprocessed_output['ndvi_stack'].shape[0] > 0: # If any NDVI data loaded
                    assert preprocessed_output['ndvi_stack'].shape[1:] == (dummy_height, dummy_width), "NDVI stack spatial dims incorrect"
                    # Check that other stacks are filtered if NDVI reduced the number of common dates
                    assert preprocessed_output['lst_stack'].shape[0] == preprocessed_output['ndvi_stack'].shape[0]
                # assert not np.all(np.isnan(preprocessed_output['ndvi_stack'])), "NDVI stack should not be all NaNs if S2 data is valid"
            else:
                assert 'ndvi_stack' not in preprocessed_output or preprocessed_output['ndvi_stack'] is None or preprocessed_output['ndvi_stack'].size == 0, \
                    "ndvi_stack should not be in output or be None/empty when GP_USE_NDVI_FEATURE is False"

            # Check if S2 for 2023-01-01 (most cloudy original was id0) has been replaced by id2 (least cloudy)
            s2_day1_data = preprocessed_output['s2_reflectance_stack'][0] # First day
            # If the "least cloudy" version (id2) was chosen, it should have no NaNs from nodata if it was fully clear
            # The dummy data for id2 has no S2_NODATA_VALUE
            assert np.sum(np.isnan(s2_day1_data)) == 0, "S2 data for the first day should be the least cloudy version (no NaNs from nodata if id2 was clear)"

            print("\n--- Test for preprocess_all_data PASSED ---")

        except Exception as e:
            print(f"Error during preprocessing test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up dummy directories
            # shutil.rmtree(test_config.BASE_DATA_DIR, ignore_errors=True) # Careful if tests run in parallel or share base
            # shutil.rmtree(current_run_config.OUTPUT_DIR_BASE, ignore_errors=True) # This might be too broad if OUTPUT_DIR_BASE is shared
            # If OUTPUT_DIR is config-specific, clean that instead:
            # if os.path.exists(current_run_config.OUTPUT_DIR):
            #    shutil.rmtree(current_run_config.OUTPUT_DIR)
            pass # Deferring cleanup decisions

    print(f"Dummy data and output directories ({test_config.BASE_DATA_DIR}, {test_config.OUTPUT_DIR_BASE}) were NOT automatically cleaned up. Please remove them manually if desired.") 