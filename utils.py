"""
Utility functions for the DELAG project.
"""
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RasterioResampling # Alias to avoid conflict

def normalize_coordinates(x_coords: np.ndarray, y_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Normalize x and y coordinates to the range [0, 1].

    Args:
        x_coords (np.ndarray): Array of x coordinates.
        y_coords (np.ndarray): Array of y coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
            Normalized x coordinates, normalized y coordinates,
            fitted x scaler, and fitted y scaler.
    """
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    # Reshape if 1D array for scaler
    x_coords_reshaped = x_coords.reshape(-1, 1) if x_coords.ndim == 1 else x_coords
    y_coords_reshaped = y_coords.reshape(-1, 1) if y_coords.ndim == 1 else y_coords

    norm_x = x_scaler.fit_transform(x_coords_reshaped).reshape(x_coords.shape)
    norm_y = y_scaler.fit_transform(y_coords_reshaped).reshape(y_coords.shape)

    return norm_x, norm_y, x_scaler, y_scaler

def create_output_directories(config):
    """
    Creates necessary output directories defined in the config.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.RECONSTRUCTED_LST_PATH, exist_ok=True)
    os.makedirs(config.UNCERTAINTY_MAPS_PATH, exist_ok=True)
    print(f"Output directories created/ensured at {config.OUTPUT_DIR}")

def resample_raster(input_path: str, output_path: str, target_resolution: tuple[float, float], target_crs: str = None, resampling_method: RasterioResampling = RasterioResampling.nearest):
    """
    Resamples a raster to a target resolution and optionally a target CRS.

    Args:
        input_path (str): Path to the input raster file.
        output_path (str): Path to save the resampled raster.
        target_resolution (tuple[float, float]): Target resolution (x_res, y_res).
        target_crs (str, optional): Target CRS (e.g., 'EPSG:32632'). If None, uses source CRS.
        resampling_method (RasterioResampling, optional): Resampling method. Defaults to nearest.
    """
    with rasterio.open(input_path) as src:
        dst_crs = rasterio.CRS.from_string(target_crs) if target_crs else src.crs
        
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=target_resolution
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method
                )
    print(f"Resampled {input_path} to {output_path} with resolution {target_resolution}")

def align_rasters(reference_raster_path: str, raster_to_align_path: str, output_aligned_path: str, resampling_method: RasterioResampling = RasterioResampling.nearest):
    """
    Aligns a raster to a reference raster's grid (extent, resolution, CRS).

    Args:
        reference_raster_path (str): Path to the reference raster.
        raster_to_align_path (str): Path to the raster to be aligned.
        output_aligned_path (str): Path to save the aligned raster.
        resampling_method (RasterioResampling, optional): Resampling method. Defaults to nearest.
    """
    with rasterio.open(reference_raster_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        ref_bounds = ref.bounds
        ref_meta = ref.meta.copy()

    with rasterio.open(raster_to_align_path) as src:
        # Update meta for the output file based on the reference
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": ref_height,
            "width": ref_width,
            "transform": ref_transform,
            "crs": ref_crs
        })
        
        # Create an empty array for the output aligned raster
        # Using src.count to handle multi-band rasters correctly
        dst_array = np.empty((src.count, ref_height, ref_width), dtype=src.dtypes[0])


        with rasterio.open(output_aligned_path, 'w', **out_meta) as dst:
             for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=dst_array[i-1], # reproject into the pre-allocated array band
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling_method
                )
             dst.write(dst_array) # Write all bands at once

    print(f"Aligned {raster_to_align_path} to {reference_raster_path} and saved to {output_aligned_path}")

def get_day_of_year(dates: np.ndarray) -> np.ndarray:
    """
    Convert an array of datetime64 objects or YYYY-MM-DD strings to day of year (1-366).
    """
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates)
    if dates.dtype != '<M8[D]': # Check if not already datetime64[D]
        dates = dates.astype('datetime64[D]')
    
    # pandas Series can simplify datetime operations
    try:
        import pandas as pd
        return pd.Series(dates).dt.dayofyear.values
    except ImportError:
        # Manual calculation if pandas is not available or preferred
        # This is a simplified version, for more robust handling, pandas is recommended
        year_starts = dates.astype('datetime64[Y]')
        day_of_year = (dates - year_starts).astype('timedelta64[D]').astype(int) + 1
        return day_of_year

def save_array_as_geotiff(data_array: np.ndarray, reference_geotiff_path: str, output_path: str, nodata_value=None):
    """
    Saves a NumPy array as a GeoTIFF file, using metadata from a reference GeoTIFF.
    Assumes data_array is 2D (single band) or 3D (bands, height, width).
    """
    with rasterio.open(reference_geotiff_path) as ref:
        profile = ref.profile
        profile['driver'] = 'GTiff' # Ensure GeoTIFF format
        
        if data_array.ndim == 2:
            profile['count'] = 1
            data_array_to_write = data_array[np.newaxis, :, :] # Add band dimension
        elif data_array.ndim == 3:
            profile['count'] = data_array.shape[0]
            data_array_to_write = data_array
        else:
            raise ValueError(f"Input data_array must be 2D or 3D, got {data_array.ndim}D")

        profile['dtype'] = data_array.dtype
        if nodata_value is not None:
            profile['nodata'] = nodata_value

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data_array_to_write)
    print(f"Saved array to {output_path}")

def visualize_reconstruction_results(
    original_lst_day: np.ndarray, 
    reconstructed_lst_day: np.ndarray, 
    total_variance_day: np.ndarray, 
    reference_grid_path: str, # For extent/aspect if needed, or just for consistency
    output_dir: str,
    date_str: str,
    roi_name: str,
    save_plot: bool = True
):
    """
    Visualizes the original LST, reconstructed LST, and total variance for a specific day.

    Args:
        original_lst_day (np.ndarray): 2D array of original LST data (with NaNs).
        reconstructed_lst_day (np.ndarray): 2D array of reconstructed LST data.
        total_variance_day (np.ndarray): 2D array of total variance data.
        reference_grid_path (str): Path to a reference GeoTIFF (can be used for extent/aspect or context).
        output_dir (str): Directory to save the visualization.
        date_str (str): Date string for titles and filename (e.g., 'YYYYMMDD').
        roi_name (str): Name of the ROI for filename uniqueness.
        save_plot (bool): Whether to save the plot to a file.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import contextily as cx # For adding basemaps, optional enhancement
    except ImportError:
        print("Matplotlib or contextily is not installed. Skipping visualization.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"LST Reconstruction Results for {roi_name} - {date_str}", fontsize=16)

    # Determine common min/max for LST plots for consistent color scaling
    valid_lst_values = np.concatenate([
        original_lst_day[~np.isnan(original_lst_day)].flatten(),
        reconstructed_lst_day[~np.isnan(reconstructed_lst_day)].flatten()
    ])
    lst_min, lst_max = (np.min(valid_lst_values), np.max(valid_lst_values)) if valid_lst_values.size > 0 else (270, 320)

    # 1. Original LST
    im1 = axes[0].imshow(original_lst_day, cmap='plasma', vmin=lst_min, vmax=lst_max)
    axes[0].set_title("Original LST (with clouds as NaN)")
    axes[0].set_xlabel("X-coordinate")
    axes[0].set_ylabel("Y-coordinate")
    fig.colorbar(im1, ax=axes[0], orientation='horizontal', label='Temperature (K)')
    axes[0].grid(True, alpha=0.3)

    # 2. Reconstructed LST
    im2 = axes[1].imshow(reconstructed_lst_day, cmap='plasma', vmin=lst_min, vmax=lst_max)
    axes[1].set_title("Reconstructed LST")
    axes[1].set_xlabel("X-coordinate")
    axes[1].set_ylabel("Y-coordinate")
    fig.colorbar(im2, ax=axes[1], orientation='horizontal', label='Temperature (K)')
    axes[1].grid(True, alpha=0.3)

    # 3. Total Variance
    # Determine min/max for variance plot, ensuring it's not too skewed by outliers
    valid_var_values = total_variance_day[~np.isnan(total_variance_day)].flatten()
    var_min, var_max = (np.min(valid_var_values), np.percentile(valid_var_values, 98)) if valid_var_values.size > 0 else (0, 1)
    if var_max <= var_min: # Handle case where variance is constant or all NaN
        var_max = var_min + 1e-6 if valid_var_values.size > 0 else 1.0
        var_min = var_min -1e-6 if valid_var_values.size > 0 and var_min >0 else 0.0
        if var_max <= var_min: var_max = var_min + 0.1 # Final fallback for constant zero variance
            
    im3 = axes[2].imshow(total_variance_day, cmap='magma', vmin=var_min, vmax=var_max)
    axes[2].set_title("Total Variance")
    axes[2].set_xlabel("X-coordinate")
    axes[2].set_ylabel("Y-coordinate")
    fig.colorbar(im3, ax=axes[2], orientation='horizontal', label='Variance (K^2)')
    axes[2].grid(True, alpha=0.3)
    
    # Attempt to add basemap if contextily is available and reference grid provides CRS/extent
    # This is an enhancement and might require further adjustments based on CRS.
    try:
        with rasterio.open(reference_grid_path) as ref_src:
            extent = [ref_src.bounds.left, ref_src.bounds.right, ref_src.bounds.bottom, ref_src.bounds.top]
            for ax in axes:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                # cx.add_basemap(ax, crs=ref_src.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5)
                # Note: Contextily might require reprojection of data if not in web mercator.
                # For simplicity, this example just sets extent. True geographic overlay is more complex.
                pass # Placeholder for more advanced basemap integration if desired
    except Exception as e:
        print(f"Could not use reference grid for plot extent/basemap: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle

    if save_plot:
        plot_subdir = os.path.join(output_dir, "plots")
        os.makedirs(plot_subdir, exist_ok=True)
        filename = os.path.join(plot_subdir, f"reconstruction_vis_{roi_name}_{date_str}.png")
        plt.savefig(filename, dpi=150)
        print(f"Saved visualization to {filename}")
        plt.close(fig) # Close the figure to free memory
    else:
        plt.show()

def plot_s2_rgb(s2_slice_bands_hw, ax, title="S2 RGB", band_indices_rgb=(2,1,0)):
    """Helper to plot an S2 RGB image from a (bands, height, width) slice."""
    if s2_slice_bands_hw is None or s2_slice_bands_hw.shape[0] < max(band_indices_rgb) + 1:
        ax.text(0.5, 0.5, 'S2 Data Not Available', horizontalalignment='center', verticalalignment='center')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Extract R, G, B bands
    # Assuming band_indices_rgb provides (Red_idx, Green_idx, Blue_idx)
    red = s2_slice_bands_hw[band_indices_rgb[0], :, :]
    green = s2_slice_bands_hw[band_indices_rgb[1], :, :]
    blue = s2_slice_bands_hw[band_indices_rgb[2], :, :]

    # Stack to (height, width, channels)
    rgb_image = np.stack([red, green, blue], axis=-1)

    # Normalize/clip for display (simple percentile clip)
    # Replace NaNs with a value (e.g., 0) before percentile calculation to avoid issues, or use np.nanpercentile
    rgb_image_no_nan = np.nan_to_num(rgb_image, nan=0.0)
    
    # Calculate percentiles for each channel independently on non-NaN data if possible
    # For simplicity here, applying a general clip or normalization
    # A more robust approach would handle NaNs carefully in percentile calculation
    # and potentially apply per-channel contrast stretching.
    
    # Simple clip based on overall min/max of each channel after NaN handling
    # or use fixed percentile like 2% and 98% to enhance contrast if data range is large.
    # For reflectance data (0-1 theoretically, but can be higher/lower due to processing)
    # We can clip to a reasonable range e.g. 0 to 0.3-0.4 for visualization
    
    # Apply a robust scaling: map 2nd to 98th percentile to 0-1 for each band
    # This helps with atmospheric effects or outliers making images too dark/light.
    scaled_rgb_image = np.zeros_like(rgb_image_no_nan, dtype=float)
    for i in range(3):
        band_data = rgb_image_no_nan[..., i]
        if np.any(band_data): # Check if band has non-zero data
            min_val = np.percentile(band_data[band_data > -np.inf], 2) # use > -np.inf to handle potential -inf from nan_to_num if input was weird
            max_val = np.percentile(band_data[band_data < np.inf], 98)
            if max_val <= min_val: # Prevent division by zero or negative range
                scaled_rgb_image[..., i] = np.clip((band_data - min_val) / (1e-6), 0, 1)
            else:
                scaled_rgb_image[..., i] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
        else:
            scaled_rgb_image[..., i] = 0 # Black if band is all zero
            
    # Handle any remaining NaNs that might have slipped through (e.g. if original was all NaN)
    scaled_rgb_image = np.nan_to_num(scaled_rgb_image, nan=0.0)
    
    ax.imshow(scaled_rgb_image)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

def visualize_daily_stacks_comparison(
    lst_observed_stack: np.ndarray, 
    reconstructed_lst_stack: np.ndarray, 
    s2_reflectance_stack: np.ndarray, # (time, bands, H, W)
    ndvi_stack: np.ndarray, # (time, H, W), can be None
    common_dates: list, 
    output_base_dir: str, # e.g., config.OUTPUT_DIR
    roi_name: str,
    app_config: 'config', # Added to access GP_USE_NDVI_FEATURE
    s2_rgb_indices: tuple = (2, 1, 0), # (R, G, B) assuming B2,B3,B4,B8 -> B4=idx 2, B3=idx 1, B2=idx 0
    max_days_to_plot: int = 10, # Limit the number of columns for readability
    lst_contrast_percentiles: tuple = (2, 98) # Percentiles for LST contrast stretching (e.g., 2nd and 98th)
):
    """
    Visualizes a comparison of observed LST, reconstructed LST, and either S2 RGB or NDVI images 
    across multiple days in a grid plot.

    Args:
        lst_observed_stack (np.ndarray): (time, H, W) stack of observed LST.
        reconstructed_lst_stack (np.ndarray): (time, H, W) stack of reconstructed LST.
        s2_reflectance_stack (np.ndarray): (time, bands, H, W) stack of S2 reflectance.
        ndvi_stack (np.ndarray): (time, H, W) stack of NDVI data, can be None.
        common_dates (list): List of datetime objects corresponding to the time dimension.
        output_base_dir (str): Base output directory (e.g., config.OUTPUT_DIR).
        roi_name (str): Name of the ROI for filenames and titles.
        app_config (config): The application configuration object.
        s2_rgb_indices (tuple): Indices for R, G, B bands in the s2_reflectance_stack's band dimension.
        max_days_to_plot (int): Maximum number of days (columns) to plot.
        lst_contrast_percentiles (tuple): Lower and upper percentiles to clip LST data for contrast enhancement.
                                         Set to (0, 100) or None to use absolute min/max.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping daily stacks comparison visualization.")
        return

    num_available_days = lst_observed_stack.shape[0]
    num_days = min(num_available_days, max_days_to_plot)

    if num_days == 0:
        print("No data available to visualize for daily stacks comparison.")
        return

    fig, axes = plt.subplots(3, num_days, figsize=(num_days * 4, 3 * 4), squeeze=False)
    # squeeze=False ensures axes is always 2D, even if num_days=1
    
    plot_title_suffix = "S2 RGB"
    if getattr(app_config, 'GP_USE_NDVI_FEATURE', False) and ndvi_stack is not None:
        plot_title_suffix = "NDVI"

    fig.suptitle(f"Daily Comparison for {roi_name} (Observed LST, Reconstructed LST, {plot_title_suffix})", fontsize=16, y=0.99)

    # Determine common min/max for LST plots for consistent color scaling across days
    valid_obs_lst_values = lst_observed_stack[~np.isnan(lst_observed_stack)]
    valid_recon_lst_values = reconstructed_lst_stack[~np.isnan(reconstructed_lst_stack)]
    
    lst_min_val, lst_max_val = (270, 320) # Default fallback

    if valid_obs_lst_values.size > 0 and valid_recon_lst_values.size > 0:
        combined_lst_values = np.concatenate((valid_obs_lst_values, valid_recon_lst_values))
        if lst_contrast_percentiles and len(lst_contrast_percentiles) == 2:
            p_low, p_high = lst_contrast_percentiles
            lst_min_val = np.percentile(combined_lst_values, p_low)
            lst_max_val = np.percentile(combined_lst_values, p_high)
            print(f"LST visualization: Using percentile ({p_low}%, {p_high}%) scaling: min={lst_min_val:.2f}, max={lst_max_val:.2f}")
        else:
            lst_min_val = np.min(combined_lst_values)
            lst_max_val = np.max(combined_lst_values)
            print(f"LST visualization: Using absolute min/max scaling: min={lst_min_val:.2f}, max={lst_max_val:.2f}")
    elif valid_obs_lst_values.size > 0:
        if lst_contrast_percentiles and len(lst_contrast_percentiles) == 2:
            p_low, p_high = lst_contrast_percentiles
            lst_min_val = np.percentile(valid_obs_lst_values, p_low)
            lst_max_val = np.percentile(valid_obs_lst_values, p_high)
        else:
            lst_min_val = np.min(valid_obs_lst_values)
            lst_max_val = np.max(valid_obs_lst_values)
    elif valid_recon_lst_values.size > 0:
        if lst_contrast_percentiles and len(lst_contrast_percentiles) == 2:
            p_low, p_high = lst_contrast_percentiles
            lst_min_val = np.percentile(valid_recon_lst_values, p_low)
            lst_max_val = np.percentile(valid_recon_lst_values, p_high)
        else:
            lst_min_val = np.min(valid_recon_lst_values)
            lst_max_val = np.max(valid_recon_lst_values)
    
    # Ensure min_val is not greater than max_val after percentile clipping, can happen with near-constant images
    if lst_min_val >= lst_max_val:
        lst_min_val = lst_max_val - 1 # Arbitrary small difference to make imshow happy
        print(f"Warning: LST min_val >= max_val after percentile clip. Adjusted to min={lst_min_val:.2f}, max={lst_max_val:.2f}")

    for i in range(num_days):
        date_str = common_dates[i].strftime('%Y-%m-%d')
        
        # Row 0: Observed LST
        ax_obs = axes[0, i]
        im_obs = ax_obs.imshow(lst_observed_stack[i], cmap='coolwarm', vmin=lst_min_val, vmax=lst_max_val)
        ax_obs.set_title(f"{date_str}\nObserved LST (K)")
        ax_obs.axis('off')
        if i == 0: ax_obs.set_ylabel("Observed LST", fontsize=12)

        # Row 1: Reconstructed LST
        ax_recon = axes[1, i]
        im_recon = ax_recon.imshow(reconstructed_lst_stack[i], cmap='coolwarm', vmin=lst_min_val, vmax=lst_max_val)
        ax_recon.set_title(f"Reconstructed LST (K)") # Date is already in column title from above
        ax_recon.axis('off')
        if i == 0: ax_recon.set_ylabel("Reconstructed LST", fontsize=12)

        # Row 2: S2 RGB or NDVI
        ax_s2_or_ndvi = axes[2, i]
        use_ndvi_plot = getattr(app_config, 'GP_USE_NDVI_FEATURE', False)

        if use_ndvi_plot and ndvi_stack is not None and ndvi_stack.shape[0] > i:
            im_ndvi = ax_s2_or_ndvi.imshow(ndvi_stack[i], cmap='RdYlGn', vmin=-1, vmax=1)
            ax_s2_or_ndvi.set_title(f"NDVI")
            ax_s2_or_ndvi.axis('off')
            # Optional: Add a colorbar for NDVI if desired, though often the range is standard
            # if i == num_days - 1: # Add colorbar to the last plot
            #     fig.colorbar(im_ndvi, ax=ax_s2_or_ndvi, orientation='vertical', label='NDVI', fraction=0.046, pad=0.04)
        elif s2_reflectance_stack is not None and s2_reflectance_stack.shape[0] > i:
            plot_s2_rgb(s2_reflectance_stack[i], ax_s2_or_ndvi, title=f"S2 RGB", band_indices_rgb=s2_rgb_indices)
            # plot_s2_rgb already calls axis('off')
        else:
            ax_s2_or_ndvi.text(0.5, 0.5, 'Image Data Not Available', horizontalalignment='center', verticalalignment='center')
            ax_s2_or_ndvi.set_title(f"{plot_title_suffix}")
            ax_s2_or_ndvi.axis('off')

        if i == 0:
            y_label_row2 = "NDVI" if use_ndvi_plot and ndvi_stack is not None else "S2 RGB"
            ax_s2_or_ndvi.set_ylabel(y_label_row2, fontsize=12)

    # Add a single colorbar for LST plots, if desired, or individual ones are fine too.
    # For simplicity, no shared colorbar here. Each plot implicitly shows its scale via vmin/vmax.
    # If a shared colorbar is needed:
    # fig.colorbar(im_recon, ax=axes.ravel().tolist(), shrink=0.6, aspect=30, orientation='horizontal', label='Temperature (K)', pad=0.05)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle

    viz_output_dir = os.path.join(output_base_dir, "viz")
    os.makedirs(viz_output_dir, exist_ok=True)
    filename = os.path.join(viz_output_dir, f"daily_comparison_{roi_name}.jpg")
    
    try:
        plt.savefig(filename, dpi=150, format='jpg', quality=90)
        print(f"Saved daily comparison visualization to {filename}")
    except Exception as e:
        print(f"Error saving daily comparison plot as JPG: {e}. Trying PNG...")
        try:
            png_filename = os.path.join(viz_output_dir, f"daily_comparison_{roi_name}.png")
            plt.savefig(png_filename, dpi=150, format='png')
            print(f"Saved daily comparison visualization to {png_filename}")
        except Exception as ep:
            print(f"Error saving daily comparison plot as PNG: {ep}")
            
    plt.close(fig) # Close the figure to free memory 