import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np

# 1. Define paths to your data folders
base_data_path = "DELAG_LST/KhanhXuan_BuonMaThuot_DakLak/"
# s2_folder = os.path.join(base_data_path, "s2_images")
ndvi_folder = os.path.join(base_data_path, "ndvi_infer")
lst_folder = os.path.join(base_data_path, "lst")
era5_folder = os.path.join(base_data_path, "era5")

# 2. Helper function to extract dates from filenames
def extract_date_from_filename(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    return None

# 3. Get all file paths and extract dates
all_files = []
for folder in [ndvi_folder, lst_folder, era5_folder]:
    for f_name in os.listdir(folder):
        if f_name.endswith(".tif"):
            full_path = os.path.join(folder, f_name)
            date = extract_date_from_filename(f_name)
            if date:
                all_files.append({"path": full_path, "date": date, "folder": folder})

# Sort files by date
all_files.sort(key=lambda x: x["date"])

if not all_files:
    print("No TIFF files found. Please check the paths and file names.")
    exit()

# Group files by 7-day intervals
start_date_period = all_files[0]["date"]
grouped_files = defaultdict(lambda: defaultdict(list))

for file_info in all_files:
    current_date = file_info["date"]
    # Determine the week group
    delta_days = (current_date - start_date_period).days
    week_number = delta_days // 7
    period_start_date = start_date_period + timedelta(days=week_number * 7)
    
    folder_type = os.path.basename(file_info["folder"])
    grouped_files[period_start_date][folder_type].append(file_info["path"])


# 4. Iterate through each 7-day group and plot
for period_start, types_in_period in grouped_files.items():
    period_end = period_start + timedelta(days=6)
    print(f"\nProcessing period: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")

    # Determine the number of subplots needed for this period
    # Max 2 (s2) + 1 (ndvi) + 1 (lst) + 2 (era5) = 6 plots per image file in the period
    # We will take the first image of each type for simplicity in this example.
    # For a more complex scenario, one might average images within the period or create a mosaic.

    plot_count = 0
    if types_in_period.get("s2_images"): plot_count += 2 # RGB and N
    if types_in_period.get("ndvi_infer"): plot_count += 1
    if types_in_period.get("lst"): plot_count += 1
    if types_in_period.get("era5"): plot_count += 2 # Two bands

    if plot_count == 0:
        print(f"  No images to plot for this period.")
        continue

    fig, axes = plt.subplots(1, plot_count, figsize=(5 * plot_count, 5))
    if plot_count == 1: # Matplotlib returns a single Axes object if only one subplot
        axes = [axes]
    
    fig.suptitle(f"Images for {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}", fontsize=16)
    
    current_ax_idx = 0

    # Plot S2 images (RGB and N)
    if "s2_images" in types_in_period and types_in_period["s2_images"]:
        s2_path = types_in_period["s2_images"][0] # Taking the first S2 image in the period
        print(f"  Processing S2 image: {s2_path}")
        try:
            with rasterio.open(s2_path) as src:
                if src.count >= 4: # Ensure there are at least 4 bands (R,G,B,N)
                    # Read bands: Assuming order is R, G, B, N
                    # Adjust band indices if they are different (e.g., B, G, R, N)
                    # Sentinel-2 often has B2(Blue), B3(Green), B4(Red), B8(NIR)
                    # For simplicity, using 1,2,3 for RGB and 4 for NIR
                    red_raw = src.read(1).astype(np.float32)
                    green_raw = src.read(2).astype(np.float32)
                    blue_raw = src.read(3).astype(np.float32)
                    nir = src.read(4).astype(np.float32) # NIR band

                    # The following print statements are commented out but can be useful for debugging band statistics.
                    # print(f"    Raw Red band stats: min={np.nanmin(red_raw):.2f}, max={np.nanmax(red_raw):.2f}, mean={np.nanmean(red_raw):.2f}, dtype={red_raw.dtype}, shape={red_raw.shape}")
                    # print(f"    Raw Green band stats: min={np.nanmin(green_raw):.2f}, max={np.nanmax(green_raw):.2f}, mean={np.nanmean(green_raw):.2f}, dtype={green_raw.dtype}, shape={green_raw.shape}")
                    # print(f"    Raw Blue band stats: min={np.nanmin(blue_raw):.2f}, max={np.nanmax(blue_raw):.2f}, mean={np.nanmean(blue_raw):.2f}, dtype={blue_raw.dtype}, shape={blue_raw.shape}")
                    
                    # # Check for all NaNs in raw bands (useful for debugging)
                    # if np.all(np.isnan(red_raw)) or np.all(np.isnan(green_raw)) or np.all(np.isnan(blue_raw)):
                    #     print("    WARNING: One or more raw RGB bands are all NaN.")

                    # Normalize for display (simple min-max scaling)
                    def normalize_percentile(band, band_name="", p_low=2, p_high=98):
                        non_nan_values = band[~np.isnan(band)]
                        if non_nan_values.size == 0:
                            # print(f"      WARNING: {band_name} band is all NaN. Returning as is.")
                            return band # Return the all-NaN band

                        low_val, high_val = np.percentile(non_nan_values, [p_low, p_high])
                        # print(f"      Normalizing {band_name} with percentile clip: original min={np.nanmin(band):.2f}, max={np.nanmax(band):.2f}")
                        # print(f"      Percentiles: p{p_low}={low_val:.2f}, p{p_high}={high_val:.2f}")
                    
                        if low_val == high_val: # Avoid division by zero if percentiles are the same
                            # print(f"      WARNING: {band_name} band percentiles {p_low}-{p_high} are constant. Clamping to 0 or 0.5.")
                            # This case means most data is constant within this percentile range.
                            # We can scale based on whether the constant value is high or low relative to typical S2 ranges.
                            # For simplicity, let's scale it to 0.5 if not NaN.
                            # Or, more simply, return a band of zeros or NaNs if it's all constant.
                            return np.full_like(band, 0.5 if not np.isnan(low_val) else np.nan)

                        # Clip values to the percentile range
                        clipped_band = np.clip(band, low_val, high_val)
                        
                        # Now normalize the clipped band to 0-1
                        normalized_band = (clipped_band - low_val) / (high_val - low_val + 1e-8)
                        return normalized_band

                    red_norm = normalize_percentile(red_raw, "Red") # Normalize Red band
                    green_norm = normalize_percentile(green_raw, "Green") # Normalize Green band
                    blue_norm = normalize_percentile(blue_raw, "Blue") # Normalize Blue band

                    # The following print statements are commented out but can be useful for debugging normalized band statistics.
                    # print(f"    Normalized Red band stats: min={np.nanmin(red_norm):.2f}, max={np.nanmax(red_norm):.2f}, mean={np.nanmean(red_norm):.2f}")
                    # print(f"    Normalized Green band stats: min={np.nanmin(green_norm):.2f}, max={np.nanmax(green_norm):.2f}, mean={np.nanmean(green_norm):.2f}")
                    # print(f"    Normalized Blue band stats: min={np.nanmin(blue_norm):.2f}, max={np.nanmax(blue_norm):.2f}, mean={np.nanmean(blue_norm):.2f}")

                    rgb_image = np.dstack((red_norm, green_norm, blue_norm))
                    
                    # Check if normalized image is all NaN
                    if np.all(np.isnan(rgb_image)):
                        # print("    WARNING: Normalized RGB image is all NaN. Plot will be blank.") # Useful debug for blank plots
                        pass # Added pass to fix empty block
                    
                    # Plot RGB
                    ax_rgb = axes[current_ax_idx]
                    ax_rgb.imshow(rgb_image)
                    ax_rgb.set_title(f"S2 RGB\n{os.path.basename(s2_path)}")
                    ax_rgb.axis('off')
                    current_ax_idx += 1

                    # Plot NIR
                    ax_nir = axes[current_ax_idx]
                    show(nir, ax=ax_nir, cmap='viridis', title=f"S2 NIR Band\n{os.path.basename(s2_path)}")
                    ax_nir.axis('off')
                    current_ax_idx += 1
                else:
                    print(f"  S2 image {os.path.basename(s2_path)} has less than 4 bands.")
                    if current_ax_idx < len(axes): axes[current_ax_idx].axis('off'); current_ax_idx +=1 # Skip first plot
                    if current_ax_idx < len(axes): axes[current_ax_idx].axis('off'); current_ax_idx +=1 # Skip second plot


        except Exception as e:
                print(f"  Error processing S2 image {s2_path}: {e}")
                if current_ax_idx < len(axes): axes[current_ax_idx].axis('off'); current_ax_idx +=1 
                if current_ax_idx < len(axes): axes[current_ax_idx].axis('off'); current_ax_idx +=1


    # Plot NDVI images
    if "ndvi_infer" in types_in_period and types_in_period["ndvi_infer"]:
        ndvi_path = types_in_period["ndvi_infer"][0]
        try:
            with rasterio.open(ndvi_path) as src:
                ndvi_band = src.read(1)
                ax_ndvi = axes[current_ax_idx]
                show(ndvi_band, ax=ax_ndvi, cmap='RdYlGn', title=f"NDVI\n{os.path.basename(ndvi_path)}")
                ax_ndvi.axis('off')
                current_ax_idx += 1
        except Exception as e:
            print(f"  Error processing NDVI image {ndvi_path}: {e}")
            if current_ax_idx < len(axes): axes[current_ax_idx].axis('off'); current_ax_idx +=1


    # Plot LST images
    if "lst" in types_in_period and types_in_period["lst"]:
        lst_path = types_in_period["lst"][0]
        try:
            with rasterio.open(lst_path) as src:
                lst_band = src.read(1) # Kelvin
                # Optionally convert to Celsius: lst_band_celsius = lst_band - 273.15
                ax_lst = axes[current_ax_idx]
                show(lst_band, ax=ax_lst, cmap='coolwarm', title=f"LST (K)\n{os.path.basename(lst_path)}")
                ax_lst.axis('off')
                current_ax_idx += 1
        except Exception as e:
            print(f"  Error processing LST image {lst_path}: {e}")
            if current_ax_idx < len(axes): axes[current_ax_idx].axis('off'); current_ax_idx +=1

    # Plot ERA5 images
    if "era5" in types_in_period and types_in_period["era5"]:
        era5_path = types_in_period["era5"][0]
        try:
            with rasterio.open(era5_path) as src:
                if src.count >= 2:
                    era5_band1 = src.read(1) # Kelvin
                    era5_band2 = src.read(2) # Kelvin
                    # Optionally convert to Celsius
                    
                    ax_era1 = axes[current_ax_idx]
                    show(era5_band1, ax=ax_era1, cmap='plasma', title=f"ERA5 Band 1 (K)\n{os.path.basename(era5_path)}")
                    ax_era1.axis('off')
                    current_ax_idx += 1

                    ax_era2 = axes[current_ax_idx]
                    show(era5_band2, ax=ax_era2, cmap='plasma', title=f"ERA5 Band 2 (K)\n{os.path.basename(era5_path)}")
                    ax_era2.axis('off')
                    current_ax_idx += 1
                else:
                    print(f"  ERA5 image {os.path.basename(era5_path)} has less than 2 bands.")
                    if current_ax_idx < len(axes): axes[current_ax_idx].axis('off'); current_ax_idx +=1
                    if current_ax_idx < len(axes): axes[current_ax_idx].axis('off'); current_ax_idx +=1


        except Exception as e:
            print(f"  Error processing ERA5 image {era5_path}: {e}")
            if current_ax_idx < len(axes): axes[current_ax_idx].axis('off'); current_ax_idx +=1
            if current_ax_idx < len(axes): axes[current_ax_idx].axis('off'); current_ax_idx +=1
            
    # Hide any unused subplots
    for i in range(current_ax_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.savefig(f"/mnt/ssd1tb/code/nhatvm/DELAG/DELAG_LST/KhanhXuan_BuonMaThuot_DakLak/visualize/plot_period_{period_start.strftime('%Y-%m-%d')}.png")

print("\nFinished processing all periods.")