"""
Configuration file for the DELAG project.
"""
import os

# --- User Defined Paths for a Single ROI --- 
# Base directory containing all ROI folders
BASE_DATA_DIR = "/mnt/ssd1tb/code/nhatvm/data_lst_16days/" # USER TO VERIFY/SET THIS

# Name of the specific ROI folder to process from BASE_DATA_DIR
ROI_NAME = "KhanhXuan_BuonMaThuot_DakLak" # USER TO SET THIS to one of the subfolders

# Construct full paths for the selected ROI
ROI_BASE_PATH = os.path.join(BASE_DATA_DIR, ROI_NAME)

LANDSAT_LST_SUBDIR = "lst"
ERA5_SKIN_TEMP_SUBDIR = "era5"
SENTINEL2_REFLECTANCE_SUBDIR = "s2_images"

LANDSAT_LST_PATH = os.path.join(ROI_BASE_PATH, LANDSAT_LST_SUBDIR)
ERA5_SKIN_TEMP_PATH = os.path.join(ROI_BASE_PATH, ERA5_SKIN_TEMP_SUBDIR)
SENTINEL2_REFLECTANCE_PATH = os.path.join(ROI_BASE_PATH, SENTINEL2_REFLECTANCE_SUBDIR)
# COORDINATES_PATH is not directly used as a path, coordinates are derived from a reference raster.

# --- Output Paths ---
# It might be good to include ROI_NAME in the output directory structure too
OUTPUT_DIR_BASE = "output/"
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, ROI_NAME)
RECONSTRUCTED_LST_PATH = os.path.join(OUTPUT_DIR, "reconstructed_lst/")
UNCERTAINTY_MAPS_PATH = os.path.join(OUTPUT_DIR, "uncertainty_maps/")
EVALUATION_RESULTS_PATH = "evaluation_results.json" # This will be saved inside the ROI-specific OUTPUT_DIR

# --- Data Parameters ---
TARGET_RESOLUTION = 30  # meters
START_DATE = "2023-01-01" # Example, user should define for the ROI
END_DATE = "2024-12-31" # Example, user should define for the ROI
DAYS_OF_YEAR = 365 # Assuming non-leap year for simplicity in ATC, adjust if needed
LST_NODATA_VALUE = -9999 # Value indicating no data or cloud in LST files
S2_NODATA_VALUE = -9999 # NoData value in S2 files

# --- ATC Model Hyperparameters ---
ATC_LEARNING_RATE = 0.1
ATC_EPOCHS = 1200
ATC_ENSEMBLE_SNAPSHOTS = 200
ATC_SNAPSHOT_INTERVAL = 4 # Save every 4 epochs
ATC_ENSEMBLE_START_EPOCH = ATC_EPOCHS - (ATC_ENSEMBLE_SNAPSHOTS * ATC_SNAPSHOT_INTERVAL)
MIN_CLEAR_OBS_ATC = 10 # Minimum number of clear sky observations to train an ATC model for a pixel

# --- GP Model Hyperparameters ---
# S2 bands will be annual/period means: Red, Green, Blue, NIR
GP_RESIDUAL_FEATURES = ['s2_red_mean', 's2_green_mean', 's2_blue_mean', 's2_nir_mean', 'norm_x', 'norm_y']
GP_LEARNING_RATE_INITIAL = 0.05
GP_EPOCHS_INITIAL = 50
GP_LEARNING_RATE_FINAL = 0.005
GP_EPOCHS_FINAL = 10
GP_MINI_BATCH_SIZE = 1024
GP_NUM_INDUCING_POINTS = 512

# --- Evaluation Parameters ---
EVAL_HOLDOUT_PERCENTAGE = 0.20 # For heavily cloudy scenario
# EVAL_SIMULATED_CLOUD_COVER_PERCENTAGE = [0.1, 0.3, 0.5, 0.7, 0.9] # Retained if needed

# --- General ---
RANDOM_SEED = 42
DEVICE = "cuda" # "cuda" if GPU is available, else "cpu" 