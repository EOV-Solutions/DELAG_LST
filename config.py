"""
Configuration file for the DELAG project.
"""
import os
import numpy as np

# --- User Defined Paths for a Single ROI --- 
# Base directory containing all ROI folders
BASE_DATA_DIR = "/mnt/hdd12tb/code/nhatvm/DELAG/DELAG_LST" # USER TO VERIFY/SET THIS

# Name of the specific ROI folder to process from BASE_DATA_DIR
ROI_NAME = "KhanhXuan_BuonMaThuot_DakLak" # USER TO SET THIS to one of the subfolders

# Construct full paths for the selected ROI
ROI_BASE_PATH = os.path.join(BASE_DATA_DIR, ROI_NAME)

LANDSAT_LST_SUBDIR = "lst"
ERA5_SKIN_TEMP_SUBDIR = "era5"
SENTINEL2_REFLECTANCE_SUBDIR = "s2_images"
NDVI_INFER_SUBDIR = "ndvi_infer"

LANDSAT_LST_PATH = os.path.join(ROI_BASE_PATH, LANDSAT_LST_SUBDIR)
ERA5_SKIN_TEMP_PATH = os.path.join(ROI_BASE_PATH, ERA5_SKIN_TEMP_SUBDIR)
SENTINEL2_REFLECTANCE_PATH = os.path.join(ROI_BASE_PATH, SENTINEL2_REFLECTANCE_SUBDIR)
NDVI_INFER_PATH = os.path.join(ROI_BASE_PATH, NDVI_INFER_SUBDIR)
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
LST_NODATA_VALUE = np.nan # Value indicating no data or cloud in LST files
S2_NODATA_VALUE = np.nan # NoData value in S2 files
KNN_N_NEIGHBORS = 2 # Number of neighbors for KNN imputation
S2_INTERPOLATION_ITERATIONS = 5 # Number of iterations for S2 interpolation
INTERPOLATE_ERA5 = True  # Set to False to disable ERA5 interpolation
INTERPOLATE_S2 = False      # Set to False to disable S2 interpolation
GP_USE_TEMPORAL_MEAN_S2_FEATURES = False # Use temporal mean of S2 bands for GP features
GP_USE_NDVI_FEATURE = True # Set to True to use NDVI as a feature instead of S2 bands
# Define S2 band indices (0-indexed) for NDVI calculation if GP_USE_NDVI_FEATURE is True
# Assuming S2 bands are typically [Blue, Green, Red, NIR, ...]
# S2_RED_INDEX = 2  # Example: Corresponds to the 3rd band (Red)
# S2_NIR_INDEX = 3  # Example: Corresponds to the 4th band (NIR)

# --- ATC Model Hyperparameters ---
ATC_LEARNING_RATE = 0.001
ATC_EPOCHS = 1200
ATC_ENSEMBLE_SNAPSHOTS = 200
ATC_SNAPSHOT_INTERVAL = 4 # Save every 4 epochs
ATC_ENSEMBLE_START_EPOCH = ATC_EPOCHS - (ATC_ENSEMBLE_SNAPSHOTS * ATC_SNAPSHOT_INTERVAL)
MIN_CLEAR_OBS_ATC = 20 # Minimum number of clear sky observations to train an ATC model for a pixel
ATC_N_JOBS = 4  # Use all available CPU cores for ATC training. Set to 1 for no parallelization, or a specific number e.g., 4.

# --- GP Model Hyperparameters ---
# S2 bands will be annual/period means: Red, Green, Blue, NIR
GP_RESIDUAL_FEATURES = ['s2_red_mean', 's2_green_mean', 's2_blue_mean', 's2_nir_mean', 'norm_x', 'norm_y']
GP_LEARNING_RATE_INITIAL = 0.05
GP_EPOCHS_INITIAL = 100
GP_LEARNING_RATE_FINAL = 0.005
GP_EPOCHS_FINAL = 50
GP_MINI_BATCH_SIZE = 1024
GP_NUM_INDUCING_POINTS = 512

# --- Evaluation Parameters ---
EVAL_HOLDOUT_PERCENTAGE = 0.20 # For heavily cloudy scenario
# EVAL_SIMULATED_CLOUD_COVER_PERCENTAGE = [0.1, 0.3, 0.5, 0.7, 0.9] # Retained if needed

# --- General ---
RANDOM_SEED = 42
DEVICE = "cuda" # "cuda" if GPU is available, else "cpu" 