"""
Configuration file for the DELAG project.
"""
import os
import numpy as np
import datetime

# --- User Defined Paths for a Single ROI --- 
# Base directory containing all ROI folders
BASE_DATA_DIR = "/mnt/hdd12tb/code/nhatvm/DELAG/DELAG_LST/download_data_v3" # USER TO VERIFY/SET THIS

# Name of the specific ROI folder to process from BASE_DATA_DIR
ROI_NAME = "KhanhXuan_BuonMaThuot_DakLak" # USER TO SET THIS to one of the subfolders

# Construct full paths for the selected ROI
ROI_BASE_PATH = os.path.join(BASE_DATA_DIR, ROI_NAME)

# --- LST Data Source Selection ---
USE_LST_LVL2 = False # SET TO True TO USE 'lst_lvl2' folder, False for 'lst'
LST_LVL1_SUBDIR = "lst"
LST_LVL2_SUBDIR = "lst_lvl2"

# Dynamically select the LST subdirectory based on the flag
LANDSAT_LST_SUBDIR = LST_LVL2_SUBDIR if USE_LST_LVL2 else LST_LVL1_SUBDIR

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
CURRENT_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LST_SOURCE_TAG = "lvl2" if USE_LST_LVL2 else "lvl1"
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, f"{ROI_NAME}_{LST_SOURCE_TAG}_{CURRENT_TIMESTAMP}")
RECONSTRUCTED_LST_PATH = os.path.join(OUTPUT_DIR, "reconstructed_lst/")
UNCERTAINTY_MAPS_PATH = os.path.join(OUTPUT_DIR, "uncertainty_maps/")
EVALUATION_RESULTS_PATH = "evaluation_results.json" # This will be saved inside the ROI-specific OUTPUT_DIR

# --- Data Parameters ---
TARGET_RESOLUTION = 30  # meters
START_DATE = "2015-01-01" # Example, user should define for the ROI
END_DATE = "2025-01-01" # Example, user should define for the ROI
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

# --- LST Outlier Detection ---
# Method can be 'percentile', 'mad', 'trend_detect', 'sudden_change', 'iqr', or 'none' to disable.
LST_OUTLIER_METHOD = 'percentile' 

# Parameters for 'percentile' method
LST_PERCENTILE_LOWER = 10 # Lower percentile to clip
LST_PERCENTILE_UPPER = 90 # Upper percentile to clip

# Parameter for 'mad' (Median Absolute Deviation) method
LST_MAD_THRESHOLD = 3.5 # Modified Z-score threshold

# Parameters for 'trend_detect' method (per-pixel seasonal decomposition)
# This method is computationally expensive.
LST_TREND_MIN_OBS = 20 # Min clear observations for a pixel to run trend detection
LST_TREND_RESID_THRESHOLD = 5.0 # Threshold for residual standard deviations to be an outlier

# Parameter for 'sudden_change' method
LST_SUDDEN_CHANGE_THRESHOLD_K = 10.0 # Absolute LST change in Kelvin between consecutive clear views to be flagged as an outlier.

# Parameter for 'iqr' (box plot) method
LST_IQR_MULTIPLIER = 1.5 # The multiplier for the IQR to determine outlier bounds.

# --- Spatial Sampling for Training ---
SPATIAL_TRAINING_SAMPLE_PERCENTAGE = 1 # Fraction of pixels to use for training (1.0 = all pixels)
MIN_PIXELS_FOR_SPATIAL_SAMPLING = 100     # Minimum number of pixels if SPATIAL_TRAINING_SAMPLE_PERCENTAGE < 1.0

# --- ATC Model Hyperparameters ---
ATC_LEARNING_RATE = 0.1
ATC_EPOCHS = 1200
ATC_INIT_SEARCH_TRIALS = 30
ATC_INIT_SEARCH_EPOCHS = 200
ATC_WEIGHT_DECAY = 1e-3  # L2 regularization strength
ATC_LR_SCHEDULER_PATIENCE = 15 # Patience for ReduceLROnPlateau
ATC_LR_SCHEDULER_FACTOR = 0.1   # Factor for ReduceLROnPlateau
ATC_LR_SCHEDULER_MIN_LR = 1e-4  # Minimum LR for ReduceLROnPlateau
ATC_ENSEMBLE_SNAPSHOTS = 200
ATC_SNAPSHOT_INTERVAL = 4 # Save every 4 epochs
ATC_ENSEMBLE_START_EPOCH = ATC_EPOCHS - (ATC_ENSEMBLE_SNAPSHOTS * ATC_SNAPSHOT_INTERVAL)
MIN_CLEAR_OBS_ATC = 30 # Minimum number of clear sky observations to train an ATC model for a pixel
ATC_N_JOBS = 16  # Use all available CPU cores for ATC training. Set to 1 for no parallelization, or a specific number e.g., 4.
ATC_LOSS_LOGGING_INTERVAL = 100 # Log loss every N epochs for map generation

# --- GP Model Hyperparameters ---
# S2 bands will be annual/period means: Red, Green, Blue, NIR
GP_RESIDUAL_FEATURES = ['s2_red_mean', 's2_green_mean', 's2_blue_mean', 's2_nir_mean', 'norm_x', 'norm_y']
GP_LEARNING_RATE_INITIAL = 0.05
GP_EPOCHS_INITIAL = 50
GP_LEARNING_RATE_FINAL = 0.005
GP_EPOCHS_FINAL = 20
GP_MINI_BATCH_SIZE = 1024
GP_NUM_INDUCING_POINTS = 512
GP_LOSS_LOGGING_INTERVAL = 10 # Log GP loss every N epochs for plot

# --- Evaluation Parameters ---
EVAL_HOLDOUT_PERCENTAGE = 0.20 # For heavily cloudy scenario
# EVAL_SIMULATED_CLOUD_COVER_PERCENTAGE = [0.1, 0.3, 0.5, 0.7, 0.9] # Retained if needed
MAX_DAYS_FOR_DAILY_VISUALIZATION_PLOT = 10 # Max days for the daily comparison plot

# --- General ---
RANDOM_SEED = 42
DEVICE = "cpu" # General default device, "cuda" if GPU is available, else "cpu". ATC and GP models will use specific settings below.
ATC_DEVICE = "cpu"  # Device for ATC model: "cuda" or "cpu"
GP_DEVICE = "cuda"   # Device for GP model: "cuda" or "cpu"

MODEL_WEIGHTS_PATH = os.path.join(BASE_DATA_DIR, "output_models", f"{ROI_NAME}_{LST_SOURCE_TAG}_{CURRENT_TIMESTAMP}")
GP_MODEL_WEIGHT_FILENAME = "gp_model_and_likelihood.pth" # Filename for saved GP model 