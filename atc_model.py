"""
Enhanced Annual Temperature Cycle (ATC) model using PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from joblib import Parallel, delayed # Added for parallelization
import config # Assuming your config.py is in the same directory or accessible

class EnhancedATCModel(nn.Module):
    """
    Enhanced ATC model for a single pixel or a batch of pixels.
    T_ATC(d) = C + A * cos(2*pi/DOY_MAX * (d - phi)) + b * T_ERA5(d)
    Parameters C, A, phi, b are learnable.
    """
    def __init__(self, initial_params: dict = None):
        super().__init__()
        # Parameters: C (mean temp), A (amplitude), phi (phase shift), b (ERA5 coefficient)
        # Initialize with some reasonable defaults or provided values
        # These will be per-pixel, so if processing a batch of pixels, these should be vectors.
        # For a single pixel model instance:
        self.C = nn.Parameter(torch.tensor(initial_params.get('C', 290.0) if initial_params else 290.0)) # Avg temp in Kelvin
        self.A = nn.Parameter(torch.tensor(initial_params.get('A', 10.0) if initial_params else 10.0))   # Amplitude in Kelvin
        self.phi = nn.Parameter(torch.tensor(initial_params.get('phi', 180.0) if initial_params else 180.0)) # Phase shift in days
        self.b = nn.Parameter(torch.tensor(initial_params.get('b', 0.5) if initial_params else 0.5))     # ERA5 coefficient
        
        self.days_in_year = config.DAYS_OF_YEAR # From config file

    def forward(self, doy: torch.Tensor, t_era5: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ATC model.

        Args:
            doy (torch.Tensor): Day of year (1 to self.days_in_year).
            t_era5 (torch.Tensor): ERA5 skin temperature for the corresponding doy.

        Returns:
            torch.Tensor: Predicted LST (T_ATC).
        """
        term_cos = torch.cos(2 * np.pi / self.days_in_year * (doy - self.phi))
        t_atc = self.C + self.A * term_cos + self.b * t_era5
        return t_atc

def train_atc_model_pixelwise(
    pixel_lst_clear: np.ndarray, 
    pixel_doy_clear: np.ndarray, 
    pixel_era5_clear: np.ndarray,
    app_config: 'config' # Using string to avoid circular import if config imports this
) -> tuple[EnhancedATCModel, list[dict]]:
    """
    Trains the Enhanced ATC model for a single pixel using its clear-sky observations.

    Args:
        pixel_lst_clear (np.ndarray): Clear-sky LST observations for the pixel (1D array).
        pixel_doy_clear (np.ndarray): Corresponding day of year for LST_clear (1D array).
        pixel_era5_clear (np.ndarray): Corresponding ERA5 skin temperature for LST_clear (1D array).
        app_config: Configuration object.

    Returns:
        tuple[EnhancedATCModel, list[dict]]: 
            - The trained ATC model for the pixel.
            - A list of model state_dict snapshots for ensemble.
    """
    device = torch.device(app_config.DEVICE if torch.cuda.is_available() else "cpu")

    # Convert inputs to tensors
    lst_tensor = torch.from_numpy(pixel_lst_clear).float().to(device)
    doy_tensor = torch.from_numpy(pixel_doy_clear).float().to(device)
    era5_tensor = torch.from_numpy(pixel_era5_clear).float().to(device)

    # Initialize model and optimizer
    # Potentially initialize params based on data statistics (e.g., C ~ mean(LST), A ~ std(LST))
    initial_C = np.nanmean(pixel_lst_clear) if len(pixel_lst_clear) > 0 else 290.0
    initial_A = np.nanstd(pixel_lst_clear) if len(pixel_lst_clear) > 1 else 10.0
    initial_phi = 180.0 # Mid-year peak, can be further refined
    initial_b = 0.5
    # Ensure initial_params are not NaN or Inf
    initial_params = {
        'C': float(initial_C) if np.isfinite(initial_C) else 290.0,
        'A': float(initial_A) if np.isfinite(initial_A) and initial_A > 0 else 10.0,
        'phi': float(initial_phi),
        'b': float(initial_b)
    }

    atc_model = EnhancedATCModel(initial_params=initial_params).to(device)
    optimizer = optim.Adam(atc_model.parameters(), lr=app_config.ATC_LEARNING_RATE)
    criterion = nn.MSELoss()

    snapshots = []

    for epoch in range(app_config.ATC_EPOCHS):
        atc_model.train()
        optimizer.zero_grad()
        
        predictions = atc_model(doy_tensor, era5_tensor)
        loss = criterion(predictions, lst_tensor)
        
        loss.backward()
        if torch.isnan(loss).any():
            print(f"  ATC_MODEL_DIAGNOSTIC: NaN loss at epoch {epoch} for a pixel. Stopping training for this pixel.")
            return None, []
        optimizer.step()

        # Store snapshots for ensemble
        if epoch >= app_config.ATC_ENSEMBLE_START_EPOCH and (epoch - app_config.ATC_ENSEMBLE_START_EPOCH) % app_config.ATC_SNAPSHOT_INTERVAL == 0:
            snapshots.append({k: v.clone().cpu().detach() for k, v in atc_model.state_dict().items()})
            if len(snapshots) >= app_config.ATC_ENSEMBLE_SNAPSHOTS:
                break # Collected enough snapshots
        
        # Optional: print progress for long training per pixel, but can be verbose
        # if (epoch + 1) % 200 == 0:
        #     print(f'  Epoch {epoch+1}/{app_config.ATC_EPOCHS}, Loss: {loss.item():.4f}')

    return atc_model, snapshots

# New worker function for parallel processing - MODIFIED FOR TRAINING ONLY
def _train_pixel_atc_worker(r, c,
                                pixel_lst_all_times_slice,
                                pixel_era5_all_times_slice,
                                doy_stack_all_days_numpy,
                                # pixel_era5_for_prediction_numpy_slice, # No longer needed for training worker
                                # doy_for_prediction_numpy, # No longer needed for training worker
                                app_config,
                                num_times_for_output # Not used directly by worker, but passed by old caller. Can be removed if caller is also updated.
                                ):
    """
    Worker function to train ATC for a single pixel and return snapshots.
    Designed to be called by joblib.Parallel.
    """
    worker_device_str = app_config.DEVICE
    if app_config.DEVICE.lower() == "cuda" and getattr(app_config, 'ATC_N_JOBS', -1) != 1:
        worker_device_str = "cpu"
    device = torch.device(worker_device_str if torch.cuda.is_available() and worker_device_str == "cuda" else "cpu")

    # If all ERA5 data for this pixel (for training period) is NaN, ATC cannot train reliably.
    if np.isnan(pixel_era5_all_times_slice).all():
        # print(f"  ATC_MODEL_DIAGNOSTIC (Worker): All ERA5 data is NaN for pixel ({r},{c}) during training period. Skipping ATC training.")
        # Return empty list of snapshots
        return r, c, []

    clear_sky_indices = np.where(~np.isnan(pixel_lst_all_times_slice))[0]
    
    pixel_lst_clear = pixel_lst_all_times_slice[clear_sky_indices]
    pixel_doy_clear = doy_stack_all_days_numpy[clear_sky_indices]
    pixel_era5_clear = pixel_era5_all_times_slice[clear_sky_indices]
    
    valid_data_mask = ~np.isnan(pixel_lst_clear) & ~np.isnan(pixel_era5_clear)
    pixel_lst_clear_valid = pixel_lst_clear[valid_data_mask]
    pixel_doy_clear_valid = pixel_doy_clear[valid_data_mask]
    pixel_era5_clear_valid = pixel_era5_clear[valid_data_mask]

    model_snapshots = []

    if len(pixel_lst_clear_valid) >= app_config.MIN_CLEAR_OBS_ATC:
        if not (np.isnan(pixel_lst_clear_valid).any() or \
                np.isnan(pixel_doy_clear_valid).any() or \
                np.isnan(pixel_era5_clear_valid).any()):
            
            trained_model_pixel, model_snapshots_from_train = train_atc_model_pixelwise(
                pixel_lst_clear_valid, pixel_doy_clear_valid, pixel_era5_clear_valid, app_config
            )
            if trained_model_pixel and model_snapshots_from_train:
                model_snapshots = model_snapshots_from_train
    
    # If training failed or insufficient data, snapshots list will be empty or not meet criteria.
    # We still need to return it, the collector can decide how to handle default/empty snapshots.
    # If no snapshots were generated (e.g. training failed, not enough data), use default ones as a fallback
    # so that the save function can expect a consistent number of snapshots.
    if not model_snapshots or len(model_snapshots) < app_config.ATC_ENSEMBLE_SNAPSHOTS:
        default_initial_C = np.nanmean(pixel_lst_clear_valid) if len(pixel_lst_clear_valid) > 0 else 290.0
        default_initial_A = np.nanstd(pixel_lst_clear_valid) if len(pixel_lst_clear_valid) > 1 else 10.0
        default_initial_C = float(default_initial_C) if np.isfinite(default_initial_C) else 290.0
        default_initial_A = float(default_initial_A) if np.isfinite(default_initial_A) and default_initial_A > 1e-6 else 10.0
        default_params = {
            'C': default_initial_C, 'A': default_initial_A,
            'phi': 180.0, 'b': 0.5
        }
        # Create one default snapshot, then duplicate it if necessary to match expected number
        # This ensures that the saving function receives a consistent structure.
        # The parameters are already detached and on CPU from train_atc_model_pixelwise.
        # For default, ensure they are tensors before calling state_dict or creating them directly.
        default_snapshot_params = {
            'C': torch.tensor(default_params['C']),
            'A': torch.tensor(default_params['A']),
            'phi': torch.tensor(default_params['phi']),
            'b': torch.tensor(default_params['b'])
        }
        
        # Fill up to the required number of snapshots with this default
        # If model_snapshots was empty, it gets filled. If partially filled, it's topped up.
        # This ensures downstream processing (saving) can expect a fixed number of snapshots.
        num_needed_snapshots = app_config.ATC_ENSEMBLE_SNAPSHOTS
        while len(model_snapshots) < num_needed_snapshots:
            model_snapshots.append({k: v.clone().cpu() for k,v in default_snapshot_params.items()})


    # The worker now only returns the raw snapshots for this pixel
    return r, c, model_snapshots

# MODIFIED to collect snapshots instead of direct prediction
def train_and_collect_all_atc_snapshots(preprocessed_data: dict, app_config: 'config') -> dict:
    """
    Trains ATC models for all pixels and collects their parameter snapshots.
    Args:
        preprocessed_data (dict): Dictionary containing preprocessed data.
                                  Expected keys: "lst_stack", "era5_stack", "doy_stack".
        app_config: Configuration object.
    Returns:
        dict: A dictionary mapping (row, col) tuples to a list of model state_dict snapshots.
              Example: {(0,0): [snap1_statedict, snap2_statedict,...], (0,1): [...]}
    """
    lst_stack = preprocessed_data["lst_stack"] # (time, height, width)
    era5_stack = preprocessed_data["era5_stack"] # (time, height, width)
    doy_stack_numpy = preprocessed_data["doy_stack"] # (time,) numpy array for DOYs of observations
    
    num_times_obs, height, width = lst_stack.shape # num_times_obs is for the observation period

    print(f"Preparing arguments for parallel ATC model training (snapshot collection) for {height*width} pixels...")
    print(f"Image size: {height}x{width}. Observation timeline: {num_times_obs} steps.")

    tasks_args_list = []
    for r in range(height):
        for c in range(width):
            tasks_args_list.append(
                (r, c,
                 lst_stack[:, r, c].copy(),
                 era5_stack[:, r, c].copy(),
                 doy_stack_numpy.copy(), # This is doy_stack_all_days_numpy for the worker
                 # doy_for_prediction_numpy.copy(), # Removed, worker doesn't predict
                 app_config,
                 num_times_obs # Pass original num_times for context, though worker might not use it directly.
                )
            )

    n_jobs = getattr(app_config, 'ATC_N_JOBS', -1)
    print(f"Starting parallel ATC training (snapshot collection) with n_jobs={n_jobs}...")
    
    delayed_jobs = [delayed(_train_pixel_atc_worker)(*task_args) for task_args in tasks_args_list]
    
    results = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(
        tqdm(delayed_jobs, desc="Training ATC Models (Pixels) & Collecting Snapshots", total=len(delayed_jobs))
    )
    num_processed_tasks = len(results)
    print(f"Parallel ATC snapshot collection finished. Processed {num_processed_tasks} pixel tasks.")

    all_pixel_snapshots = {}
    print("Collecting snapshots from parallel ATC training...")
    for r_res, c_res, snapshots_for_pixel in tqdm(results, desc="Organizing Snapshots"):
        all_pixel_snapshots[(r_res, c_res)] = snapshots_for_pixel
    
    print("Finished collecting all ATC model snapshots.")
    return all_pixel_snapshots


def save_atc_snapshots(all_pixel_snapshots: dict, filepath: str, image_height: int, image_width: int, num_snapshots_expected: int):
    """
    Saves the collected ATC model parameter snapshots to a compressed NPZ file.

    Args:
        all_pixel_snapshots (dict): Dict mapping (r,c) to list of state_dict snapshots.
                                    Each state_dict contains 'C', 'A', 'phi', 'b' as tensors.
        filepath (str): Path to save the .npz file.
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        num_snapshots_expected (int): The number of snapshots expected per pixel (e.g., app_config.ATC_ENSEMBLE_SNAPSHOTS).
    """
    # Initialize stacks for each parameter
    # Shape: (num_snapshots, height, width)
    C_stack = np.full((num_snapshots_expected, image_height, image_width), np.nan, dtype=np.float32)
    A_stack = np.full((num_snapshots_expected, image_height, image_width), np.nan, dtype=np.float32)
    phi_stack = np.full((num_snapshots_expected, image_height, image_width), np.nan, dtype=np.float32)
    b_stack = np.full((num_snapshots_expected, image_height, image_width), np.nan, dtype=np.float32)

    print(f"Structuring snapshots for saving. Expected snapshots per pixel: {num_snapshots_expected}")
    
    for (r, c), snapshots_list in tqdm(all_pixel_snapshots.items(), desc="Processing pixels for saving"):
        if not snapshots_list:
            print(f"Warning: No snapshots found for pixel ({r},{c}). Will be NaNs in saved file.")
            # NaNs are already the default from np.full
            continue
        if len(snapshots_list) != num_snapshots_expected:
            print(f"Warning: Pixel ({r},{c}) has {len(snapshots_list)} snapshots, expected {num_snapshots_expected}. This might indicate an issue or default filling.")
            # Continue processing with what's available, np.full handles missing ones if list is shorter after all.
            # The worker _train_pixel_atc_worker should ideally ensure num_snapshots_expected are returned.

        for idx, state_dict in enumerate(snapshots_list):
            if idx >= num_snapshots_expected: # Should not happen if worker is correct
                print(f"Warning: Pixel ({r},{c}) had more snapshots than expected. Truncating.")
                break
            # Ensure values are numpy floats for saving, detaching if they are tensors
            C_stack[idx, r, c] = state_dict['C'].cpu().numpy() if torch.is_tensor(state_dict['C']) else float(state_dict['C'])
            A_stack[idx, r, c] = state_dict['A'].cpu().numpy() if torch.is_tensor(state_dict['A']) else float(state_dict['A'])
            phi_stack[idx, r, c] = state_dict['phi'].cpu().numpy() if torch.is_tensor(state_dict['phi']) else float(state_dict['phi'])
            b_stack[idx, r, c] = state_dict['b'].cpu().numpy() if torch.is_tensor(state_dict['b']) else float(state_dict['b'])

    print(f"Saving snapshot stacks to {filepath}...")
    np.savez_compressed(filepath, 
                        C_snapshots=C_stack, 
                        A_snapshots=A_stack, 
                        phi_snapshots=phi_stack, 
                        b_snapshots=b_stack)
    print(f"Snapshots saved successfully.")


def load_atc_snapshots(filepath: str) -> dict:
    """
    Loads ATC model parameter snapshots from an NPZ file.

    Args:
        filepath (str): Path to the .npz file.

    Returns:
        dict: A dictionary with keys 'C_snapshots', 'A_snapshots', 'phi_snapshots', 'b_snapshots',
              each mapping to a NumPy array of shape (num_snapshots, height, width).
    """
    print(f"Loading snapshots from {filepath}...")
    data = np.load(filepath)
    print("Snapshots loaded.")
    return {
        "C_snapshots": data['C_snapshots'],
        "A_snapshots": data['A_snapshots'],
        "phi_snapshots": data['phi_snapshots'],
        "b_snapshots": data['b_snapshots']
    }

def predict_atc_from_loaded_snapshots(
    loaded_snapshots_data: dict, 
    doy_for_prediction_numpy: np.ndarray, 
    era5_for_prediction_numpy: np.ndarray, # This is the full era5_stack (time, height, width)
    app_config: 'config'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs ATC prediction using loaded snapshots for all pixels.

    Args:
        loaded_snapshots_data (dict): Data loaded by load_atc_snapshots.
        doy_for_prediction_numpy (np.ndarray): DOY for the prediction timeline (1D array, time_pred).
        era5_for_prediction_numpy (np.ndarray): ERA5 data for the prediction timeline (time_pred, height, width).
        app_config: Configuration object.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - atc_predictions_mean (np.ndarray): Mean ATC predictions (time_pred, height, width).
            - atc_predictions_variance (np.ndarray): Variance of ATC predictions (time_pred, height, width).
    """
    C_snaps = loaded_snapshots_data['C_snapshots'] # (num_snaps, height, width)
    A_snaps = loaded_snapshots_data['A_snapshots']
    phi_snaps = loaded_snapshots_data['phi_snapshots']
    b_snaps = loaded_snapshots_data['b_snapshots']

    num_snapshots, height, width = C_snaps.shape
    num_times_pred = len(doy_for_prediction_numpy)

    print(f"Starting ATC prediction from loaded snapshots. Image: {height}x{width}, Snapshots: {num_snapshots}, Pred Timesteps: {num_times_pred}")

    # Output arrays
    atc_predictions_mean = np.full((num_times_pred, height, width), np.nan, dtype=np.float32)
    atc_predictions_variance = np.full((num_times_pred, height, width), np.nan, dtype=np.float32)

    # Determine device for prediction models (can be global if predictions are sequential per pixel)
    # For parallelizing prediction itself (not done here), device management would be per-worker.
    device = torch.device(app_config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # Convert DOY for prediction to a tensor once
    doy_for_prediction_tensor = torch.from_numpy(doy_for_prediction_numpy).float().to(device) # (time_pred)

    # Create a single model instance to reuse
    # Initialize with dummy params, will be overwritten by loaded snapshots
    temp_model = EnhancedATCModel(initial_params={'C':0,'A':0,'phi':0,'b':0}).to(device)
    temp_model.eval() # Set to evaluation mode

    for r in tqdm(range(height), desc="Predicting ATC (Rows)"):
        for c in range(width):
            # Per-pixel ERA5 for prediction timeline
            pixel_era5_pred_numpy = era5_for_prediction_numpy[:, r, c] # (time_pred)
            pixel_era5_pred_tensor = torch.from_numpy(pixel_era5_pred_numpy).float().to(device) # (time_pred)

            # Check if all snapshot parameters for this pixel are NaN (e.g., if source data was all NaN)
            if np.isnan(C_snaps[:, r, c]).all():
                # print(f"Skipping prediction for pixel ({r},{c}) as all its C_snapshots are NaN.")
                # Outputs will remain NaN as initialized
                continue
            
            # If ERA5 for prediction is all NaN for this pixel, predictions will be NaN
            if torch.isnan(pixel_era5_pred_tensor).all():
                # print(f"ERA5 for prediction is all NaN for pixel ({r},{c}). Predictions will be NaN.")
                # Outputs will remain NaN as initialized
                continue

            pixel_ensemble_predictions_list = [] # List to hold (time_pred) arrays

            for snap_idx in range(num_snapshots):
                # Load parameters for the current snapshot and pixel
                current_params = {
                    'C': torch.tensor(C_snaps[snap_idx, r, c], device=device),
                    'A': torch.tensor(A_snaps[snap_idx, r, c], device=device),
                    'phi': torch.tensor(phi_snaps[snap_idx, r, c], device=device),
                    'b': torch.tensor(b_snaps[snap_idx, r, c], device=device)
                }
                
                # Check if any parameter for this specific snapshot is NaN. If so, this snapshot can't predict.
                if any(torch.isnan(p) for p in current_params.values()):
                    # This snapshot's prediction will be all NaNs for this pixel
                    nan_preds_for_snapshot = np.full(num_times_pred, np.nan, dtype=np.float32)
                    pixel_ensemble_predictions_list.append(nan_preds_for_snapshot)
                    continue

                temp_model.load_state_dict(current_params)
                
                with torch.no_grad():
                    preds_tensor = temp_model(doy_for_prediction_tensor, pixel_era5_pred_tensor)
                    pixel_ensemble_predictions_list.append(preds_tensor.cpu().numpy()) # Store (time_pred)

            if pixel_ensemble_predictions_list:
                # Stack along a new axis (axis 0: snapshots) -> (num_snapshots, time_pred)
                pixel_ensemble_stack = np.stack(pixel_ensemble_predictions_list, axis=0)
                
                # Calculate mean and variance across snapshots, handling NaNs
                atc_predictions_mean[:, r, c] = np.nanmean(pixel_ensemble_stack, axis=0)
                atc_predictions_variance[:, r, c] = np.nanvar(pixel_ensemble_stack, axis=0)
            # If list is empty (e.g., all params were NaN), outputs remain NaN

    print("Finished ATC prediction from loaded snapshots.")
    return atc_predictions_mean, atc_predictions_variance

# Original train_all_atc_models is now split.
# Keep the original main function as a template for how to use the new functions.
if __name__ == '__main__':
    # This is a placeholder for testing.
    # It requires preprocessed_data and app_config.
    print("ATC Model module main execution - for testing (Two-Phase).")

    # Dummy config and data setup (simplified from data_preprocessing.py example)
    class DummyConfig(config.Config):
        def __init__(self):
            super().__init__() # Initialize base if it has setup
            self.DEVICE = "cpu" # Force CPU for testing simplicity with PyTorch
            self.ATC_LEARNING_RATE = 0.01
            self.ATC_EPOCHS = 100 # Original: 1000
            self.ATC_ENSEMBLE_SNAPSHOTS = 5 # Original: 50
            self.ATC_SNAPSHOT_INTERVAL = 10 # Original: 10
            self.ATC_ENSEMBLE_START_EPOCH = self.ATC_EPOCHS - (self.ATC_ENSEMBLE_SNAPSHOTS * self.ATC_SNAPSHOT_INTERVAL)
            if self.ATC_ENSEMBLE_START_EPOCH < 0: self.ATC_ENSEMBLE_START_EPOCH = 0 # Ensure non-negative
            self.DAYS_OF_YEAR = 365
            self.MIN_CLEAR_OBS_ATC = 10
            self.ATC_N_JOBS = -1 # Use all available cores for parallel training part

    dummy_config = DummyConfig()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Create dummy preprocessed data
    num_times_obs, height, width = 20, 3, 3 # Small for testing observations
    num_times_pred = 25 # Potentially different timeline for prediction

    lst_stack_raw_dummy = np.random.rand(num_times_obs, height, width).astype(np.float32) * 15 + 280 # K
    cloud_locations_dummy = np.random.rand(num_times_obs, height, width) < 0.3
    lst_stack = np.where(cloud_locations_dummy, np.nan, lst_stack_raw_dummy)
    
    era5_stack_obs = np.random.rand(num_times_obs, height, width).astype(np.float32) * 10 + 275 # K for training
    doy_stack_obs = np.arange(1, num_times_obs + 1).astype(np.int32) # DOY for training period

    # Ensure pixel (0,0) has enough clear data for training
    lst_stack[:dummy_config.MIN_CLEAR_OBS_ATC+2, 0, 0] = lst_stack_raw_dummy[:dummy_config.MIN_CLEAR_OBS_ATC+2, 0, 0]
    # Ensure pixel (1,1) is all NaN for training LST
    lst_stack[:, 1, 1] = np.nan
    # Ensure pixel (1,2) has all NaN ERA5 for training
    era5_stack_obs[:, 1, 2] = np.nan


    preprocessed_data_dummy_train = {
        "lst_stack": lst_stack,
        "era5_stack": era5_stack_obs, # Use observation ERA5
        "doy_stack": doy_stack_obs,    # Use observation DOY
    }
    
    snapshots_filepath = "dummy_atc_snapshots.npz"

    # --- Phase 1: Training and Saving Snapshots ---
    print("\n--- Running Phase 1: ATC Model Training and Snapshot Saving ---")
    try:
        all_snapshots = train_and_collect_all_atc_snapshots(preprocessed_data_dummy_train, dummy_config)
        
        # Check if snapshots were collected for pixel (0,0)
        if (0,0) in all_snapshots and all_snapshots[(0,0)]:
            print(f"Snapshots collected for pixel (0,0): {len(all_snapshots[(0,0)])} snapshots.")
            # Check parameter values of the first snapshot for (0,0)
            # print(f"First snapshot for (0,0): C={all_snapshots[(0,0)][0]['C'].item():.2f}, A={all_snapshots[(0,0)][0]['A'].item():.2f}")
        else:
            print("Warning: No snapshots collected for pixel (0,0).")

        # Check for pixel (1,1) (all LST NaN) - should have default/fallback snapshots
        if (1,1) in all_snapshots and all_snapshots[(1,1)]:
             print(f"Snapshots for pixel (1,1) (all LST NaN): {len(all_snapshots[(1,1)])} snapshots (should be defaults).")
             # print(f"First snapshot for (1,1): C={all_snapshots[(1,1)][0]['C'].item():.2f}")
        else:
            print("Warning: No snapshots for pixel (1,1).")

        # Check for pixel (1,2) (all ERA5 NaN during training) - should have default/fallback snapshots
        if (1,2) in all_snapshots and all_snapshots[(1,2)]:
             print(f"Snapshots for pixel (1,2) (all ERA5 NaN): {len(all_snapshots[(1,2)])} snapshots (should be defaults).")
        else:
            print("Warning: No snapshots for pixel (1,2).")

        save_atc_snapshots(all_snapshots, snapshots_filepath, height, width, dummy_config.ATC_ENSEMBLE_SNAPSHOTS)
        print(f"Snapshots saved to {snapshots_filepath}")

    except Exception as e:
        print(f"Error during dummy ATC training/saving phase: {e}")
        import traceback
        traceback.print_exc()

    # --- Phase 2: Loading Snapshots and Predicting ---
    print("\n--- Running Phase 2: Loading ATC Snapshots and Predicting ---")
    
    # Create dummy ERA5 and DOY for the prediction period
    era5_for_prediction_dummy = np.random.rand(num_times_pred, height, width).astype(np.float32) * 12 + 273 # K
    doy_for_prediction_dummy = np.arange(num_times_obs + 1, num_times_obs + 1 + num_times_pred).astype(np.int32) # e.g., subsequent days

    # Simulate a case where ERA5 for prediction is all NaN for one pixel (e.g. 0,1)
    era5_for_prediction_dummy[:, 0, 1] = np.nan
    
    if os.path.exists(snapshots_filepath):
        try:
            loaded_snapshot_data = load_atc_snapshots(snapshots_filepath)
            # Basic check of loaded data structure
            if 'C_snapshots' in loaded_snapshot_data:
                 print(f"Loaded C_snapshots shape: {loaded_snapshot_data['C_snapshots'].shape}") # Expected: (num_snaps, H, W)
            
            atc_preds, atc_vars = predict_atc_from_loaded_snapshots(
                loaded_snapshot_data, 
                doy_for_prediction_dummy, 
                era5_for_prediction_dummy, 
                dummy_config
            )
            print(f"Prediction phase complete. Predictions shape: {atc_preds.shape}, Variance shape: {atc_vars.shape}")

            # Check predictions for specific pixels:
            # Pixel (0,0) - should have valid predictions (assuming training was okay)
            if not np.all(np.isnan(atc_preds[:, 0, 0])):
                print("Pixel (0,0) has valid predictions.")
            else:
                print("Pixel (0,0) has all NaN predictions. Check training data or prediction logic.")
            
            # Pixel (1,1) - LST was all NaN during training, so snapshots are defaults. Predictions might be generic.
            # Its params are defaults; if ERA5 is valid, it should predict.
            if not np.all(np.isnan(atc_preds[:, 1, 1])):
                 print("Pixel (1,1) (trained on default LST) has predictions.")
            else:
                 print("Pixel (1,1) has all NaN predictions. Check default param handling or ERA5 pred data.")

            # Pixel (1,2) - ERA5 was all NaN during training. Snapshots are defaults. Predictions might be generic.
            if not np.all(np.isnan(atc_preds[:, 1, 2])):
                 print("Pixel (1,2) (trained on default ERA5) has predictions.")
            else:
                 print("Pixel (1,2) has all NaN predictions. Check default param handling or ERA5 pred data.")

            # Pixel (0,1) - ERA5 for prediction is all NaN
            if np.all(np.isnan(atc_preds[:, 0, 1])):
                print("Pixel (0,1) correctly has all NaN predictions due to NaN ERA5 in prediction period.")
            else:
                print("Pixel (0,1) did NOT have all NaN predictions. Check NaN ERA5 handling in prediction.")


        except Exception as e:
            print(f"Error during dummy ATC prediction phase: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up dummy file
            # os.remove(snapshots_filepath) 
            # print(f"Cleaned up {snapshots_filepath}") # Keep for inspection for now
            pass
    else:
        print(f"Snapshot file {snapshots_filepath} not found. Skipping prediction phase.")

    print("\nFinished ATC Model module main execution (Two-Phase).") 