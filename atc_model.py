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
import utils # Added for the new plotting function


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

    def forward(self, doy: torch.Tensor, t_era5_1: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ATC model.

        Args:
            doy (torch.Tensor): Day of year (1 to self.days_in_year).
            t_era5_1 (torch.Tensor): ERA5 band 1 values for the corresponding doy.

        Returns:
            torch.Tensor: Predicted LST (T_ATC).
        """
        term_cos = torch.cos(2 * np.pi / self.days_in_year * (doy - self.phi))
        t_atc = self.C + self.A * term_cos + self.b * t_era5_1
        return t_atc

def train_atc_model_pixelwise(
    pixel_lst_clear: np.ndarray, 
    pixel_doy_clear: np.ndarray, 
    pixel_era5_clear: np.ndarray,
    app_config: 'config',
    pixel_identifier: str = "" # Keep for potential diagnostic messages
) -> tuple[EnhancedATCModel | None, list[dict], dict[str, list[float]]]: # Model can be None, return dict for losses
    """
    Trains the Enhanced ATC model for a single pixel using a two-phase approach.
    Phase 1: Search for optimal initial parameters by running short training trials.
    Phase 2: Train the model fully starting with the best initial parameters found.

    Args:
        pixel_lst_clear (np.ndarray): Clear-sky LST observations for the pixel (1D array).
        pixel_doy_clear (np.ndarray): Corresponding day of year for LST_clear (1D array).
        pixel_era5_clear (np.ndarray): Corresponding ERA5 skin temperature for LST_clear (1D array).
        app_config: Configuration object.
        pixel_identifier (str, optional): Identifier for the pixel for logging.

    Returns:
        tuple[EnhancedATCModel | None, list[dict], dict[str, list[float]]]: 
            - The trained ATC model for the pixel.
            - A list of model state_dict snapshots for ensemble.
            - A dictionary containing lists of mean 'train' and 'val' losses for each logging interval.
    """
    device = torch.device(app_config.ATC_DEVICE if torch.cuda.is_available() else "cpu")
    loss_fn = nn.MSELoss()

    # --- Data Split ---
    num_samples = len(pixel_lst_clear)
    # Initialize loss dictionary for return in case of early exit
    loss_logging_interval_setup = getattr(app_config, 'ATC_LOSS_LOGGING_INTERVAL', 100)
    total_epochs_setup = app_config.ATC_EPOCHS
    num_loss_intervals_setup = (total_epochs_setup + loss_logging_interval_setup - 1) // loss_logging_interval_setup
    default_losses_dict = {
        'train': [np.nan] * num_loss_intervals_setup,
        'val': [np.nan] * num_loss_intervals_setup
    }

    if num_samples < 2: # Not enough data to split
        return None, [], default_losses_dict

    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * 0.9)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    if len(train_indices) == 0 or len(val_indices) == 0:
        # Not enough data for a meaningful train/val split, use all for training.
        train_indices = indices
        val_indices = [] # No validation set

    # Convert all inputs to tensors first
    lst_tensor_full = torch.from_numpy(pixel_lst_clear).float().to(device)
    doy_tensor_full = torch.from_numpy(pixel_doy_clear).float().to(device)
    era5_tensor_1_full = torch.from_numpy(pixel_era5_clear[:, 0]).float().to(device)

    # Create train and val tensors from indices
    lst_tensor_train = lst_tensor_full[train_indices]
    doy_tensor_train = doy_tensor_full[train_indices]
    era5_tensor_1_train = era5_tensor_1_full[train_indices]

    if val_indices.size > 0:
        lst_tensor_val = lst_tensor_full[val_indices]
        doy_tensor_val = doy_tensor_full[val_indices]
        era5_tensor_1_val = era5_tensor_1_full[val_indices]
    else:
        lst_tensor_val, doy_tensor_val, era5_tensor_1_val = None, None, None


    # --- Phase 1: Search for best initial parameters (using training data only) ---
    init_search_trials = getattr(app_config, 'ATC_INIT_SEARCH_TRIALS', 20)
    init_search_epochs = getattr(app_config, 'ATC_INIT_SEARCH_EPOCHS', 40)
    
    best_initial_params = None
    best_loss = float('inf')

    # Base values for randomization from training data
    initial_C_base = np.nanmean(pixel_lst_clear[train_indices]) if len(train_indices) > 0 else 290.0
    initial_A_base = np.nanstd(pixel_lst_clear[train_indices]) if len(train_indices) > 1 else 10.0
    initial_phi_base = 180.0
    initial_b_base = 0.1
    
    initial_C_base_val = float(initial_C_base) if np.isfinite(initial_C_base) else 290.0
    initial_A_base_val = float(initial_A_base) if np.isfinite(initial_A_base) and initial_A_base > 0 else 10.0

    for _ in range(init_search_trials):
        # Randomize initial parameters as requested: base * random_float(0-1)
        trial_initial_params = {
            'C': initial_C_base_val * np.random.rand()/2,
            'A': initial_A_base_val * np.random.rand()/2,
            'phi': initial_phi_base * np.random.rand()/2,
            'b': initial_b_base * np.random.rand()/2,
        }
        
        # Short training for this trial
        atc_model_trial = EnhancedATCModel(initial_params=trial_initial_params).to(device)
        optimizer_trial = optim.Adam(atc_model_trial.parameters(), lr=app_config.ATC_LEARNING_RATE)
        trial_final_loss = float('inf')

        for _ in range(init_search_epochs):
            atc_model_trial.train()
            optimizer_trial.zero_grad()
            
            predictions = atc_model_trial(doy_tensor_train, era5_tensor_1_train)
            loss = loss_fn(predictions, lst_tensor_train)
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                trial_final_loss = float('inf')
                break # This trial with these initial params failed

            loss.backward()
            optimizer_trial.step()
            trial_final_loss = loss.item()
        
        if not np.isinf(trial_final_loss) and trial_final_loss < best_loss:
            best_loss = trial_final_loss
            best_initial_params = trial_initial_params

    # Fallback to deterministic initialization if search fails to find any valid params
    if best_initial_params is None:
        best_initial_params = {
            'C': initial_C_base_val,
            'A': initial_A_base_val,
            'phi': initial_phi_base,
            'b': initial_b_base,
        }

    # --- Phase 2: Full training with best initial parameters ---
    atc_model = EnhancedATCModel(initial_params=best_initial_params).to(device)
    optimizer = optim.Adam(atc_model.parameters(), 
                           lr=app_config.ATC_LEARNING_RATE, 
                           weight_decay=getattr(app_config, 'ATC_WEIGHT_DECAY', 0.0))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=getattr(app_config, 'ATC_LR_SCHEDULER_FACTOR', 0.1),
        patience=getattr(app_config, 'ATC_LR_SCHEDULER_PATIENCE', 10),
        min_lr=getattr(app_config, 'ATC_LR_SCHEDULER_MIN_LR', 1e-6)
    )

    snapshots = []
    
    # Loss logging setup
    loss_logging_interval = getattr(app_config, 'ATC_LOSS_LOGGING_INTERVAL', 100)
    total_epochs = app_config.ATC_EPOCHS
    num_loss_intervals = (total_epochs + loss_logging_interval - 1) // loss_logging_interval
    interval_losses_output = {
        'train': [np.nan] * num_loss_intervals,
        'val': [np.nan] * num_loss_intervals
    }
    
    current_interval_train_losses = []

    phase2_epochs = total_epochs - init_search_epochs
    if phase2_epochs < 0:
        phase2_epochs = 0
        
    last_global_epoch = -1 # To handle the case where phase2_epochs is 0

    for epoch in range(phase2_epochs):
        global_epoch = epoch + init_search_epochs
        last_global_epoch = global_epoch

        atc_model.train()
        optimizer.zero_grad()
        
        predictions = atc_model(doy_tensor_train, era5_tensor_1_train)
        
        loss = loss_fn(predictions, lst_tensor_train)
        
        loss.backward()
        if torch.isnan(loss).any() or torch.isinf(loss).any(): # Added check for inf loss
            # On failure in phase 2, return current losses (mostly NaNs).
            return None, [], interval_losses_output

        optimizer.step()
        
        current_interval_train_losses.append(loss.item())

        # Step the scheduler based on training loss
        scheduler.step(loss)

        # Log loss at interval
        if (global_epoch + 1) % loss_logging_interval == 0:
            current_interval_idx = global_epoch // loss_logging_interval
            if current_interval_train_losses and current_interval_idx < num_loss_intervals:
                interval_losses_output['train'][current_interval_idx] = np.mean(current_interval_train_losses)
            current_interval_train_losses = [] # Reset for next interval

            # Calculate validation loss for the interval
            if lst_tensor_val is not None:
                atc_model.eval() # Switch to evaluation mode
                with torch.no_grad():
                    val_predictions = atc_model(doy_tensor_val, era5_tensor_1_val)
                    val_loss = loss_fn(val_predictions, lst_tensor_val)
                    if current_interval_idx < num_loss_intervals:
                        interval_losses_output['val'][current_interval_idx] = val_loss.item()
                atc_model.train() # Switch back to training mode

        # Store snapshots for ensemble
        if (global_epoch >= app_config.ATC_ENSEMBLE_START_EPOCH and 
            (global_epoch - app_config.ATC_ENSEMBLE_START_EPOCH) % app_config.ATC_SNAPSHOT_INTERVAL == 0):
            snapshots.append({k: v.clone().cpu().detach() for k, v in atc_model.state_dict().items()})
            if len(snapshots) >= app_config.ATC_ENSEMBLE_SNAPSHOTS:
                # If we break early, handle the last partial loss interval
                if current_interval_train_losses:
                    current_interval_idx = global_epoch // loss_logging_interval
                    if current_interval_idx < num_loss_intervals:
                        # Log train loss
                        interval_losses_output['train'][current_interval_idx] = np.mean(current_interval_train_losses)
                        # Log val loss one last time if possible
                        if lst_tensor_val is not None:
                            atc_model.eval()
                            with torch.no_grad():
                                val_predictions = atc_model(doy_tensor_val, era5_tensor_1_val)
                                val_loss = loss_fn(val_predictions, lst_tensor_val)
                                if current_interval_idx < num_loss_intervals:
                                    interval_losses_output['val'][current_interval_idx] = val_loss.item()
                            atc_model.train()
                break 
    
    # Handle final partial interval if training finished all epochs
    if current_interval_train_losses and last_global_epoch != -1:
        current_interval_idx = last_global_epoch // loss_logging_interval
        if current_interval_idx < num_loss_intervals and np.isnan(interval_losses_output['train'][current_interval_idx]):
             interval_losses_output['train'][current_interval_idx] = np.mean(current_interval_train_losses)
             # Note: Validation loss is only calculated at interval boundaries, so no need to calculate it here.

    return atc_model, snapshots, interval_losses_output

# MODIFIED for returning structured interval losses
def _train_pixel_atc_worker(
    r: int, c: int,
    pixel_lst_all_times_slice: np.ndarray,
    pixel_era5_all_times_slice: np.ndarray,
    doy_stack_all_days_numpy: np.ndarray,
    app_config: 'config',
    num_times_for_output: int 
) -> tuple[int, int, list[dict], dict[str, list[float]]]: # Return dict of loss lists
    """
    Worker function to train ATC for a single pixel, return snapshots and interval losses.
    """
    worker_device_str = app_config.ATC_DEVICE
    if app_config.ATC_DEVICE.lower() == "cuda" and getattr(app_config, 'ATC_N_JOBS', -1) != 1:
        worker_device_str = "cpu" 
    
    device = torch.device(worker_device_str if torch.cuda.is_available() and worker_device_str.lower() == "cuda" else "cpu")

    pixel_id_str = f"Pixel ({r},{c})"
    
    # Initialize default interval losses (all NaNs) as a dictionary
    loss_logging_interval = getattr(app_config, 'ATC_LOSS_LOGGING_INTERVAL', 100)
    num_loss_intervals = (app_config.ATC_EPOCHS + loss_logging_interval - 1) // loss_logging_interval
    default_interval_losses = {
        'train': [np.nan] * num_loss_intervals,
        'val': [np.nan] * num_loss_intervals
    }

    if np.isnan(pixel_era5_all_times_slice).all():
        return r, c, [], default_interval_losses # Return empty snapshots and default losses

    clear_sky_indices = np.where(~np.isnan(pixel_lst_all_times_slice))[0]
    pixel_lst_clear = pixel_lst_all_times_slice[clear_sky_indices]
    pixel_doy_clear = doy_stack_all_days_numpy[clear_sky_indices]
    pixel_era5_clear = pixel_era5_all_times_slice[clear_sky_indices]
    
    mask_lst = ~np.isnan(pixel_lst_clear)
    mask_era5_1 = ~np.isnan(pixel_era5_clear[:, 0])
    mask_era5_2 = ~np.isnan(pixel_era5_clear[:, 1])
    valid_data_mask = mask_lst & mask_era5_1 & mask_era5_2
    pixel_lst_clear_valid = pixel_lst_clear[valid_data_mask]
    pixel_doy_clear_valid = pixel_doy_clear[valid_data_mask]
    pixel_era5_clear_valid = pixel_era5_clear[valid_data_mask]

    model_snapshots = []
    pixel_interval_losses_final = {k: v[:] for k, v in default_interval_losses.items()} # Deep copy

    if len(pixel_lst_clear_valid) >= app_config.MIN_CLEAR_OBS_ATC:
        if not (np.isnan(pixel_lst_clear_valid).any() or \
                np.isnan(pixel_doy_clear_valid).any() or \
                np.isnan(pixel_era5_clear_valid).any()):
            
            trained_model_pixel, model_snaps_from_train, interval_losses_from_train = train_atc_model_pixelwise(
                pixel_lst_clear_valid, pixel_doy_clear_valid, pixel_era5_clear_valid, app_config,
                pixel_identifier=pixel_id_str
            )
            if trained_model_pixel and model_snaps_from_train:
                model_snapshots = model_snaps_from_train
            # Use interval_losses_from_train regardless of whether model was trained
            pixel_interval_losses_final = interval_losses_from_train 
    
    # Ensure snapshots list is padded if needed (as before)
    if not model_snapshots or len(model_snapshots) < app_config.ATC_ENSEMBLE_SNAPSHOTS:
        default_initial_C = np.nanmean(pixel_lst_clear_valid) if len(pixel_lst_clear_valid) > 0 else 290.0
        default_initial_A = np.nanstd(pixel_lst_clear_valid) if len(pixel_lst_clear_valid) > 1 else 10.0
        default_initial_C = float(default_initial_C) if np.isfinite(default_initial_C) else 290.0
        default_initial_A = float(default_initial_A) if np.isfinite(default_initial_A) and default_initial_A > 1e-6 else 10.0
        default_params_for_snapshot = {
            'C': torch.tensor(default_initial_C), 'A': torch.tensor(default_initial_A),
            'phi': torch.tensor(180.0), 'b': torch.tensor(0.5),
        }
        num_needed_snapshots = app_config.ATC_ENSEMBLE_SNAPSHOTS
        while len(model_snapshots) < num_needed_snapshots:
            model_snapshots.append({k: v.clone().cpu() for k,v in default_params_for_snapshot.items()})

    return r, c, model_snapshots, pixel_interval_losses_final

# MODIFIED to collect and return dict of loss maps
def train_and_collect_all_atc_snapshots(
    preprocessed_data: dict, app_config: 'config'
) -> tuple[dict[tuple[int, int], list[dict]], dict[str, np.ndarray]]: # Return dict of loss maps
    """
    Trains ATC models for all pixels, collects snapshots, and interval loss maps.
    Returns:
        tuple[dict[tuple[int, int], list[dict]], dict[str, np.ndarray]]:
            - all_pixel_snapshots: Dict mapping (r,c) to list of state_dict snapshots.
            - interval_loss_maps: Dict with 'train' and 'val' keys, each mapping to a 
                                  NumPy array (num_intervals, height, width) of mean losses.
    """
    lst_stack = preprocessed_data["lst_stack"]
    era5_stack = preprocessed_data["era5_stack"]
    doy_stack_numpy = preprocessed_data["doy_stack"]
    training_pixel_mask = preprocessed_data.get("training_pixel_mask")
    
    num_times_obs, height, width = lst_stack.shape

    # --- START: Plot input data timeseries overview --- ADDED BLOCK ---
    if getattr(app_config, 'PLOT_INPUT_TIMESERIES_OVERVIEW', True): # Check a config flag if you want to make it optional
        print("\nGenerating input data timeseries overview plot...")
        try:
            utils.plot_input_data_timeseries_overview(
                doy_stack_numpy=doy_stack_numpy,
                lst_stack=lst_stack,
                era5_stack=era5_stack, # era5_stack is (time, 2, H, W)
                training_pixel_mask=training_pixel_mask, # Can be None
                output_dir=app_config.OUTPUT_DIR,
                roi_name=preprocessed_data.get('roi_name', 'UnknownROI')
            )
        except Exception as e:
            print(f"Warning: Failed to generate input data timeseries overview plot. Error: {e}")
            import traceback
            traceback.print_exc()
    # --- END: Plot input data timeseries overview ---

    if training_pixel_mask is None:
        training_pixel_mask = np.ones((height, width), dtype=bool)
    elif not isinstance(training_pixel_mask, np.ndarray) or training_pixel_mask.shape != (height, width):
        print(f"Warning: training_pixel_mask has unexpected type/shape ({type(training_pixel_mask)}, {training_pixel_mask.shape if isinstance(training_pixel_mask, np.ndarray) else 'N/A'}). Defaulting to training all pixels.")
        training_pixel_mask = np.ones((height, width), dtype=bool)

    num_pixels_to_train = np.sum(training_pixel_mask)
    # print(f"Preparing arguments for parallel ATC model training (snapshot & loss collection) for {num_pixels_to_train} selected pixels...")

    tasks_args_list = []
    for r_iter in range(height):
        for c_iter in range(width):
            if training_pixel_mask[r_iter, c_iter]:
                tasks_args_list.append(
                    (r_iter, c_iter,
                     lst_stack[:, r_iter, c_iter].copy(),
                     era5_stack[:, :, r_iter, c_iter].copy(),
                     doy_stack_numpy.copy(),
                     app_config,
                     num_times_obs
                    )
                )

    n_jobs = getattr(app_config, 'ATC_N_JOBS', -1)
    # print(f"Starting parallel ATC training (snapshot & loss collection) with n_jobs={n_jobs}...")
    
    delayed_jobs = [delayed(_train_pixel_atc_worker)(*task_args) for task_args in tasks_args_list]
    
    results = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(
        tqdm(delayed_jobs, desc="Training ATC & Collecting Losses/Snapshots", total=len(delayed_jobs))
    )
    # print(f"Parallel ATC snapshot & loss collection finished. Processed {len(results)} pixel tasks.")

    all_pixel_snapshots = {}
    
    # Prepare structure for interval loss maps (train and val)
    loss_logging_interval = getattr(app_config, 'ATC_LOSS_LOGGING_INTERVAL', 100)
    num_loss_intervals = (app_config.ATC_EPOCHS + loss_logging_interval -1) // loss_logging_interval
    interval_loss_maps = {
        'train': np.full((num_loss_intervals, height, width), np.nan, dtype=np.float32),
        'val': np.full((num_loss_intervals, height, width), np.nan, dtype=np.float32)
    }

    print("Collecting snapshots and interval losses from parallel ATC training...")
    for r_res, c_res, snapshots_for_pixel, interval_losses_dict in tqdm(results, desc="Organizing Results"):
        all_pixel_snapshots[(r_res, c_res)] = snapshots_for_pixel
        
        # Unpack the dictionary of losses and populate the respective maps
        train_losses = interval_losses_dict.get('train', [])
        val_losses = interval_losses_dict.get('val', [])

        if len(train_losses) == num_loss_intervals:
            for interval_idx in range(num_loss_intervals):
                interval_loss_maps['train'][interval_idx, r_res, c_res] = train_losses[interval_idx]
        
        if len(val_losses) == num_loss_intervals:
            for interval_idx in range(num_loss_intervals):
                interval_loss_maps['val'][interval_idx, r_res, c_res] = val_losses[interval_idx]
    
    # For pixels not in training_pixel_mask, their entries in interval_loss_maps will remain NaN.
    
    print("Finished collecting all ATC model snapshots and interval losses.")
    return all_pixel_snapshots, interval_loss_maps

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
            print(f"Warning: Pixel ({r},{c}) has {len(snapshots_list)} snapshots, expected {num_snapshots_expected}.")
            # Continue processing with what's available, np.full handles missing ones if list is shorter after all.
            # The worker _train_pixel_atc_worker should ideally ensure num_snapshots_expected are returned.

        for idx, state_dict in enumerate(snapshots_list):
            if idx >= num_snapshots_expected: 
                break
            # Gracefully get parameters; MLP state_dicts won't have these specific keys.
            C_stack[idx, r, c] = state_dict.get('C', np.nan) if torch.is_tensor(state_dict.get('C', np.nan)) else float(state_dict.get('C', np.nan))
            A_stack[idx, r, c] = state_dict.get('A', np.nan) if torch.is_tensor(state_dict.get('A', np.nan)) else float(state_dict.get('A', np.nan))
            phi_stack[idx, r, c] = state_dict.get('phi', np.nan) if torch.is_tensor(state_dict.get('phi', np.nan)) else float(state_dict.get('phi', np.nan))
            b_stack[idx, r, c] = state_dict.get('b', np.nan) if torch.is_tensor(state_dict.get('b', np.nan)) else float(state_dict.get('b', np.nan))

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
    # For now, assume new files won't have 'd_snapshots'. Prediction code needs to handle this.
    # Also, this structure is for ATC model. MLP snapshots would be raw state_dicts.
    loaded_data = {
        "C_snapshots": data.get('C_snapshots'), # Use .get() in case file is from older version or MLP (where this won't be)
        "A_snapshots": data.get('A_snapshots'),
        "phi_snapshots": data.get('phi_snapshots'),
        "b_snapshots": data.get('b_snapshots'),
    }
    # It would be good to save/load ATC_MODEL_TYPE with the snapshots.
    # For now, prediction function assumes ATC or needs modification for MLP.
    return loaded_data

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
            - atc_predictions_variance (np.ndarray): Ensemble variance of ATC predictions (time_pred, height, width).
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
    device = torch.device(app_config.ATC_DEVICE if torch.cuda.is_available() else "cpu")
    
    # Convert DOY for prediction to a tensor once
    doy_for_prediction_tensor = torch.from_numpy(doy_for_prediction_numpy).float().to(device) # (time_pred)

    # Create a single model instance to reuse
    # Initialize with dummy params, will be overwritten by loaded snapshots
    temp_model_initial_params = {'C':0.0,'A':0.0,'phi':0.0,'b':0.0}
    temp_model = EnhancedATCModel(initial_params=temp_model_initial_params).to(device)
    temp_model.eval() # Set to evaluation mode

    for r in tqdm(range(height), desc="Predicting ATC (Rows)"):
        for c in range(width):
            # Per-pixel ERA5 bands for prediction timeline
            pixel_era5_pred_band1 = era5_for_prediction_numpy[:, 0, r, c]  # (time_pred)
            pixel_era5_pred_tensor1 = torch.from_numpy(pixel_era5_pred_band1).float().to(device)

            # Skip if no model snapshots or ERA5 parameters are all NaN
            if np.isnan(C_snaps[:, r, c]).all():
                continue
            if torch.isnan(pixel_era5_pred_tensor1).all():
                continue

            pixel_ensemble_predictions_list = [] # List to hold (time_pred) arrays for mean predictions

            for snap_idx in range(num_snapshots):
                # Load parameters for the current snapshot and pixel
                current_C = C_snaps[snap_idx, r, c]
                current_A = A_snaps[snap_idx, r, c]
                current_phi = phi_snaps[snap_idx, r, c]
                current_b = b_snaps[snap_idx, r, c]

                current_params = {
                    'C': torch.tensor(current_C, device=device),
                    'A': torch.tensor(current_A, device=device),
                    'phi': torch.tensor(current_phi, device=device),
                    'b': torch.tensor(current_b, device=device),
                }
                
                # Check if any parameter for this specific snapshot is NaN. If so, this snapshot can't predict.
                if any(torch.isnan(p.data) for p in current_params.values()): # Check .data for Parameter objects
                    # This snapshot's prediction will be all NaNs for this pixel
                    nan_preds_for_snapshot = np.full(num_times_pred, np.nan, dtype=np.float32)
                    pixel_ensemble_predictions_list.append(nan_preds_for_snapshot)
                    continue

                temp_model.load_state_dict(current_params)
                
                with torch.no_grad():
                    # Model forward now only returns the mean prediction
                    preds_tensor = temp_model(doy_for_prediction_tensor, pixel_era5_pred_tensor1)
                    pixel_ensemble_predictions_list.append(preds_tensor.cpu().numpy()) # Store (time_pred)
                    

            if pixel_ensemble_predictions_list:
                # Stack along a new axis (axis 0: snapshots) -> (num_snapshots, time_pred)
                pixel_ensemble_preds_stack = np.stack(pixel_ensemble_predictions_list, axis=0)
                
                # Calculate mean of predictions and variance of predictions (ensemble uncertainty)
                mean_of_predictions = np.nanmean(pixel_ensemble_preds_stack, axis=0)
                variance_of_predictions = np.nanvar(pixel_ensemble_preds_stack, axis=0)

                atc_predictions_mean[:, r, c] = mean_of_predictions
                atc_predictions_variance[:, r, c] = variance_of_predictions
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
            self.DEVICE = "cpu" # General device
            self.ATC_DEVICE = "cpu" # Specific for ATC
            self.GP_DEVICE = "cpu"  # Specific for GP (though not used directly in ATC tests)
            self.ATC_LEARNING_RATE = 0.01
            self.ATC_EPOCHS = 100 # Original: 1000 -> Reduced for faster test
            self.ATC_INIT_SEARCH_TRIALS = 5 # Add search trials for the test
            self.ATC_INIT_SEARCH_EPOCHS = 10 # Add search epochs for the test
            self.ATC_ENSEMBLE_SNAPSHOTS = 5
            self.ATC_SNAPSHOT_INTERVAL = 10
            self.ATC_ENSEMBLE_START_EPOCH = self.ATC_EPOCHS - (self.ATC_ENSEMBLE_SNAPSHOTS * self.ATC_SNAPSHOT_INTERVAL)
            if self.ATC_ENSEMBLE_START_EPOCH < 0: self.ATC_ENSEMBLE_START_EPOCH = 0 
            self.DAYS_OF_YEAR = 365
            self.MIN_CLEAR_OBS_ATC = 10
            self.ATC_N_JOBS = 1 # Test with 1 first for easier debugging of loss collection
            self.ATC_LOSS_LOGGING_INTERVAL = 20 # Log loss every 20 epochs for test
            self.ATC_WEIGHT_DECAY = 1e-5
            self.ATC_LR_SCHEDULER_FACTOR = 0.5
            self.ATC_LR_SCHEDULER_PATIENCE = 5
            self.ATC_LR_SCHEDULER_MIN_LR = 1e-7

    dummy_config = DummyConfig()
    torch.manual_seed(config.RANDOM_SEED if hasattr(config, 'RANDOM_SEED') else 42)
    np.random.seed(config.RANDOM_SEED if hasattr(config, 'RANDOM_SEED') else 42)

    num_times_obs, height, width = 20, 3, 3 
    num_times_pred = 25 

    lst_stack_raw_dummy = np.random.rand(num_times_obs, height, width).astype(np.float32) * 15 + 280 
    cloud_locations_dummy = np.random.rand(num_times_obs, height, width) < 0.3
    lst_stack = np.where(cloud_locations_dummy, np.nan, lst_stack_raw_dummy)
    
    era5_stack_obs = np.random.rand(num_times_obs, height, width).astype(np.float32) * 10 + 275 
    doy_stack_obs = np.arange(1, num_times_obs + 1).astype(np.int32) 

    lst_stack[:dummy_config.MIN_CLEAR_OBS_ATC+2, 0, 0] = lst_stack_raw_dummy[:dummy_config.MIN_CLEAR_OBS_ATC+2, 0, 0]
    lst_stack[:, 1, 1] = np.nan
    era5_stack_obs[:, 1, 2] = np.nan

    # Dummy training_pixel_mask (train all for this simple test)
    training_pixel_mask_dummy = np.ones((height,width), dtype=bool)
    # training_pixel_mask_dummy[0,1] = False # Example of not training one pixel

    preprocessed_data_dummy_train = {
        "lst_stack": lst_stack,
        "era5_stack": era5_stack_obs, 
        "doy_stack": doy_stack_obs,
        "training_pixel_mask": training_pixel_mask_dummy, # Add the mask
        "roi_name": "DummyROI" # For snapshot filename
    }
    
    snapshots_filepath = "dummy_atc_snapshots.npz"

    print("\n--- Running Phase 1: ATC Model Training and Snapshot/Loss Saving ---")
    try:
        all_snapshots, collected_interval_loss_maps = train_and_collect_all_atc_snapshots(preprocessed_data_dummy_train, dummy_config)
        
        print(f"Collected interval loss maps shape: {collected_interval_loss_maps.shape}")
        # Expected shape: (ATC_EPOCHS // ATC_LOSS_LOGGING_INTERVAL, height, width)
        # For dummy_config: (100 // 20 = 5, 3, 3)
        expected_loss_intervals = (dummy_config.ATC_EPOCHS + dummy_config.ATC_LOSS_LOGGING_INTERVAL -1) // dummy_config.ATC_LOSS_LOGGING_INTERVAL
        assert collected_interval_loss_maps.shape == (expected_loss_intervals, height, width), "Loss maps shape mismatch"
        
        # Check loss for pixel (0,0) - which should have trained
        print("Losses for pixel (0,0) over intervals:", collected_interval_loss_maps[:, 0, 0])
        assert not np.all(np.isnan(collected_interval_loss_maps[:, 0, 0])), "Pixel (0,0) should have valid loss values"
        
        # Check loss for pixel (1,1) - all LST NaN, so training skipped, losses should be NaN
        print("Losses for pixel (1,1) over intervals (expect NaNs):", collected_interval_loss_maps[:, 1, 1])
        assert np.all(np.isnan(collected_interval_loss_maps[:, 1, 1])), "Pixel (1,1) (all LST NaN) should have all NaN losses"

        save_atc_snapshots(all_snapshots, snapshots_filepath, height, width, dummy_config.ATC_ENSEMBLE_SNAPSHOTS)
        print(f"Snapshots saved to {snapshots_filepath}")

    except Exception as e:
        print(f"Error during dummy ATC training/saving phase: {e}")
        import traceback
        traceback.print_exc()

    # Phase 2 (Prediction) remains the same... (omitted for brevity in this diff, but should still work)
    print("\n--- Running Phase 2: Loading ATC Snapshots and Predicting ---")
    # ... (rest of the if __name__ block for prediction) ...
    if os.path.exists(snapshots_filepath):
        try:
            loaded_snapshot_data = load_atc_snapshots(snapshots_filepath)
            era5_for_prediction_dummy = np.random.rand(num_times_pred, height, width).astype(np.float32) * 12 + 273 
            doy_for_prediction_dummy = np.arange(num_times_obs + 1, num_times_obs + 1 + num_times_pred).astype(np.int32)
            
            atc_preds, atc_vars = predict_atc_from_loaded_snapshots(
                loaded_snapshot_data, 
                doy_for_prediction_dummy, 
                era5_for_prediction_dummy, 
                dummy_config
            )
            print(f"Prediction phase complete. Predictions shape: {atc_preds.shape}, Variance shape: {atc_vars.shape}")
        except Exception as e:
            print(f"Error during dummy ATC prediction phase: {e}")
            import traceback
            traceback.print_exc()
        # finally: # Keep for inspection for now
            # os.remove(snapshots_filepath) 
            # print(f"Cleaned up {snapshots_filepath}")
    else:
        print(f"Snapshot file {snapshots_filepath} not found. Skipping prediction phase.")
    print("\nFinished ATC Model module main execution (Two-Phase).") 