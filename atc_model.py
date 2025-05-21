"""
Enhanced Annual Temperature Cycle (ATC) model using PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
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

def train_all_atc_models(preprocessed_data: dict, app_config: 'config') -> tuple[np.ndarray, np.ndarray]:
    """
    Trains ATC models for all pixels in the dataset.

    Args:
        preprocessed_data (dict): Dictionary containing all preprocessed data stacks.
                                  Expected keys: "lst_stack", "era5_stack", "doy_stack".
        app_config: Configuration object.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - atc_predictions_full_timeseries (np.ndarray): ATC predictions for all pixels and all DOYs (time, height, width).
            - atc_variance_full_timeseries (np.ndarray): Variance of ATC predictions from ensemble (time, height, width).
    """
    lst_stack = preprocessed_data["lst_stack"] # (time, height, width)
    # cloud_mask_stack is no longer used here; cloud info is in LST_NODATA_VALUE
    era5_stack = preprocessed_data["era5_stack"] # (time, height, width)
    doy_stack_all_days = preprocessed_data["doy_stack"] # (time,) - for all days in the period
    
    num_times, height, width = lst_stack.shape
    device = torch.device(app_config.DEVICE if torch.cuda.is_available() else "cpu")

    # Output arrays for predictions and variance for the entire time series
    # These will be based on the full DOY range, not just observed days
    all_doys_tensor = torch.arange(1, app_config.DAYS_OF_YEAR + 1, dtype=torch.float32).to(device)
    
    # We need ERA5 for all DOYs. The input era5_stack corresponds to observed dates.
    # For simplicity, if we need ERA5 for a full year cycle for prediction, 
    # we might need a representative annual cycle of ERA5 or use the available time series and repeat/average.
    # The paper implies T_ATC(d) uses T_ERA5(d). So, for prediction over a full year cycle (all_doys_tensor),
    # we need a corresponding T_ERA5 for those DOYs.
    # Let's assume era5_stack covers the days for which we want to predict. 
    # If predicting for a generic year, one might average ERA5 per DOY.
    # Here, doy_stack_all_days corresponds to era5_stack and lst_stack timeline.
    
    doy_for_prediction = torch.from_numpy(doy_stack_all_days).float().to(device)
    era5_for_prediction = torch.from_numpy(era5_stack).float().to(device) # (time, height, width)

    atc_predictions_full_timeseries = np.full((num_times, height, width), np.nan, dtype=np.float32)
    atc_variance_full_timeseries = np.full((num_times, height, width), np.nan, dtype=np.float32)

    print(f"Training ATC models for {height*width} pixels...")
    for r in tqdm(range(height), desc="ATC Training Rows"):
        for c in range(width):
            pixel_lst_all_times = lst_stack[:, r, c] # This lst_stack comes from preprocessed_data, contains np.nan for nodata
            # pixel_cloud_mask_all_times = cloud_mask_stack[:, r, c] # No longer used
            pixel_era5_all_times = era5_stack[:, r, c]
            # doy_stack_all_days is already 1D

            # Select clear-sky observations for training this pixel's ATC model
            # Clear sky is where LST (from preprocessed_data) is not np.nan
            # clear_sky_indices = np.where(pixel_lst_all_times != app_config.LST_NODATA_VALUE)[0] # Old logic
            clear_sky_indices = np.where(~np.isnan(pixel_lst_all_times))[0] # Corrected logic
            
            if len(clear_sky_indices) < app_config.MIN_CLEAR_OBS_ATC: # Minimum observations to train
                # print(f"Skipping pixel ({r},{c}) due to insufficient clear data: {len(clear_sky_indices)} points")
                continue

            pixel_lst_clear = pixel_lst_all_times[clear_sky_indices]
            pixel_doy_clear = doy_stack_all_days[clear_sky_indices]
            pixel_era5_clear = pixel_era5_all_times[clear_sky_indices]
            
            # Filter out NaNs that might exist even if cloud mask is clear (e.g. sensor issues)
            valid_data_mask = ~np.isnan(pixel_lst_clear) & ~np.isnan(pixel_era5_clear)
            pixel_lst_clear = pixel_lst_clear[valid_data_mask]
            pixel_doy_clear = pixel_doy_clear[valid_data_mask]
            pixel_era5_clear = pixel_era5_clear[valid_data_mask]

            if len(pixel_lst_clear) < 10:
                # print(f"Skipping pixel ({r},{c}) after NaN filter: {len(pixel_lst_clear)} points")
                continue
            
            _, model_snapshots = train_atc_model_pixelwise(
                pixel_lst_clear, pixel_doy_clear, pixel_era5_clear, app_config
            )

            if not model_snapshots:
                # print(f"No model snapshots for pixel ({r},{c})")
                continue

            # Generate predictions for all time steps using the ensemble of snapshots
            ensemble_predictions_pixel = [] # List to store predictions from each snapshot model
            temp_model = EnhancedATCModel().to(device) # Create a temporary model instance
            
            # Prepare ERA5 data for this pixel for all prediction time steps
            pixel_era5_for_prediction = era5_for_prediction[:, r, c] # (time,)
            
            for snapshot_params in model_snapshots:
                # Load parameters into the temporary model
                # Ensure parameters are loaded to the correct device
                device_params = {k: v.to(device) for k, v in snapshot_params.items()}
                temp_model.load_state_dict(device_params)
                temp_model.eval()
                with torch.no_grad():
                    # Predict for all DOYs in the time series for this pixel
                    preds = temp_model(doy_for_prediction, pixel_era5_for_prediction)
                    ensemble_predictions_pixel.append(preds.cpu().numpy())
            
            if ensemble_predictions_pixel:
                ensemble_predictions_stack = np.stack(ensemble_predictions_pixel, axis=0) # (num_snapshots, num_times)
                atc_predictions_full_timeseries[:, r, c] = np.mean(ensemble_predictions_stack, axis=0)
                atc_variance_full_timeseries[:, r, c] = np.var(ensemble_predictions_stack, axis=0)
    
    print("Finished training all ATC models.")
    return atc_predictions_full_timeseries, atc_variance_full_timeseries


if __name__ == '__main__':
    # This is a placeholder for testing. 
    # It requires preprocessed_data and app_config.
    print("ATC Model module main execution - for testing.")

    # Dummy config and data setup (simplified from data_preprocessing.py example)
    class DummyConfig(config.Config):
        def __init__(self):
            super().__init__() # Initialize base if it has setup
            self.DEVICE = "cpu"
            self.ATC_LEARNING_RATE = 0.01 # Faster for dummy test
            self.ATC_EPOCHS = 100       # Fewer epochs for dummy test
            self.ATC_ENSEMBLE_SNAPSHOTS = 5
            self.ATC_SNAPSHOT_INTERVAL = 10
            self.ATC_ENSEMBLE_START_EPOCH = self.ATC_EPOCHS - (self.ATC_ENSEMBLE_SNAPSHOTS * self.ATC_SNAPSHOT_INTERVAL)
            self.DAYS_OF_YEAR = 365
            self.LST_NODATA_VALUE = -9999.0 # Used for creating dummy LST *before* it becomes NaN
            self.MIN_CLEAR_OBS_ATC = 10 # Min observations for ATC training

    dummy_config = DummyConfig()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Create dummy preprocessed data
    num_times, height, width = 20, 3, 3 # Small for testing
    
    # Generate LST data first, then introduce nodata values for the dummy `lst_stack`
    # This simulates what data_preprocessing.py does: original data might have LST_NODATA_VALUE,
    # but the 'lst_stack' passed to ATC model has np.nan for these.
    lst_stack_raw_dummy = np.random.rand(num_times, height, width).astype(np.float32) * 15 + 280 # K
    # Simulate clouds/nodata by setting some values to np.nan directly in the dummy lst_stack
    cloud_locations_dummy = np.random.rand(num_times, height, width) < 0.3 # ~30% cloudy
    # lst_stack = np.where(cloud_locations, dummy_config.LST_NODATA_VALUE, lst_stack_raw) # Old dummy data creation
    lst_stack = np.where(cloud_locations_dummy, np.nan, lst_stack_raw_dummy) # Corrected dummy data for lst_stack input
    
    # cloud_mask_stack = (np.random.rand(num_times, height, width) > 0.3).astype(np.uint8) # ~70% clear # No longer needed
    era5_stack = np.random.rand(num_times, height, width).astype(np.float32) * 10 + 275 # K
    # Create a simple linear sequence for DOY for this short period
    doy_stack = np.arange(1, num_times + 1).astype(np.int32)

    # Ensure some clear data for most pixels to test training
    # For pixel (0,0), make sure there is enough clear data
    # To ensure pixel (0,0) has enough clear data:
    # lst_stack[:dummy_config.MIN_CLEAR_OBS_ATC+2, 0, 0] = np.random.rand(dummy_config.MIN_CLEAR_OBS_ATC+2).astype(np.float32) * 15 + 280
    # This should set non-NaN values
    lst_stack[:dummy_config.MIN_CLEAR_OBS_ATC+2, 0, 0] = lst_stack_raw_dummy[:dummy_config.MIN_CLEAR_OBS_ATC+2, 0, 0]
    # Ensure some other pixels might be skipped by having all NaNs
    # lst_stack[:, 1, 1] = dummy_config.LST_NODATA_VALUE # Old dummy data creation
    lst_stack[:, 1, 1] = np.nan # Corrected dummy data: all NaN for pixel (1,1)


    preprocessed_data_dummy = {
        "lst_stack": lst_stack,
        # "cloud_mask_stack": cloud_mask_stack, # Removed
        "era5_stack": era5_stack,
        "doy_stack": doy_stack,
        # Other data like S2, coords might be needed if testing full pipeline,
        # but for ATC model training, these are the core ones from its perspective.
    }

    print("Running ATC model training with dummy data...")
    try:
        atc_preds, atc_vars = train_all_atc_models(preprocessed_data_dummy, dummy_config)
        print(f"ATC Predictions shape: {atc_preds.shape}")
        print(f"ATC Variance shape: {atc_vars.shape}")
        # Check if pixel (1,1) was skipped (should be all NaNs)
        if np.all(np.isnan(atc_preds[:, 1, 1])):
            print("Pixel (1,1) correctly skipped due to all nodata values.")
        else:
            print("Pixel (1,1) was NOT skipped, check LST_NODATA_VALUE logic.")
        
        # Check if pixel (0,0) has valid data
        if not np.all(np.isnan(atc_preds[:, 0, 0])):
            print("Pixel (0,0) has valid predictions.")
        else:
            print("Pixel (0,0) has all NaN predictions, check data generation or training logic.")

    except Exception as e:
        print(f"Error during dummy ATC training: {e}")
        import traceback
        traceback.print_exc()

    print("Finished ATC Model module main execution.") 