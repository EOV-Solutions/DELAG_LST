"""
Gaussian Process (GP) model for residuals using GPyTorch.
"""
import torch
import gpytorch
import numpy as np
import pandas as pd # For feature naming consistency if needed
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import warnings # Import the standard warnings module

import config # Assuming your config.py is accessible

# Define the GP model
class ApproximateGPModelResiduals(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        """
        Approximate GP model for LST residuals.

        Args:
            inducing_points (torch.Tensor): Tensor of inducing point locations 
                                          (num_inducing_points, num_features).
        """
        # Ensure inducing_points is float32
        if inducing_points.dtype != torch.float32:
            inducing_points = inducing_points.float()
            
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        # Mean and Kernel (RBF Kernel)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass of the GP model.

        Args:
            x (torch.Tensor): Input features (batch_size, num_features).

        Returns:
            gpytorch.distributions.MultivariateNormal: Predictive distribution.
        """
        # Ensure x is float32
        if x.dtype != torch.float32:
            x = x.float()
            
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp_model(
    train_x: torch.Tensor, 
    train_y: torch.Tensor, 
    app_config: 'config'
) -> tuple[ApproximateGPModelResiduals, gpytorch.likelihoods.GaussianLikelihood]:
    """
    Trains the Approximate GP model for residuals.

    Args:
        train_x (torch.Tensor): Training features (num_samples, num_features).
        train_y (torch.Tensor): Training targets (residuals) (num_samples).
        app_config: Configuration object.

    Returns:
        tuple[ApproximateGPModelResiduals, gpytorch.likelihoods.GaussianLikelihood]:
            - Trained GP model.
            - Trained likelihood.
    """
    device = torch.device(app_config.DEVICE if torch.cuda.is_available() else "cpu")
    train_x, train_y = train_x.to(device), train_y.to(device)

    # Initialize inducing points (e.g., using a subset of train_x or k-means)
    # For simplicity, take a random subset if train_x is large enough
    num_inducing = min(app_config.GP_NUM_INDUCING_POINTS, train_x.size(0))
    inducing_points_indices = torch.randperm(train_x.size(0))[:num_inducing]
    inducing_points = train_x[inducing_points_indices, :]

    model = ApproximateGPModelResiduals(inducing_points=inducing_points).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()}, 
        {'params': likelihood.parameters()}
    ], lr=app_config.GP_LEARNING_RATE_INITIAL)

    # Use MLL for SVGP
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    # Create DataLoader for mini-batch training
    dataset = TensorDataset(train_x, train_y)
    dataloader = DataLoader(dataset, batch_size=app_config.GP_MINI_BATCH_SIZE, shuffle=True)

    print(f"Starting GP model training with {app_config.GP_EPOCHS_INITIAL + app_config.GP_EPOCHS_FINAL} epochs...")
    # Initial learning rate phase
    for i in range(app_config.GP_EPOCHS_INITIAL):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(dataloader.dataset)
        if (i + 1) % 10 == 0:
            print(f"GP Epoch {i+1}/{app_config.GP_EPOCHS_INITIAL}, Loss: {epoch_loss:.3f}, LR: {app_config.GP_LEARNING_RATE_INITIAL}")

    # Lower learning rate phase
    for param_group in optimizer.param_groups:
        param_group['lr'] = app_config.GP_LEARNING_RATE_FINAL
    
    for i in range(app_config.GP_EPOCHS_FINAL):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(dataloader.dataset)
        if (i + 1) % 2 == 0:
             print(f"GP Epoch {i+1+app_config.GP_EPOCHS_INITIAL}/{app_config.GP_EPOCHS_INITIAL + app_config.GP_EPOCHS_FINAL}, Loss: {epoch_loss:.3f}, LR: {app_config.GP_LEARNING_RATE_FINAL}")

    print("GP model training finished.")
    return model, likelihood

def predict_gp_residuals(
    model: ApproximateGPModelResiduals, 
    likelihood: gpytorch.likelihoods.GaussianLikelihood, 
    features_all_pixel_time_observations: np.ndarray, # Renamed for clarity
    app_config: 'config'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicts mean and variance of residuals for all pixels and time steps using the trained GP model.

    Args:
        model: Trained ApproximateGPModelResiduals.
        likelihood: Trained GaussianLikelihood.
        features_all_pixel_time_observations (np.ndarray): Features for all pixels and time steps
                                                            (num_total_observations, num_features), 
                                                            where num_total_observations = time * height * width.
        app_config: Configuration object.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - gp_mean_flat (np.ndarray): Predicted mean of residuals (num_total_observations,).
            - gp_variance_flat (np.ndarray): Predicted variance of residuals (num_total_observations,).
    """
    device = torch.device(app_config.DEVICE if torch.cuda.is_available() else "cpu")
    model.eval()
    likelihood.eval()

    features_tensor = torch.from_numpy(features_all_pixel_time_observations).float().to(device)
    
    gp_mean_preds = []
    gp_variance_preds = []

    # Predict in batches to avoid OOM for large feature sets
    pred_batch_size = app_config.GP_MINI_BATCH_SIZE 

    print(f"Predicting GP residuals for {features_all_pixel_time_observations.shape[0]} observations...")
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in tqdm(range(0, features_tensor.size(0), pred_batch_size), desc="GP Prediction Batches"):
            batch_features = features_tensor[i:i+pred_batch_size, :]
            predictions = likelihood(model(batch_features))
            gp_mean_preds.append(predictions.mean.cpu().numpy())
            gp_variance_preds.append(predictions.variance.cpu().numpy())
    
    gp_mean_flat = np.concatenate(gp_mean_preds)
    gp_variance_flat = np.concatenate(gp_variance_preds)
    
    return gp_mean_flat, gp_variance_flat

def prepare_gp_training_data(preprocessed_data: dict, atc_predictions: np.ndarray, app_config: 'config') -> tuple[torch.Tensor, torch.Tensor, np.ndarray, tuple[int, int, int]]:
    """
    Prepares training data (features and residuals) for the GP model from clear-sky pixels,
    and features for all pixels and time steps for prediction.
    Assumes lst_stack in preprocessed_data has np.nan for no-data/cloudy pixels.
    S2 stack is (time, num_bands, height, width).

    Args:
        preprocessed_data (dict): Dictionary from data_preprocessing.py.
                                  Expected: 'lst_stack', 's2_reflectance_stack', 'lon_coords', 'lat_coords'.
        atc_predictions (np.ndarray): ATC model predictions (time, height, width).
        app_config: Config object.

    Returns:
        tuple[torch.Tensor, torch.Tensor, np.ndarray, tuple[int, int, int]]:
            - train_x (torch.Tensor): Features for GP training (num_clear_samples, num_features).
            - train_y (torch.Tensor): Residuals for GP training (num_clear_samples,).
            - features_all_pixel_time_flat (np.ndarray): Flattened features for ALL pixels and ALL time steps
                                                           (time*height*width, num_features) for prediction.
            - original_dims (tuple[int, int, int]): Original (num_times, height, width) of the grid.
    """
    lst_stack = preprocessed_data['lst_stack'] # (time, height, width), np.nan for nodata
    s2_stack_time_bands_hw = preprocessed_data['s2_reflectance_stack'] # (time, num_bands, height, width)
    lon_coords = preprocessed_data['lon_coords'] # (height, width), normalized
    lat_coords = preprocessed_data['lat_coords'] # (height, width), normalized
    ndvi_stack = preprocessed_data.get('ndvi_stack') # (time, height, width), may be None

    num_times, height, width = lst_stack.shape
    if s2_stack_time_bands_hw.shape[0] != num_times or \
       s2_stack_time_bands_hw.shape[2] != height or \
       s2_stack_time_bands_hw.shape[3] != width:
        raise ValueError("Shape mismatch between LST stack and S2 stack (time, H, W dimensions)")
    num_s2_bands = s2_stack_time_bands_hw.shape[1]

    # Determine GP features based on config
    use_ndvi_feature = getattr(app_config, 'GP_USE_NDVI_FEATURE', False)
    use_temporal_mean_s2 = getattr(app_config, 'GP_USE_TEMPORAL_MEAN_S2_FEATURES', False)

    current_s2_features_for_gp = None
    num_gp_features = 0

    if use_ndvi_feature and ndvi_stack is not None:
        print("Using NDVI as a feature for GP model.")
        num_gp_features = 1 + 2 # NDVI + lon + lat
        # The ndvi_stack itself will be indexed per t,r,c later
    elif use_temporal_mean_s2:
        print("Using temporal mean of S2 bands for GP features.")
        # Calculate temporal mean of S2 bands. Result shape: (num_bands, height, width)
        # RuntimeWarning: Mean of empty slice, can be ignored if NaNs are handled as expected.
        with warnings.catch_warnings(): # Use standard warnings module
            warnings.filterwarnings('ignore', r'Mean of empty slice') # Use standard warnings module
            current_s2_features_for_gp = np.nanmean(s2_stack_time_bands_hw, axis=0) # Shape: (num_bands, height, width)
        num_gp_features = num_s2_bands + 2
    else:
        print("Using instantaneous S2 bands for GP features.")
        current_s2_features_for_gp = s2_stack_time_bands_hw # Shape: (time, num_bands, height, width)
        num_gp_features = num_s2_bands + 2

    if num_gp_features == 0:
        raise ValueError("Could not determine GP features based on configuration. num_gp_features is 0.")

    residuals = lst_stack - atc_predictions 
    residuals[np.isnan(lst_stack)] = np.nan 

    train_x_list = []
    train_y_list = []
    features_all_pixel_time_list = [] # For storing all (t,h,w) features for final prediction

    print("Preparing GP training data and prediction features...")
    for t in tqdm(range(num_times), desc="Processing Time Steps for GP Data"):
        for r in range(height):
            for c in range(width):
                pixel_features_list = []
                valid_pixel_features = True

                if use_ndvi_feature and ndvi_stack is not None:
                    ndvi_val = ndvi_stack[t, r, c]
                    if np.isnan(ndvi_val):
                        valid_pixel_features = False
                    pixel_features_list.append(ndvi_val)
                elif use_temporal_mean_s2:
                    # current_s2_features_for_gp has shape (num_bands, height, width)
                    s2_pixel_vals = current_s2_features_for_gp[:, r, c]
                    if np.isnan(s2_pixel_vals).any():
                        valid_pixel_features = False
                    pixel_features_list.extend(s2_pixel_vals.tolist())
                else: # Instantaneous S2
                    # current_s2_features_for_gp has shape (time, num_bands, height, width)
                    s2_pixel_vals = current_s2_features_for_gp[t, :, r, c]
                    if np.isnan(s2_pixel_vals).any():
                        valid_pixel_features = False
                    pixel_features_list.extend(s2_pixel_vals.tolist())
                
                coords_rc = [lon_coords[r, c], lat_coords[r, c]]
                full_pixel_features = pixel_features_list + coords_rc
                features_all_pixel_time_list.append(full_pixel_features)

                # Check for NaNs in residuals AND ensure pixel_features were valid for training data
                if not np.isnan(residuals[t, r, c]) and valid_pixel_features:
                    train_x_list.append(full_pixel_features)
                    train_y_list.append(residuals[t, r, c])
    
    if not train_x_list:
        # Handle case with no clear sky data for training GP - this would be problematic
        print("Warning: No clear-sky data points (with valid features) found to train the GP model. Predictions will be based on untrained GP (prior only).")
        # Use the dynamically determined num_gp_features
        train_x = torch.empty(0, num_gp_features, dtype=torch.float32)
        train_y = torch.empty(0, dtype=torch.float32)
    else:
        train_x = torch.tensor(train_x_list, dtype=torch.float32)
        train_y = torch.tensor(train_y_list, dtype=torch.float32)

    features_all_pixel_time_flat = np.array(features_all_pixel_time_list, dtype=np.float32)
    original_dims = (num_times, height, width)

    print(f"GP training features shape: {train_x.shape}")
    print(f"GP training targets shape: {train_y.shape}")
    print(f"GP prediction features shape (flat): {features_all_pixel_time_flat.shape}")

    return train_x, train_y, features_all_pixel_time_flat, original_dims


def train_and_predict_all_gp_residuals(preprocessed_data: dict, atc_predictions:np.ndarray, app_config: 'config') -> tuple[np.ndarray, np.ndarray]:
    """
    Orchestrates GP model training and prediction of residuals for all pixels and time steps.

    Args:
        preprocessed_data (dict): Data from preprocessing.
        atc_predictions (np.ndarray): ATC model predictions (time, height, width).
        app_config: Configuration object.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - gp_mean_residuals_map (np.ndarray): Predicted mean of residuals (time, height, width).
            - gp_variance_residuals_map (np.ndarray): Predicted variance of residuals (time, height, width).
    """
    train_x, train_y, features_all_pixel_time_flat, original_dims = prepare_gp_training_data(
        preprocessed_data, atc_predictions, app_config
    )

    num_times, height, width = original_dims

    if train_x.shape[0] < app_config.GP_NUM_INDUCING_POINTS: # Check if enough data for inducing points
        print(f"Warning: Number of clear sky training points ({train_x.shape[0]}) is less than GP_NUM_INDUCING_POINTS ({app_config.GP_NUM_INDUCING_POINTS}). Adjusting inducing points.")
        # You might want to set app_config.GP_NUM_INDUCING_POINTS = train_x.shape[0] here if mutable, or handle in train_gp_model
        # For now, train_gp_model itself has a min(app_config.GP_NUM_INDUCING_POINTS, train_x.size(0)) safety

    if train_x.shape[0] == 0:
        print("Skipping GP model training and prediction as no clear-sky training data is available.")
        # Return maps full of NaNs or zeros, matching the expected shape (time, height, width)
        gp_mean_map_reshaped = np.full(original_dims, np.nan, dtype=np.float32)
        gp_variance_map_reshaped = np.full(original_dims, np.nan, dtype=np.float32)
        return gp_mean_map_reshaped, gp_variance_map_reshaped

    gp_model, likelihood = train_gp_model(train_x, train_y, app_config)
    
    # Predict for all pixels and times using features_all_pixel_time_flat
    gp_mean_flat, gp_variance_flat = predict_gp_residuals(
        gp_model, likelihood, features_all_pixel_time_flat, app_config
    )

    # Reshape flat predictions back to (time, height, width)
    gp_mean_map_reshaped = gp_mean_flat.reshape(original_dims)
    gp_variance_map_reshaped = gp_variance_flat.reshape(original_dims)

    print(f"GP mean residuals map shape: {gp_mean_map_reshaped.shape}")
    print(f"GP variance residuals map shape: {gp_variance_map_reshaped.shape}")

    return gp_mean_map_reshaped, gp_variance_map_reshaped


# --- Test block for gp_model.py ---
if __name__ == '__main__':
    # Create a dummy config (can inherit from your actual config if it has defaults)
    class DummyConfig(config.Config): # Inherit from actual config to get some defaults
        def __init__(self):
            super().__init__() # Ensure base class init is called if it does anything
            # Override specific settings for the test if needed
            self.DEVICE = "cpu"
            self.GP_NUM_INDUCING_POINTS = 64 # Smaller for faster test
            self.GP_EPOCHS_INITIAL = 3 # Minimal epochs for test
            self.GP_EPOCHS_FINAL = 2   # Minimal epochs for test
            self.GP_MINI_BATCH_SIZE = 128
            self.GP_LEARNING_RATE_INITIAL = 0.01
            self.GP_LEARNING_RATE_FINAL = 0.001
            # Define GP_RESIDUAL_FEATURES to match dummy data: 4 S2 bands + 2 coords
            # This GP_RESIDUAL_FEATURES in config might become misleading as features are now dynamic.
            # The actual number of features is determined in prepare_gp_training_data.
            self.GP_RESIDUAL_FEATURES_CONFIG_IGNORED = ['s2_b1', 's2_b2', 's2_b3', 's2_b4', 'lon', 'lat']
            self.GP_USE_TEMPORAL_MEAN_S2_FEATURES = False # Test default behavior first
            self.GP_USE_NDVI_FEATURE = False # Test default behavior
            self.S2_RED_INDEX = 2 # Example for dummy S2 [B,G,R,N]
            self.S2_NIR_INDEX = 3 # Example for dummy S2 [B,G,R,N]

    test_app_config_default = DummyConfig()
    
    # Test with GP_USE_TEMPORAL_MEAN_S2_FEATURES = True
    class DummyConfigTemporalMeanS2(DummyConfig):
        def __init__(self):
            super().__init__()
            self.GP_USE_TEMPORAL_MEAN_S2_FEATURES = True
            self.GP_USE_NDVI_FEATURE = False
            
    test_app_config_temporal_mean_s2 = DummyConfigTemporalMeanS2()

    # Test with GP_USE_NDVI_FEATURE = True
    class DummyConfigNDVI(DummyConfig):
        def __init__(self):
            super().__init__()
            self.GP_USE_TEMPORAL_MEAN_S2_FEATURES = False # Ensure this is False if NDVI is primary
            self.GP_USE_NDVI_FEATURE = True

    test_app_config_ndvi = DummyConfigNDVI()

    # Dummy preprocessed_data
    num_times_test, height_test, width_test = 3, 5, 5
    num_s2_bands_test = 4 # Matching the change

    # LST stack with some NaNs
    dummy_lst_stack = np.random.rand(num_times_test, height_test, width_test).astype(np.float32) * 10 + 290
    dummy_lst_stack[0, 0, 0] = np.nan # Simulate a cloudy pixel
    dummy_lst_stack[1, 1, 1:3] = np.nan

    # S2 stack (time, bands, height, width)
    dummy_s2_stack = np.random.rand(num_times_test, num_s2_bands_test, height_test, width_test).astype(np.float32)
    dummy_s2_stack[0, :, 0, 0] = np.nan # S2 can also have NaNs if source was NaN

    # Dummy NDVI stack (time, height, width) - will be populated by data_preprocessing if active
    # For direct testing of gp_model.py, we can simulate its presence if GP_USE_NDVI_FEATURE is true in test config
    dummy_ndvi_stack = np.random.rand(num_times_test, height_test, width_test).astype(np.float32)
    dummy_ndvi_stack[0, 0, 1] = np.nan # Simulate some NaN in NDVI

    # Coordinates
    dummy_lon_coords = np.linspace(0, 1, width_test).reshape(1, -1).repeat(height_test, axis=0)
    dummy_lat_coords = np.linspace(0, 1, height_test).reshape(-1, 1).repeat(width_test, axis=1)

    # Dummy ATC predictions (all clear for simplicity here)
    dummy_atc_predictions = np.random.rand(num_times_test, height_test, width_test).astype(np.float32) * 10 + 288

    preprocessed_data_test_base = {
        "lst_stack": dummy_lst_stack,
        "s2_reflectance_stack": dummy_s2_stack, 
        "lon_coords": dummy_lon_coords,
        "lat_coords": dummy_lat_coords,
        # "cloud_mask_stack": dummy_cloud_mask # No longer used
    }

    print("--- Testing GP Model module --- ")
    
    # Test scenarios
    test_configs_to_run = {
        "Instantaneous_S2": test_app_config_default,
        "Temporal_Mean_S2": test_app_config_temporal_mean_s2,
        "NDVI_Feature": test_app_config_ndvi
    }

    for test_name, current_test_config in test_configs_to_run.items():
        print(f"\\n--- Running GP Model Test Scenario: {test_name} ---")
        
        # Add dummy NDVI to preprocessed_data if this test scenario uses NDVI
        current_preprocessed_data_test = preprocessed_data_test_base.copy()
        if getattr(current_test_config, 'GP_USE_NDVI_FEATURE', False):
            current_preprocessed_data_test['ndvi_stack'] = dummy_ndvi_stack
            print("  (Added dummy NDVI stack for this test scenario)")
        else:
            # Ensure ndvi_stack is not present if not testing NDVI, or is None
            current_preprocessed_data_test['ndvi_stack'] = None 

        try:
            # Test data preparation
            train_x_out, train_y_out, features_pred_flat_out, original_dims_out = prepare_gp_training_data(
                current_preprocessed_data_test, dummy_atc_predictions, current_test_config
            )
            print(f"prepare_gp_training_data output shapes for {test_name}:")
            print(f"  train_x: {train_x_out.shape}")
            print(f"  train_y: {train_y_out.shape}")
            print(f"  features_pred_flat: {features_pred_flat_out.shape}")
            print(f"  original_dims: {original_dims_out}")

            expected_total_obs = num_times_test * height_test * width_test
            # Determine expected number of features based on the current test config
            if getattr(current_test_config, 'GP_USE_NDVI_FEATURE', False):
                expected_num_features_for_test = 1 + 2 # NDVI + lon + lat
            else:
                expected_num_features_for_test = num_s2_bands_test + 2 # S2 bands + lon + lat
            
            assert features_pred_flat_out.shape == (expected_total_obs, expected_num_features_for_test), \
                f"Shape mismatch for prediction features in {test_name}. Expected ({expected_total_obs}, {expected_num_features_for_test}), Got {features_pred_flat_out.shape}"
            assert original_dims_out == (num_times_test, height_test, width_test), f"Original dimensions mismatch in {test_name}"
            assert train_x_out.shape[1] == expected_num_features_for_test, \
                f"Number of features in train_x is incorrect for {test_name}. Expected {expected_num_features_for_test}, Got {train_x_out.shape[1]}"

            # Basic check on training data points (should be less than total if there are NaNs in LST or features)
            # This check becomes more complex with conditional features, so focusing on shapes primarily.
            print(f"Number of training points for {test_name}: {train_x_out.shape[0]}")

            # Test full pipeline: train_and_predict_all_gp_residuals
            if train_x_out.shape[0] > 0: # Only run full train/predict if there is training data
                gp_mean_map, gp_var_map = train_and_predict_all_gp_residuals(
                    current_preprocessed_data_test, dummy_atc_predictions, current_test_config
                )
                print(f"train_and_predict_all_gp_residuals output shapes for {test_name}:")
                print(f"  gp_mean_map: {gp_mean_map.shape}")
                print(f"  gp_var_map: {gp_var_map.shape}")
                assert gp_mean_map.shape == (num_times_test, height_test, width_test)
                assert gp_var_map.shape == (num_times_test, height_test, width_test)
                print(f"GP module test completed successfully for {test_name}.")
            else:
                print("Skipping full GP train/predict test as no training data was generated (e.g., all LST was NaN).")
                # Test the case where no training data is available
                dummy_lst_all_nan = np.full_like(dummy_lst_stack, np.nan)
                preprocessed_data_all_nan_lst = current_preprocessed_data_test.copy()
                preprocessed_data_all_nan_lst["lst_stack"] = dummy_lst_all_nan
                gp_mean_map_no_train, gp_var_map_no_train = train_and_predict_all_gp_residuals(
                    preprocessed_data_all_nan_lst, dummy_atc_predictions, current_test_config
                )
                assert np.all(np.isnan(gp_mean_map_no_train)), f"Mean map should be all NaN if no training data ({test_name})"
                assert np.all(np.isnan(gp_var_map_no_train)), f"Variance map should be all NaN if no training data ({test_name})"
                print(f"GP module test for no training data completed successfully for {test_name}.")

        except Exception as e:
            print(f"Error during GP module test ({test_name}): {e}")
            import traceback
            traceback.print_exc()

    print("Finished GP Model module main execution.") 