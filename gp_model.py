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
import os # Import os module for file operations

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

def _train_gp_model_internal( # Renamed to internal, takes inducing_points
    train_x: torch.Tensor, 
    train_y: torch.Tensor, 
    inducing_points: torch.Tensor, # Added inducing_points as direct argument
    app_config: 'config'
) -> tuple[ApproximateGPModelResiduals, gpytorch.likelihoods.GaussianLikelihood, list[float]]:
    """
    Internal function to train the Approximate GP model for residuals.

    Args:
        train_x (torch.Tensor): Training features (num_samples, num_features).
        train_y (torch.Tensor): Training targets (residuals) (num_samples).
        inducing_points (torch.Tensor): Pre-determined inducing points.
        app_config: Configuration object.

    Returns:
        tuple[ApproximateGPModelResiduals, gpytorch.likelihoods.GaussianLikelihood, list[float]]:
            - Trained GP model.
            - Trained likelihood.
            - List of mean losses for each logging interval.
    """
    device = torch.device(app_config.DEVICE if torch.cuda.is_available() else "cpu")
    train_x, train_y = train_x.to(device), train_y.to(device)
    inducing_points = inducing_points.to(device) # Ensure inducing points are on the correct device

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
    
    # Loss logging setup
    total_epochs_gp = app_config.GP_EPOCHS_INITIAL + app_config.GP_EPOCHS_FINAL
    loss_logging_interval_gp = getattr(app_config, 'GP_LOSS_LOGGING_INTERVAL', 10)
    num_loss_intervals_gp = (total_epochs_gp + loss_logging_interval_gp -1) // loss_logging_interval_gp
    interval_losses_output_gp = [np.nan] * num_loss_intervals_gp
    current_interval_losses_gp = []
    current_interval_idx_gp = 0
    global_epoch_count = 0

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
        current_interval_losses_gp.append(epoch_loss) # Store average loss for this epoch
        global_epoch_count += 1

        if global_epoch_count % loss_logging_interval_gp == 0:
            if current_interval_losses_gp:
                interval_losses_output_gp[current_interval_idx_gp] = np.mean(current_interval_losses_gp)
            current_interval_losses_gp = []
            current_interval_idx_gp += 1

        if (i + 1) % 10 == 0: # Keep some print statements for feedback during long training
            print(f"GP Epoch (Initial LR) {i+1}/{app_config.GP_EPOCHS_INITIAL}, Avg Loss: {epoch_loss:.3f}, LR: {app_config.GP_LEARNING_RATE_INITIAL}")

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
        current_interval_losses_gp.append(epoch_loss) # Store average loss for this epoch
        global_epoch_count += 1

        if global_epoch_count % loss_logging_interval_gp == 0:
            if current_interval_losses_gp:
                interval_losses_output_gp[current_interval_idx_gp] = np.mean(current_interval_losses_gp)
            current_interval_losses_gp = []
            if current_interval_idx_gp < num_loss_intervals_gp: # Prevent index out of bounds if last interval is full
                 current_interval_idx_gp += 1

        if (i + 1) % 2 == 0:
             print(f"GP Epoch (Final LR) {i+1+app_config.GP_EPOCHS_INITIAL}/{total_epochs_gp}, Avg Loss: {epoch_loss:.3f}, LR: {app_config.GP_LEARNING_RATE_FINAL}")

    # Handle any remaining losses in the last partial interval
    if current_interval_losses_gp and current_interval_idx_gp < num_loss_intervals_gp:
        interval_losses_output_gp[current_interval_idx_gp] = np.mean(current_interval_losses_gp)

    print("GP model training finished.")
    return model, likelihood, interval_losses_output_gp

def save_gp_model(model: ApproximateGPModelResiduals, likelihood: gpytorch.likelihoods.GaussianLikelihood, inducing_points: torch.Tensor, filepath: str, interval_losses: list[float] = None):
    """
    Saves the GP model state_dict, likelihood state_dict, inducing points, and optionally interval losses.
    Args:
        model: Trained GP model.
        likelihood: Trained likelihood.
        inducing_points (torch.Tensor): Inducing points used for the model.
        filepath (str): Path to save the model components (.pth).
        interval_losses (list[float], optional): List of mean losses per interval. If provided, saved to a .npy file with similar name.
    """
    state = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'inducing_points': inducing_points.cpu() # Save inducing points on CPU
    }
    torch.save(state, filepath)
    print(f"GP model, likelihood, and inducing points saved to {filepath}")

    if interval_losses is not None:
        loss_filepath = filepath.replace('.pth', '_interval_losses.npy') # More descriptive filename
        try:
            np.save(loss_filepath, np.array(interval_losses))
            print(f"GP interval losses saved to {loss_filepath}")
        except Exception as e:
            print(f"Error saving GP interval losses to {loss_filepath}: {e}")

def load_gp_model(filepath: str, app_config: 'config') -> tuple[ApproximateGPModelResiduals, gpytorch.likelihoods.GaussianLikelihood]:
    """
    Loads the GP model state_dict, likelihood state_dict, and inducing points.
    Initializes a new model instance with these components.

    Args:
        filepath (str): Path to the saved model components.
        app_config: Configuration object.

    Returns:
        tuple[ApproximateGPModelResiduals, gpytorch.likelihoods.GaussianLikelihood]:
            - Loaded and initialized GP model.
            - Loaded and initialized likelihood.
    """
    device = torch.device(app_config.DEVICE if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GP model file not found at {filepath}")
        
    state = torch.load(filepath, map_location=device)
    
    inducing_points = state['inducing_points'].to(device)
    
    model = ApproximateGPModelResiduals(inducing_points=inducing_points).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    
    model.load_state_dict(state['model_state_dict'])
    likelihood.load_state_dict(state['likelihood_state_dict'])
    
    print(f"GP model, likelihood, and inducing points loaded from {filepath}")
    return model, likelihood

def load_gp_interval_losses(model_filepath: str) -> list[float] | None:
    """
    Loads GP interval losses if the corresponding _interval_losses.npy file exists.

    Args:
        model_filepath (str): Path to the primary GP model .pth file.

    Returns:
        list[float] | None: List of losses, or None if file not found or error occurs.
    """
    loss_filepath = model_filepath.replace('.pth', '_interval_losses.npy')
    if os.path.exists(loss_filepath):
        try:
            losses = np.load(loss_filepath)
            print(f"GP interval losses loaded from {loss_filepath}")
            return list(losses)
        except Exception as e:
            print(f"Error loading GP interval losses from {loss_filepath}: {e}")
            return None
    else:
        # print(f"GP interval loss file not found: {loss_filepath}") # Optional: for debugging
        return None

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

def prepare_gp_training_data(preprocessed_data: dict, atc_predictions: np.ndarray, app_config: 'config') -> tuple[torch.Tensor, torch.Tensor, np.ndarray, tuple[int, int, int], torch.Tensor]:
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
        tuple[torch.Tensor, torch.Tensor, np.ndarray, tuple[int, int, int], torch.Tensor]:
            - train_x (torch.Tensor): Features for GP training (num_clear_samples, num_features).
            - train_y (torch.Tensor): Residuals for GP training (num_clear_samples,).
            - features_all_pixel_time_flat (np.ndarray): Flattened features for ALL pixels and ALL time steps
                                                           (time*height*width, num_features) for prediction.
            - original_dims (tuple[int, int, int]): Original (num_times, height, width) of the grid.
            - initial_inducing_points (torch.Tensor): Initial inducing points based on training data.
    """
    lst_stack = preprocessed_data['lst_stack'] # (time, height, width), np.nan for nodata
    s2_stack_time_bands_hw = preprocessed_data['s2_reflectance_stack'] # (time, num_bands, height, width)
    lon_coords = preprocessed_data['lon_coords'] # (height, width), normalized
    lat_coords = preprocessed_data['lat_coords'] # (height, width), normalized
    ndvi_stack = preprocessed_data.get('ndvi_stack') # (time, height, width), may be None
    training_pixel_mask = preprocessed_data.get('training_pixel_mask') # (height, width) bool array or None

    num_times, height, width = lst_stack.shape
    if s2_stack_time_bands_hw.shape[0] != num_times or \
       s2_stack_time_bands_hw.shape[2] != height or \
       s2_stack_time_bands_hw.shape[3] != width:
        raise ValueError("Shape mismatch between LST stack and S2 stack (time, H, W dimensions)")
    num_s2_bands = s2_stack_time_bands_hw.shape[1]

    if training_pixel_mask is None:
        print("No training_pixel_mask found in preprocessed_data for GP, defaulting to considering all pixels for training data selection.")
        training_pixel_mask = np.ones((height, width), dtype=bool)
    elif not isinstance(training_pixel_mask, np.ndarray) or training_pixel_mask.shape != (height, width):
        print(f"Warning: training_pixel_mask for GP has unexpected type/shape. Defaulting to considering all pixels.")
        training_pixel_mask = np.ones((height, width), dtype=bool)

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

                # Check for NaNs in residuals AND ensure pixel_features were valid AND pixel is in training_pixel_mask
                if training_pixel_mask[r, c]: # Only consider this pixel for training if it's in the mask
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

    # Determine initial inducing points from train_x
    num_inducing = min(app_config.GP_NUM_INDUCING_POINTS, train_x.size(0))
    if train_x.size(0) > 0:
        inducing_points_indices = torch.randperm(train_x.size(0))[:num_inducing]
        initial_inducing_points = train_x[inducing_points_indices, :]
    else: # Handle case with no training data, create dummy inducing points matching feature count
        initial_inducing_points = torch.empty(0, num_gp_features, dtype=torch.float32)

    features_all_pixel_time_flat = np.array(features_all_pixel_time_list, dtype=np.float32)
    original_dims = (num_times, height, width)

    print(f"GP training features shape: {train_x.shape}")
    print(f"GP training targets shape: {train_y.shape}")
    print(f"GP prediction features shape (flat): {features_all_pixel_time_flat.shape}")
    print(f"Initial inducing points shape: {initial_inducing_points.shape}")

    return train_x, train_y, features_all_pixel_time_flat, original_dims, initial_inducing_points

def train_and_save_gp_model(preprocessed_data: dict, atc_predictions:np.ndarray, app_config: 'config'):
    """
    Orchestrates GP model training and saves the trained model and its interval losses.

    Args:
        preprocessed_data (dict): Data from preprocessing.
        atc_predictions (np.ndarray): ATC model predictions (time, height, width).
        app_config: Configuration object.
    """
    train_x, train_y, _, _, inducing_points_initial = prepare_gp_training_data(
        preprocessed_data, atc_predictions, app_config
    )

    if train_x.shape[0] < app_config.GP_NUM_INDUCING_POINTS and train_x.shape[0] > 0:
        print(f"Warning: Number of clear sky training points ({train_x.shape[0]}) is less than GP_NUM_INDUCING_POINTS ({app_config.GP_NUM_INDUCING_POINTS}). Using all training points as inducing points.")
        # inducing_points_initial will already be capped at train_x.shape[0] by prepare_gp_training_data

    if train_x.shape[0] == 0:
        print("Skipping GP model training as no clear-sky training data is available.")
        # Optionally, save a "dummy" or "untrained" model state if downstream requires a file
        # For now, we'll just skip saving if no training occurs.
        return

    gp_model_trained, likelihood_trained, gp_interval_losses = _train_gp_model_internal(train_x, train_y, inducing_points_initial, app_config)
    
    # Save the model
    # Inducing points to save are model.variational_strategy.inducing_points (these are the learned ones)
    learned_inducing_points = gp_model_trained.variational_strategy.inducing_points.data.clone().detach()

    gp_model_save_path = os.path.join(app_config.MODEL_WEIGHTS_PATH, app_config.GP_MODEL_WEIGHT_FILENAME)
    save_gp_model(gp_model_trained, likelihood_trained, learned_inducing_points, gp_model_save_path, interval_losses=gp_interval_losses)

def load_and_predict_gp_residuals(preprocessed_data: dict, atc_predictions: np.ndarray, app_config: 'config') -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a pre-trained GP model and predicts residuals for all pixels and time steps.

    Args:
        preprocessed_data (dict): Data from preprocessing. Used to get features for prediction.
        atc_predictions (np.ndarray): ATC model predictions (time, height, width). Used for feature prep.
        app_config: Configuration object.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - gp_mean_residuals_map (np.ndarray): Predicted mean of residuals (time, height, width).
            - gp_variance_residuals_map (np.ndarray): Predicted variance of residuals (time, height, width).
    """
    # Prepare data mainly to get `features_all_pixel_time_flat` and `original_dims`
    # train_x, train_y, and initial_inducing_points from this call are not used for prediction if loading a model.
    _, _, features_all_pixel_time_flat, original_dims, _ = prepare_gp_training_data(
        preprocessed_data, atc_predictions, app_config
    )
    
    gp_model_load_path = os.path.join(app_config.MODEL_WEIGHTS_PATH, app_config.GP_MODEL_WEIGHT_FILENAME)
    
    if not os.path.exists(gp_model_load_path):
        print(f"GP model file not found at {gp_model_load_path}. Cannot perform GP prediction.")
        print("This might happen if GP training was skipped due to no clear data.")
        print("Returning NaN maps for GP residuals.")
        num_times, height, width = original_dims
        gp_mean_map_reshaped = np.full(original_dims, np.nan, dtype=np.float32)
        gp_variance_map_reshaped = np.full(original_dims, np.nan, dtype=np.float32)
        return gp_mean_map_reshaped, gp_variance_map_reshaped

    gp_model, likelihood = load_gp_model(gp_model_load_path, app_config)
    
    gp_mean_flat, gp_variance_flat = predict_gp_residuals(
        gp_model, likelihood, features_all_pixel_time_flat, app_config
    )

    # Reshape flat predictions back to (time, height, width)
    gp_mean_map_reshaped = gp_mean_flat.reshape(original_dims)
    gp_variance_map_reshaped = gp_variance_flat.reshape(original_dims)

    print(f"GP mean residuals map shape (from loaded model): {gp_mean_map_reshaped.shape}")
    print(f"GP variance residuals map shape (from loaded model): {gp_variance_map_reshaped.shape}")

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
            self.GP_LOSS_LOGGING_INTERVAL = 2 # For testing, log more frequently

    test_app_config_default = DummyConfig()
    
    # Test with GP_USE_TEMPORAL_MEAN_S2_FEATURES = True
    class DummyConfigTemporalMeanS2(DummyConfig):
        def __init__(self):
            super().__init__()
            self.GP_USE_TEMPORAL_MEAN_S2_FEATURES = True
            self.GP_USE_NDVI_FEATURE = False
            self.GP_LOSS_LOGGING_INTERVAL = 2 # Override for test
            
    test_app_config_temporal_mean_s2 = DummyConfigTemporalMeanS2()

    # Test with GP_USE_NDVI_FEATURE = True
    class DummyConfigNDVI(DummyConfig):
        def __init__(self):
            super().__init__()
            self.GP_USE_TEMPORAL_MEAN_S2_FEATURES = False # Ensure this is False if NDVI is primary
            self.GP_USE_NDVI_FEATURE = True
            self.GP_LOSS_LOGGING_INTERVAL = 2 # Override for test

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
            train_x_out, train_y_out, features_pred_flat_out, original_dims_out, initial_inducing_points_out = prepare_gp_training_data(
                current_preprocessed_data_test, dummy_atc_predictions, current_test_config
            )
            print(f"prepare_gp_training_data output shapes for {test_name}:")
            print(f"  train_x: {train_x_out.shape}")
            print(f"  train_y: {train_y_out.shape}")
            print(f"  features_pred_flat: {features_pred_flat_out.shape}")
            print(f"  original_dims: {original_dims_out}")
            print(f"  initial_inducing_points: {initial_inducing_points_out.shape}")

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
            # This old function is now split. We test train_and_save then load_and_predict
            
            # Path for dummy model saving
            dummy_gp_model_filename = f"dummy_gp_model_{test_name}.pth"
            current_test_config.GP_MODEL_WEIGHT_FILENAME = dummy_gp_model_filename # Override for test
            dummy_model_weights_dir = "dummy_gp_weights_output"
            os.makedirs(dummy_model_weights_dir, exist_ok=True)
            current_test_config.MODEL_WEIGHTS_PATH = dummy_model_weights_dir

            if train_x_out.shape[0] > 0: # Only run full train/predict if there is training data
                print(f"Testing train_and_save_gp_model for {test_name}...")
                train_and_save_gp_model(
                    current_preprocessed_data_test, dummy_atc_predictions, current_test_config
                )
                dummy_gp_model_path = os.path.join(current_test_config.MODEL_WEIGHTS_PATH, current_test_config.GP_MODEL_WEIGHT_FILENAME)
                assert os.path.exists(dummy_gp_model_path), f"GP model file was not saved for {test_name} at {dummy_gp_model_path}"
                
                # Test loading interval losses
                loaded_losses = load_gp_interval_losses(dummy_gp_model_path)
                assert loaded_losses is not None, f"GP interval losses could not be loaded for {test_name}"
                expected_num_gp_loss_intervals = (current_test_config.GP_EPOCHS_INITIAL + current_test_config.GP_EPOCHS_FINAL + current_test_config.GP_LOSS_LOGGING_INTERVAL -1) // current_test_config.GP_LOSS_LOGGING_INTERVAL
                assert len(loaded_losses) == expected_num_gp_loss_intervals, f"Loaded GP losses have incorrect length for {test_name}. Expected {expected_num_gp_loss_intervals}, got {len(loaded_losses)}"
                print(f"Loaded GP losses for {test_name}: {loaded_losses}")

                print(f"Model saved for {test_name}. Now testing load_and_predict_gp_residuals...")

                gp_mean_map, gp_var_map = load_and_predict_gp_residuals(
                    current_preprocessed_data_test, dummy_atc_predictions, current_test_config
                )
                
                print(f"load_and_predict_gp_residuals output shapes for {test_name}:")
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
                
                print(f"Testing train_and_save_gp_model with NO training data for {test_name}...")
                # This should not save a file if training is skipped.
                train_and_save_gp_model(
                    preprocessed_data_all_nan_lst, dummy_atc_predictions, current_test_config
                )
                # Ensure file for this specific "no data" scenario does not exist if train_and_save skips saving
                dummy_no_data_model_file = os.path.join(current_test_config.MODEL_WEIGHTS_PATH, current_test_config.GP_MODEL_WEIGHT_FILENAME)
                assert not os.path.exists(dummy_no_data_model_file.replace('.pth', '_interval_losses.npy')), \
                    f"GP loss file for NO DATA scenario ({test_name}) should not exist if training was skipped."

                print(f"Testing load_and_predict_gp_residuals with NO training data (expecting NaNs due to no model or prior) for {test_name}...")
                gp_mean_map_no_train, gp_var_map_no_train = load_and_predict_gp_residuals(
                    preprocessed_data_all_nan_lst, dummy_atc_predictions, current_test_config
                )

                assert np.all(np.isnan(gp_mean_map_no_train)), f"Mean map should be all NaN if no training data ({test_name})"
                assert np.all(np.isnan(gp_var_map_no_train)), f"Variance map should be all NaN if no training data ({test_name})"

        except Exception as e:
            print(f"Error during GP module test ({test_name}): {e}")
            import traceback
            traceback.print_exc()
    
    # Clean up dummy model weights directory
    import shutil
    if os.path.exists(dummy_model_weights_dir):
        shutil.rmtree(dummy_model_weights_dir)
        print(f"Cleaned up dummy GP weights directory: {dummy_model_weights_dir}")

    print("Finished GP Model module main execution.") 