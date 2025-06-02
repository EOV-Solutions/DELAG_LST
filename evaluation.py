"""
Model evaluation functions for the DELAG project.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os

import config

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """
    Calculates MAE, RMSE, R2, and Bias between true and predicted values.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        prefix (str, optional): Prefix for metric names in the output dictionary.

    Returns:
        dict: Dictionary of calculated metrics.
    """
    # Flatten arrays and remove NaNs
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    valid_mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
    if not np.any(valid_mask):
        print(f"Warning: No valid (non-NaN) pairs for metric calculation with prefix '{prefix}'.")
        return {
            f"{prefix}mae": np.nan,
            f"{prefix}rmse": np.nan,
            f"{prefix}r2": np.nan,
            f"{prefix}bias": np.nan,
            f"{prefix}count": 0
        }

    y_true_valid = y_true_flat[valid_mask]
    y_pred_valid = y_pred_flat[valid_mask]

    if len(y_true_valid) == 0:
        print(f"Warning: No valid data points after filtering NaNs for prefix '{prefix}'.")
        return {
            f"{prefix}mae": np.nan,
            f"{prefix}rmse": np.nan,
            f"{prefix}r2": np.nan,
            f"{prefix}bias": np.nan,
            f"{prefix}count": 0
        }

    metrics = {
        f"{prefix}mae": mean_absolute_error(y_true_valid, y_pred_valid),
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        f"{prefix}r2": r2_score(y_true_valid, y_pred_valid),
        f"{prefix}bias": np.mean(y_pred_valid - y_true_valid),
        f"{prefix}count": len(y_true_valid)
    }
    return metrics

def evaluate_reconstruction_simulated_clouds(
    model_predicted_lst: np.ndarray,
    observed_lst_clear: np.ndarray, 
    app_config: 'config'
) -> dict:
    """
    Evaluates model-predicted LST against clear-sky observations.

    Args:
        model_predicted_lst (np.ndarray): The model's direct LST predictions (T, H, W).
        observed_lst_clear (np.ndarray): Original LST observations, including only clear-sky values (NaN elsewhere) (T, H, W).
        app_config: Configuration object.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    print("Evaluating reconstruction under simulated cloud conditions...")
    
    # For this strategy, we are interested in locations that WERE originally clear.
    # We compare the original clear LST (ground truth) with the reconstructed LST at these specific locations.
    
    # Clear pixels are where observed_lst_clear is not NaN.
    clear_pixels_mask = ~np.isnan(observed_lst_clear)
    
    true_values = observed_lst_clear[clear_pixels_mask]
    pred_values = model_predicted_lst[clear_pixels_mask]

    simulated_metrics = calculate_metrics(true_values, pred_values, prefix="simulated_clouds_")
    print(f"Simulated cloud evaluation metrics (model_predicted vs observed_clear): {simulated_metrics}")
    return simulated_metrics

def evaluate_reconstruction_heavily_cloudy(
    model_predicted_lst: np.ndarray,
    observed_lst_clear: np.ndarray, 
    app_config: 'config' 
) -> dict:
    """
    Evaluates performance under heavily cloudy conditions by holding out a percentage of valid observations,
    comparing model's direct predictions against these holdout clear pixels.

    Args:
        model_predicted_lst (np.ndarray): The model's direct LST predictions (T, H, W).
        observed_lst_clear (np.ndarray): Original LST observations, including only clear-sky values (NaN elsewhere) (T, H, W).
        app_config: Configuration object.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    print("Evaluating reconstruction under heavily cloudy (holdout) conditions...")
    np.random.seed(app_config.RANDOM_SEED)

    # Identify all originally clear pixels with valid observations (i.e., not NaN)
    clear_pixel_indices_time, clear_pixel_indices_row, clear_pixel_indices_col = np.where(
        ~np.isnan(observed_lst_clear)
    )

    if len(clear_pixel_indices_time) == 0:
        print("Warning: No clear pixels found for heavily cloudy evaluation.")
        return calculate_metrics(np.array([]), np.array([]), prefix="heavily_cloudy_holdout_")

    num_clear_pixels = len(clear_pixel_indices_time)
    holdout_size = int(app_config.EVAL_HOLDOUT_PERCENTAGE * num_clear_pixels)
    
    if holdout_size == 0:
        print(f"Warning: Holdout size is 0 ({app_config.EVAL_HOLDOUT_PERCENTAGE*100}% of {num_clear_pixels} clear pixels). Skipping heavily cloudy eval.")
        return calculate_metrics(np.array([]), np.array([]), prefix="heavily_cloudy_holdout_")

    # Randomly select indices for the holdout set
    holdout_indices_selector = np.random.choice(num_clear_pixels, size=holdout_size, replace=False)

    holdout_time_indices = clear_pixel_indices_time[holdout_indices_selector]
    holdout_row_indices = clear_pixel_indices_row[holdout_indices_selector]
    holdout_col_indices = clear_pixel_indices_col[holdout_indices_selector]

    true_values_holdout = observed_lst_clear[holdout_time_indices, holdout_row_indices, holdout_col_indices]
    pred_values_holdout = model_predicted_lst[holdout_time_indices, holdout_row_indices, holdout_col_indices]
    
    # Here, pred_values_holdout are from the `model_predicted_lst`.
    # This now correctly tests the model's pure predictive capability on these holdout points.

    holdout_metrics = calculate_metrics(true_values_holdout, pred_values_holdout, prefix="heavily_cloudy_holdout_")
    print(f"Heavily cloudy (holdout) evaluation metrics: {holdout_metrics}")
    return holdout_metrics

def run_all_evaluations(
    model_predicted_lst: np.ndarray,
    observed_lst_clear: np.ndarray, # This is typically preprocessed_data['lst_stack']
    app_config: 'config'
) -> dict:
    """
    Runs all defined evaluation strategies using the model's direct predictions.

    Args:
        model_predicted_lst (np.ndarray): Model's direct LST predictions (T, H, W).
        observed_lst_clear (np.ndarray): Original LST observations, with NaNs for clouds (T, H, W).
        app_config: Configuration object.

    Returns:
        dict: Combined dictionary of all evaluation metrics.
    """
    all_metrics = {}

    # Strategy 1: Compare model-predicted LST with clear-sky observations
    metrics_simulated = evaluate_reconstruction_simulated_clouds(
        model_predicted_lst, observed_lst_clear, app_config
    )
    all_metrics.update(metrics_simulated)

    # Strategy 2: Evaluate under heavily cloudy conditions (simulated holdout) using model predictions
    metrics_holdout = evaluate_reconstruction_heavily_cloudy(
        model_predicted_lst, observed_lst_clear, app_config
    )
    all_metrics.update(metrics_holdout)

    # Save metrics to a JSON file
    output_path = os.path.join(app_config.OUTPUT_DIR, app_config.EVALUATION_RESULTS_PATH)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=4, cls=NpEncoder) # Use custom encoder for numpy types
    print(f"Evaluation metrics saved to {output_path}")

    return all_metrics

class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    print("Evaluation module main execution - for testing.")

    class DummyConfig(config.Config):
        def __init__(self):
            super().__init__() # Inherit from base config
            self.RANDOM_SEED = 42
            self.EVAL_HOLDOUT_PERCENTAGE = 0.20
            self.OUTPUT_DIR = "./output_evaluation_test/"
            self.EVALUATION_RESULTS_PATH = "evaluation_results.json"
            self.LST_NODATA_VALUE = -9999.0 # For consistency, though functions use np.nan
            # Ensure parent directory for EVALUATION_RESULTS_PATH exists for dummy test
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    dummy_config = DummyConfig()
    np.random.seed(dummy_config.RANDOM_SEED)

    num_times, height, width = 10, 5, 4
    
    # Simulate observed LST (ground truth for clear pixels)
    # This should have NaNs where data is missing/cloudy
    dummy_observed_lst_raw = np.random.rand(num_times, height, width).astype(np.float32) * 20 + 280 # K
    # Introduce NaNs to simulate missing/cloudy data (approx 30% cloudy)
    cloud_locations = np.random.rand(num_times, height, width) < 0.3
    dummy_observed_lst = np.where(cloud_locations, np.nan, dummy_observed_lst_raw)

    # dummy_cloud_mask_stack = (np.random.rand(num_times, height, width) > 0.7).astype(np.uint8) # No longer used

    # Simulate model's direct predictions (these would be ATC+GP output)
    # For this test, let's make them slightly different from observed_lst even for clear areas
    dummy_model_predicted_lst = dummy_observed_lst_raw + (np.random.randn(num_times,height,width) * 0.5) # Simulate some noise/error
    # Where observed_lst was NaN (cloudy), model predictions would be purely model-driven
    nan_locations_obs = np.isnan(dummy_observed_lst)
    dummy_model_predicted_lst[nan_locations_obs] = (np.random.rand(*dummy_model_predicted_lst.shape)[nan_locations_obs].astype(np.float32) * 15 + 275)


    # Ensure some specific cases for testing:
    # Pixel (0,0,0) should be clear if not made NaN by random cloud_locations
    if np.isnan(dummy_observed_lst[0,0,0]): 
        dummy_observed_lst[0,0,0] = 290.0 # Make it clear
    dummy_model_predicted_lst[0,0,0] = 290.5 # Model prediction slightly off from clear observed
    
    # Pixel (0,0,1) should be cloudy, with a distinct reconstructed value
    dummy_observed_lst[0,0,1] = np.nan
    dummy_model_predicted_lst[0,0,1] = 285.0 # A model-predicted value for cloudy spot

    print("Dummy Observed LST (with NaNs for clouds) - Time 0:")
    print(dummy_observed_lst[0,:,:])
    print("Dummy Model Predicted LST - Time 0:")
    print(dummy_model_predicted_lst[0,:,:])

    all_eval_metrics = run_all_evaluations(
        dummy_model_predicted_lst,
        dummy_observed_lst, 
        dummy_config
    )

    print("\nAll evaluation metrics:")
    for k, v in all_eval_metrics.items():
        print(f"  {k}: {v}")

    # Example of how to verify saved JSON (optional)
    # with open(os.path.join(dummy_config.OUTPUT_DIR, dummy_config.EVALUATION_RESULTS_PATH), 'r') as f:
    #     loaded_metrics = json.load(f)
    # print("\nLoaded metrics from JSON:")
    # print(loaded_metrics)

    # Cleanup dummy output
    # import shutil
    # shutil.rmtree(dummy_config.OUTPUT_DIR)
    # print("Cleaned up dummy evaluation output directory.") 