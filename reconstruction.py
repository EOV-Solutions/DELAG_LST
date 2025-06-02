"""
Combines predictions from ATC and GP models and quantifies uncertainty.
"""
import numpy as np
import torch # For any torch specific operations if needed, though mostly numpy here
from scipy.stats import norm # For confidence intervals

import config # Assuming config.py is accessible

def combine_predictions(
    atc_predictions: np.ndarray, 
    atc_variance: np.ndarray, 
    gp_mean_residuals_map: np.ndarray, 
    gp_variance_residuals_map: np.ndarray,
    preprocessed_data: dict,
    app_config: 'config'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combines ATC and GP predictions to get reconstructed LST and total uncertainty.
    Uses np.nan in lst_observed (from preprocessed_data['lst_stack']) to identify clear/cloudy pixels.

    Args:
        atc_predictions (np.ndarray): Mean ATC predictions (time, height, width).
        atc_variance (np.ndarray): Variance from ATC ensemble (time, height, width).
        gp_mean_residuals_map (np.ndarray): Predicted mean of GP residuals (time, height, width).
        gp_variance_residuals_map (np.ndarray): Predicted variance of GP residuals (time, height, width).
        preprocessed_data (dict): Dictionary containing preprocessed data, expects 'lst_stack'.
        app_config: Configuration object.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - reconstructed_lst (np.ndarray): Final reconstructed LST (time, height, width).
            - total_variance (np.ndarray): Total variance (time, height, width).
            - ci_lower (np.ndarray): Lower 95% confidence interval (time, height, width).
            - ci_upper (np.ndarray): Upper 95% confidence interval (time, height, width).
    """
    print("Combining ATC and GP predictions...")
    lst_observed = preprocessed_data['lst_stack'] # (time, height, width)
    num_times, height, width = atc_predictions.shape

    # GP residuals are already (time, height, width) from gp_model.py
    # No longer need to expand/tile them.
    gp_mean_residuals_expanded = gp_mean_residuals_map
    gp_variance_residuals_expanded = gp_variance_residuals_map

    # Basic shape check to catch mismatches early
    if gp_mean_residuals_expanded.shape != (num_times, height, width) or \
       gp_variance_residuals_expanded.shape != (num_times, height, width):
        raise ValueError(f"Shape mismatch for GP residual maps. Expected {(num_times, height, width)}, \
                         got mean: {gp_mean_residuals_expanded.shape}, var: {gp_variance_residuals_expanded.shape}")

    reconstructed_lst = np.full_like(atc_predictions, np.nan)
    total_variance = np.full_like(atc_predictions, np.nan)

    # Iterate through time to apply logic based on cloud cover / observations
    for t in range(num_times):
        for r in range(height):
            for c in range(width):
                # Case 1: Clear-sky pixel - use observed LST, uncertainty from models (though paper implies reconstruction for all)
                # The paper seems to imply T_reconstructed = T_ATC + T_GP for partially observed days (meaning cloudy pixels)
                # and T_reconstructed = T_ATC for days with no observations (fully cloudy days/regions).
                # Let's refine this: if a pixel is clear, its "reconstruction" is the observation itself.
                # The primary goal is to fill gaps where LST is missing due to clouds.

                if not np.isnan(lst_observed[t, r, c]): # Pixel is clear if observed LST is not NaN
                # if False:
                    # Pixel is clear and has a valid observation
                    reconstructed_lst[t, r, c] = lst_observed[t, r, c]
                    # Uncertainty for clear pixels: can be just GP variance (spatial uncertainty)
                    # or a minimal value if we trust observations highly.
                    # The paper focuses on uncertainty of *reconstructed* values.
                    # For evaluation, we compare reconstructed to clear. So here, perhaps variance should be low.
                    # Let's use GP variance as within-day uncertainty even for clear pixels, ATC variance as cross-day.
                    total_variance[t, r, c] = atc_variance[t, r, c] + gp_variance_residuals_expanded[t, r, c]
                
                else: # Pixel is cloudy or missing LST data
                    # If atc_prediction itself is NaN (e.g., pixel had insufficient data for ATC model, 
                    # or underlying ERA5 was all NaN), then reconstructed LST is also NaN.
                    if np.isnan(atc_predictions[t, r, c]):
                        reconstructed_lst[t, r, c] = np.nan
                        total_variance[t, r, c] = np.nan
                        continue
                        
                    # Now, ATC prediction is valid. Check GP prediction.
                    current_gp_mean_residual = gp_mean_residuals_expanded[t, r, c]
                    current_gp_variance_residual = gp_variance_residuals_expanded[t, r, c]

                    if np.isnan(current_gp_mean_residual) or np.isnan(current_gp_variance_residual):
                        # GP prediction is NaN, use only ATC prediction and its variance.
                        reconstructed_lst[t, r, c] = atc_predictions[t, r, c]
                        # If ATC variance is also NaN (should ideally not happen if ATC pred is valid, but check)
                        if np.isnan(atc_variance[t, r, c]):
                            total_variance[t, r, c] = np.nan # Or a default high variance
                        else:
                            total_variance[t, r, c] = atc_variance[t, r, c]
                    else:
                        # Both ATC and GP predictions are valid, combine them.
                        reconstructed_lst[t, r, c] = atc_predictions[t, r, c] + current_gp_mean_residual
                        # Sum variances (assuming independence or as an approximation)
                        # Ensure atc_variance is not NaN before adding
                        if np.isnan(atc_variance[t, r, c]):
                             # This case implies ATC pred was fine but variance was NaN. 
                             # Total variance becomes GP variance or NaN if GP variance is also NaN (already handled by outer if)
                            total_variance[t, r, c] = current_gp_variance_residual 
                        else:
                            total_variance[t, r, c] = atc_variance[t, r, c] + current_gp_variance_residual

    # Ensure variances are non-negative (e.g. due to numerical precision)
    total_variance[total_variance < 0] = 0 
    total_std_dev = np.sqrt(total_variance)

    # Calculate 95% confidence intervals
    # Z-score for 95% CI is approx 1.96
    z_score = norm.ppf(0.975) # More precise: 1.95996...
    ci_lower = reconstructed_lst - z_score * total_std_dev
    ci_upper = reconstructed_lst + z_score * total_std_dev

    print("Finished combining predictions and quantifying uncertainty.")
    return reconstructed_lst, total_variance, ci_lower, ci_upper


if __name__ == '__main__':
    print("Reconstruction module main execution - for testing.")

    # Dummy config
    class DummyConfig(config.Config):
        def __init__(self):
            super().__init__() # Inherit from base config
            self.RANDOM_SEED = 42
            # Add LST_NODATA_VALUE for completeness, though not directly used by combine_predictions logic itself
            # as it expects NaNs in lst_observed_data from preprocessing.
            self.LST_NODATA_VALUE = -9999.0 
    
    dummy_config = DummyConfig()
    np.random.seed(dummy_config.RANDOM_SEED)

    # Dummy data
    num_times, height, width = 5, 3, 3
    atc_preds = np.random.rand(num_times, height, width).astype(np.float32) * 10 + 280 # K
    atc_var = np.random.rand(num_times, height, width).astype(np.float32) * 1.0 # K^2
    # GP maps should be (time, height, width) for the test to align with new expectation
    gp_mean_res_map = (np.random.rand(num_times, height, width).astype(np.float32) - 0.5) * 2 # K
    gp_var_res_map = np.random.rand(num_times, height, width).astype(np.float32) * 0.5 # K^2
    
    # cloud_mask = (np.random.rand(num_times, height, width) > 0.6).astype(np.uint8) # ~40% clear # No longer used
    # lst_observed_data directly contains NaNs for cloudy/missing pixels
    lst_observed_data = np.copy(atc_preds) + (np.random.randn(num_times, height, width) * 0.5) 
    # Simulate cloudy pixels by introducing NaNs. About 40% cloudy.
    cloud_locations = np.random.rand(num_times, height, width) <= 0.4 
    lst_observed_data[cloud_locations] = np.nan

    # Ensure some specific pixels for testing clear/cloudy logic
    # Pixel (0,0,0) should be clear if not made NaN by random cloud_locations
    if np.isnan(lst_observed_data[0,0,0]): # If it became cloudy by chance, make it clear
        lst_observed_data[0,0,0] = atc_preds[0,0,0] + (np.random.randn() * 0.5)
    # Pixel (0,1,1) should be cloudy
    lst_observed_data[0,1,1] = np.nan 

    # Make ATC preds/vars NaN for some pixels to simulate insufficient training data (e.g., pixel 0,0,1)
    atc_preds[:, 0, 1] = np.nan 
    atc_var[:, 0, 1] = np.nan

    dummy_preprocessed_data = {
        'lst_stack': lst_observed_data
        # Other keys not strictly needed by this simplified test of combine_predictions
    }

    print("Original LST (some NaN due to clouds):")
    print(lst_observed_data[0, :, :]) # Print first time slice
    # print("Cloud mask (0=clear, 1=cloud):") # No longer used
    # print(cloud_mask[0, :, :])
    print("ATC predictions (some NaN due to insufficient training data):")
    print(atc_preds[0, :, :])
    print("GP mean residuals map:")
    print(gp_mean_res_map)

    try:
        recon_lst, tot_var, ci_low, ci_up = combine_predictions(
            atc_preds, atc_var, 
            gp_mean_res_map, gp_var_res_map, 
            # cloud_mask, # Removed from arguments
            dummy_preprocessed_data, dummy_config
        )

        print(f"Reconstructed LST shape: {recon_lst.shape}")
        print("Reconstructed LST (first time slice):")
        print(recon_lst[0, :, :])
        print(f"Total variance shape: {tot_var.shape}")
        print("Total variance (first time slice):")
        print(tot_var[0, :, :])
        print(f"CI Lower shape: {ci_low.shape}")
        print("CI Lower (first time slice):")
        print(ci_low[0, :, :])
        print(f"CI Upper shape: {ci_up.shape}")
        print("CI Upper (first time slice):")
        print(ci_up[0, :, :])
        
        # Check if NaNs propagated correctly
        assert np.isnan(recon_lst[0,0,1]), "NaNs from ATC pred not propagated to reconstructed LST"
        assert np.isnan(tot_var[0,0,1]), "NaNs from ATC var not propagated to total variance"

        # Check a clear pixel - (0,0,0) is made to be clear if it wasn't already
        # Ensure it was actually clear after potential random assignment and fixing
        if not np.isnan(lst_observed_data[0,0,0]):
            assert np.isclose(recon_lst[0,0,0], lst_observed_data[0,0,0]), \
                f"Clear pixel LST {recon_lst[0,0,0]} not matching observed {lst_observed_data[0,0,0]}"
        else:
            print("Skipping clear pixel check as lst_observed_data[0,0,0] ended up NaN by chance and fix failed?")
        
        # Check a cloudy pixel (where ATC was not NaN) - (0,1,1) is made to be cloudy
        # And ensure ATC for this pixel is not NaN for the test
        atc_preds[0,1,1] = 285.0 # Ensure ATC pred is valid for this cloudy test pixel
        atc_var[0,1,1] = 0.5
        if np.isnan(lst_observed_data[0,1,1]) and not np.isnan(atc_preds[0,1,1]):
            expected_cloudy_val = atc_preds[0,1,1] + gp_mean_res_map[0,1,1] # Use t=0 for GP map
            assert np.isclose(recon_lst[0,1,1], expected_cloudy_val), \
                f"Cloudy pixel LST {recon_lst[0,1,1]} incorrect, expected {expected_cloudy_val}"
        else:
            print("Skipping cloudy pixel check due to unexpected NaNs in test data setup for (0,1,1).")

        print("Reconstruction test completed successfully.")

    except Exception as e:
        print(f"Error during reconstruction test: {e}")
        import traceback
        traceback.print_exc() 