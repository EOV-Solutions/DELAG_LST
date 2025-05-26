"""
Main script to run the DELAG LST reconstruction pipeline.
"""
import numpy as np
import torch
import os

# Import project modules
import config
import utils
import data_preprocessing
import atc_model
import gp_model
import reconstruction
import evaluation

def main():
    """Main pipeline execution function."""
    print("Starting DELAG LST Reconstruction Pipeline...")
    
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
        # Potentially set deterministic algorithms if desired, though might impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # 0. Create output directories (idempotent)
    utils.create_output_directories(config)

    # 1. Data Preprocessing
    print("\nStep 1: Data Preprocessing")
    try:
        preprocessed_data = data_preprocessing.preprocess_all_data(config)
        # Example: save a preprocessed stack if needed for inspection (optional)
        # utils.save_array_as_geotiff(preprocessed_data['lst_stack'][0], 
        #                               preprocessed_data['reference_grid_path'], 
        #                               os.path.join(config.OUTPUT_DIR, 'temp_first_lst_processed.tif'))
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return # Stop pipeline if preprocessing fails
    print("Data preprocessing completed.")

    # 2. Train ATC Model and Get Predictions/Variance
    print("\nStep 2: ATC Model Training and Prediction")
    try:
        atc_mean_predictions, atc_variance = atc_model.train_all_atc_models(
            preprocessed_data, config
        )
        # Save ATC outputs (optional, for debugging or intermediate results)
        # utils.save_array_as_geotiff(atc_mean_predictions, preprocessed_data['reference_grid_path'], 
        #                               os.path.join(config.OUTPUT_DIR, 'atc_mean_predictions.tif'))
        # utils.save_array_as_geotiff(atc_variance, preprocessed_data['reference_grid_path'], 
        #                               os.path.join(config.OUTPUT_DIR, 'atc_variance.tif'))
    except Exception as e:
        print(f"Error during ATC model training/prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    print("ATC model processing completed.")
    # --- START DIAGNOSTIC BLOCK FOR ATC ---
    if 'lst_stack' in preprocessed_data and atc_mean_predictions is not None:
        nan_in_lst_stack = np.isnan(preprocessed_data['lst_stack'])
        nan_in_atc_pred = np.isnan(atc_mean_predictions)
        total_nan_in_lst = np.sum(nan_in_lst_stack)
        total_nan_in_atc = np.sum(nan_in_atc_pred)
        print(f"  DIAGNOSTIC: Total NaNs in input LST stack: {total_nan_in_lst}")
        print(f"  DIAGNOSTIC: Total NaNs in ATC mean predictions: {total_nan_in_atc}")
        if total_nan_in_lst > 0 and total_nan_in_atc > 0:
            matching_nans_atc = np.sum(nan_in_lst_stack & nan_in_atc_pred)
            print(f"  DIAGNOSTIC: NaN locations in LST stack that are also NaN in ATC predictions: {matching_nans_atc}")
            if matching_nans_atc > 0 and matching_nans_atc == total_nan_in_atc and matching_nans_atc >= np.sum(nan_in_lst_stack): # Check if all NaNs in ATC are from LST
                 print("  WARNING: ATC predictions appear to carry over NaNs from the input LST stack. Gap-filling by ATC may not be effective.")
        elif total_nan_in_atc == 0 and total_nan_in_lst > 0:
            print("  INFO: ATC predictions do not contain NaNs, suggesting it might be performing gap-filling.")
    # --- END DIAGNOSTIC BLOCK FOR ATC ---

    # 3. Train GP Model for Residuals and Get Predictions/Variance
    print("\nStep 3: GP Model Training and Prediction for Residuals")
    try:
        gp_mean_residuals_map, gp_variance_residuals_map = gp_model.train_and_predict_all_gp_residuals(
            preprocessed_data, atc_mean_predictions, config
        )
        # Save GP outputs (optional)
        # utils.save_array_as_geotiff(gp_mean_residuals_map, preprocessed_data['reference_grid_path'], 
        #                               os.path.join(config.OUTPUT_DIR, 'gp_mean_residuals_map.tif'))
        # utils.save_array_as_geotiff(gp_variance_residuals_map, preprocessed_data['reference_grid_path'], 
        #                               os.path.join(config.OUTPUT_DIR, 'gp_variance_residuals_map.tif'))
    except Exception as e:
        print(f"Error during GP model training/prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    print("GP model processing for residuals completed.")

    # 4. Combine Predictions and Quantify Uncertainty
    print("\nStep 4: Combining Predictions and Quantifying Uncertainty")
    try:
        reconstructed_lst, total_variance, ci_lower, ci_upper = reconstruction.combine_predictions(
            atc_predictions=atc_mean_predictions,
            atc_variance=atc_variance,
            gp_mean_residuals_map=gp_mean_residuals_map,
            gp_variance_residuals_map=gp_variance_residuals_map,
            preprocessed_data=preprocessed_data,
            app_config=config
        )
    except Exception as e:
        print(f"Error during final reconstruction and uncertainty quantification: {e}")
        import traceback
        traceback.print_exc()
        return
    print("Final reconstruction and uncertainty quantification completed.")
    # --- START DIAGNOSTIC BLOCK FOR RECONSTRUCTION ---
    if 'lst_stack' in preprocessed_data and reconstructed_lst is not None:
        nan_in_lst_stack = np.isnan(preprocessed_data['lst_stack'])
        nan_in_reconstructed = np.isnan(reconstructed_lst)
        total_nan_in_lst = np.sum(nan_in_lst_stack)
        total_nan_in_reconstructed = np.sum(nan_in_reconstructed)
        print(f"  DIAGNOSTIC: Total NaNs in input LST stack: {total_nan_in_lst}")
        print(f"  DIAGNOSTIC: Total NaNs in final reconstructed LST: {total_nan_in_reconstructed}")
        if total_nan_in_lst > 0 and total_nan_in_reconstructed > 0:
            matching_nans_reconstructed = np.sum(nan_in_lst_stack & nan_in_reconstructed)
            print(f"  DIAGNOSTIC: NaN locations in LST stack that are also NaN in reconstructed LST: {matching_nans_reconstructed}")
            if matching_nans_reconstructed > 0 and matching_nans_reconstructed == total_nan_in_reconstructed and matching_nans_reconstructed >= np.sum(nan_in_lst_stack):
                print("  WARNING: Final reconstructed LST appears to carry over NaNs from the input LST stack. Overall gap-filling may not be effective.")
        elif total_nan_in_reconstructed == 0 and total_nan_in_lst > 0:
             print("  INFO: Final reconstructed LST does not contain NaNs, suggesting pipeline might be performing gap-filling.")
    # --- END DIAGNOSTIC BLOCK FOR RECONSTRUCTION ---

    # 5. Save Reconstructed LST and Uncertainty Products
    print("\nStep 5: Saving Outputs")
    try:
        # Save reconstructed LST for each time step
        num_times = reconstructed_lst.shape[0]
        dates = preprocessed_data['common_dates']
        for t in range(num_times):
            date_str = dates[t].strftime('%Y%m%d')
            
            # Reconstructed LST
            recon_filename = os.path.join(config.RECONSTRUCTED_LST_PATH, f"LST_RECON_{date_str}.tif")
            utils.save_array_as_geotiff(
                data_array=reconstructed_lst[t, :, :],
                reference_geotiff_path=preprocessed_data['reference_grid_path'],
                output_path=recon_filename,
                nodata_value=np.nan # Or a specific nodata value if preferred
            )
            
            # Total Variance
            var_filename = os.path.join(config.UNCERTAINTY_MAPS_PATH, f"LST_TOTAL_VARIANCE_{date_str}.tif")
            utils.save_array_as_geotiff(
                data_array=total_variance[t, :, :],
                reference_geotiff_path=preprocessed_data['reference_grid_path'],
                output_path=var_filename,
                nodata_value=np.nan
            )
            
            # Confidence Intervals (Optional - can be large)
            # ci_low_filename = os.path.join(config.UNCERTAINTY_MAPS_PATH, f"LST_CI_LOWER_{date_str}.tif")
            # utils.save_array_as_geotiff(ci_lower[t,:,:], preprocessed_data['reference_grid_path'], ci_low_filename, nodata_value=np.nan)
            # ci_up_filename = os.path.join(config.UNCERTAINTY_MAPS_PATH, f"LST_CI_UPPER_{date_str}.tif")
            # utils.save_array_as_geotiff(ci_upper[t,:,:], preprocessed_data['reference_grid_path'], ci_up_filename, nodata_value=np.nan)
        print(f"Saved reconstructed LST and variance maps to {config.RECONSTRUCTED_LST_PATH} and {config.UNCERTAINTY_MAPS_PATH}")

    except Exception as e:
        print(f"Error during saving outputs: {e}")
        import traceback
        traceback.print_exc()
        # Continue to evaluation even if saving fails for some reason

    # 6. Evaluate Model Performance
    print("\nStep 6: Model Evaluation")
    try:
        all_eval_metrics = evaluation.run_all_evaluations(
            reconstructed_lst=reconstructed_lst,
            observed_lst_clear=preprocessed_data['lst_stack'], # Original LST with NaNs for clouds
            app_config=config
        )
        print("\nFinal Evaluation Metrics:")
        for k, v in all_eval_metrics.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()

    # Visualize daily comparison stacks (Observed LST, Reconstructed LST, S2 RGB)
    if reconstructed_lst.shape[0] > 0 and \
       preprocessed_data.get('lst_stack') is not None and \
       preprocessed_data.get('s2_reflectance_stack') is not None and \
       preprocessed_data.get('common_dates'):
        print("\nVisualizing daily comparison stacks (Observed LST vs Reconstructed LST)...")
        try:
            # Assuming S2 bands are [B2, B3, B4, B8], so RGB indices are (B4=2, B3=1, B2=0)
            s2_rgb_indices_param = getattr(config, 'S2_RGB_INDICES', (2, 1, 0)) 
            max_days_plot_param = getattr(config, 'MAX_DAYS_FOR_DAILY_VISUALIZATION_PLOT', 7)

            utils.visualize_daily_stacks_comparison(
                lst_observed_stack=preprocessed_data['lst_stack'],
                reconstructed_lst_stack=reconstructed_lst,
                s2_reflectance_stack=preprocessed_data['s2_reflectance_stack'],
                common_dates=preprocessed_data['common_dates'],
                output_base_dir=config.OUTPUT_DIR, 
                roi_name=preprocessed_data.get('roi_name', 'UnknownROI'),
                s2_rgb_indices=s2_rgb_indices_param,
                max_days_to_plot=max_days_plot_param
            )
        except Exception as e:
            print(f"Error during daily comparison visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping daily comparison visualization as not all required data stacks are available.")
    
    print("\nDELAG LST Reconstruction Pipeline Completed.")

if __name__ == '__main__':
    # Before running, ensure that:
    # 1. All necessary libraries (numpy, pandas, torch, gpytorch, rasterio, scikit-learn, tqdm) are installed.
    # 2. Paths in `config.py` (LANDSAT_LST_PATH, ERA5_SKIN_TEMP_PATH, etc.) are correctly set to your data locations.
    # 3. Your input data files follow a naming convention that can be discovered by the `glob` patterns in `data_preprocessing.py`,
    #    or modify those patterns to match your data.
    #    Example: Landsat LST files like LST_YYYYMMDD.tif, Cloud Masks like MASK_YYYYMMDD.tif, ERA5 like ERA5_YYYYMMDD.tif,
    #             Sentinel-2 annual means like S2_ANNUAL_MEAN_RED.tif, etc.
    # 4. You have a GPU if `config.DEVICE` is set to "cuda", otherwise change to "cpu".

    # This main script will not run the dummy data generation from individual modules.
    # It expects real data paths to be configured.
    main() 