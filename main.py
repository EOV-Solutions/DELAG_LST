"""
Main script to run the DELAG LST reconstruction pipeline.
"""
import numpy as np
import torch
import os
import json

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

    # --- START DIAGNOSTIC BLOCK FOR PREPROCESSED DATA ---
    if 'era5_stack' in preprocessed_data:
        era5_nan_count = np.isnan(preprocessed_data['era5_stack']).sum()
        era5_total_count = preprocessed_data['era5_stack'].size
        era5_nan_percentage = (era5_nan_count / era5_total_count) * 100 if era5_total_count > 0 else 0
        print(f"  DIAGNOSTIC (PREPROCESSING): ERA5 stack has {era5_nan_count} NaNs out of {era5_total_count} values ({era5_nan_percentage:.2f}%).")
        if era5_nan_percentage > 0:
            print("    INFO: These NaNs in ERA5 will likely propagate to ATC predictions if the 'b' coefficient is non-zero.")
    if 's2_reflectance_stack' in preprocessed_data:
        s2_nan_count = np.isnan(preprocessed_data['s2_reflectance_stack']).sum()
        s2_total_count = preprocessed_data['s2_reflectance_stack'].size
        s2_nan_percentage = (s2_nan_count / s2_total_count) * 100 if s2_total_count > 0 else 0
        print(f"  DIAGNOSTIC (PREPROCESSING): S2 reflectance stack has {s2_nan_count} NaNs out of {s2_total_count} values ({s2_nan_percentage:.2f}%).")
    # --- END DIAGNOSTIC BLOCK FOR PREPROCESSED DATA ---

    # 2. Train ATC Model and Get Predictions/Variance
    print("\nStep 2: ATC Model Training and Prediction")
    try:
        # Phase 2.1: Train ATC models and save snapshots
        print("  Phase 2.1: Training ATC models and collecting snapshots/losses...")
        # Now expects two return values: snapshots and interval loss maps
        all_pixel_snapshots, interval_loss_maps_array = atc_model.train_and_collect_all_atc_snapshots(
            preprocessed_data, config
        )

        # --- Plot Mean ATC Training Loss ---
        if interval_loss_maps_array is not None and interval_loss_maps_array.size > 0:
            mean_losses_over_intervals = np.nanmean(interval_loss_maps_array, axis=(1, 2)) # Mean over H, W for each interval
            num_intervals = interval_loss_maps_array.shape[0]
            loss_logging_interval = getattr(config, 'ATC_LOSS_LOGGING_INTERVAL', 100)
            epoch_ticks = [(i + 1) * loss_logging_interval for i in range(num_intervals)]
            
            # Ensure epoch_ticks doesn't exceed ATC_EPOCHS if it was not a perfect multiple
            if epoch_ticks and epoch_ticks[-1] > config.ATC_EPOCHS:
                # Adjust the last tick or handle how epochs are displayed if desired.
                # For simplicity, we can cap it or let it be if it represents the *end* of an interval beyond total epochs.
                # The plot function can clarify this in its x-label.
                pass 

            utils.plot_mean_atc_loss_over_intervals(
                mean_interval_losses=list(mean_losses_over_intervals),
                epoch_intervals_x_axis=epoch_ticks,
                output_dir=config.OUTPUT_DIR, # Main output dir, plot_mean_atc_loss will put it in a 'viz' subdir
                roi_name=preprocessed_data.get('roi_name', 'UnknownROI'),
                loss_logging_interval=loss_logging_interval
            )
        else:
            print("  Skipping ATC mean loss plot as interval_loss_maps_array is None or empty.")
        # --- End Plot Mean ATC Training Loss ---

        # Define path for saving snapshots
        # Ensure MODEL_WEIGHTS_PATH is defined in your config and the directory exists
        # utils.create_output_directories should ideally create this too if it's under OUTPUT_DIR
        snapshots_filename = f"atc_snapshots_{preprocessed_data.get('roi_name', 'all')}.npz"
        snapshots_filepath = os.path.join(config.MODEL_WEIGHTS_PATH, snapshots_filename)
        
        print(f"  Saving ATC snapshots to {snapshots_filepath}...")
        # Get image dimensions from one of the stacks, e.g., LST stack
        _, height, width = preprocessed_data['lst_stack'].shape 
        atc_model.save_atc_snapshots(
            all_pixel_snapshots, 
            snapshots_filepath,
            image_height=height,
            image_width=width,
            num_snapshots_expected=config.ATC_ENSEMBLE_SNAPSHOTS
        )
        print("  ATC snapshots saved.")

        # Phase 2.2: Load ATC snapshots and predict
        print("  Phase 2.2: Loading ATC snapshots and performing prediction...")
        loaded_snapshots_data = atc_model.load_atc_snapshots(snapshots_filepath)
        
        # For prediction, we use the full timeline DOY and ERA5 from preprocessed_data
        # as the target prediction timeline.
        doy_for_prediction = preprocessed_data["doy_stack"]
        era5_for_prediction = preprocessed_data["era5_stack"]

        atc_mean_predictions, atc_variance = atc_model.predict_atc_from_loaded_snapshots(
            loaded_snapshots_data,
            doy_for_prediction_numpy=doy_for_prediction,
            era5_for_prediction_numpy=era5_for_prediction,
            app_config=config
        )
        
        # Optional: save ATC outputs (for debugging or intermediate results)
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
    print("\nStep 3: GP Model Training for Residuals and Saving Model")
    try:
        # Phase 3.1: Train GP model and save it (also saves interval losses internally)
        gp_model.train_and_save_gp_model(
            preprocessed_data, atc_mean_predictions, config
        )
        print("GP model training and saving completed.")

        # --- Plot Mean GP Training Loss ---
        gp_model_filepath = os.path.join(config.MODEL_WEIGHTS_PATH, config.GP_MODEL_WEIGHT_FILENAME)
        gp_interval_losses = gp_model.load_gp_interval_losses(gp_model_filepath)
        
        if gp_interval_losses:
            num_gp_intervals = len(gp_interval_losses)
            gp_loss_logging_interval = getattr(config, 'GP_LOSS_LOGGING_INTERVAL', 10)
            # Calculate epoch ticks based on total epochs for GP = GP_EPOCHS_INITIAL + GP_EPOCHS_FINAL
            total_gp_epochs = config.GP_EPOCHS_INITIAL + config.GP_EPOCHS_FINAL
            gp_epoch_ticks = [(i + 1) * gp_loss_logging_interval for i in range(num_gp_intervals)]
            # Cap ticks at total_gp_epochs if needed, though interval logic should align
            if gp_epoch_ticks and gp_epoch_ticks[-1] > total_gp_epochs and num_gp_intervals * gp_loss_logging_interval > total_gp_epochs:
                 # This can happen if the last interval is partial. The plot x-label clarifies.
                 pass 

            utils.plot_mean_gp_loss_over_intervals(
                mean_interval_losses=gp_interval_losses, # Already a list of means
                epoch_intervals_x_axis=gp_epoch_ticks,
                output_dir=config.OUTPUT_DIR,
                roi_name=preprocessed_data.get('roi_name', 'UnknownROI'),
                loss_logging_interval=gp_loss_logging_interval
            )
        else:
            print("  Skipping GP mean loss plot as interval losses were not found or loaded.")
        # --- End Plot Mean GP Training Loss ---

        # Phase 3.2: Load GP model and predict residuals
        print("\nStep 3.2: Loading GP Model and Predicting Residuals")
        gp_mean_residuals_map, gp_variance_residuals_map = gp_model.load_and_predict_gp_residuals(
            preprocessed_data, atc_mean_predictions, config
        )

        # Optional: Save GP outputs (for debugging or intermediate results)
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
    # This step produces the 'reconstructed_lst' that is used for saving and visualization,
    # which correctly incorporates observed data for clear pixels.
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

    # Create the 'model_predictions_for_eval' for fair evaluation
    # This represents the model's raw output before merging with observed data.
    print("\nPreparing model's raw predictions for evaluation purposes...")
    # Start with ATC predictions
    model_predictions_for_eval = np.copy(atc_mean_predictions)
    
    # Add GP residuals where they are valid
    # If gp_mean_residuals_map is None or all NaN (e.g. GP failed/skipped), this won't add anything or add NaNs
    if gp_mean_residuals_map is not None:
        # Ensure gp_mean_residuals_map is not all NaNs before attempting to add
        if not np.all(np.isnan(gp_mean_residuals_map)):
            # Where gp_mean_residuals_map is NaN, adding it will result in NaN, which is fine.
            # Where atc_mean_predictions is NaN, result will be NaN.
            model_predictions_for_eval = atc_mean_predictions + gp_mean_residuals_map
        else:
            print("  GP mean residuals map is all NaN; using only ATC predictions for model evaluation output.")
            # model_predictions_for_eval already holds atc_mean_predictions
    else:
        print("  GP mean residuals map is None; using only ATC predictions for model evaluation output.")
        # model_predictions_for_eval already holds atc_mean_predictions

    # 6. Evaluate Model Performance
    print("\nStep 6: Model Evaluation")
    try:
        all_eval_metrics = evaluation.run_all_evaluations(
            model_predicted_lst=model_predictions_for_eval, # Use the model's raw predictions
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
        print("\nVisualizing daily comparison stacks (Observed LST vs Predicted LST vs Reconstructed LST)...")
        try:
            # Assuming S2 bands are [B2, B3, B4, B8], so RGB indices are (B4=2, B3=1, B2=0)
            s2_rgb_indices_param = getattr(config, 'S2_RGB_INDICES', (2, 1, 0)) 
            max_days_plot_param = getattr(config, 'MAX_DAYS_FOR_DAILY_VISUALIZATION_PLOT', 10)

            utils.visualize_daily_stacks_comparison(
                lst_observed_stack=preprocessed_data['lst_stack'],
                model_predicted_lst_stack=model_predictions_for_eval, # ADDED: Pass model's direct predictions
                reconstructed_lst_stack=reconstructed_lst,
                era5_stack=preprocessed_data['era5_stack'], # ADDED: Pass ERA5 stack
                s2_reflectance_stack=preprocessed_data['s2_reflectance_stack'],
                ndvi_stack=preprocessed_data.get('ndvi_stack'), # Pass NDVI stack, could be None
                common_dates=preprocessed_data['common_dates'],
                output_base_dir=config.OUTPUT_DIR, 
                roi_name=preprocessed_data.get('roi_name', 'UnknownROI'),
                app_config=config, # Pass the config object
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

    # Save the configuration used for this run
    print("\nStep 7: Saving Run Configuration")
    try:
        config_dict = {key: getattr(config, key) for key in dir(config) if not key.startswith('__') and not callable(getattr(config, key))}
        config_filename = os.path.join(config.OUTPUT_DIR, 'run_config.json')
        with open(config_filename, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str) # Use default=str to handle non-serializable types like Path objects if any
        print(f"Run configuration saved to {config_filename}")
    except Exception as e:
        print(f"Error saving run configuration: {e}")
        import traceback
        traceback.print_exc()

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