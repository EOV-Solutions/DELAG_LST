"""
Hyperparameter tuning for the DELAG project.

This script performs hyperparameter optimization by:
1. Separating data into train/test based on dates (train: 2015-2024, test: 2024-2025)
2. Using SPATIAL_TRAINING_SAMPLE_PERCENTAGE for masking to speed up training
3. Training both ATC and GP models for each hyperparameter set
4. Evaluating on test set using methods from evaluation.py
5. Saving results as JSON mapping hyperparameter sets to results
"""

import numpy as np
import pandas as pd
import json
import os
import itertools
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
from tqdm import tqdm
import torch

# Import project modules
import config
import data_preprocessing
import atc_model
import gp_model
import reconstruction
import evaluation
import utils


class HyperparameterTuner:
    """
    Main hyperparameter tuning class for DELAG models.
    """
    
    def __init__(self, base_config):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            base_config: Base configuration object
        """
        self.base_config = base_config
        self.results = {}
        
        # Set up output directories
        self.tuning_output_dir = os.path.join(
            base_config.OUTPUT_DIR_BASE, 
            f"hyperparameter_tuning_{base_config.ROI_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.tuning_output_dir, exist_ok=True)
        
        print(f"Hyperparameter tuning output directory: {self.tuning_output_dir}")
    
    def split_data_by_date(self, preprocessed_data: Dict) -> Tuple[Dict, Dict]:
        """
        Split data into train (2015-2024) and test (2024-2025) sets based on dates.
        
        Args:
            preprocessed_data: Dictionary containing preprocessed data
            
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        print("Splitting data by date...")
        
        common_dates = preprocessed_data['common_dates']
        
        # Define date boundaries
        train_start = pd.to_datetime("2015-01-01")
        train_end = pd.to_datetime("2024-01-01")
        test_start = pd.to_datetime("2024-01-01")
        test_end = pd.to_datetime("2025-01-01")
        
        # Find indices for train and test sets
        train_indices = []
        test_indices = []
        
        for i, date in enumerate(common_dates):
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            if train_start <= date <= train_end:
                train_indices.append(i)
            if test_start <= date <= test_end:
                test_indices.append(i)
        
        print(f"Train dates: {len(train_indices)} ({train_start} to {train_end})")
        print(f"Test dates: {len(test_indices)} ({test_start} to {test_end})")
        
        if len(train_indices) == 0:
            raise ValueError("No training data found in the specified date range")
        if len(test_indices) == 0:
            raise ValueError("No test data found in the specified date range")
        
        # Create train data subset
        train_data = {}
        for key, value in preprocessed_data.items():
            if key == 'common_dates':
                train_data[key] = [common_dates[i] for i in train_indices]
            elif key in ['lst_stack', 'era5_stack', 's2_reflectance_stack', 'doy_stack']:
                if hasattr(value, 'shape') and len(value.shape) >= 1:
                    train_data[key] = value[train_indices]
                else:
                    train_data[key] = value
            elif key == 'ndvi_stack' and value is not None:
                train_data[key] = value[train_indices]
            else:
                train_data[key] = value
        
        # Create test data subset
        test_data = {}
        for key, value in preprocessed_data.items():
            if key == 'common_dates':
                test_data[key] = [common_dates[i] for i in test_indices]
            elif key in ['lst_stack', 'era5_stack', 's2_reflectance_stack', 'doy_stack']:
                if hasattr(value, 'shape') and len(value.shape) >= 1:
                    test_data[key] = value[test_indices]
                else:
                    test_data[key] = value
            elif key == 'ndvi_stack' and value is not None:
                test_data[key] = value[test_indices]
            else:
                test_data[key] = value
        
        return train_data, test_data
    
    def create_hyperparameter_sets(self) -> List[Dict]:
        """
        Create different hyperparameter combinations to test.
        
        Returns:
            List of hyperparameter dictionaries
        """
        print("Creating hyperparameter sets...")
        
        # Define hyperparameter ranges
        atc_params = {
            'ATC_LEARNING_RATE': [0.01, 0.05, 0.1, 0.2],
            'ATC_EPOCHS': [5000, 10000, 15000, 20000],
            'ATC_WEIGHT_DECAY': [1e-4, 1e-3, 1e-2],
            'ATC_ENSEMBLE_SNAPSHOTS': [100, 200, 300],
        }
        
        gp_params = {
            'GP_LEARNING_RATE_INITIAL': [0.05, 0.1, 0.2],
            'GP_EPOCHS_INITIAL': [30, 50, 70],
            'GP_EPOCHS_FINAL': [10, 20, 30],
            'GP_NUM_INDUCING_POINTS': [512, 1024, 2048],
            'GP_MINI_BATCH_SIZE': [512, 1024, 2048],
        }
        
        # Create a subset of combinations for feasible tuning
        # Full grid search would be too expensive, so we sample combinations
        hyperparameter_sets = []
        
        # Strategy 1: Test ATC parameters with default GP parameters
        atc_combinations = [
            {'ATC_LEARNING_RATE': 0.05, 'ATC_EPOCHS': 10000, 'ATC_WEIGHT_DECAY': 1e-3, 'ATC_ENSEMBLE_SNAPSHOTS': 200},
            {'ATC_LEARNING_RATE': 0.1, 'ATC_EPOCHS': 15000, 'ATC_WEIGHT_DECAY': 1e-4, 'ATC_ENSEMBLE_SNAPSHOTS': 300},
            {'ATC_LEARNING_RATE': 0.2, 'ATC_EPOCHS': 5000, 'ATC_WEIGHT_DECAY': 1e-2, 'ATC_ENSEMBLE_SNAPSHOTS': 100},
            {'ATC_LEARNING_RATE': 0.01, 'ATC_EPOCHS': 20000, 'ATC_WEIGHT_DECAY': 1e-3, 'ATC_ENSEMBLE_SNAPSHOTS': 200},
        ]
        
        for atc_params_combo in atc_combinations:
            hyperparameter_sets.append({
                **atc_params_combo,
                'GP_LEARNING_RATE_INITIAL': 0.1,
                'GP_EPOCHS_INITIAL': 50,
                'GP_EPOCHS_FINAL': 20,
                'GP_NUM_INDUCING_POINTS': 1024,
                'GP_MINI_BATCH_SIZE': 1024,
                'set_type': 'atc_focused'
            })
        
        # Strategy 2: Test GP parameters with default ATC parameters
        gp_combinations = [
            {'GP_LEARNING_RATE_INITIAL': 0.05, 'GP_EPOCHS_INITIAL': 30, 'GP_EPOCHS_FINAL': 10, 
             'GP_NUM_INDUCING_POINTS': 512, 'GP_MINI_BATCH_SIZE': 1024},
            {'GP_LEARNING_RATE_INITIAL': 0.1, 'GP_EPOCHS_INITIAL': 70, 'GP_EPOCHS_FINAL': 30, 
             'GP_NUM_INDUCING_POINTS': 2048, 'GP_MINI_BATCH_SIZE': 2048},
            {'GP_LEARNING_RATE_INITIAL': 0.2, 'GP_EPOCHS_INITIAL': 50, 'GP_EPOCHS_FINAL': 20, 
             'GP_NUM_INDUCING_POINTS': 1024, 'GP_MINI_BATCH_SIZE': 512},
        ]
        
        for gp_params_combo in gp_combinations:
            hyperparameter_sets.append({
                'ATC_LEARNING_RATE': 0.1,
                'ATC_EPOCHS': 15000,
                'ATC_WEIGHT_DECAY': 1e-3,
                'ATC_ENSEMBLE_SNAPSHOTS': 200,
                **gp_params_combo,
                'set_type': 'gp_focused'
            })
        
        # Strategy 3: Test some joint optimizations
        joint_combinations = [
            {
                'ATC_LEARNING_RATE': 0.1, 'ATC_EPOCHS': 10000, 'ATC_WEIGHT_DECAY': 1e-3, 'ATC_ENSEMBLE_SNAPSHOTS': 200,
                'GP_LEARNING_RATE_INITIAL': 0.1, 'GP_EPOCHS_INITIAL': 50, 'GP_EPOCHS_FINAL': 20,
                'GP_NUM_INDUCING_POINTS': 1024, 'GP_MINI_BATCH_SIZE': 1024,
                'set_type': 'joint_default'
            },
            {
                'ATC_LEARNING_RATE': 0.05, 'ATC_EPOCHS': 15000, 'ATC_WEIGHT_DECAY': 1e-4, 'ATC_ENSEMBLE_SNAPSHOTS': 300,
                'GP_LEARNING_RATE_INITIAL': 0.05, 'GP_EPOCHS_INITIAL': 70, 'GP_EPOCHS_FINAL': 30,
                'GP_NUM_INDUCING_POINTS': 2048, 'GP_MINI_BATCH_SIZE': 1024,
                'set_type': 'joint_conservative'
            },
            {
                'ATC_LEARNING_RATE': 0.2, 'ATC_EPOCHS': 5000, 'ATC_WEIGHT_DECAY': 1e-2, 'ATC_ENSEMBLE_SNAPSHOTS': 100,
                'GP_LEARNING_RATE_INITIAL': 0.2, 'GP_EPOCHS_INITIAL': 30, 'GP_EPOCHS_FINAL': 10,
                'GP_NUM_INDUCING_POINTS': 512, 'GP_MINI_BATCH_SIZE': 512,
                'set_type': 'joint_aggressive'
            },
        ]
        
        hyperparameter_sets.extend(joint_combinations)
        
        # Add set IDs for tracking
        for i, param_set in enumerate(hyperparameter_sets):
            param_set['set_id'] = f"set_{i+1:03d}"
        
        print(f"Created {len(hyperparameter_sets)} hyperparameter sets")
        return hyperparameter_sets
    
    def create_config_from_hyperparams(self, hyperparams: Dict):
        """
        Create a config object with specific hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameters
            
        Returns:
            Modified config object
        """
        # Create a new config class that inherits from the base config
        class TuningConfig:
            pass
        
        tuning_config = TuningConfig()
        
        # Copy all attributes from base config, filtering out non-copyable ones
        for attr_name in dir(self.base_config):
            if not attr_name.startswith('_'):  # Skip private attributes
                try:
                    attr_value = getattr(self.base_config, attr_name)
                    # Skip modules, functions, classes, and other complex objects
                    if not callable(attr_value) and not hasattr(attr_value, '__module__'):
                        setattr(tuning_config, attr_name, attr_value)
                except (TypeError, AttributeError):
                    # Skip attributes that can't be accessed or set
                    continue
        
        # Update with hyperparameters
        for key, value in hyperparams.items():
            if key not in ['set_id', 'set_type']:
                setattr(tuning_config, key, value)
        
        # Update dependent parameters
        if hasattr(tuning_config, 'ATC_ENSEMBLE_SNAPSHOTS') and hasattr(tuning_config, 'ATC_SNAPSHOT_INTERVAL'):
            tuning_config.ATC_ENSEMBLE_START_EPOCH = max(0, 
                tuning_config.ATC_EPOCHS - (tuning_config.ATC_ENSEMBLE_SNAPSHOTS * tuning_config.ATC_SNAPSHOT_INTERVAL))
        
        # Create unique output paths for this hyperparameter set
        set_id = hyperparams.get('set_id', 'unknown')
        tuning_config.OUTPUT_DIR = os.path.join(self.tuning_output_dir, set_id)
        tuning_config.MODEL_WEIGHTS_PATH = os.path.join(tuning_config.OUTPUT_DIR, "model_weights")
        os.makedirs(tuning_config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(tuning_config.MODEL_WEIGHTS_PATH, exist_ok=True)
        
        return tuning_config
    
    def train_models(self, train_data: Dict, tuning_config) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train both ATC and GP models with given configuration.
        
        Args:
            train_data: Training data dictionary
            tuning_config: Configuration with hyperparameters
            
        Returns:
            Tuple of (atc_predictions, gp_predictions) for the training data
        """
        print(f"Training models for set {tuning_config.OUTPUT_DIR}...")
        
        try:
            # Train ATC model
            print("  Training ATC model...")
            all_pixel_snapshots, interval_loss_maps = atc_model.train_and_collect_all_atc_snapshots(
                train_data, tuning_config
            )
            
            # Save ATC snapshots
            snapshots_filename = f"atc_snapshots_{train_data.get('roi_name', 'train')}.npz"
            snapshots_filepath = os.path.join(tuning_config.MODEL_WEIGHTS_PATH, snapshots_filename)
            
            _, height, width = train_data['lst_stack'].shape
            atc_model.save_atc_snapshots(
                all_pixel_snapshots, 
                snapshots_filepath,
                image_height=height,
                image_width=width,
                num_snapshots_expected=tuning_config.ATC_ENSEMBLE_SNAPSHOTS
            )
            
            # Load ATC snapshots and predict on training data
            loaded_snapshots_data = atc_model.load_atc_snapshots(snapshots_filepath)
            atc_mean_predictions, atc_variance = atc_model.predict_atc_from_loaded_snapshots(
                loaded_snapshots_data,
                doy_for_prediction_numpy=train_data["doy_stack"],
                era5_for_prediction_numpy=train_data["era5_stack"],
                app_config=tuning_config
            )
            
            # Train GP model
            print("  Training GP model...")
            gp_model.train_and_save_gp_model(
                train_data, atc_mean_predictions, tuning_config
            )
            
            # Load GP model and predict residuals on training data
            gp_mean_residuals_map, gp_variance_residuals_map = gp_model.load_and_predict_gp_residuals(
                train_data, atc_mean_predictions, tuning_config
            )
            
            # Return pure model predictions (ATC + GP residuals) for evaluation
            pure_model_predictions_train = atc_mean_predictions + gp_mean_residuals_map
            
            return pure_model_predictions_train, atc_mean_predictions
            
        except Exception as e:
            print(f"  Error training models: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def predict_on_test(self, test_data: Dict, tuning_config) -> np.ndarray:
        """
        Make predictions on test data using trained models.
        
        Args:
            test_data: Test data dictionary
            tuning_config: Configuration with hyperparameters
            
        Returns:
            Pure model predictions (ATC + GP residuals) on test data
        """
        print(f"Making predictions on test data...")
        
        try:
            # Load ATC model and predict
            snapshots_filename = f"atc_snapshots_{test_data.get('roi_name', 'train')}.npz"
            snapshots_filepath = os.path.join(tuning_config.MODEL_WEIGHTS_PATH, snapshots_filename)
            
            loaded_snapshots_data = atc_model.load_atc_snapshots(snapshots_filepath)
            atc_mean_predictions, atc_variance = atc_model.predict_atc_from_loaded_snapshots(
                loaded_snapshots_data,
                doy_for_prediction_numpy=test_data["doy_stack"],
                era5_for_prediction_numpy=test_data["era5_stack"],
                app_config=tuning_config
            )
            
            # Load GP model and predict residuals
            gp_mean_residuals_map, gp_variance_residuals_map = gp_model.load_and_predict_gp_residuals(
                test_data, atc_mean_predictions, tuning_config
            )
            
            # Return pure model predictions (ATC + GP residuals) WITHOUT observed data substitution
            # This is what we want to evaluate against observed data
            pure_model_predictions = atc_mean_predictions + gp_mean_residuals_map
            
            return pure_model_predictions
            
        except Exception as e:
            print(f"  Error making predictions: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def evaluate_predictions(self, predictions: np.ndarray, test_data: Dict, tuning_config) -> Dict:
        """
        Evaluate predictions using methods from evaluation.py.
        
        Args:
            predictions: Model predictions
            test_data: Test data dictionary
            tuning_config: Configuration object
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("  Evaluating predictions...")
        
        try:
            # Use evaluation methods from evaluation.py
            evaluation_metrics = evaluation.run_all_evaluations(
                model_predicted_lst=predictions,
                observed_lst_clear=test_data['lst_stack'],
                app_config=tuning_config
            )
            
            return evaluation_metrics
            
        except Exception as e:
            print(f"  Error evaluating predictions: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_hyperparameter_tuning(self) -> Dict:
        """
        Run the complete hyperparameter tuning process.
        
        Returns:
            Dictionary mapping hyperparameter sets to results
        """
        print("Starting hyperparameter tuning...")
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        preprocessed_data = data_preprocessing.preprocess_all_data(self.base_config)
        
        # Split data by date
        train_data, test_data = self.split_data_by_date(preprocessed_data)
        
        # Create hyperparameter sets
        hyperparameter_sets = self.create_hyperparameter_sets()
        
        # Run tuning for each hyperparameter set
        for i, hyperparams in enumerate(hyperparameter_sets):
            set_id = hyperparams['set_id']
            print(f"\n{'='*60}")
            print(f"Processing hyperparameter set {i+1}/{len(hyperparameter_sets)}: {set_id}")
            print(f"Hyperparameters: {hyperparams}")
            print(f"{'='*60}")
            
            try:
                # Create config with hyperparameters
                tuning_config = self.create_config_from_hyperparams(hyperparams)
                
                # Train models
                model_predictions_train, atc_predictions_train = self.train_models(train_data, tuning_config)
                
                # Predict on test data (pure model predictions, not reconstruction)
                model_predictions_test = self.predict_on_test(test_data, tuning_config)
                
                # Evaluate pure model predictions against observed data
                evaluation_metrics = self.evaluate_predictions(model_predictions_test, test_data, tuning_config)
                
                # Store results
                self.results[set_id] = {
                    'hyperparameters': hyperparams,
                    'evaluation_metrics': evaluation_metrics,
                    'status': 'completed'
                }
                
                print(f"  Completed successfully!")
                print(f"  Key metrics: MAE={evaluation_metrics.get('simulated_clouds_mae', 'N/A'):.4f}, "
                      f"RMSE={evaluation_metrics.get('simulated_clouds_rmse', 'N/A'):.4f}, "
                      f"R2={evaluation_metrics.get('simulated_clouds_r2', 'N/A'):.4f}")
                
            except Exception as e:
                print(f"  Failed with error: {e}")
                self.results[set_id] = {
                    'hyperparameters': hyperparams,
                    'evaluation_metrics': {},
                    'status': 'failed',
                    'error': str(e)
                }
                continue
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """
        Save hyperparameter tuning results to JSON file.
        """
        results_path = os.path.join(self.tuning_output_dir, "hyperparameter_tuning_results.json")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy_types(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=4, default=str)
        
        print(f"\nResults saved to: {results_path}")
        
        # Create summary
        self.create_summary()
    
    def create_summary(self):
        """
        Create a summary of hyperparameter tuning results.
        """
        summary_path = os.path.join(self.tuning_output_dir, "tuning_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("Hyperparameter Tuning Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            completed_runs = sum(1 for result in self.results.values() if result['status'] == 'completed')
            failed_runs = sum(1 for result in self.results.values() if result['status'] == 'failed')
            
            f.write(f"Total runs: {len(self.results)}\n")
            f.write(f"Completed: {completed_runs}\n")
            f.write(f"Failed: {failed_runs}\n\n")
            
            # Best results
            if completed_runs > 0:
                # Find best results by different metrics
                metrics_to_check = ['simulated_clouds_mae', 'simulated_clouds_rmse', 'simulated_clouds_r2']
                
                for metric in metrics_to_check:
                    f.write(f"Best {metric.upper()}:\n")
                    
                    valid_results = []
                    for set_id, result in self.results.items():
                        if (result['status'] == 'completed' and 
                            metric in result['evaluation_metrics'] and
                            not np.isnan(result['evaluation_metrics'][metric])):
                            valid_results.append((set_id, result['evaluation_metrics'][metric], result['hyperparameters']))
                    
                    if valid_results:
                        if metric == 'simulated_clouds_r2':
                            # Higher is better for R2
                            best_set_id, best_value, best_hyperparams = max(valid_results, key=lambda x: x[1])
                        else:
                            # Lower is better for MAE, RMSE
                            best_set_id, best_value, best_hyperparams = min(valid_results, key=lambda x: x[1])
                        
                        f.write(f"  Set ID: {best_set_id}\n")
                        f.write(f"  Value: {best_value:.6f}\n")
                        f.write(f"  Hyperparameters: {best_hyperparams}\n\n")
                    else:
                        f.write(f"  No valid results for {metric}\n\n")
            
            # Detailed results
            f.write("Detailed Results:\n")
            f.write("-" * 30 + "\n\n")
            
            for set_id, result in self.results.items():
                f.write(f"Set ID: {set_id}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Hyperparameters: {result['hyperparameters']}\n")
                
                if result['status'] == 'completed':
                    metrics = result['evaluation_metrics']
                    f.write(f"Metrics:\n")
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                
                f.write("\n" + "-" * 20 + "\n\n")
        
        print(f"Summary saved to: {summary_path}")


def test_config_creation():
    """
    Test function to verify config creation works properly.
    """
    print("Testing config creation...")
    try:
        tuner = HyperparameterTuner(config)
        
        # Test creating a config with sample hyperparameters
        test_hyperparams = {
            'ATC_LEARNING_RATE': 0.1,
            'ATC_EPOCHS': 1000,
            'GP_LEARNING_RATE_INITIAL': 0.05,
            'set_id': 'test_001',
            'set_type': 'test'
        }
        
        test_config = tuner.create_config_from_hyperparams(test_hyperparams)
        print(f"✓ Config creation successful")
        print(f"  ATC_LEARNING_RATE: {getattr(test_config, 'ATC_LEARNING_RATE', 'Missing')}")
        print(f"  OUTPUT_DIR: {getattr(test_config, 'OUTPUT_DIR', 'Missing')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to run hyperparameter tuning.
    """
    print("DELAG Hyperparameter Tuning")
    print("=" * 50)
    
    # Test config creation first
    if not test_config_creation():
        print("Config creation test failed. Please fix the issues before running full tuning.")
        return
    
    print("\nConfig creation test passed. Starting hyperparameter tuning...")
    
    # Initialize tuner with base config
    tuner = HyperparameterTuner(config)
    
    # Run hyperparameter tuning
    results = tuner.run_hyperparameter_tuning()
    
    print("\nHyperparameter tuning completed!")
    print(f"Results saved in: {tuner.tuning_output_dir}")


if __name__ == "__main__":
    main() 