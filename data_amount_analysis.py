"""
Data Amount Analysis for the DELAG project.

This script analyzes how much training data is needed for good model performance by:
1. Splitting data: train (2015-2024), test (2024-2025)
2. Progressively reducing training data by 20% each step
3. Training both ATC and GP models for each data amount
4. Evaluating on test set using pure model predictions
5. Storing results for plotting performance vs data amount
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
from tqdm import tqdm

# Import project modules
import config
import data_preprocessing
import atc_model
import gp_model
import evaluation


class DataAmountAnalyzer:
    """
    Analyzes model performance with different amounts of training data.
    """
    
    def __init__(self, base_config):
        """
        Initialize the data amount analyzer.
        
        Args:
            base_config: Base configuration object
        """
        self.base_config = base_config
        self.results = []
        
        # Set up output directory
        self.analysis_output_dir = os.path.join(
            base_config.OUTPUT_DIR_BASE, 
            f"data_amount_analysis_{base_config.ROI_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.analysis_output_dir, exist_ok=True)
        
        print(f"Data amount analysis output directory: {self.analysis_output_dir}")
    
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
            
            if train_start <= date < train_end:
                train_indices.append(i)
            if test_start <= date < test_end:
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
    
    def reduce_training_data(self, train_data: Dict, reduction_factor: float) -> Dict:
        """
        Reduce the amount of training data by temporal subsampling.
        
        Args:
            train_data: Original training data dictionary
            reduction_factor: Factor to reduce data (0.8 means keep 80%)
            
        Returns:
            Dictionary with reduced training data
        """
        print(f"Reducing training data by factor {reduction_factor:.2f}")
        
        original_length = len(train_data['common_dates'])
        new_length = int(original_length * reduction_factor)
        
        if new_length < 10:  # Minimum threshold
            print(f"Warning: Reduced data length ({new_length}) is very small")
        
        # Create indices for temporal subsampling (evenly spaced)
        if new_length >= original_length:
            selected_indices = list(range(original_length))
        else:
            # Evenly space the selected indices across the full range
            selected_indices = np.linspace(0, original_length - 1, new_length, dtype=int)
        
        print(f"Selected {len(selected_indices)} time points out of {original_length}")
        
        # Create reduced data subset
        reduced_data = {}
        for key, value in train_data.items():
            if key == 'common_dates':
                reduced_data[key] = [train_data['common_dates'][i] for i in selected_indices]
            elif key in ['lst_stack', 'era5_stack', 's2_reflectance_stack', 'doy_stack']:
                if hasattr(value, 'shape') and len(value.shape) >= 1:
                    reduced_data[key] = value[selected_indices]
                else:
                    reduced_data[key] = value
            elif key == 'ndvi_stack' and value is not None:
                reduced_data[key] = value[selected_indices]
            else:
                reduced_data[key] = value
        
        return reduced_data
    
    def create_analysis_config(self, step_id: str):
        """
        Create a config object for this analysis step.
        
        Args:
            step_id: Identifier for this analysis step
            
        Returns:
            Modified config object
        """
        # Create a new config class
        class AnalysisConfig:
            pass
        
        analysis_config = AnalysisConfig()
        
        # Copy all attributes from base config, filtering out non-copyable ones
        for attr_name in dir(self.base_config):
            if not attr_name.startswith('_'):  # Skip private attributes
                try:
                    attr_value = getattr(self.base_config, attr_name)
                    # Skip modules, functions, classes, and other complex objects
                    if not callable(attr_value) and not hasattr(attr_value, '__module__'):
                        setattr(analysis_config, attr_name, attr_value)
                except (TypeError, AttributeError):
                    # Skip attributes that can't be accessed or set
                    continue
        
        # Create unique output paths for this analysis step
        analysis_config.OUTPUT_DIR = os.path.join(self.analysis_output_dir, step_id)
        analysis_config.MODEL_WEIGHTS_PATH = os.path.join(analysis_config.OUTPUT_DIR, "model_weights")
        os.makedirs(analysis_config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(analysis_config.MODEL_WEIGHTS_PATH, exist_ok=True)
        
        return analysis_config
    
    def train_and_evaluate_models(self, train_data: Dict, test_data: Dict, analysis_config) -> Dict:
        """
        Train both ATC and GP models and evaluate on test data.
        
        Args:
            train_data: Training data dictionary
            test_data: Test data dictionary
            analysis_config: Configuration for this analysis step
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            print("  Training ATC model...")
            # Train ATC model
            all_pixel_snapshots, interval_loss_maps = atc_model.train_and_collect_all_atc_snapshots(
                train_data, analysis_config
            )
            
            # Save ATC snapshots
            snapshots_filename = f"atc_snapshots_{train_data.get('roi_name', 'analysis')}.npz"
            snapshots_filepath = os.path.join(analysis_config.MODEL_WEIGHTS_PATH, snapshots_filename)
            
            _, height, width = train_data['lst_stack'].shape
            atc_model.save_atc_snapshots(
                all_pixel_snapshots, 
                snapshots_filepath,
                image_height=height,
                image_width=width,
                num_snapshots_expected=analysis_config.ATC_ENSEMBLE_SNAPSHOTS
            )
            
            # Load ATC snapshots and predict on test data
            loaded_snapshots_data = atc_model.load_atc_snapshots(snapshots_filepath)
            atc_mean_predictions, atc_variance = atc_model.predict_atc_from_loaded_snapshots(
                loaded_snapshots_data,
                doy_for_prediction_numpy=test_data["doy_stack"],
                era5_for_prediction_numpy=test_data["era5_stack"],
                app_config=analysis_config
            )
            
            print("  Training GP model...")
            # Train GP model
            gp_model.train_and_save_gp_model(
                train_data, 
                # Use ATC predictions on training data for GP training
                atc_model.predict_atc_from_loaded_snapshots(
                    loaded_snapshots_data,
                    doy_for_prediction_numpy=train_data["doy_stack"],
                    era5_for_prediction_numpy=train_data["era5_stack"],
                    app_config=analysis_config
                )[0],  # Get mean predictions only
                analysis_config
            )
            
            # Load GP model and predict residuals on test data
            gp_mean_residuals_map, gp_variance_residuals_map = gp_model.load_and_predict_gp_residuals(
                test_data, atc_mean_predictions, analysis_config
            )
            
            # Combine ATC and GP predictions (pure model predictions, not reconstruction)
            pure_model_predictions = atc_mean_predictions + gp_mean_residuals_map
            
            print("  Evaluating predictions...")
            # Evaluate pure model predictions against observed test data
            evaluation_metrics = evaluation.run_all_evaluations(
                model_predicted_lst=pure_model_predictions,
                observed_lst_clear=test_data['lst_stack'],
                app_config=analysis_config
            )
            
            return evaluation_metrics
            
        except Exception as e:
            print(f"  Error in training/evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_data_amount_analysis(self) -> List[Dict]:
        """
        Run the complete data amount analysis.
        
        Returns:
            List of results for each data amount step
        """
        print("Starting data amount analysis...")
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        preprocessed_data = data_preprocessing.preprocess_all_data(self.base_config)
        
        # Split data by date
        full_train_data, test_data = self.split_data_by_date(preprocessed_data)
        
        # Define reduction factors (5 steps as requested)
        reduction_factors = [1.0, 0.8, 0.64, 0.512, 0.4096]  # Each step reduces by 20%
        
        print(f"Will test {len(reduction_factors)} different data amounts:")
        for i, factor in enumerate(reduction_factors):
            print(f"  Step {i+1}: {factor*100:.1f}% of original training data")
        
        # Run analysis for each data amount
        for step, reduction_factor in enumerate(reduction_factors):
            step_id = f"step_{step+1:02d}_data_{reduction_factor*100:.1f}pct"
            
            print(f"\n{'='*60}")
            print(f"Data Amount Analysis Step {step+1}/{len(reduction_factors)}")
            print(f"Using {reduction_factor*100:.1f}% of training data")
            print(f"Step ID: {step_id}")
            print(f"{'='*60}")
            
            try:
                # Create config for this step
                analysis_config = self.create_analysis_config(step_id)
                
                # Reduce training data
                if reduction_factor == 1.0:
                    current_train_data = full_train_data
                else:
                    current_train_data = self.reduce_training_data(full_train_data, reduction_factor)
                
                # Train and evaluate models
                evaluation_metrics = self.train_and_evaluate_models(
                    current_train_data, test_data, analysis_config
                )
                
                # Calculate data statistics
                original_train_points = len(full_train_data['common_dates'])
                current_train_points = len(current_train_data['common_dates'])
                test_points = len(test_data['common_dates'])
                
                # Store results
                step_result = {
                    'step': step + 1,
                    'step_id': step_id,
                    'reduction_factor': reduction_factor,
                    'data_percentage': reduction_factor * 100,
                    'original_train_points': original_train_points,
                    'current_train_points': current_train_points,
                    'test_points': test_points,
                    'evaluation_metrics': evaluation_metrics,
                    'status': 'completed' if evaluation_metrics else 'failed'
                }
                
                self.results.append(step_result)
                
                if evaluation_metrics:
                    print(f"  Completed successfully!")
                    print(f"  Training points: {current_train_points} ({reduction_factor*100:.1f}%)")
                    print(f"  Key metrics: MAE={evaluation_metrics.get('simulated_clouds_mae', 'N/A'):.4f}, "
                          f"RMSE={evaluation_metrics.get('simulated_clouds_rmse', 'N/A'):.4f}, "
                          f"R2={evaluation_metrics.get('simulated_clouds_r2', 'N/A'):.4f}")
                else:
                    print(f"  Failed to complete evaluation")
                
            except Exception as e:
                print(f"  Failed with error: {e}")
                step_result = {
                    'step': step + 1,
                    'step_id': step_id,
                    'reduction_factor': reduction_factor,
                    'data_percentage': reduction_factor * 100,
                    'original_train_points': len(full_train_data['common_dates']),
                    'current_train_points': 0,
                    'test_points': len(test_data['common_dates']),
                    'evaluation_metrics': {},
                    'status': 'failed',
                    'error': str(e)
                }
                self.results.append(step_result)
                continue
        
        # Save results and create plots
        self.save_results()
        self.create_plots()
        
        return self.results
    
    def save_results(self):
        """
        Save data amount analysis results to JSON file.
        """
        results_path = os.path.join(self.analysis_output_dir, "data_amount_analysis_results.json")
        
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
    
    def create_plots(self):
        """
        Create line charts showing performance vs data amount.
        """
        print("Creating performance plots...")
        
        # Extract data for plotting
        completed_results = [r for r in self.results if r['status'] == 'completed']
        
        if not completed_results:
            print("No completed results to plot")
            return
        
        data_percentages = [r['data_percentage'] for r in completed_results]
        train_points = [r['current_train_points'] for r in completed_results]
        
        # Extract metrics
        metrics_to_plot = ['simulated_clouds_mae', 'simulated_clouds_rmse', 'simulated_clouds_r2']
        metrics_data = {}
        
        for metric in metrics_to_plot:
            metrics_data[metric] = []
            for r in completed_results:
                value = r['evaluation_metrics'].get(metric, np.nan)
                metrics_data[metric].append(value if not np.isnan(value) else None)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Performance vs Training Data Amount\nROI: {self.base_config.ROI_NAME}', fontsize=16)
        
        # Plot 1: MAE vs Data Amount
        ax1 = axes[0, 0]
        if metrics_data['simulated_clouds_mae']:
            ax1.plot(data_percentages, metrics_data['simulated_clouds_mae'], 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Training Data Percentage (%)')
            ax1.set_ylabel('MAE (K)')
            ax1.set_title('Mean Absolute Error vs Training Data Amount')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: RMSE vs Data Amount
        ax2 = axes[0, 1]
        if metrics_data['simulated_clouds_rmse']:
            ax2.plot(data_percentages, metrics_data['simulated_clouds_rmse'], 'o-', linewidth=2, markersize=8, color='orange')
            ax2.set_xlabel('Training Data Percentage (%)')
            ax2.set_ylabel('RMSE (K)')
            ax2.set_title('Root Mean Square Error vs Training Data Amount')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: R² vs Data Amount
        ax3 = axes[1, 0]
        if metrics_data['simulated_clouds_r2']:
            ax3.plot(data_percentages, metrics_data['simulated_clouds_r2'], 'o-', linewidth=2, markersize=8, color='green')
            ax3.set_xlabel('Training Data Percentage (%)')
            ax3.set_ylabel('R²')
            ax3.set_title('R² Score vs Training Data Amount')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training Points vs Performance (MAE)
        ax4 = axes[1, 1]
        if metrics_data['simulated_clouds_mae']:
            ax4.plot(train_points, metrics_data['simulated_clouds_mae'], 'o-', linewidth=2, markersize=8, color='red')
            ax4.set_xlabel('Number of Training Time Points')
            ax4.set_ylabel('MAE (K)')
            ax4.set_title('MAE vs Number of Training Points')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.analysis_output_dir, "data_amount_analysis_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {plot_path}")
        
        # Create summary table
        self.create_summary_table()
    
    def create_summary_table(self):
        """
        Create a summary table of results.
        """
        summary_path = os.path.join(self.analysis_output_dir, "analysis_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("Data Amount Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"ROI: {self.base_config.ROI_NAME}\n")
            f.write(f"Spatial sampling: {self.base_config.SPATIAL_TRAINING_SAMPLE_PERCENTAGE*100:.1f}%\n\n")
            
            # Table header
            f.write(f"{'Step':<6} {'Data%':<8} {'Points':<8} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Status':<10}\n")
            f.write("-" * 70 + "\n")
            
            for result in self.results:
                step = result['step']
                data_pct = result['data_percentage']
                points = result['current_train_points']
                status = result['status']
                
                if status == 'completed':
                    metrics = result['evaluation_metrics']
                    mae = metrics.get('simulated_clouds_mae', np.nan)
                    rmse = metrics.get('simulated_clouds_rmse', np.nan)
                    r2 = metrics.get('simulated_clouds_r2', np.nan)
                    
                    f.write(f"{step:<6} {data_pct:<8.1f} {points:<8} {mae:<8.4f} {rmse:<8.4f} {r2:<8.4f} {status:<10}\n")
                else:
                    f.write(f"{step:<6} {data_pct:<8.1f} {points:<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {status:<10}\n")
        
        print(f"Summary table saved to: {summary_path}")


def main():
    """
    Main function to run data amount analysis.
    """
    print("DELAG Data Amount Analysis")
    print("=" * 50)
    
    # Initialize analyzer with base config
    analyzer = DataAmountAnalyzer(config)
    
    # Run data amount analysis
    results = analyzer.run_data_amount_analysis()
    
    print("\nData amount analysis completed!")
    print(f"Results saved in: {analyzer.analysis_output_dir}")
    
    # Print quick summary
    completed_results = [r for r in results if r['status'] == 'completed']
    if completed_results:
        print(f"\nCompleted {len(completed_results)} out of {len(results)} analysis steps")
        print("Quick summary:")
        for result in completed_results:
            mae = result['evaluation_metrics'].get('simulated_clouds_mae', 'N/A')
            data_pct = result['data_percentage']
            points = result['current_train_points']
            print(f"  {data_pct:5.1f}% data ({points:3d} points): MAE = {mae:.4f}")


if __name__ == "__main__":
    main() 