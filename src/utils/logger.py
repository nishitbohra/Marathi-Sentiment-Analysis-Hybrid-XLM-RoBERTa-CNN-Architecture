"""
Logger utility for experiment tracking.

Provides simple logging functionality for tracking experiments,
metrics, and model performance.
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class ExperimentLogger:
    """
    Simple logger for tracking experiments and metrics.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to save logs
        
    Attributes:
        experiment_name: Name of the experiment
        log_dir: Directory for logs
        log_file: Path to log file
        metrics_file: Path to metrics CSV file
    """
    
    def __init__(self, experiment_name: str, log_dir: str = "results/logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        
        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set file paths
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}_metrics.csv")
        
        # Initialize log file
        self.log(f"Experiment: {experiment_name}")
        self.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 70)
    
    def log(self, message: str, print_console: bool = True) -> None:
        """
        Log a message to file and optionally print to console.
        
        Args:
            message: Message to log
            print_console: Whether to print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # Write to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
        
        # Print to console
        if print_console:
            print(message)
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        epoch: Optional[int] = None,
        phase: str = "train"
    ) -> None:
        """
        Log metrics to CSV file.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Epoch number (optional)
            phase: Training phase ('train', 'val', 'test')
        """
        # Prepare row
        row = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'phase': phase
        }
        
        if epoch is not None:
            row['epoch'] = epoch
        
        row.update(metrics)
        
        # Check if file exists
        file_exists = os.path.exists(self.metrics_file)
        
        # Write to CSV
        with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
        
        # Log summary
        metrics_str = ', '.join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                for k, v in metrics.items()])
        self.log(f"{phase.upper()} Metrics - {metrics_str}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration to JSON file.
        
        Args:
            config: Configuration dictionary
        """
        config_file = os.path.join(self.log_dir, f"{self.experiment_name}_config.json")
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        self.log(f"Configuration saved to: {config_file}")
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """
        Log model information.
        
        Args:
            model_info: Dictionary with model information
        """
        self.log("\nModel Information:")
        for key, value in model_info.items():
            self.log(f"  {key}: {value}")
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ) -> None:
        """
        Log summary for an epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            epoch_time: Time taken for epoch
        """
        self.log(f"\nEpoch {epoch} Summary:")
        self.log(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        self.log(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
        self.log(f"  Time: {epoch_time:.2f}s")
    
    def log_final_results(self, results: Dict[str, Any]) -> None:
        """
        Log final results.
        
        Args:
            results: Dictionary of final results
        """
        self.log("\n" + "=" * 70)
        self.log("FINAL RESULTS")
        self.log("=" * 70)
        
        for key, value in results.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.4f}")
            else:
                self.log(f"  {key}: {value}")
        
        self.log("=" * 70)
        self.log(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
            filename: Custom filename (optional)
        """
        if filename is None:
            filename = f"{self.experiment_name}_results.json"
        
        results_file = os.path.join(self.log_dir, filename)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.log(f"Results saved to: {results_file}")


def create_logger(experiment_name: str, log_dir: str = "results/logs") -> ExperimentLogger:
    """
    Factory function to create a logger.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for logs
        
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(experiment_name, log_dir)


if __name__ == "__main__":
    """Test logger functionality."""
    
    print("Testing ExperimentLogger...\n")
    
    # Create logger
    logger = create_logger("test_experiment", "results/logs")
    
    # Log some messages
    logger.log("Starting test experiment")
    logger.log("Loading data...")
    
    # Log configuration
    config = {
        'model': 'test_model',
        'learning_rate': 0.001,
        'batch_size': 32
    }
    logger.log_config(config)
    
    # Log metrics
    logger.log_metrics({
        'loss': 0.5,
        'accuracy': 0.85
    }, epoch=1, phase='train')
    
    # Log final results
    logger.log_final_results({
        'test_accuracy': 0.87,
        'test_f1': 0.86
    })
    
    print("\n✅ Logger tests passed!")
    print(f"   Log file: {logger.log_file}")
    print(f"   Metrics file: {logger.metrics_file}")
