import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tabulate import tabulate


def setup_logger(
    model_name, model_family, model_variant, device, test_size=0.3, random_state=42
):
    """Setup logger for the experiment"""
    # Create logs directory if it doesn't exist
    logs_dir = "ordinary_steering_logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        logs_dir, f"run_{model_family}_{model_variant}_{test_size}test_{timestamp}"
    )
    os.makedirs(run_dir)

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create file handler
    log_file = os.path.join(run_dir, "run.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log run parameters
    logger.info("=" * 50)
    logger.info("Starting new run with parameters:")
    logger.info(f"Model: {model_name}")
    logger.info(f"Model Family: {model_family}")
    logger.info(f"Model Variant: {model_variant}")
    logger.info(f"Test Size: {test_size}")
    logger.info(f"Random State: {random_state}")
    logger.info(f"Device: {device}")
    logger.info("=" * 50)

    return logger, run_dir, os.path.join(run_dir, "run")


def print_results_summary(
    results,
    steering_coefficients,
    model_name,
    model_family,
    model_variant,
    run_dir=None,
):
    """Print comprehensive results summary table with all metrics and their distributions."""
    # Create summary dictionary
    summary = {
        "Model Information": {
            "Model Name": model_name,
            "Model Family": model_family,
            "Model Variant": model_variant,
            "Number of Layers": len(results),
            "Steering Coefficients": steering_coefficients,
        }
    }

    # Initialize metrics storage
    metrics = {
        # CCS performance metrics
        "accuracy": [],
        "silhouette": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "decision_confidence": [],
        # Steering metrics
        "similarity_change": [],
        "path_length": [],
        "semantic_consistency": [],
        "steering_alignment": [],
        # Agreement and contradiction metrics
        "agreement_score": [],
        "contradiction_index": [],
        "ideal_distance": [],
        # Stability metrics
        "representation_stability": [],
        # Layer-specific metrics
        "class_separability": [],
        "subspace_separation": [],
        "subspace_variance": [],
    }

    # Collect all metrics across layers and coefficients
    for layer_idx in results:
        layer_data = results[layer_idx]

        # Add layer-specific metrics
        metrics["class_separability"].append(layer_data["class_separability"])
        metrics["subspace_separation"].append(
            layer_data["subspace_analysis"]["top_component_separation"]
        )
        metrics["subspace_variance"].append(
            layer_data["subspace_analysis"]["variance_explained"]
        )

        for coef in steering_coefficients:
            coef_data = layer_data[f"coef_{coef}"]
            for metric in metrics:
                if metric in coef_data:
                    metrics[metric].append(coef_data[metric])

    # Calculate statistics for each metric
    for metric in metrics:
        if metrics[metric]:  # Only calculate if we have data
            values = metrics[metric]
            summary[f"{metric.capitalize()} Statistics"] = {
                "Mean": np.mean(values),
                "Std": np.std(values),
                "Min": np.min(values),
                "Max": np.max(values),
                "Median": np.median(values),
                "Q1": np.percentile(values, 25),
                "Q3": np.percentile(values, 75),
            }

    # Create detailed layer-wise summary
    layer_summary = []
    for layer_idx in results:
        layer_data = results[layer_idx]
        for coef in steering_coefficients:
            coef_data = layer_data[f"coef_{coef}"]
            summary_row = {
                "Layer": layer_idx + 1,
                "Steering Coefficient": coef,
                "Class Separability": layer_data["class_separability"],
                "Subspace Separation": layer_data["subspace_analysis"][
                    "top_component_separation"
                ],
                "Subspace Variance": layer_data["subspace_analysis"][
                    "variance_explained"
                ],
            }
            # Add all coefficient-specific metrics
            for metric, value in coef_data.items():
                summary_row[metric.replace("_", " ").title()] = value
            layer_summary.append(summary_row)

    # Convert to DataFrame for pretty printing
    layer_df = pd.DataFrame(layer_summary)

    # Format the summary for display
    summary_str = "\n" + "=" * 80 + "\n"
    summary_str += "EXPERIMENT RESULTS SUMMARY\n"
    summary_str += "=" * 80 + "\n\n"

    # Model Information
    summary_str += "MODEL INFORMATION\n"
    summary_str += "-" * 40 + "\n"
    for key, value in summary["Model Information"].items():
        summary_str += f"{key}: {value}\n"
    summary_str += "\n"

    # Metric Statistics
    for metric, stats in summary.items():
        if "Statistics" in metric:
            summary_str += f"{metric}\n"
            summary_str += "-" * 40 + "\n"
            for stat_name, value in stats.items():
                summary_str += f"{stat_name}: {value:.4f}\n"
            summary_str += "\n"

    # Layer-wise Results
    summary_str += "LAYER-WISE RESULTS\n"
    summary_str += "-" * 80 + "\n"
    # Convert DataFrame to list of lists for tabulate
    headers = layer_df.columns.tolist()
    data = layer_df.values.tolist()
    summary_str += tabulate(data, headers=headers, tablefmt="grid", floatfmt=".4f")
    summary_str += "\n"

    # Print to terminal
    print(summary_str)

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_summary_{model_family}_{model_variant}_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write(summary_str)

    # Save detailed results as JSON
    json_filename = f"detailed_results_{model_family}_{model_variant}_{timestamp}.json"
    with open(json_filename, "w") as f:
        json.dump(results, f, indent=2)

    return summary_str
