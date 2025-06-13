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
    """Setup logger for the experiment - CHANGED: Only handles logging setup, no plotting"""
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


def log_results_summary(
    results,
    steering_coefficients,
    model_name,
    model_family,
    model_variant,
    run_dir=None,
):
    """Log comprehensive results summary - CHANGED: Only handles logging, no plotting or detailed analysis

    Args:
        results: List or dictionary with results for each layer and steering coefficient
        steering_coefficients: List of steering coefficients used
        model_name: Name of the model
        model_family: Family of the model
        model_variant: Variant of the model
        run_dir: Directory to save results (optional)

    Returns:
        Summary string, filename, json_filename
    """
    # Handle the case where results is a tuple (from train_ccs_with_steering)
    if isinstance(results, tuple) and len(results) >= 1:
        # Extract the first element (actual results list) from the tuple
        results = results[0]

    # Convert list to dict if results is a list
    results_dict = {}
    if isinstance(results, (list, tuple)):
        for i, layer_result in enumerate(results):
            if "layer_idx" in layer_result:
                results_dict[layer_result["layer_idx"]] = layer_result
            else:
                results_dict[i] = layer_result
    else:
        results_dict = results

    # Create summary dictionary
    summary = {
        "Model Information": {
            "Model Name": model_name,
            "Model Family": model_family,
            "Model Variant": model_variant,
            "Number of Layers": len(results_dict),
            "Steering Coefficients": steering_coefficients,
        }
    }

    # Initialize metrics storage for all possible metrics
    metrics = {
        # CCS performance metrics
        "accuracy": [],
        "silhouette": [],
        "auc": [],
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
        "subspace_analysis": [],
    }

    for layer_idx, layer_data in results_dict.items():
        # Convert layer_idx to int if it's a string
        if isinstance(layer_idx, str) and layer_idx.isdigit():
            layer_idx = int(layer_idx)

        # Add layer-specific metrics if available
        if "class_separability" in layer_data:
            metrics["class_separability"].append(layer_data["class_separability"])

        if "subspace_analysis" in layer_data:
            if "top_component_separation" in layer_data["subspace_analysis"]:
                metrics["subspace_separation"].append(
                    layer_data["subspace_analysis"]["top_component_separation"]
                )
            if "variance_explained" in layer_data["subspace_analysis"]:
                metrics["subspace_variance"].append(
                    layer_data["subspace_analysis"]["variance_explained"]
                )

        # Extract metrics for each steering coefficient
        for coef in steering_coefficients:
            coef_key = f"coef_{coef}"
            if coef_key in layer_data:
                coef_data = layer_data[coef_key]
                for metric in metrics:
                    if metric in coef_data:
                        metrics[metric].append(coef_data[metric])

    # Filter out empty metrics
    metrics = {k: v for k, v in metrics.items() if v}

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
    # Ensure results_dict is always a dictionary
    if not isinstance(results_dict, dict):
        print(
            f"Warning: results_dict is not a dictionary for layer summary. Type: {type(results_dict)}"
        )
        results_dict = {}  # Use empty dict as fallback

    for layer_idx, layer_data in results_dict.items():
        # Convert layer_idx to int if it's a string
        if isinstance(layer_idx, str) and layer_idx.isdigit():
            layer_idx = int(layer_idx)

        for coef in steering_coefficients:
            coef_key = f"coef_{coef}"
            if coef_key in layer_data:
                coef_data = layer_data[coef_key]
                summary_row = {
                    "Layer": layer_idx,
                    "Steering Coefficient": coef,
                }

                # Add layer-specific metrics if available
                if "class_separability" in layer_data:
                    summary_row["Class Separability"] = layer_data["class_separability"]

                if "subspace_analysis" in layer_data:
                    if "top_component_separation" in layer_data["subspace_analysis"]:
                        summary_row["Subspace Separation"] = layer_data[
                            "subspace_analysis"
                        ]["top_component_separation"]
                    if "variance_explained" in layer_data["subspace_analysis"]:
                        summary_row["Subspace Variance"] = layer_data[
                            "subspace_analysis"
                        ]["variance_explained"]

                # Add all coefficient-specific metrics
                for metric, value in coef_data.items():
                    summary_row[metric.replace("_", " ").title()] = value

                layer_summary.append(summary_row)

    # Convert to DataFrame for pretty printing
    layer_df = pd.DataFrame(layer_summary)

    # Sort by layer and coefficient for better readability
    if not layer_df.empty:
        if "Layer" in layer_df.columns and "Steering Coefficient" in layer_df.columns:
            layer_df = layer_df.sort_values(by=["Layer", "Steering Coefficient"])

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
                if isinstance(value, (int, float)):
                    summary_str += f"{stat_name}: {value:.4f}\n"
                else:
                    summary_str += f"{stat_name}: {value}\n"
            summary_str += "\n"

    # Layer-wise Results
    summary_str += "LAYER-WISE RESULTS\n"
    summary_str += "-" * 80 + "\n"
    # Convert DataFrame to list of lists for tabulate
    if not layer_df.empty:
        headers = layer_df.columns.tolist()
        data = layer_df.values.tolist()
        summary_str += tabulate(data, headers=headers, tablefmt="grid", floatfmt=".4f")
    else:
        summary_str += "No layer-wise results available\n"
    summary_str += "\n"

    # Print to terminal
    print(summary_str)

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = run_dir if run_dir else "."
    filename = os.path.join(
        results_dir, f"results_summary_{model_family}_{model_variant}_{timestamp}.txt"
    )

    with open(filename, "w") as f:
        f.write(summary_str)

    # Save detailed results as JSON
    json_filename = os.path.join(
        results_dir, f"detailed_results_{model_family}_{model_variant}_{timestamp}.json"
    )

    # Convert numpy arrays and other non-serializable objects to lists
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(
        results_dict if isinstance(results, list) else results
    )
    with open(json_filename, "w") as f:
        json.dump(serializable_results, f, indent=2)

    return summary_str, filename, json_filename
