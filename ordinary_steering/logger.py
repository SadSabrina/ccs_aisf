import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from ordinary_steering_metrics import plot_coefficient_sweep_lines_comparison
from tabulate import tabulate
from tr_plotting import (
    plot_all_decision_boundaries,
    plot_all_layer_vectors,
    plot_all_strategies_all_steering_vectors,
    plot_performance_across_layers,
    plot_vectors_all_strategies,
)


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
    layer_data=None,
    all_strategy_data=None,
    all_steering_vectors=None,
):
    """Print comprehensive results summary table with all metrics and their distributions.

    Args:
        results: List or dictionary with results for each layer and steering coefficient
        steering_coefficients: List of steering coefficients used
        model_name: Name of the model
        model_family: Family of the model
        model_variant: Variant of the model
        run_dir: Directory to save results and plots (if None, will not save plots)
        layer_data: List of layer data for visualization (optional)
        all_strategy_data: Dictionary with data for all embedding strategies (optional)
        all_steering_vectors: Dictionary of different steering vectors with their properties (optional)

    Returns:
        Summary string
    """
    # Create plots directory if saving plots
    plots_dir = None
    if run_dir:
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Saving plots and results to {plots_dir}")

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

    # Generate plots if we have run_dir
    if plots_dir:
        summary_str += "GENERATED PLOTS\n"
        summary_str += "-" * 40 + "\n"

        # Generate plots using available data
        plot_paths = []

        # 1. Performance across layers plot
        perf_plot_path = os.path.join(plots_dir, "performance_accuracy.png")
        plot_performance_across_layers(
            results=results_dict, metric="accuracy", save_path=perf_plot_path
        )
        plot_paths.append(("Performance Accuracy", perf_plot_path))

        if "silhouette" in metrics:
            silhouette_plot_path = os.path.join(plots_dir, "performance_silhouette.png")
            plot_performance_across_layers(
                results=results_dict,
                metric="silhouette",
                save_path=silhouette_plot_path,
            )
            plot_paths.append(("Performance Silhouette", silhouette_plot_path))

        if "auc" in metrics:
            auc_plot_path = os.path.join(plots_dir, "performance_auc.png")
            plot_performance_across_layers(
                results=results_dict, metric="auc", save_path=auc_plot_path
            )
            plot_paths.append(("Performance AUC", auc_plot_path))

        # 2. Coefficient sweep plots
        available_metrics = [
            m
            for m in [
                "accuracy",
                "silhouette",
                "similarity_change",
                "path_length",
                "semantic_consistency",
                "steering_alignment",
                "agreement_score",
                "contradiction_index",
                "representation_stability",
                "f1",
            ]
            if m in metrics
        ]

        if available_metrics:
            sweep_plot_path = os.path.join(plots_dir, "coefficient_sweep.png")
            plot_coefficient_sweep_lines_comparison(
                results=results_dict,
                metrics=available_metrics,
                save_path=sweep_plot_path,
            )
            plot_paths.append(("Coefficient Sweep", sweep_plot_path))

        # 3. Layer vectors plot (if layer_data is provided)
        if layer_data:
            # Check if layer_data has the right structure for plotting
            if isinstance(layer_data, (list, dict)) and len(layer_data) > 0:
                # Validate layer data first
                valid_layer_data = []
                if isinstance(layer_data, dict):
                    layer_data_to_check = [
                        layer_data[k] for k in sorted(layer_data.keys())
                    ]
                else:
                    layer_data_to_check = layer_data

                for i, data in enumerate(layer_data_to_check):
                    if not isinstance(data, dict):
                        print(
                            f"Warning: Layer data at index {i} is not a dictionary. Type: {type(data)}"
                        )
                        continue

                    required_keys = [
                        "hate_mean_vector",
                        "safe_mean_vector",
                        "steering_vector",
                    ]
                    if not all(key in data for key in required_keys):
                        available_keys = (
                            list(data.keys()) if isinstance(data, dict) else "N/A"
                        )
                        print(
                            f"Warning: Layer data {i} missing required keys. Available keys: {available_keys}"
                        )
                        continue

                    # Check vector types
                    if not isinstance(data["hate_mean_vector"], np.ndarray):
                        print(
                            f"Warning: hate_mean_vector for layer {i} is not a numpy array. Type: {type(data['hate_mean_vector'])}"
                        )
                        continue

                    if not isinstance(data["safe_mean_vector"], np.ndarray):
                        print(
                            f"Warning: safe_mean_vector for layer {i} is not a numpy array. Type: {type(data['safe_mean_vector'])}"
                        )
                        continue

                    if not isinstance(data["steering_vector"], np.ndarray):
                        print(
                            f"Warning: steering_vector for layer {i} is not a numpy array. Type: {type(data['steering_vector'])}"
                        )
                        continue

                    valid_layer_data.append(data)

                if valid_layer_data:
                    vectors_plot_path = os.path.join(plots_dir, "all_layer_vectors.png")
                    plot_all_layer_vectors(results=valid_layer_data, save_dir=plots_dir)
                    plot_paths.append(("Layer Vectors", vectors_plot_path))
                else:
                    print("Warning: No valid layer data for plotting vectors")
            else:
                print(
                    f"Warning: layer_data has invalid structure. Type: {type(layer_data)}, Length: {len(layer_data) if hasattr(layer_data, '__len__') else 'N/A'}"
                )

            # Decision boundaries also depend on layer_data structure
            if isinstance(layer_data, (list, dict)) and len(layer_data) > 0:
                # Validate layer data for decision boundaries
                valid_boundary_data = []
                if isinstance(layer_data, dict):
                    layer_data_to_check = [
                        layer_data[k] for k in sorted(layer_data.keys())
                    ]
                else:
                    layer_data_to_check = layer_data

                for i, data in enumerate(layer_data_to_check):
                    if not isinstance(data, dict):
                        print(
                            f"Warning: Layer data at index {i} is not a dictionary. Type: {type(data)}"
                        )
                        continue

                    required_keys = [
                        "ccs",
                        "hate_vectors",
                        "safe_vectors",
                        "steering_vector",
                    ]
                    if not all(key in data for key in required_keys):
                        available_keys = (
                            list(data.keys()) if isinstance(data, dict) else "N/A"
                        )
                        print(
                            f"Warning: Layer data {i} missing required keys for decision boundaries. Available keys: {available_keys}"
                        )
                        continue

                    valid_boundary_data.append(data)

                if valid_boundary_data:
                    decision_boundaries_path = os.path.join(
                        plots_dir, "all_decision_boundaries.png"
                    )
                    plot_all_decision_boundaries(
                        layers_data=valid_boundary_data,
                        log_base=os.path.join(plots_dir, "all_decision_boundaries"),
                    )
                    plot_paths.append(("Decision Boundaries", decision_boundaries_path))
                else:
                    print(
                        "Warning: No valid layer data for plotting decision boundaries"
                    )
            else:
                print(
                    f"Warning: layer_data has invalid structure for decision boundary plotting. Type: {type(layer_data)}"
                )

        # 4. Strategy vectors plot (if all_strategy_data and all_steering_vectors are provided)
        if all_strategy_data and all_steering_vectors:
            # Validate both data structures
            if not isinstance(all_strategy_data, (dict, list)) or not isinstance(
                all_steering_vectors, (dict, list)
            ):
                print(
                    f"Warning: Invalid types for strategy data plotting - all_strategy_data: {type(all_strategy_data)}, all_steering_vectors: {type(all_steering_vectors)}"
                )
            else:
                valid_layers = range(len(results_dict))
                for layer_idx in valid_layers:
                    # Extract data for this layer
                    if isinstance(all_strategy_data, list) and 0 <= layer_idx < len(
                        all_strategy_data
                    ):
                        layer_strategy_data = all_strategy_data[layer_idx]
                    elif (
                        isinstance(all_strategy_data, dict)
                        and layer_idx in all_strategy_data
                    ):
                        layer_strategy_data = all_strategy_data[layer_idx]
                    else:
                        print(f"Warning: No strategy data for layer {layer_idx}")
                        continue

                    if isinstance(all_steering_vectors, list) and 0 <= layer_idx < len(
                        all_steering_vectors
                    ):
                        layer_steering_vectors = all_steering_vectors[layer_idx]
                    elif (
                        isinstance(all_steering_vectors, dict)
                        and layer_idx in all_steering_vectors
                    ):
                        layer_steering_vectors = all_steering_vectors[layer_idx]
                    else:
                        print(f"Warning: No steering vectors for layer {layer_idx}")
                        continue

                    # Check if at least one strategy exists
                    if (
                        not isinstance(layer_strategy_data, dict)
                        or len(layer_strategy_data) == 0
                    ):
                        print(
                            f"Warning: Invalid strategy data format for layer {layer_idx}. Type: {type(layer_strategy_data)}"
                        )
                        continue

                    if (
                        not isinstance(layer_steering_vectors, dict)
                        or len(layer_steering_vectors) == 0
                    ):
                        print(
                            f"Warning: Invalid steering vectors format for layer {layer_idx}. Type: {type(layer_steering_vectors)}"
                        )
                        continue

                    # Plot vectors for all strategies
                    strategy_plot_path = os.path.join(
                        plots_dir, f"layer_{layer_idx}_all_strategies_vectors.png"
                    )
                    plot_vectors_all_strategies(
                        layer_idx=layer_idx,
                        all_strategy_data=layer_strategy_data,
                        current_strategy="last-token",
                        save_path=strategy_plot_path,
                        all_steering_vectors=layer_steering_vectors,
                    )
                    plot_paths.append(
                        (f"Layer {layer_idx} Strategies", strategy_plot_path)
                    )

                    # All strategies, all steering vectors comprehensive plot
                    comprehensive_path = os.path.join(
                        plots_dir,
                        f"layer_{layer_idx}_all_strategies_all_steering_vectors.png",
                    )
                    plot_all_strategies_all_steering_vectors(
                        plot_dir=plots_dir,
                        layer_idx=layer_idx,
                        representations=layer_strategy_data,
                        all_steering_vectors_by_strategy=layer_steering_vectors,
                    )
                    plot_paths.append(
                        (f"Layer {layer_idx} Comprehensive", comprehensive_path)
                    )

        # Add plots to summary
        for plot_name, plot_path in plot_paths:
            rel_path = os.path.relpath(plot_path, run_dir) if run_dir else plot_path
            summary_str += f"{plot_name}: {rel_path}\n"
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
