import os

# Suppress specific warnings from scikit-learn PCA
# import warnings
# warnings.filterwarnings("ignore", message="invalid value encountered in matmul")
# warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
import numpy as np
from metrics_ordinary_steering import (
    plot_coefficient_sweep_lines_comparison,
)
from metrics_plotter import (
    plot_all_decision_boundaries,
    plot_all_layer_vectors,
    plot_all_strategies_all_steering_vectors,
    plot_performance_across_layers,
)


def generate_comprehensive_analysis_and_plots(
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
    """Generate comprehensive analysis plots and visualizations - CHANGED: Moved from logger.py

    This function handles all the plotting and detailed analysis that was previously
    mixed into the logger. Now it's properly separated.

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
        Dictionary with paths to generated plots
    """
    plot_paths = {}

    # Create plots directory if saving plots
    plots_dir = None
    if run_dir:
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Saving plots to {plots_dir}")

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

    # Initialize metrics storage for plotting
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

    for layer_idx, layer_data_item in results_dict.items():
        # Convert layer_idx to int if it's a string
        if isinstance(layer_idx, str) and layer_idx.isdigit():
            layer_idx = int(layer_idx)

        # Add layer-specific metrics if available
        if "class_separability" in layer_data_item:
            metrics["class_separability"].append(layer_data_item["class_separability"])

        if "subspace_analysis" in layer_data_item:
            if "top_component_separation" in layer_data_item["subspace_analysis"]:
                metrics["subspace_separation"].append(
                    layer_data_item["subspace_analysis"]["top_component_separation"]
                )
            if "variance_explained" in layer_data_item["subspace_analysis"]:
                metrics["subspace_variance"].append(
                    layer_data_item["subspace_analysis"]["variance_explained"]
                )

        # Extract metrics for each steering coefficient
        for coef in steering_coefficients:
            coef_key = f"coef_{coef}"
            if coef_key in layer_data_item:
                coef_data = layer_data_item[coef_key]
                for metric in metrics:
                    if metric in coef_data:
                        metrics[metric].append(coef_data[metric])

    # Filter out empty metrics
    metrics = {k: v for k, v in metrics.items() if v}

    if not plots_dir:
        print("No run_dir provided, skipping plot generation")
        return plot_paths

    print("GENERATING COMPREHENSIVE PLOTS AND ANALYSIS")
    print("-" * 60)

    # 1. Performance across layers plot
    if results_dict:
        # CHANGED: Convert results_dict to proper format for plotting
        # The plotting function expects a list of layer results, not a nested dict structure

        # First, let's understand the structure we're dealing with
        print(f"Debug: results_dict type: {type(results_dict)}")
        if isinstance(results_dict, dict) and results_dict:
            first_key = list(results_dict.keys())[0]
            first_value = results_dict[first_key]
            print(
                f"Debug: First key: {first_key}, First value type: {type(first_value)}"
            )

        # The results_dict appears to be strategy -> pair -> layer -> coef structure
        # We need to flatten this for the plotting function
        try:
            # Try to create a flattened structure for plotting
            flattened_results = []

            # Find available strategies and get the first one for plotting
            available_strategies = list(results_dict.keys())
            if not available_strategies:
                print("Warning: No strategies found in results")
            else:
                first_strategy = available_strategies[0]
                strategy_results = results_dict[first_strategy]

                if isinstance(strategy_results, dict):
                    # Get available pairs and use the first one
                    available_pairs = list(strategy_results.keys())
                    if not available_pairs:
                        print("Warning: No pairs found in strategy results")
                    else:
                        first_pair = available_pairs[0]
                        pair_results = strategy_results[first_pair]

                        if isinstance(pair_results, dict):
                            # Now iterate through layers
                            for layer_idx in sorted(pair_results.keys()):
                                layer_result = pair_results[layer_idx]
                                if isinstance(layer_result, dict):
                                    # Create a proper layer result structure
                                    layer_entry = {"layer_idx": layer_idx}

                                    # Add coefficient-specific data
                                    for coef_key, coef_data in layer_result.items():
                                        if (
                                            isinstance(coef_data, dict)
                                            and "metrics" in coef_data
                                        ):
                                            layer_entry[coef_key] = coef_data["metrics"]

                                    flattened_results.append(layer_entry)

            if flattened_results:
                perf_plot_path = os.path.join(plots_dir, "performance_accuracy.png")
                plot_performance_across_layers(
                    results=flattened_results,
                    metric="accuracy",
                    save_path=perf_plot_path,
                )
                plot_paths["Performance Accuracy"] = perf_plot_path

                if "silhouette" in metrics:
                    silhouette_plot_path = os.path.join(
                        plots_dir, "performance_silhouette.png"
                    )
                    plot_performance_across_layers(
                        results=flattened_results,
                        metric="silhouette",
                        save_path=silhouette_plot_path,
                    )
                    plot_paths["Performance Silhouette"] = silhouette_plot_path

                if "auc" in metrics:
                    auc_plot_path = os.path.join(plots_dir, "performance_auc.png")
                    plot_performance_across_layers(
                        results=flattened_results, metric="auc", save_path=auc_plot_path
                    )
                    plot_paths["Performance AUC"] = auc_plot_path
            else:
                print("Warning: Could not create flattened results for plotting")

        except Exception as e:
            print(f"Error creating performance plots: {e}")
            print("Skipping performance plots due to data structure issues")

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

    if available_metrics and results_dict:
        try:
            # CHANGED: Use the same flattened results structure for coefficient sweep
            if "flattened_results" in locals() and flattened_results:
                sweep_plot_path = os.path.join(plots_dir, "coefficient_sweep.png")
                plot_coefficient_sweep_lines_comparison(
                    results=flattened_results,
                    metrics=available_metrics,
                    save_path=sweep_plot_path,
                )
                plot_paths["Coefficient Sweep"] = sweep_plot_path
            else:
                print(
                    "Warning: No flattened results available for coefficient sweep plot"
                )
        except Exception as e:
            print(f"Error creating coefficient sweep plot: {e}")

    # 3. Layer vectors plot (if layer_data is provided)
    if layer_data:
        if isinstance(layer_data, (list, dict)) and len(layer_data) > 0:
            # Validate layer data first
            valid_layer_data = []
            if isinstance(layer_data, dict):
                layer_data_to_check = [layer_data[k] for k in sorted(layer_data.keys())]
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
                plot_paths["Layer Vectors"] = vectors_plot_path
            else:
                print("Warning: No valid layer data for plotting vectors")
        else:
            print(
                f"Warning: layer_data has invalid structure. Type: {type(layer_data)}, Length: {len(layer_data) if hasattr(layer_data, '__len__') else 'N/A'}"
            )

    # 4. Decision boundaries (if layer_data is provided)
    if layer_data and isinstance(layer_data, (list, dict)) and len(layer_data) > 0:
        # Validate layer data for decision boundaries
        valid_boundary_data = []
        if isinstance(layer_data, dict):
            layer_data_to_check = [layer_data[k] for k in sorted(layer_data.keys())]
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
                available_keys = list(data.keys()) if isinstance(data, dict) else "N/A"
                print(
                    f"Warning: Layer data {i} missing required keys for decision boundaries. Available keys: {available_keys}"
                )
                continue

            valid_boundary_data.append(data)

        if valid_boundary_data:
            # Get first layer index for the plot name
            first_layer_idx = 0
            if isinstance(layer_data, dict):
                first_layer_idx = sorted(layer_data.keys())[0]

            decision_boundaries_path = os.path.join(
                plots_dir, f"layer_{first_layer_idx}_all_decision_boundaries.png"
            )
            plot_all_decision_boundaries(
                representations={
                    strategy: {
                        "hate_yes": layer["hate_vectors"],
                        "hate_no": layer["hate_vectors"],
                        "safe_yes": layer["safe_vectors"],
                        "safe_no": layer["safe_vectors"],
                    }
                    for strategy in ["last-token"]
                    for layer in valid_boundary_data
                },
                steering_vectors={
                    "last-token": {
                        "combined": {
                            "vector": layer["steering_vector"],
                            "color": "#00FF00",
                            "label": "Steering Vector",
                        }
                        for layer in valid_boundary_data
                    }
                },
                ccs_models={
                    "last-token": {
                        i: layer["ccs"] for i, layer in enumerate(valid_boundary_data)
                    }
                },
                save_dir=os.path.join(plots_dir, "all_decision_boundaries"),
            )
            plot_paths["Decision Boundaries"] = decision_boundaries_path
        else:
            print("Warning: No valid layer data for plotting decision boundaries")

    # 5. Strategy vectors plot (if all_strategy_data and all_steering_vectors are provided)
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
                plot_paths[f"Layer {layer_idx} Comprehensive"] = comprehensive_path

    print("PLOT GENERATION SUMMARY")
    print("-" * 40)
    for plot_name, plot_path in plot_paths.items():
        rel_path = os.path.relpath(plot_path, run_dir) if run_dir else plot_path
        print(f"{plot_name}: {rel_path}")
    print()

    return plot_paths


def analyze_steering_effectiveness(results_dict, steering_coefficients):
    """Analyze how effective different steering coefficients are - CHANGED: Moved from logger.py

    Args:
        results_dict: Dictionary of results by layer
        steering_coefficients: List of steering coefficients

    Returns:
        Dictionary with effectiveness analysis
    """
    effectiveness_analysis = {}

    for layer_idx, layer_data in results_dict.items():
        layer_analysis = {}

        # Get baseline metrics (coefficient 0.0)
        baseline_metrics = None
        if "coef_0.0" in layer_data:
            baseline_metrics = layer_data["coef_0.0"]
        elif "coef_0" in layer_data:
            baseline_metrics = layer_data["coef_0"]

        if baseline_metrics:
            baseline_acc = baseline_metrics.get("accuracy", 0.0)
            layer_analysis["baseline_accuracy"] = baseline_acc

            # Compare with other coefficients
            coefficient_effects = {}
            for coef in steering_coefficients:
                if coef == 0.0:
                    continue

                coef_key = f"coef_{coef}"
                if coef_key in layer_data:
                    coef_metrics = layer_data[coef_key]
                    coef_acc = coef_metrics.get("accuracy", 0.0)

                    coefficient_effects[coef] = {
                        "accuracy": coef_acc,
                        "accuracy_change": coef_acc - baseline_acc,
                        "relative_change": (coef_acc - baseline_acc) / baseline_acc
                        if baseline_acc > 0
                        else 0.0,
                    }

            layer_analysis["coefficient_effects"] = coefficient_effects

        effectiveness_analysis[layer_idx] = layer_analysis

    return effectiveness_analysis


def print_effectiveness_summary(effectiveness_analysis):
    """Print a summary of steering effectiveness - CHANGED: Moved from logger.py"""
    print("\nSTEERING EFFECTIVENESS ANALYSIS")
    print("=" * 50)

    for layer_idx, layer_analysis in effectiveness_analysis.items():
        print(f"\nLayer {layer_idx}:")
        print("-" * 20)

        baseline_acc = layer_analysis.get("baseline_accuracy", 0.0)
        print(f"Baseline Accuracy: {baseline_acc:.4f}")

        coefficient_effects = layer_analysis.get("coefficient_effects", {})
        for coef, effects in coefficient_effects.items():
            acc_change = effects["accuracy_change"]
            rel_change = effects["relative_change"]

            change_symbol = "↑" if acc_change > 0 else "↓" if acc_change < 0 else "="

            print(
                f"  Coef {coef:4.1f}: {effects['accuracy']:.4f} ({change_symbol}{abs(acc_change):.4f}, {rel_change:+.2%})"
            )

            # Interpretation
            if acc_change > 0.05:
                print(
                    f"    → Steering coefficient {coef} SIGNIFICANTLY IMPROVES performance!"
                )
            elif acc_change > 0.01:
                print(f"    → Steering coefficient {coef} improves performance")
            elif acc_change > -0.01:
                print(f"    → Steering coefficient {coef} maintains performance")
            elif acc_change > -0.05:
                print(f"    → Steering coefficient {coef} slightly hurts performance")
            else:
                print(
                    f"    → Steering coefficient {coef} significantly hurts performance"
                )


def generate_layer_comparison_analysis(results_dict, steering_coefficients):
    """Generate analysis comparing performance across layers - CHANGED: New function for detailed analysis"""
    print("\nLAYER COMPARISON ANALYSIS")
    print("=" * 50)

    # Collect metrics across layers
    layer_metrics = {}
    for layer_idx, layer_data in results_dict.items():
        layer_metrics[layer_idx] = {}

        for coef in steering_coefficients:
            coef_key = f"coef_{coef}"
            if coef_key in layer_data:
                layer_metrics[layer_idx][coef] = layer_data[coef_key].get(
                    "accuracy", 0.0
                )

    # Find best performing layers for each coefficient
    print("\nBest Performing Layers by Steering Coefficient:")
    print("-" * 50)

    for coef in steering_coefficients:
        layer_performances = []
        for layer_idx, metrics in layer_metrics.items():
            if coef in metrics:
                layer_performances.append((layer_idx, metrics[coef]))

        if layer_performances:
            # Sort by performance (descending)
            layer_performances.sort(key=lambda x: x[1], reverse=True)
            best_layer, best_acc = layer_performances[0]
            worst_layer, worst_acc = layer_performances[-1]

            print(f"Coefficient {coef:4.1f}:")
            print(f"  Best:  Layer {best_layer} (Accuracy: {best_acc:.4f})")
            print(f"  Worst: Layer {worst_layer} (Accuracy: {worst_acc:.4f})")
            print(f"  Range: {best_acc - worst_acc:.4f}")

    # Find most steering-robust layers (least affected by steering)
    print("\nSteering Robustness Analysis:")
    print("-" * 40)

    layer_robustness = {}
    for layer_idx, metrics in layer_metrics.items():
        if 0.0 in metrics:  # Has baseline
            baseline = metrics[0.0]
            deviations = []

            for coef in steering_coefficients:
                if coef != 0.0 and coef in metrics:
                    deviation = abs(metrics[coef] - baseline)
                    deviations.append(deviation)

            if deviations:
                avg_deviation = np.mean(deviations)
                max_deviation = max(deviations)
                layer_robustness[layer_idx] = {
                    "avg_deviation": avg_deviation,
                    "max_deviation": max_deviation,
                    "baseline": baseline,
                }

    # Sort by robustness (least deviation = most robust)
    sorted_robustness = sorted(
        layer_robustness.items(), key=lambda x: x[1]["avg_deviation"]
    )

    print("Layers ranked by robustness to steering (most robust first):")
    for layer_idx, robustness in sorted_robustness:
        print(
            f"  Layer {layer_idx}: Avg deviation = {robustness['avg_deviation']:.4f}, "
            f"Max deviation = {robustness['max_deviation']:.4f}, "
            f"Baseline = {robustness['baseline']:.4f}"
        )


def comprehensive_results_analysis(
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
    """Main function that orchestrates all analysis and plotting - CHANGED: Main entry point"""
    print("\n🔍 STARTING COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 60)

    # Convert results to proper format
    if isinstance(results, tuple) and len(results) >= 1:
        results = results[0]

    results_dict = {}
    if isinstance(results, (list, tuple)):
        for i, layer_result in enumerate(results):
            if "layer_idx" in layer_result:
                results_dict[layer_result["layer_idx"]] = layer_result
            else:
                results_dict[i] = layer_result
    else:
        results_dict = results

    # 1. Generate all plots and visualizations
    print("📊 Generating plots and visualizations...")
    plot_paths = generate_comprehensive_analysis_and_plots(
        results=results,
        steering_coefficients=steering_coefficients,
        model_name=model_name,
        model_family=model_family,
        model_variant=model_variant,
        run_dir=run_dir,
        layer_data=layer_data,
        all_strategy_data=all_strategy_data,
        all_steering_vectors=all_steering_vectors,
    )

    # 2. Analyze steering effectiveness
    print("🎯 Analyzing steering effectiveness...")
    effectiveness_analysis = analyze_steering_effectiveness(
        results_dict, steering_coefficients
    )
    print_effectiveness_summary(effectiveness_analysis)

    # 3. Generate layer comparison analysis
    print("🔄 Generating layer comparison analysis...")
    generate_layer_comparison_analysis(results_dict, steering_coefficients)

    # 4. Generate summary statistics
    print("📈 Summary Statistics:")
    print("-" * 30)

    total_layers = len(results_dict)
    total_coefficients = len(steering_coefficients)
    total_experiments = total_layers * total_coefficients

    print(f"Total layers analyzed: {total_layers}")
    print(f"Total coefficients tested: {total_coefficients}")
    print(f"Total experiments conducted: {total_experiments}")
    print(f"Plots generated: {len(plot_paths)}")

    if run_dir:
        print(f"All results saved to: {run_dir}")

    print("\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 60)

    return {
        "plot_paths": plot_paths,
        "effectiveness_analysis": effectiveness_analysis,
        "results_dict": results_dict,
    }
