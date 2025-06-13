import os

# Suppress specific warnings from scikit-learn PCA and numerical operations
# import warnings
# warnings.filterwarnings("ignore", message="invalid value encountered in matmul")
# warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# # Remove VisibleDeprecationWarning since it may not be available in all NumPy versions
# warnings.filterwarnings("ignore", category=FutureWarning)
# # Silence all UserWarnings (commonly used for deprecation and configuration warnings)
# warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA

# Set the style for all plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")  # Use a more vibrant color palette

# Custom color palette for better visualization
CUSTOM_PALETTE = {
    "hate": "#FF6B6B",  # Coral red
    "safe": "#4ECDC4",  # Turquoise
    "steering": "#45B7D1",  # Sky blue
    "background": "#F7F7F7",  # Light gray
    "grid": "#E0E0E0",  # Lighter gray for grid
}

# Define consistent colors and markers for each category
CATEGORY_STYLES = {
    "hate_yes": {
        "color": "#FF0000",  # Red
        "marker": "o",  # Round
        "label": "Hate Yes",
    },
    "hate_no": {
        "color": "#0000FF",  # Blue
        "marker": "o",  # Round
        "label": "Hate No",
    },
    "safe_yes": {
        "color": "#0000FF",  # Blue
        "marker": "^",  # Triangle up
        "label": "Safe Yes",
    },
    "safe_no": {
        "color": "#FF0000",  # Red
        "marker": "^",  # Triangle up
        "label": "Safe No",
    },
}


def validate_representations(representations, required_strategies=None):
    """Minimal validation - just check basic structure.

    CHANGED: Simplified validation to only check basic structure without strict data type checking.

    Args:
        representations: Dict with any structure
        required_strategies: List of required strategies, if None uses all available

    Returns:
        Tuple of (is_valid, error_message, available_strategies, available_data_types)
    """
    if not isinstance(representations, dict):
        return (
            False,
            f"representations must be dict, got {type(representations)}",
            [],
            [],
        )

    if len(representations) == 0:
        return False, "representations dictionary is empty", [], []

    # Just return True if we have some data - let individual functions handle detailed validation
    return True, "", list(representations.keys()), []


def validate_steering_vectors(steering_vectors, required_strategies=None):
    """Minimal validation - just check basic structure.

    CHANGED: Simplified validation to only check basic structure.

    Args:
        steering_vectors: Dict with any structure
        required_strategies: List of required strategies, if None uses all available

    Returns:
        Tuple of (is_valid, error_message, available_strategies, available_vector_types)
    """
    if not isinstance(steering_vectors, dict):
        return (
            False,
            f"steering_vectors must be dict, got {type(steering_vectors)}",
            [],
            [],
        )

    if len(steering_vectors) == 0:
        return False, "steering_vectors dictionary is empty", [], []

    # Just return True if we have some data - let individual functions handle detailed validation
    return True, "", list(steering_vectors.keys()), []


def apply_pca_to_data(data_arrays, normalize=True):
    """Apply PCA to combined data arrays.

    Args:
        data_arrays: List of numpy arrays to combine and apply PCA to
        normalize: Whether to normalize data before PCA

    Returns:
        Tuple of (pca_model, combined_2d_data, data_mean, data_std)
    """
    if not data_arrays:
        raise ValueError("data_arrays cannot be empty")

    # Flatten 3D arrays if needed
    flattened_arrays = []
    for i, arr in enumerate(data_arrays):
        if arr.ndim == 3:
            flattened = arr.reshape(arr.shape[0], -1)
        elif arr.ndim == 2:
            flattened = arr
        else:
            raise ValueError(f"Array {i} has unsupported shape: {arr.shape}")
        flattened_arrays.append(flattened.astype(np.float32))

    # Combine all data
    combined_data = np.vstack(flattened_arrays)

    # Normalize if requested
    if normalize:
        data_mean = np.mean(combined_data, axis=0, keepdims=True)
        data_std = np.std(combined_data, axis=0, keepdims=True) + 1e-10
        normalized_data = (combined_data - data_mean) / data_std
    else:
        normalized_data = combined_data
        data_mean = np.zeros((1, combined_data.shape[1]))
        data_std = np.ones((1, combined_data.shape[1]))

    # Apply PCA
    if normalized_data.shape[0] < 2 or normalized_data.shape[1] < 2:
        raise ValueError(f"Insufficient data for PCA: shape {normalized_data.shape}")

    # Check for zero variance
    if np.all(np.std(normalized_data, axis=0) < 1e-10):
        raise ValueError("Data has zero variance, cannot apply PCA")

    pca = PCA(n_components=2, svd_solver="full")
    data_2d = pca.fit_transform(normalized_data)

    return pca, data_2d, data_mean, data_std


def plot_performance_across_layers(results, metric="accuracy", save_path=None):
    """
    Plot performance metrics across model layers.

    CHANGED: Fixed data structure handling to prevent 'str' object has no attribute 'keys' error

    Args:
        results: List of results per layer OR dictionary with layer results
        metric: Metric to plot (e.g., "accuracy", "loss")
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # CHANGED: Better handling of different input formats
    # Convert results to a consistent format
    if isinstance(results, dict):
        # If results is a dict, convert to list format
        results_list = []
        for layer_idx in sorted(results.keys()):
            layer_result = results[layer_idx]
            if isinstance(layer_result, dict):
                # Ensure layer_idx is in the result
                if "layer_idx" not in layer_result:
                    layer_result["layer_idx"] = layer_idx
                results_list.append(layer_result)
        results = results_list

    # Validate that results is now a list
    if not isinstance(results, list):
        print(f"Error: Expected list or dict, got {type(results)}")
        return

    # Validate each item in results is a dictionary
    valid_results = []
    for i, layer_result in enumerate(results):
        if not isinstance(layer_result, dict):
            print(
                f"Warning: Layer result at index {i} is not a dictionary (type: {type(layer_result)}), skipping"
            )
            continue
        valid_results.append(layer_result)

    if not valid_results:
        print("Error: No valid layer results found")
        return

    results = valid_results

    # Extract data
    layers = []
    baseline_values = []
    coef_values = {}

    # Find all coefficients in the data
    all_coefficients = set()
    for layer_result in results:
        # CHANGED: Added validation before calling .keys()
        if not isinstance(layer_result, dict):
            continue

        for key in layer_result.keys():
            if key.startswith("coef_"):
                coef = key.split("_")[1]
                all_coefficients.add(coef)

    # Sort coefficients to ensure consistent order
    all_coefficients = sorted(all_coefficients, key=lambda x: float(x))

    for layer_result in results:
        # CHANGED: Added validation
        if not isinstance(layer_result, dict):
            continue

        layer_idx = layer_result.get("layer_idx", 0)
        layers.append(layer_idx)

        # Get baseline values (coefficient = 0.0)
        baseline_value = None
        if (
            "final_metrics" in layer_result
            and isinstance(layer_result["final_metrics"], dict)
            and "base_metrics" in layer_result["final_metrics"]
            and isinstance(layer_result["final_metrics"]["base_metrics"], dict)
        ):
            if metric in layer_result["final_metrics"]["base_metrics"]:
                baseline_value = layer_result["final_metrics"]["base_metrics"][metric]
        elif "coef_0.0" in layer_result and isinstance(layer_result["coef_0.0"], dict):
            if metric in layer_result["coef_0.0"]:
                baseline_value = layer_result["coef_0.0"][metric]
        elif "coef_0" in layer_result and isinstance(layer_result["coef_0"], dict):
            if metric in layer_result["coef_0"]:
                baseline_value = layer_result["coef_0"][metric]

        baseline_values.append(baseline_value)

        # Get values for each coefficient
        for coef in all_coefficients:
            if f"coef_{coef}" not in coef_values:
                coef_values[f"coef_{coef}"] = []

            coef_value = None
            if (
                f"coef_{coef}" in layer_result
                and isinstance(layer_result[f"coef_{coef}"], dict)
                and metric in layer_result[f"coef_{coef}"]
            ):
                coef_value = layer_result[f"coef_{coef}"][metric]

            coef_values[f"coef_{coef}"].append(coef_value)

    # Plot baseline if available
    if any(v is not None for v in baseline_values):
        plt.plot(
            layers,
            baseline_values,
            marker="o",
            linestyle="-",
            color="black",
            label="Baseline",
        )

    # Plot values for each coefficient with different colors
    colors = ["red", "green", "blue", "orange", "purple", "cyan"]
    for i, (coef_key, values) in enumerate(coef_values.items()):
        if any(v is not None for v in values):
            color_idx = i % len(colors)
            coef = coef_key.split("_")[1]
            plt.plot(
                layers,
                values,
                marker="o",
                linestyle="-",
                color=colors[color_idx],
                label=f"Coef={coef}",
            )

    # Add title and labels
    coef_list = ", ".join([c for c in all_coefficients])
    plt.title(
        f"{metric.capitalize()} Across Layers\nSteering Coefficients: {coef_list}",
        fontsize=14,
    )
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10)

    # Set x-axis to show integer ticks
    plt.xticks(layers)

    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def safe_get_dict_keys(obj, description="object"):
    """
    Safely get keys from an object, with detailed error reporting.

    CHANGED: Added helper function to prevent key access errors

    Args:
        obj: Object to get keys from
        description: Description of the object for error messages

    Returns:
        List of keys or empty list if not a dict
    """
    if isinstance(obj, dict):
        return list(obj.keys())
    else:
        raise ValueError(f"Expected {description} to be dict, got {type(obj)}")


def plot_all_layer_vectors(results, save_dir):
    """Plot all layer vectors in a grid.

    Args:
        results: List of layer results
        save_dir: Directory to save the plot.
        The plot will be saved as "{save_dir}/all_layer_vectors.png"

    Returns:
        Path to the saved plot or None if unsuccessful
    """
    # Validate input
    if not results:
        print("Error: No results provided")
        return None

    # Filter valid results with all required keys
    valid_results = []
    for i, layer_data in enumerate(results):
        if not isinstance(layer_data, dict):
            print(
                f"Warning: Layer data at index {i} is not a dictionary. Type: {type(layer_data)}"
            )
            continue

        # Check for required keys
        required_keys = ["hate_mean_vector", "safe_mean_vector", "steering_vector"]
        if not all(key in layer_data for key in required_keys):
            available_keys = (
                list(layer_data.keys()) if isinstance(layer_data, dict) else "N/A"
            )
            print(
                f"Warning: Layer data at index {i} missing required keys. Available keys: {available_keys}"
            )
            continue

        valid_results.append(layer_data)

    # Check if we have any valid results
    if not valid_results:
        print("Error: No valid layer data found. Cannot create plot.")
        return None

    # Create grid layout
    n_layers = len(valid_results)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    # Handle single subplot case
    if n_layers == 1 and n_rows == 1 and n_cols == 1:
        fig, ax = plt.subplots(figsize=(7, 6))
        axes = [ax]  # Create a list with a single axis for consistency
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        # Convert to 1D array for easier indexing
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]  # Handle case of single axis

    for i, layer_data in enumerate(valid_results):
        if i >= len(axes):
            print(f"Warning: Not enough axes for all layers. Skipping layer {i}")
            continue

        # Extract vectors and ensure correct shapes
        hate_mean = layer_data["hate_mean_vector"]
        safe_mean = layer_data["safe_mean_vector"]
        steering = layer_data["steering_vector"]

        # Validate vector shapes
        if not isinstance(hate_mean, np.ndarray):
            print(
                f"Error: hate_mean_vector for layer {i} is not a numpy array. Type: {type(hate_mean)}"
            )
            axes[i].text(0.5, 0.5, "Invalid hate_mean_vector", ha="center", va="center")
            axes[i].set_title(f"Layer {i}")
            continue

        if not isinstance(safe_mean, np.ndarray):
            print(
                f"Error: safe_mean_vector for layer {i} is not a numpy array. Type: {type(safe_mean)}"
            )
            axes[i].text(0.5, 0.5, "Invalid safe_mean_vector", ha="center", va="center")
            axes[i].set_title(f"Layer {i}")
            continue

        if not isinstance(steering, np.ndarray):
            print(
                f"Error: steering_vector for layer {i} is not a numpy array. Type: {type(steering)}"
            )
            axes[i].text(0.5, 0.5, "Invalid steering_vector", ha="center", va="center")
            axes[i].set_title(f"Layer {i}")
            continue

        # Print shapes for debugging
        print(
            f"Layer {i} shapes - hate_mean: {hate_mean.shape}, safe_mean: {safe_mean.shape}, steering: {steering.shape}"
        )

        # Ensure vectors are properly shaped for PCA
        if hate_mean.ndim > 1:
            hate_mean = hate_mean.reshape(-1)
        if safe_mean.ndim > 1:
            safe_mean = safe_mean.reshape(-1)
        if steering.ndim > 1:
            steering = steering.reshape(-1)

        # Ensure all vectors have the same shape
        min_dim = min(hate_mean.shape[0], safe_mean.shape[0], steering.shape[0])
        hate_mean = hate_mean[:min_dim]
        safe_mean = safe_mean[:min_dim]
        steering = steering[:min_dim]

        # Check for NaN or Inf values
        if (
            np.isnan(hate_mean).any()
            or np.isnan(safe_mean).any()
            or np.isnan(steering).any()
        ):
            print(f"Error: NaN values detected in vectors for layer {i}")
            axes[i].text(0.5, 0.5, "NaN values in vectors", ha="center", va="center")
            axes[i].set_title(f"Layer {i}")
            continue

        if (
            np.isinf(hate_mean).any()
            or np.isinf(safe_mean).any()
            or np.isinf(steering).any()
        ):
            print(f"Error: Infinite values detected in vectors for layer {i}")
            axes[i].text(
                0.5, 0.5, "Infinite values in vectors", ha="center", va="center"
            )
            axes[i].set_title(f"Layer {i}")
            continue

        # Stack for PCA
        vectors = np.vstack([hate_mean, safe_mean, steering])

        # Apply PCA using the utility function
        pca, vectors_2d, _, _ = apply_pca_to_data([vectors])

        # Plot the vectors
        ax = axes[i]
        for idx, (label, color) in enumerate(
            zip(
                ["Hate Mean", "Safe Mean", "Steering Vector"],
                ["#FF0000", "#0000FF", "#00FF00"],
            )
        ):
            ax.scatter(
                vectors_2d[idx, 0],
                vectors_2d[idx, 1],
                c=color,
                s=100,
                label=label,
                alpha=0.8,
            )
            ax.text(
                vectors_2d[idx, 0],
                vectors_2d[idx, 1],
                label,
                fontsize=10,
                ha="center",
                va="bottom",
            )

        # Add arrows to show direction
        ax.arrow(
            vectors_2d[0, 0],
            vectors_2d[0, 1],
            vectors_2d[2, 0] * 0.8,
            vectors_2d[2, 1] * 0.8,
            head_width=0.05,
            head_length=0.1,
            fc="#00FF00",
            ec="#00FF00",
            alpha=0.6,
        )

        # Set title and legend
        ax.set_title(f"Layer {layer_data.get('layer_idx', i)}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

    # Hide empty subplots
    for i in range(len(valid_results), len(axes)):
        axes[i].set_visible(False)

    # Set overall title
    plt.suptitle("Layer Vector Representations", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust for suptitle

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "all_layer_vectors.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved layer vectors plot to {save_path}")
    return save_path


def plot_all_decision_boundaries(
    representations,
    steering_vectors,
    ccs_models,
    save_dir,
    strategies=None,
    max_samples_per_type=50,
):
    """Plot decision boundaries for all layers and strategies.

    CHANGED: Fixed to work with actual data structure - representations[strategy] -> Dict[data_type] -> numpy array.

    Args:
        representations: Dict[strategy] -> Dict[data_type] -> numpy array (for one layer)
        steering_vectors: Dict[strategy] -> Dict[vector_type] -> {"vector": array, "color": str, "label": str}
        ccs_models: Dict[strategy] -> {layer_idx: CCS model}
        save_dir: Directory to save plots
        strategies: List of strategies to plot, if None uses all available
        max_samples_per_type: Maximum samples to plot per data type

    Returns:
        Path to the saved plot or None if unsuccessful
    """
    # Check if we have data
    if not representations or not steering_vectors or not ccs_models:
        print("Error: Missing required data for decision boundaries")
        return None

    available_strategies = list(representations.keys())
    if strategies is None:
        strategies = available_strategies

    # Filter to strategies we actually have
    strategies = [
        s
        for s in strategies
        if s in representations and s in steering_vectors and s in ccs_models
    ]

    if not strategies:
        print("Error: No valid strategies found")
        return None

    # Get layer indices from CCS models
    first_strategy = strategies[0]
    layer_indices = sorted(ccs_models[first_strategy].keys())

    if not layer_indices:
        print("Error: No layer indices found in CCS models")
        return None

    # For simplicity, just use first layer for this plot
    layer_idx = layer_indices[0]

    # Create subplot grid: strategies as columns
    n_strategies = len(strategies)
    fig, axes = plt.subplots(1, n_strategies, figsize=(6 * n_strategies, 5))

    # Handle single subplot case
    if n_strategies == 1:
        axes = [axes]

    # Plot decision boundaries for each strategy
    for strategy_idx, strategy in enumerate(strategies):
        ax = axes[strategy_idx]

        strategy_reps = representations[strategy]
        strategy_steering = steering_vectors[strategy]

        # Check if we have CCS model for this strategy and layer
        if layer_idx not in ccs_models[strategy]:
            ax.text(
                0.5,
                0.5,
                f"No CCS model\nfor layer {layer_idx}",
                ha="center",
                va="center",
            )
            ax.set_title(f"{strategy} - Layer {layer_idx}")
            continue

        ccs_model = ccs_models[strategy][layer_idx]

        # Get data for this strategy
        data_arrays = []
        data_types = ["hate_yes", "hate_no", "safe_yes", "safe_no"]

        missing_types = []
        for data_type in data_types:
            if data_type not in strategy_reps:
                missing_types.append(data_type)
                continue

            data = strategy_reps[data_type]

            # Limit samples for performance
            if len(data) > max_samples_per_type:
                indices = np.random.choice(
                    len(data), max_samples_per_type, replace=False
                )
                data = data[indices]

            data_arrays.append(data)

        if missing_types:
            ax.text(
                0.5,
                0.5,
                f"Missing data types:\n{missing_types}",
                ha="center",
                va="center",
            )
            ax.set_title(f"{strategy} - Layer {layer_idx}")
            continue

        if len(data_arrays) != 4:
            ax.text(0.5, 0.5, "Insufficient data arrays", ha="center", va="center")
            ax.set_title(f"{strategy} - Layer {layer_idx}")
            continue

        # Apply PCA to combined data
        pca, combined_2d, data_mean, data_std = apply_pca_to_data(data_arrays)

        # Split back into individual data types
        start_idx = 0
        data_2d_by_type = {}
        for i, data_type in enumerate(data_types):
            end_idx = start_idx + len(data_arrays[i])
            data_2d_by_type[data_type] = combined_2d[start_idx:end_idx]
            start_idx = end_idx

        # Create decision boundary grid
        x_min, x_max = combined_2d[:, 0].min() - 1, combined_2d[:, 0].max() + 1
        y_min, y_max = combined_2d[:, 1].min() - 1, combined_2d[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Transform grid points back to original space
        grid_original = pca.inverse_transform(grid_points)
        grid_original = (grid_original * data_std) + data_mean

        # Get predictions from CCS model
        if hasattr(ccs_model, "predict_from_vectors"):
            grid_preds, grid_probs = ccs_model.predict_from_vectors(grid_original)
        else:
            # Fallback
            grid_tensor = torch.tensor(grid_original, dtype=torch.float32)
            if hasattr(ccs_model, "device"):
                grid_tensor = grid_tensor.to(ccs_model.device)

            with torch.no_grad():
                if hasattr(ccs_model, "probe"):
                    grid_probs = (
                        torch.sigmoid(ccs_model.probe(grid_tensor)).cpu().numpy()
                    )
                else:
                    grid_probs = torch.sigmoid(ccs_model(grid_tensor)).cpu().numpy()

            grid_preds = (grid_probs > 0.5).astype(int)

        # Reshape for contour plot
        Z_probs = grid_probs.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z_probs, cmap="RdBu_r", alpha=0.3, levels=20)
        ax.contour(
            xx,
            yy,
            Z_probs,
            levels=[0.5],
            colors="black",
            linestyles="dashed",
            linewidths=2,
        )

        # Plot data points for each type
        for data_type, style in CATEGORY_STYLES.items():
            if data_type in data_2d_by_type:
                points_2d = data_2d_by_type[data_type]
                ax.scatter(
                    points_2d[:, 0],
                    points_2d[:, 1],
                    c=style["color"],
                    marker=style["marker"],
                    s=30,
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=0.5,
                    label=style["label"],
                )

        # Set labels and title
        ax.set_title(f"{strategy} - Layer {layer_idx}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")

        # Add legend only to first subplot
        if strategy_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"layer_{layer_idx}_all_decision_boundaries.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved decision boundaries plot to {save_path}")
    return save_path


def plot_all_strategies_all_steering_vectors(
    plot_dir,
    layer_idx,
    representations,
    all_steering_vectors_by_strategy,
):
    """
    Create a comprehensive visualization of all steering vectors for all embedding strategies.

    CHANGED: Fixed to work with actual data structure - representations[strategy] -> Dict[data_type] -> numpy array.

    Args:
        plot_dir: Directory to save the plots
        layer_idx: Layer index
        representations: Dict[strategy] -> Dict[data_type] -> numpy array (for one layer)
        all_steering_vectors_by_strategy: Dict[strategy] -> Dict[vector_type] -> {"vector": array, "color": str, "label": str}

    Returns:
        Path to the saved plot or None if unsuccessful
    """
    # Check if we have any data
    if not representations or not all_steering_vectors_by_strategy:
        print("Error: No data provided for plotting")
        return None

    # Get available strategies
    available_strategies = [
        s
        for s in representations.keys()
        if s in all_steering_vectors_by_strategy.keys()
    ]

    if not available_strategies:
        print("Error: No matching strategies found")
        return None

    # Define transformations that we expect
    transformations = [
        "combined",
        "hate_yes_to_safe_yes",
        "safe_no_to_hate_no",
        "hate_yes_to_hate_no",
        "safe_yes_to_safe_no",
    ]

    # Filter to only transformations available for all strategies
    available_transformations = []
    for trans in transformations:
        all_have_trans = True
        for strategy in available_strategies:
            if trans not in all_steering_vectors_by_strategy[strategy]:
                all_have_trans = False
                break
        if all_have_trans:
            available_transformations.append(trans)

    if not available_transformations:
        print("Warning: No common transformations available, using first available")
        # Just use whatever transformations we have from first strategy
        first_strategy = available_strategies[0]
        available_transformations = list(
            all_steering_vectors_by_strategy[first_strategy].keys()
        )[:3]  # Limit to 3

    # Create readable titles for transformations
    transformation_titles = {
        "combined": "(Hate Yes + Safe No) → (Safe Yes + Hate No)",
        "hate_yes_to_safe_yes": "Hate Yes → Safe Yes",
        "safe_no_to_hate_no": "Safe No → Hate No",
        "hate_yes_to_hate_no": "Hate Yes → Hate No",
        "safe_yes_to_safe_no": "Safe Yes → Safe No",
    }

    # Create figure with subplots - strategies as rows, transformations as columns
    fig, axes = plt.subplots(
        len(available_strategies),
        len(available_transformations),
        figsize=(6 * len(available_transformations), 5 * len(available_strategies)),
    )

    # Handle single subplot cases
    if len(available_strategies) == 1 and len(available_transformations) == 1:
        axes = np.array([[axes]])
    elif len(available_strategies) == 1:
        axes = np.array([axes])
    elif len(available_transformations) == 1:
        axes = np.array([[ax] for ax in axes])

    # Process each strategy and transformation
    for row_idx, strategy in enumerate(available_strategies):
        # Get representations for this strategy
        strategy_reps = representations[strategy]

        # Get steering vectors for this strategy
        strategy_steering_vectors = all_steering_vectors_by_strategy[strategy]

        # For each transformation
        for col_idx, trans_name in enumerate(available_transformations):
            ax = axes[row_idx][col_idx]

            # Check if we have required data types
            required_types = ["hate_yes", "hate_no", "safe_yes", "safe_no"]
            missing_types = [t for t in required_types if t not in strategy_reps]

            if missing_types:
                ax.text(
                    0.5,
                    0.5,
                    f"Missing data types:\n{missing_types}",
                    ha="center",
                    va="center",
                )
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            # Check if we have steering vector for this transformation
            if trans_name not in strategy_steering_vectors:
                ax.text(
                    0.5,
                    0.5,
                    f"Missing steering vector:\n{trans_name}",
                    ha="center",
                    va="center",
                )
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            # Get transformation-specific data
            if trans_name == "combined":
                source_data = np.vstack(
                    [strategy_reps["hate_yes"], strategy_reps["safe_no"]]
                )
                source_types = ["hate_yes"] * len(strategy_reps["hate_yes"]) + [
                    "safe_no"
                ] * len(strategy_reps["safe_no"])
                target_data = np.vstack(
                    [strategy_reps["safe_yes"], strategy_reps["hate_no"]]
                )
                target_types = ["safe_yes"] * len(strategy_reps["safe_yes"]) + [
                    "hate_no"
                ] * len(strategy_reps["hate_no"])
                use_combined = True
            elif trans_name == "hate_yes_to_safe_yes":
                source_data = strategy_reps["hate_yes"]
                source_type = "hate_yes"
                target_data = strategy_reps["safe_yes"]
                target_type = "safe_yes"
                use_combined = False
            elif trans_name == "safe_no_to_hate_no":
                source_data = strategy_reps["safe_no"]
                source_type = "safe_no"
                target_data = strategy_reps["hate_no"]
                target_type = "hate_no"
                use_combined = False
            elif trans_name == "hate_yes_to_hate_no":
                source_data = strategy_reps["hate_yes"]
                source_type = "hate_yes"
                target_data = strategy_reps["hate_no"]
                target_type = "hate_no"
                use_combined = False
            elif trans_name == "safe_yes_to_safe_no":
                source_data = strategy_reps["safe_yes"]
                source_type = "safe_yes"
                target_data = strategy_reps["safe_no"]
                target_type = "safe_no"
                use_combined = False
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Unknown transformation:\n{trans_name}",
                    ha="center",
                    va="center",
                )
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            # Get steering vector
            steering_vector_data = strategy_steering_vectors[trans_name]
            steering_vector = steering_vector_data["vector"]
            steering_color = steering_vector_data["color"]

            # Apply PCA to combined source and target data
            pca, combined_2d, data_mean, data_std = apply_pca_to_data(
                [source_data, target_data]
            )

            # Split back into source and target
            n_source = len(source_data)
            source_2d = combined_2d[:n_source]
            target_2d = combined_2d[n_source:]

            # Transform means and steering vector to PCA space
            source_mean = np.mean(source_data, axis=0)
            target_mean = np.mean(target_data, axis=0)

            source_mean_norm = (source_mean - data_mean[0]) / data_std[0]
            target_mean_norm = (target_mean - data_mean[0]) / data_std[0]
            steering_vector_norm = (steering_vector - data_mean[0]) / data_std[0]

            source_mean_2d = pca.transform(source_mean_norm.reshape(1, -1))[0]
            target_mean_2d = pca.transform(target_mean_norm.reshape(1, -1))[0]

            # Plot data points
            if use_combined:
                # For combined case with multiple types
                for category_type in set(source_types):
                    indices = [
                        i for i, t in enumerate(source_types) if t == category_type
                    ]
                    style = CATEGORY_STYLES[category_type]
                    ax.scatter(
                        source_2d[indices, 0],
                        source_2d[indices, 1],
                        color=style["color"],
                        alpha=0.6,
                        s=40,
                        marker=style["marker"],
                        edgecolors="black",
                        linewidths=0.5,
                        label=f"{style['label']} (source)",
                    )

                for category_type in set(target_types):
                    indices = [
                        i for i, t in enumerate(target_types) if t == category_type
                    ]
                    style = CATEGORY_STYLES[category_type]
                    ax.scatter(
                        target_2d[indices, 0],
                        target_2d[indices, 1],
                        color=style["color"],
                        alpha=0.6,
                        s=40,
                        marker=style["marker"],
                        edgecolors="white",
                        linewidths=0.5,
                        label=f"{style['label']} (target)",
                    )
            else:
                # For single-category transformations
                style_source = CATEGORY_STYLES[source_type]
                style_target = CATEGORY_STYLES[target_type]

                ax.scatter(
                    source_2d[:, 0],
                    source_2d[:, 1],
                    color=style_source["color"],
                    alpha=0.6,
                    s=40,
                    marker=style_source["marker"],
                    edgecolors="black",
                    linewidths=0.5,
                    label=f"{style_source['label']} (source)",
                )

                ax.scatter(
                    target_2d[:, 0],
                    target_2d[:, 1],
                    color=style_target["color"],
                    alpha=0.6,
                    s=40,
                    marker=style_target["marker"],
                    edgecolors="white",
                    linewidths=0.5,
                    label=f"{style_target['label']} (target)",
                )

            # Plot source and target centroids
            ax.scatter(
                source_mean_2d[0],
                source_mean_2d[1],
                color="black",
                s=100,
                marker="*",
                label="Source Mean",
            )
            ax.scatter(
                target_mean_2d[0],
                target_mean_2d[1],
                color="white",
                s=100,
                marker="*",
                edgecolors="black",
                label="Target Mean",
            )

            # Plot steering vector as arrow from source to target mean
            vector_scale = 0.5
            ax.arrow(
                source_mean_2d[0],
                source_mean_2d[1],
                (target_mean_2d[0] - source_mean_2d[0]) * vector_scale,
                (target_mean_2d[1] - source_mean_2d[1]) * vector_scale,
                head_width=0.2,
                head_length=0.3,
                fc=steering_color,
                ec=steering_color,
                linewidth=2,
                length_includes_head=True,
                label="Direction",
            )

            # Set title and labels
            ax.set_title(
                f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
            )
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(
        plot_dir, f"layer_{layer_idx}_all_strategies_all_steering_vectors.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comprehensive steering vector plot to {output_path}")
    return output_path


def visualize_decision_boundary(
    ccs,
    hate_vectors,
    safe_vectors,
    steering_vector,
    log_base=None,
    layer_idx=None,
    strategy=None,
    steering_coefficient=None,
    pair_type=None,
):
    """Visualize the decision boundary of the CCS probe in the steering vector direction.

    CHANGED: Removed all dummy fallbacks and added proper validation.

    Args:
        ccs: CCS probe with predict_from_vectors method
        hate_vectors: Vectors for hate content (combined hate_yes + safe_no)
        safe_vectors: Vectors for safe content (combined safe_yes + hate_no)
        steering_vector: The calculated steering vector
        log_base: Base path for saving the plot
        layer_idx: Optional layer index for title
        strategy: Optional embedding strategy (last-token, first-token, mean)
        steering_coefficient: Optional steering coefficient value
        pair_type: Optional data pair type
    """
    # Validate inputs
    if not hasattr(ccs, "predict_from_vectors"):
        raise ValueError("CCS model must have predict_from_vectors method")

    if not isinstance(hate_vectors, np.ndarray) or hate_vectors.size == 0:
        raise ValueError("hate_vectors must be non-empty numpy array")

    if not isinstance(safe_vectors, np.ndarray) or safe_vectors.size == 0:
        raise ValueError("safe_vectors must be non-empty numpy array")

    if not isinstance(steering_vector, np.ndarray) or steering_vector.size == 0:
        raise ValueError("steering_vector must be non-empty numpy array")

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

    # Ensure arrays are float32 and properly shaped
    hate_vectors = hate_vectors.astype(np.float32)
    safe_vectors = safe_vectors.astype(np.float32)
    steering_vector = steering_vector.astype(np.float32)

    # Flatten if needed
    if hate_vectors.ndim == 3:
        hate_vectors = hate_vectors.reshape(hate_vectors.shape[0], -1)
    if safe_vectors.ndim == 3:
        safe_vectors = safe_vectors.reshape(safe_vectors.shape[0], -1)
    if steering_vector.ndim > 1:
        steering_vector = steering_vector.reshape(-1)

    # Normalize steering vector
    steering_norm = np.linalg.norm(steering_vector)
    if steering_norm <= 1e-10:
        raise ValueError("Steering vector has zero norm")
    steering_vector = steering_vector / steering_norm

    # Apply PCA to combined data for visualization
    pca, X_2d, data_mean, data_std = apply_pca_to_data([hate_vectors, safe_vectors])

    # Split back into hate and safe
    n_hate = len(hate_vectors)
    hate_2d = X_2d[:n_hate]
    safe_2d = X_2d[n_hate:]

    # Create labels
    labels = np.concatenate([np.zeros(n_hate), np.ones(len(safe_vectors))])

    # Create grid for decision boundary
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Transform grid points back to original space
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_points)
    grid_original = (grid_original * data_std) + data_mean

    # Get predictions on grid
    grid_preds, grid_confidences = ccs.predict_from_vectors(
        grid_original.astype(np.float32)
    )
    grid_preds = grid_preds.reshape(xx.shape)
    grid_confidences = grid_confidences.reshape(xx.shape)

    # Plot decision boundary
    contour_fill = ax.contourf(
        xx, yy, grid_preds, alpha=0.6, cmap="RdBu_r", levels=np.linspace(0, 1, 11)
    )

    # Add decision boundary line
    ax.contour(
        xx,
        yy,
        grid_confidences,
        levels=[0.5],
        colors="black",
        linestyles="dashed",
        linewidths=2,
    )

    # Add colorbar
    plt.colorbar(contour_fill, ax=ax, label="Prediction (0=Hate, 1=Safe)")

    # Plot data points
    colors = np.array(["#D7263D", "#1B98E0"])  # Red for hate, blue for safe
    for label, color in zip([0, 1], colors):
        idx = labels == label
        points = hate_2d if label == 0 else safe_2d
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=color,
            edgecolor="k",
            s=70,
            alpha=0.85,
            label="Hate (hate_yes + safe_no)"
            if label == 0
            else "Safe (safe_yes + hate_no)",
        )

    # Transform and plot steering vector
    steering_norm_pca = (steering_vector - data_mean[0]) / data_std[0]
    steering_2d = pca.transform(steering_norm_pca.reshape(1, -1))[0]

    # Scale for visibility
    scale = 0.3 * min(x_max - x_min, y_max - y_min)
    ax.arrow(
        0,
        0,
        steering_2d[0] * scale,
        steering_2d[1] * scale,
        color="black",
        width=0.01,
        head_width=0.1,
        head_length=0.1,
        length_includes_head=True,
        label="Steering Direction",
    )

    # Build title
    title_parts = ["CCS Probe Decision Boundary"]
    if layer_idx is not None:
        title_parts.append(f"Layer {layer_idx}")
    if strategy is not None:
        title_parts.append(f"Strategy: {strategy}")
    if steering_coefficient is not None:
        title_parts.append(f"Coef: {steering_coefficient}")
    if pair_type is not None:
        title_parts.append(f"Pair: {pair_type}")

    plt.title(" | ".join(title_parts), fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save if path provided
    if log_base:
        filename = f"{log_base}.png"
        if pair_type:
            filename = f"{log_base}_{pair_type}.png"
        if steering_coefficient is not None:
            filename = filename.replace(".png", f"_coef_{steering_coefficient}.png")

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        return filename

    return fig


def visualize_detailed_decision_boundary(
    ccs,
    hate_yes_vectors,
    hate_no_vectors,
    safe_yes_vectors,
    safe_no_vectors,
    steering_vector,
    layer_idx,
    log_base=None,
    strategy=None,
    steering_coefficient=None,
    pair_type=None,
):
    """Create a detailed decision boundary visualization that shows all four data types.

    CHANGED: Removed all dummy fallbacks and added proper validation.

    Args:
        ccs: CCS probe with predict_from_vectors method
        hate_yes_vectors: Vectors for hate content with "yes"
        hate_no_vectors: Vectors for hate content with "no"
        safe_yes_vectors: Vectors for safe content with "yes"
        safe_no_vectors: Vectors for safe content with "no"
        steering_vector: The calculated steering vector
        layer_idx: Layer index for title
        log_base: Base path for saving the plot
        strategy: Optional embedding strategy
        steering_coefficient: Optional steering coefficient value
        pair_type: Optional data pair type
    """
    # Validate inputs
    if not hasattr(ccs, "predict_from_vectors"):
        raise ValueError("CCS model must have predict_from_vectors method")

    required_arrays = [
        ("hate_yes_vectors", hate_yes_vectors),
        ("hate_no_vectors", hate_no_vectors),
        ("safe_yes_vectors", safe_yes_vectors),
        ("safe_no_vectors", safe_no_vectors),
        ("steering_vector", steering_vector),
    ]

    for name, arr in required_arrays:
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            raise ValueError(f"{name} must be non-empty numpy array")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Process arrays
    arrays = [hate_yes_vectors, hate_no_vectors, safe_yes_vectors, safe_no_vectors]
    processed_arrays = []

    for arr in arrays:
        arr = arr.astype(np.float32)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        processed_arrays.append(arr)

    # Normalize steering vector
    steering_vector = steering_vector.astype(np.float32)
    if steering_vector.ndim > 1:
        steering_vector = steering_vector.reshape(-1)

    steering_norm = np.linalg.norm(steering_vector)
    if steering_norm <= 1e-10:
        raise ValueError("Steering vector has zero norm")
    steering_vector = steering_vector / steering_norm

    # Apply PCA
    pca, combined_2d, data_mean, data_std = apply_pca_to_data(processed_arrays)

    # Split back into categories
    start_idx = 0
    categories_2d = {}
    category_names = ["hate_yes", "hate_no", "safe_yes", "safe_no"]

    for i, name in enumerate(category_names):
        end_idx = start_idx + len(processed_arrays[i])
        categories_2d[name] = combined_2d[start_idx:end_idx]
        start_idx = end_idx

    # Create labels (0=hate_yes, 1=hate_no, 2=safe_yes, 3=safe_no)
    labels = []
    for i, name in enumerate(category_names):
        labels.extend([i] * len(processed_arrays[i]))
    labels = np.array(labels)

    # Create decision boundary grid
    x_min, x_max = combined_2d[:, 0].min() - 1, combined_2d[:, 0].max() + 1
    y_min, y_max = combined_2d[:, 1].min() - 1, combined_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Transform grid to original space
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_points)
    grid_original = (grid_original * data_std) + data_mean

    # Get predictions
    grid_preds, grid_confidences = ccs.predict_from_vectors(
        grid_original.astype(np.float32)
    )
    grid_preds = grid_preds.reshape(xx.shape)
    grid_confidences = grid_confidences.reshape(xx.shape)

    # Plot decision boundary
    contour_fill = ax.contourf(
        xx, yy, grid_preds, alpha=0.5, cmap="RdBu_r", levels=np.linspace(0, 1, 11)
    )
    ax.contour(
        xx,
        yy,
        grid_confidences,
        levels=[0.5],
        colors="black",
        linestyles="dashed",
        linewidths=2,
    )

    # Add colorbar
    plt.colorbar(contour_fill, ax=ax, label="Prediction (0=Hate, 1=Safe)")

    # Plot each category
    category_colors = {
        "hate_yes": "#FF0000",  # Red
        "hate_no": "#0000FF",  # Blue
        "safe_yes": "#00CCFF",  # Light Blue
        "safe_no": "#FF00CC",  # Pink
    }

    category_markers = {
        "hate_yes": "o",  # Circle
        "hate_no": "s",  # Square
        "safe_yes": "^",  # Triangle up
        "safe_no": "v",  # Triangle down
    }

    for name, points_2d in categories_2d.items():
        ax.scatter(
            points_2d[:, 0],
            points_2d[:, 1],
            c=category_colors[name],
            marker=category_markers[name],
            s=70,
            alpha=0.85,
            edgecolor="black",
            linewidths=0.5,
            label=CATEGORY_STYLES[name]["label"],
        )

    # Transform and plot steering vector
    steering_norm_pca = (steering_vector - data_mean[0]) / data_std[0]
    steering_2d = pca.transform(steering_norm_pca.reshape(1, -1))[0]

    scale = 0.3 * min(x_max - x_min, y_max - y_min)
    ax.arrow(
        0,
        0,
        steering_2d[0] * scale,
        steering_2d[1] * scale,
        color="black",
        width=0.01,
        head_width=0.1,
        head_length=0.1,
        length_includes_head=True,
        label="Steering Direction",
    )

    # Build title
    title_parts = ["Detailed CCS Decision Boundary", f"Layer {layer_idx}"]
    if strategy is not None:
        title_parts.append(f"Strategy: {strategy}")
    if steering_coefficient is not None:
        title_parts.append(f"Coef: {steering_coefficient}")
    if pair_type is not None:
        title_parts.append(f"Pair: {pair_type}")

    plt.title(" | ".join(title_parts), fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save if path provided
    if log_base:
        filename = f"{log_base}_detailed_decision_boundary.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        return filename

    return fig
