import os

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


def plot_performance_across_layers(results, metric="accuracy", save_path=None):
    """
    Plot performance metrics across model layers.

    Args:
        results: List of results per layer
        metric: Metric to plot (e.g., "accuracy", "loss")
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Extract data
    layers = []
    baseline_values = []
    coef_values = {}

    # Find all coefficients in the data
    all_coefficients = set()
    for layer_result in results:
        for key in layer_result.keys():
            if key.startswith("coef_"):
                coef = key.split("_")[1]
                all_coefficients.add(coef)

    # Sort coefficients to ensure consistent order
    all_coefficients = sorted(all_coefficients, key=lambda x: float(x))

    for layer_result in results:
        layer_idx = layer_result["layer_idx"]
        layers.append(layer_idx)

        # Get baseline values (coefficient = 0.0)
        if (
            "final_metrics" in layer_result
            and "base_metrics" in layer_result["final_metrics"]
        ):
            if metric in layer_result["final_metrics"]["base_metrics"]:
                baseline_values.append(
                    layer_result["final_metrics"]["base_metrics"][metric]
                )
            else:
                baseline_values.append(None)
        else:
            baseline_values.append(None)

        # Get values for each coefficient
        for coef in all_coefficients:
            if f"coef_{coef}" not in coef_values:
                coef_values[f"coef_{coef}"] = []

            if (
                f"coef_{coef}" in layer_result
                and metric in layer_result[f"coef_{coef}"]
            ):
                coef_values[f"coef_{coef}"].append(layer_result[f"coef_{coef}"][metric])
            else:
                coef_values[f"coef_{coef}"].append(None)

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


def plot_all_layer_vectors(results, save_dir):
    """Plot all layer vectors in a grid.

    Args:
        results: List of layer results
        save_dir: Directory to save the plot.
        The plot will be saved as "{save_dir}/all_layer_vectors.png"

    Returns:
        Path to the saved plot or None if unsuccessful
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    # Check if results is valid
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

        # Standardize before PCA
        vectors_mean = np.mean(vectors, axis=0, keepdims=True)
        vectors_std = (
            np.std(vectors, axis=0, keepdims=True) + 1e-10
        )  # Avoid divide by zero
        vectors_norm = (vectors - vectors_mean) / vectors_std

        # Check if PCA can be performed
        if vectors_norm.shape[0] < 2 or vectors_norm.shape[1] < 2:
            print(
                f"Error: Insufficient data for PCA for layer {i}: shape {vectors_norm.shape}"
            )
            axes[i].text(
                0.5, 0.5, "Insufficient data for PCA", ha="center", va="center"
            )
            axes[i].set_title(f"Layer {i}")
            continue

        # Check if data has sufficient variance
        if np.all(np.abs(np.std(vectors_norm, axis=0)) < 1e-6):
            print(f"Error: Insufficient variance in data for layer {i}")
            axes[i].text(0.5, 0.5, "Insufficient variance", ha="center", va="center")
            axes[i].set_title(f"Layer {i}")
            continue

        # Perform PCA - if it fails, handle it gracefully without try-except
        if not np.all(np.isfinite(vectors_norm)):
            print(f"Error: Non-finite values after normalization for layer {i}")
            axes[i].text(
                0.5,
                0.5,
                "Non-finite values after normalization",
                ha="center",
                va="center",
            )
            axes[i].set_title(f"Layer {i}")
            continue

        # Run PCA
        pca = PCA(n_components=2, svd_solver="full")
        vectors_2d = pca.fit_transform(vectors_norm)

        # Check if PCA output is valid
        if not np.all(np.isfinite(vectors_2d)):
            print(f"Error: PCA produced non-finite values for layer {i}")
            axes[i].text(
                0.5, 0.5, "PCA produced non-finite values", ha="center", va="center"
            )
            axes[i].set_title(f"Layer {i}")
            continue

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

    Args:
        ccs: CCS probe
        hate_vectors: Vectors for hate content
        safe_vectors: Vectors for safe content
        steering_vector: The calculated steering vector
        log_base: Base path for saving the plot.
        layer_idx: Optional layer index for title
        strategy: Optional embedding strategy (last-token, first-token, mean)
        steering_coefficient: Optional steering coefficient value
        pair_type: Optional data pair type (e.g., hate_yes_to_safe_yes)
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

    # Make sure inputs are numpy arrays for plotting
    if isinstance(hate_vectors, torch.Tensor):
        hate_vectors = hate_vectors.detach().cpu().numpy()
    if isinstance(safe_vectors, torch.Tensor):
        safe_vectors = safe_vectors.detach().cpu().numpy()
    if isinstance(steering_vector, torch.Tensor):
        steering_vector = steering_vector.detach().cpu().numpy()

    # Ensure arrays are float32
    hate_vectors = hate_vectors.astype(np.float32)
    safe_vectors = safe_vectors.astype(np.float32)
    steering_vector = steering_vector.astype(np.float32)

    # Normalize steering vector
    steering_norm = np.linalg.norm(steering_vector)
    if steering_norm > 1e-10:
        steering_vector = steering_vector / steering_norm

    # Ensure steering vector is flattened
    steering_vector_flat = steering_vector.flatten()

    # Combine data and make sure everything is 2D
    if len(hate_vectors.shape) == 3:
        hate_vectors_2d = hate_vectors.reshape(hate_vectors.shape[0], -1)
    else:
        hate_vectors_2d = hate_vectors

    if len(safe_vectors.shape) == 3:
        safe_vectors_2d = safe_vectors.reshape(safe_vectors.shape[0], -1)
    else:
        safe_vectors_2d = safe_vectors

    # Stack the vectors
    X_combined = np.vstack([hate_vectors_2d, safe_vectors_2d])
    labels = np.concatenate(
        [np.zeros(len(hate_vectors_2d)), np.ones(len(safe_vectors_2d))]
    )

    # Project data to 2D for visualization
    # First component: steering vector direction
    # Second component: PCA of residuals
    projections = []

    # Project onto steering vector - ensure proper shapes for dot product
    projection1 = np.array(
        [np.dot(x.flatten(), steering_vector_flat) for x in X_combined]
    )
    projections.append(projection1)

    # Compute residuals - ensure they're 2D for PCA
    residuals = X_combined - np.outer(projection1, steering_vector_flat)

    # Add small regularization to avoid numerical issues
    epsilon = 1e-8
    residuals += np.random.normal(0, epsilon, residuals.shape)

    # Find second direction (orthogonal to steering vector)
    pca = PCA(n_components=1, svd_solver="arpack")
    pca.fit(residuals)
    second_direction = pca.components_[0]

    # Project onto second direction - ensure proper shapes again
    projection2 = np.array(
        [np.dot(x.flatten(), second_direction.flatten()) for x in X_combined]
    )
    projections.append(projection2)

    # Create 2D projections
    X_2d = np.column_stack(projections)

    # Create grid for decision boundary
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Reconstruct grid points in original space
    grid_points = np.array([xx.ravel(), yy.ravel()]).T
    original_space = np.outer(grid_points[:, 0], steering_vector_flat) + np.outer(
        grid_points[:, 1], second_direction.flatten()
    )

    # Ensure original_space is float32
    original_space = original_space.astype(np.float32)

    # Predict on grid points using predict_from_vectors instead of predict
    # This avoids the issue with tokenization since we're passing vectors directly
    grid_preds, grid_confidences = ccs.predict_from_vectors(original_space)
    grid_preds = grid_preds.reshape(xx.shape)
    grid_confidences = grid_confidences.reshape(xx.shape)

    # Plot decision boundary with higher contrast
    contour_fill = ax.contourf(
        xx, yy, grid_preds, alpha=0.6, cmap="RdBu_r", levels=np.linspace(0, 1, 11)
    )

    # Add explicit decision boundary line (where probability = 0.5)
    decision_boundary = ax.contour(
        xx,
        yy,
        grid_confidences,
        levels=[0.5],
        colors="black",
        linestyles="dashed",
        linewidths=2,
    )

    # Add colorbar to show prediction confidence
    plt.colorbar(contour_fill, ax=ax, label="Prediction (0=Hate, 1=Safe)")

    # Plot data points
    colors = np.array(["#D7263D", "#1B98E0"])  # Red for hate, blue for safe
    edge_color = "k"  # Black edge for contrast

    for label, color in zip([0, 1], colors):
        idx = labels == label
        ax.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            c=color,
            edgecolor=edge_color,
            s=70,
            alpha=0.85,
            label="Hate (hate_yes + safe_no)"
            if label == 0
            else "Safe (safe_yes + hate_no)",
        )

    # Add steering vector direction
    ax.arrow(
        0,
        0,
        -1,
        0,
        color="black",
        width=0.01,
        head_width=0.1,
        head_length=0.1,
        length_includes_head=True,
        label="Steering Direction",
    )

    # Build a more detailed title with all available information
    title_parts = ["CCS Probe Decision Boundary"]

    if layer_idx is not None:
        title_parts.append(f"Layer {layer_idx}")

    if strategy is not None:
        title_parts.append(f"Strategy: {strategy}")

    if steering_coefficient is not None:
        title_parts.append(f"Coef: {steering_coefficient}")

    if pair_type is not None:
        title_parts.append(f"Pair: {pair_type}")

    # Join all parts with pipes
    title = " | ".join(title_parts)
    plt.title(title, fontsize=14)

    # Add axis labels
    plt.xlabel("Steering Direction", fontsize=12)
    plt.ylabel("Orthogonal Direction", fontsize=12)

    # Add legend
    plt.legend(fontsize=10)

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Add description box explaining the plot
    description = """Description: This plot shows the decision boundary of the CCS probe in the space defined by the steering vector and its orthogonal complement.

    Data Categories:
    - Red points (Hate): Combined category of hate_yes (hate speech with "Yes") and safe_no (safe speech with "No")
    - Blue points (Safe): Combined category of safe_yes (safe speech with "Yes") and hate_no (hate speech with "No")

    Ideal Case:
    - Clear separation between hate (red) and safe (blue) content
    - Decision boundary should be roughly perpendicular to the steering direction
    - Points should cluster into two distinct groups

    Interpretation:
    - The steering vector direction (horizontal axis) shows how content changes when steered
    - The orthogonal direction (vertical axis) shows variations that preserve the steering effect
    - The decision boundary (colored regions) shows where the probe switches between hate and safe predictions
    - A clear boundary indicates the probe can reliably distinguish between content types

    Note: The steering vector points from hate to safe direction because it's calculated as safe_mean - hate_mean.
    When applying positive steering coefficients, content moves in this direction (toward safe classification).
    """

    # Add text box with description
    plt.figtext(
        0.5,
        0.01,
        description,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        wrap=True,
    )

    # Adjust layout to make room for description
    plt.tight_layout(rect=(0, 0.15, 1, 0.95))

    # Save figure if log_base is provided
    if log_base:
        if pair_type:
            filename = f"{log_base}_{pair_type}"
            if steering_coefficient is not None:
                filename += f"_coef_{steering_coefficient}"
            filename += ".png"
        else:
            filename = f"{log_base}.png"

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    return plt.gcf()


def plot_all_decision_boundaries(layers_data, log_base=None):
    """Plot decision boundaries for all layers as subplots in a single figure.

    Args:
        layers_data: List of layer data
        log_base: Base path for saving the plot.
        Example: "plots/all_decision_boundaries" (will save as "plots/all_decision_boundaries.png")
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    # Check if layers_data is valid
    if not layers_data or not isinstance(layers_data, (list, dict)):
        print(f"Error: Invalid layers_data structure. Type: {type(layers_data)}")
        return None

    # Convert to list if it's a dict
    valid_layers = []
    if isinstance(layers_data, dict):
        for key in sorted(layers_data.keys()):
            valid_layers.append(layers_data[key])
    else:
        # Filter out invalid entries
        for i, layer in enumerate(layers_data):
            if not isinstance(layer, dict):
                print(
                    f"Warning: Layer data at index {i} is not a dictionary. Type: {type(layer)}"
                )
                continue

            # Check for required keys
            required_keys = ["ccs", "hate_vectors", "safe_vectors", "steering_vector"]
            if not all(key in layer for key in required_keys):
                available_keys = (
                    list(layer.keys()) if isinstance(layer, dict) else "N/A"
                )
                print(
                    f"Warning: Layer {i} missing required keys. Available keys: {available_keys}"
                )
                continue

            valid_layers.append(layer)

    if not valid_layers:
        print("Error: No valid layer data found. Cannot create plot.")
        return None

    n_layers = len(valid_layers)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    # Create figure and axis grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Handle case with only one subplot
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Plot each layer's decision boundary
    for i, layer_data in enumerate(valid_layers):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx, col_idx]

        # Extract required data
        ccs = layer_data["ccs"]
        hate_vectors = layer_data["hate_vectors"]
        safe_vectors = layer_data["safe_vectors"]
        steering_vector = layer_data["steering_vector"]

        # Validate data types
        if not hasattr(ccs, "predict_from_vectors"):
            print(
                f"Error: Layer {i} - CCS object does not have predict_from_vectors method"
            )
            ax.text(0.5, 0.5, "Invalid CCS object", ha="center", va="center")
            ax.set_title(f"Layer {i}")
            continue

        if not isinstance(hate_vectors, np.ndarray):
            print(
                f"Error: Layer {i} - hate_vectors is not a numpy array. Type: {type(hate_vectors)}"
            )
            ax.text(0.5, 0.5, "Invalid hate_vectors", ha="center", va="center")
            ax.set_title(f"Layer {i}")
            continue

        if not isinstance(safe_vectors, np.ndarray):
            print(
                f"Error: Layer {i} - safe_vectors is not a numpy array. Type: {type(safe_vectors)}"
            )
            ax.text(0.5, 0.5, "Invalid safe_vectors", ha="center", va="center")
            ax.set_title(f"Layer {i}")
            continue

        if not isinstance(steering_vector, np.ndarray):
            print(
                f"Error: Layer {i} - steering_vector is not a numpy array. Type: {type(steering_vector)}"
            )
            ax.text(0.5, 0.5, "Invalid steering_vector", ha="center", va="center")
            ax.set_title(f"Layer {i}")
            continue

        # Get original shapes for reshaping
        if hate_vectors.ndim > 2:
            hate_shape = hate_vectors.shape[1:]
            hate_vectors = hate_vectors.reshape(hate_vectors.shape[0], -1)
        else:
            hate_shape = None

        if safe_vectors.ndim > 2:
            safe_shape = safe_vectors.shape[1:]
            safe_vectors = safe_vectors.reshape(safe_vectors.shape[0], -1)
        else:
            safe_shape = None

        # Ensure steering vector is properly shaped
        if steering_vector.ndim > 1:
            steering_vector = steering_vector.reshape(-1)

        # Sample a subset for visualization to avoid crowding
        max_samples = 100
        if hate_vectors.shape[0] > max_samples:
            indices = np.random.choice(
                hate_vectors.shape[0], max_samples, replace=False
            )
            hate_subset = hate_vectors[indices]
        else:
            hate_subset = hate_vectors

        if safe_vectors.shape[0] > max_samples:
            indices = np.random.choice(
                safe_vectors.shape[0], max_samples, replace=False
            )
            safe_subset = safe_vectors[indices]
        else:
            safe_subset = safe_vectors

        # Combine all vectors for PCA
        all_vectors = np.vstack([hate_subset, safe_subset])

        # Check for valid PCA input
        if all_vectors.shape[0] < 2 or all_vectors.shape[1] < 2:
            print(
                f"Error: Layer {i} - Insufficient data for PCA. Shape: {all_vectors.shape}"
            )
            ax.text(0.5, 0.5, "Insufficient data for PCA", ha="center", va="center")
            ax.set_title(f"Layer {i}")
            continue

        # Apply PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        all_vectors_2d = pca.fit_transform(all_vectors)

        # Split back into hate and safe
        hate_vectors_2d = all_vectors_2d[: hate_subset.shape[0]]
        safe_vectors_2d = all_vectors_2d[hate_subset.shape[0] :]

        # Project steering vector to 2D
        steering_vector_2d = pca.transform(steering_vector.reshape(1, -1))[0]

        # Create a grid of points for decision boundary
        x_min, x_max = all_vectors_2d[:, 0].min() - 1, all_vectors_2d[:, 0].max() + 1
        y_min, y_max = all_vectors_2d[:, 1].min() - 1, all_vectors_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Convert grid points back to original space
        grid_original = pca.inverse_transform(grid_points)

        # Get predictions for the grid
        if hate_shape is not None and grid_original.shape[1] != np.prod(hate_shape):
            print(f"Error: Layer {i} - Dimension mismatch in grid transform")
            ax.text(0.5, 0.5, "Dimension mismatch", ha="center", va="center")
            ax.set_title(f"Layer {i}")
            continue

        # Make predictions with CCS model
        grid_tensor = torch.tensor(
            grid_original, dtype=torch.float32, device=ccs.device
        )

        with torch.no_grad():
            grid_preds = ccs.probe(grid_tensor).cpu().numpy()
            # Reshape to match grid
            Z = grid_preds.reshape(xx.shape)

        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, cmap="RdBu", alpha=0.3)

        # Plot data points
        ax.scatter(
            hate_vectors_2d[:, 0],
            hate_vectors_2d[:, 1],
            c="red",
            label="Hate",
            alpha=0.6,
            edgecolors="k",
        )
        ax.scatter(
            safe_vectors_2d[:, 0],
            safe_vectors_2d[:, 1],
            c="blue",
            label="Safe",
            alpha=0.6,
            edgecolors="k",
        )

        # Plot steering vector
        ax.arrow(
            0,
            0,
            steering_vector_2d[0],
            steering_vector_2d[1],
            head_width=0.15,
            head_length=0.2,
            fc="green",
            ec="green",
            label="Steering Vector",
        )

        # Add labels, title, legend
        ax.set_title(f"Layer {i}")
        if i == 0:  # Only add legend to first subplot
            ax.legend()

        if i % n_cols == 0:  # First column
            ax.set_ylabel("PC2")
        if i >= (n_rows - 1) * n_cols:  # Last row
            ax.set_xlabel("PC1")

    # Hide unused subplots
    for i in range(len(valid_layers), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()

    # Save figure if log_base is provided
    if log_base:
        if not os.path.exists(os.path.dirname(log_base)):
            os.makedirs(os.path.dirname(log_base), exist_ok=True)
        save_path = f"{log_base}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved decision boundaries plot to {save_path}")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_vectors_all_strategies(
    layer_idx,
    all_strategy_data,
    current_strategy="last-token",
    save_path=None,
    all_steering_vectors=None,
):
    """
    Plot vectors for all three embedding strategies (mean, last-token, first-token) in a single figure.

    Args:
        layer_idx: Layer index for title
        all_strategy_data: Dictionary with data for all strategies
        current_strategy: The current strategy used in the model ("mean", "last-token", or "first-token")
        save_path: Path to save the plot.
            Example: "plots/layer_0_all_strategies_vectors.png"
        all_steering_vectors: Dict of different steering vectors with their properties

    Returns:
        Example: "plots/layer_0_all_strategies_vectors.png"
    """
    # Create figure with 1 row and 3 columns for the three strategies
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Define the three strategies and titles
    strategies = ["mean", "last-token", "first-token"]
    strategy_titles = [
        "Mean Embedding",
        "Last Token Embedding",
        "First Token Embedding",
    ]

    # Define consistent colors and markers for each category
    category_styles = {
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

    # Print information about all_steering_vectors
    if all_steering_vectors is not None:
        print(f"All steering vectors keys: {list(all_steering_vectors.keys())}")
    else:
        print("No all_steering_vectors provided")

    # Plot each strategy in its own subplot
    for i, (strategy, title) in enumerate(zip(strategies, strategy_titles)):
        ax = axes[i]

        # Get data for this strategy
        if strategy not in all_strategy_data:
            print(f"Strategy {strategy} not found in all_strategy_data")
            continue

        data = all_strategy_data[strategy]
        print(f"Data keys for strategy {strategy}: {list(data.keys())}")

        # Access individual category data
        hate_yes_vectors = data.get("hate_yes", None)
        hate_no_vectors = data.get("hate_no", None)
        safe_yes_vectors = data.get("safe_yes", None)
        safe_no_vectors = data.get("safe_no", None)

        # Print information about vector shapes
        for name, vectors in [
            ("hate_yes", hate_yes_vectors),
            ("hate_no", hate_no_vectors),
            ("safe_yes", safe_yes_vectors),
            ("safe_no", safe_no_vectors),
        ]:
            if vectors is not None:
                print(f"{strategy} - {name} shape: {vectors.shape}")
            else:
                print(f"{strategy} - {name} is None")

        # Make sure we have all necessary vectors
        if (
            hate_yes_vectors is None
            or hate_yes_vectors.size == 0
            or hate_no_vectors is None
            or hate_no_vectors.size == 0
            or safe_yes_vectors is None
            or safe_yes_vectors.size == 0
            or safe_no_vectors is None
            or safe_no_vectors.size == 0
        ):
            print(
                f"Warning: Missing some vector types for {strategy}. Using available data."
            )
            # Fall back to combined hate/safe if needed
            hate_vectors = data.get("hate", None)
            safe_vectors = data.get("safe", None)

            if (
                hate_vectors is not None
                and hate_vectors.size > 0
                and safe_vectors is not None
                and safe_vectors.size > 0
            ):
                # For standard datasets, we'll split each category (hate/safe) into two
                # equal parts to simulate hate_yes/hate_no and safe_yes/safe_no
                half_hate = len(hate_vectors) // 2
                half_safe = len(safe_vectors) // 2

                # Split each category
                hate_yes_vectors = hate_vectors[:half_hate]
                hate_no_vectors = hate_vectors[half_hate:]
                safe_yes_vectors = safe_vectors[:half_safe]
                safe_no_vectors = safe_vectors[half_safe:]

                print("Using fallback with hate/safe data:")
                print(
                    f"  hate_yes: {hate_yes_vectors.shape if hate_yes_vectors is not None else None}"
                )
                print(
                    f"  hate_no: {hate_no_vectors.shape if hate_no_vectors is not None else None}"
                )
                print(
                    f"  safe_yes: {safe_yes_vectors.shape if safe_yes_vectors is not None else None}"
                )
                print(
                    f"  safe_no: {safe_no_vectors.shape if safe_no_vectors is not None else None}"
                )

        # Properly flatten vectors if they're 3D - critical for PCA
        if hate_yes_vectors is not None and len(hate_yes_vectors.shape) == 3:
            hate_yes_vectors = hate_yes_vectors.reshape(hate_yes_vectors.shape[0], -1)
        if hate_no_vectors is not None and len(hate_no_vectors.shape) == 3:
            hate_no_vectors = hate_no_vectors.reshape(hate_no_vectors.shape[0], -1)
        if safe_yes_vectors is not None and len(safe_yes_vectors.shape) == 3:
            safe_yes_vectors = safe_yes_vectors.reshape(safe_yes_vectors.shape[0], -1)
        if safe_no_vectors is not None and len(safe_no_vectors.shape) == 3:
            safe_no_vectors = safe_no_vectors.reshape(safe_no_vectors.shape[0], -1)

        # Stack all vectors for PCA - handle empty arrays
        all_vectors_list = []
        if hate_yes_vectors is not None and hate_yes_vectors.size > 0:
            # Ensure float32 for numerical stability
            all_vectors_list.append(hate_yes_vectors.astype(np.float32))
        if hate_no_vectors is not None and hate_no_vectors.size > 0:
            all_vectors_list.append(hate_no_vectors.astype(np.float32))
        if safe_yes_vectors is not None and safe_yes_vectors.size > 0:
            all_vectors_list.append(safe_yes_vectors.astype(np.float32))
        if safe_no_vectors is not None and safe_no_vectors.size > 0:
            all_vectors_list.append(safe_no_vectors.astype(np.float32))

        if not all_vectors_list:
            ax.text(
                0.5,
                0.5,
                "No data available for this strategy",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            continue

        all_vectors = np.vstack(all_vectors_list)

        # Check for empty data
        if all_vectors.size == 0:
            ax.text(
                0.5,
                0.5,
                "No data available for this strategy",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            continue

        # Add tiny jitter to avoid numerical issues
        epsilon = 1e-8
        all_vectors_reg = all_vectors + np.random.normal(0, epsilon, all_vectors.shape)

        # Normalize vectors to avoid numerical issues in PCA
        all_vectors_mean = np.mean(all_vectors_reg, axis=0, keepdims=True)
        all_vectors_std = np.std(all_vectors_reg, axis=0, keepdims=True) + 1e-10
        all_vectors_norm = (all_vectors_reg - all_vectors_mean) / all_vectors_std

        # Prepare mean vectors
        mean_vectors = []
        mean_vectors_categories = []  # Keep track of which category each mean is from

        # Calculate mean vectors for each category and add them to the list if they exist
        if hate_yes_vectors is not None and hate_yes_vectors.size > 0:
            hate_yes_mean = (
                np.mean(hate_yes_vectors, axis=0).reshape(1, -1).astype(np.float32)
            )
            mean_vectors.append(hate_yes_mean)
            mean_vectors_categories.append("hate_yes")
        else:
            hate_yes_mean = None

        if hate_no_vectors is not None and hate_no_vectors.size > 0:
            hate_no_mean = (
                np.mean(hate_no_vectors, axis=0).reshape(1, -1).astype(np.float32)
            )
            mean_vectors.append(hate_no_mean)
            mean_vectors_categories.append("hate_no")
        else:
            hate_no_mean = None

        if safe_yes_vectors is not None and safe_yes_vectors.size > 0:
            safe_yes_mean = (
                np.mean(safe_yes_vectors, axis=0).reshape(1, -1).astype(np.float32)
            )
            mean_vectors.append(safe_yes_mean)
            mean_vectors_categories.append("safe_yes")
        else:
            safe_yes_mean = None

        if safe_no_vectors is not None and safe_no_vectors.size > 0:
            safe_no_mean = (
                np.mean(safe_no_vectors, axis=0).reshape(1, -1).astype(np.float32)
            )
            mean_vectors.append(safe_no_mean)
            mean_vectors_categories.append("safe_no")
        else:
            safe_no_mean = None

        # Add steering vectors
        steering_vectors_list = []
        steering_vector_names = []
        steering_vector_colors = []
        steering_vector_labels = []

        if all_steering_vectors is not None:
            for name, data in all_steering_vectors.items():
                if "vector" in data and data["vector"] is not None:
                    # Make sure the steering vector is properly flattened
                    vector = data["vector"].astype(np.float32)
                    if len(vector.shape) == 3:
                        vector = vector.reshape(vector.shape[0], -1)
                    elif len(vector.shape) == 1:
                        vector = vector.reshape(1, -1)

                    steering_vectors_list.append(vector)
                    steering_vector_names.append(name)
                    steering_vector_colors.append(data.get("color", "#00FF00"))
                    steering_vector_labels.append(data.get("label", name))
        else:
            # Extract steering vector from data
            steering_vector = data.get("steering_vector", None)
            if steering_vector is not None:
                # Make sure the steering vector is properly flattened
                steering_vector = steering_vector.astype(np.float32)
                if len(steering_vector.shape) == 3:
                    steering_vector = steering_vector.reshape(
                        steering_vector.shape[0], -1
                    )
                elif len(steering_vector.shape) == 1:
                    steering_vector = steering_vector.reshape(1, -1)

                steering_vectors_list.append(steering_vector)
                steering_vector_names.append("Steering Vector")
                steering_vector_colors.append("#00FF00")
                steering_vector_labels.append("Steering Vector")

        print(
            f"Number of mean vectors: {len(mean_vectors)}, steering vectors: {len(steering_vectors_list)}"
        )

        # Check if we have any vectors to show
        if not mean_vectors and not steering_vectors_list:
            ax.text(
                0.5,
                0.5,
                "No vectors available for PCA in this strategy",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            continue

        # Stack all vectors for PCA projection - proper dimensionality handling
        all_mean_steering_vectors_list = mean_vectors + steering_vectors_list
        if all_mean_steering_vectors_list:
            all_mean_steering_vectors = np.vstack(all_mean_steering_vectors_list)
        else:
            continue  # Skip if no vectors to show

        # Ensure proper normalization with same parameters as data points
        all_mean_steering_vectors_reg = (
            all_mean_steering_vectors - all_vectors_mean
        ) / all_vectors_std

        # Apply PCA to get 2D projections using robust approach
        # Apply PCA to reduce to 2D with robust solver
        pca = PCA(n_components=2, svd_solver="arpack")
        pca.fit(all_vectors_norm)  # Fit on all data points

        # Transform both data points and vectors
        all_2d = pca.transform(all_vectors_norm)
        mean_2d = pca.transform(all_mean_steering_vectors_reg)

        print(f"PCA successful for {strategy}")

        # Split the projected points back to their categories
        start_idx = 0
        hate_yes_2d = np.array([])
        hate_no_2d = np.array([])
        safe_yes_2d = np.array([])
        safe_no_2d = np.array([])

        # Extract 2D coordinates for each category
        if hate_yes_vectors is not None and hate_yes_vectors.size > 0:
            end_idx = start_idx + len(hate_yes_vectors)
            hate_yes_2d = all_2d[start_idx:end_idx]
            start_idx = end_idx

        if hate_no_vectors is not None and hate_no_vectors.size > 0:
            end_idx = start_idx + len(hate_no_vectors)
            hate_no_2d = all_2d[start_idx:end_idx]
            start_idx = end_idx

        if safe_yes_vectors is not None and safe_yes_vectors.size > 0:
            end_idx = start_idx + len(safe_yes_vectors)
            safe_yes_2d = all_2d[start_idx:end_idx]
            start_idx = end_idx

        if safe_no_vectors is not None and safe_no_vectors.size > 0:
            end_idx = start_idx + len(safe_no_vectors)
            safe_no_2d = all_2d[start_idx:end_idx]

        # Plot each category separately with consistent styling
        for category, points_2d in [
            ("hate_yes", hate_yes_2d),
            ("hate_no", hate_no_2d),
            ("safe_yes", safe_yes_2d),
            ("safe_no", safe_no_2d),
        ]:
            if len(points_2d) > 0:
                style = category_styles[category]
                ax.scatter(
                    points_2d[:, 0],
                    points_2d[:, 1],
                    color=style["color"],
                    alpha=0.6,
                    marker=style["marker"],
                    s=40,
                    edgecolors="black",
                    linewidths=0.5,
                    label=style["label"],
                    zorder=5,
                )

        # Plot mean vectors
        for idx, (category, mean_point) in enumerate(
            zip(mean_vectors_categories, mean_2d[: len(mean_vectors)])
        ):
            style = category_styles[category]

            # Each category has its own label, ensuring each appears separately in the legend
            unique_label = f"{style['label']} Mean"
            print(
                f"Plotting {unique_label} at coordinates: ({mean_point[0]:.4f}, {mean_point[1]:.4f})"
            )

            ax.quiver(
                0,
                0,
                mean_point[0],
                mean_point[1],
                color=style["color"],
                label=unique_label,
                scale_units="xy",
                scale=1,
                width=0.015,
                headwidth=5,
                headlength=7,
                zorder=50,
                alpha=1.0,
            )

        # Plot steering vectors
        offset = len(mean_vectors)
        for idx, (name, color, label, vector_point) in enumerate(
            zip(
                steering_vector_names,
                steering_vector_colors,
                steering_vector_labels,
                mean_2d[offset : offset + len(steering_vectors_list)],
            )
        ):
            print(
                f"Plotting steering vector {name} at coordinates: ({vector_point[0]:.4f}, {vector_point[1]:.4f})"
            )
            ax.quiver(
                0,
                0,
                vector_point[0],
                vector_point[1],
                color=color,
                label=label,
                scale_units="xy",
                scale=1,
                width=0.015,
                headwidth=5,
                headlength=7,
                zorder=50,
            )

        # Set axis properties
        ax.set_title(f"{title} (Layer {layer_idx})")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Calculate axis limits from all data points
        all_points_list = []
        for points in [hate_yes_2d, hate_no_2d, safe_yes_2d, safe_no_2d]:
            if len(points) > 0:
                all_points_list.append(points)

        if all_points_list:
            all_points = np.vstack(all_points_list)
            x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Add a red border around the current strategy subplot
        if strategy == current_strategy:
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

    # Add explanation text
    plt.figtext(
        0.5,
        0.01,
        "These plots show the distribution of different statement types in 2D PCA space.\n"
        "Red circles: Hate statements with 'Yes', Blue circles: Hate statements with 'No'\n"
        "Blue triangles: Safe statements with 'Yes', Red triangles: Safe statements with 'No'\n"
        "Arrows show mean vectors and steering vectors.\n"
        "The red border indicates the embedding strategy used in the current run.",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.97))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved vector plot with all strategies to {save_path}")

    plt.close()  # Close the plot to avoid memory issues
    return fig


def combine_steering_plots(plot_dir, layer_idx, strategy="last-token"):
    """
    Create a combined plot with all steering vector types in a single figure with subplots in one row.

    Args:
        plot_dir: Directory to save the combined plot
        layer_idx: Layer index
        strategy: Embedding strategy used

    Returns:
        fig, axes, combined_path where combined_path will be "{plot_dir}/layer_{layer_idx}_{strategy}_all_steering_combined.png"
    """

    import matplotlib.pyplot as plt

    # Define the transformation types to include
    transformations = [
        "combined",
        "hate_yes_to_safe_yes",
        "safe_no_to_hate_no",
        "hate_no_to_hate_yes",
        "safe_yes_to_safe_no",
    ]

    # Create names for the transformations with readable titles
    transformation_titles = {
        "combined": "(Hate Yes + Safe No) → (Safe Yes + Hate No)",
        "hate_yes_to_safe_yes": "Hate Yes → Safe Yes",
        "safe_no_to_hate_no": "Safe No → Hate No",
        "hate_no_to_hate_yes": "Hate No → Hate Yes",
        "safe_yes_to_safe_no": "Safe Yes → Safe No",
    }

    # Create figure with subplots in one row
    fig, axes = plt.subplots(
        1, len(transformations), figsize=(7 * len(transformations), 6)
    )

    # Set figure title
    fig.suptitle(
        f"Steering Vectors - Layer {layer_idx} ({strategy} strategy)", fontsize=16
    )

    # Set each subplot title
    for i, trans in enumerate(transformations):
        axes[i].set_title(transformation_titles[trans], fontsize=12)
        axes[i].axis(
            "off"
        )  # Placeholder, actual plots will be filled by plot_individual_steering_vectors

    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.95))  # Make room for the suptitle

    # Save the combined figure path for returning
    combined_path = os.path.join(
        plot_dir, f"layer_{layer_idx}_{strategy}_all_steering_combined.png"
    )

    return fig, axes, combined_path


def plot_individual_steering_vectors(
    plot_dir,
    layer_idx,
    all_steering_vectors,
    hate_yes_vectors,
    hate_no_vectors,
    safe_yes_vectors,
    safe_no_vectors,
    strategy="last-token",
):
    """
    Create a combined visualization for all 5 steering vector types in a single figure with subplots.

    Args:
        plot_dir: Directory to save the plots. Example: "plots/"
        layer_idx: Layer index
        all_steering_vectors: Dictionary of steering vectors
        hate_yes_vectors: Vectors for hate_yes statements
        safe_yes_vectors: Vectors for safe_yes statements
        hate_no_vectors: Vectors for hate_no statements
        safe_no_vectors: Vectors for safe_no statements
        strategy: Embedding strategy

    Returns:
        combined_path: Path to the saved plot. Example: "plots/layer_0_last-token_all_steering_combined.png"
        combined_path: Path to the saved plot. Example: "plots/layer_0_first-token_all_steering_combined.png"
        combined_path: Path to the saved plot. Example: "plots/layer_0_mean_all_steering_combined.png"
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    # Ensure that all vector inputs are valid
    if any(
        v is None
        for v in [hate_yes_vectors, hate_no_vectors, safe_yes_vectors, safe_no_vectors]
    ):
        print("Skipping steering vector plots because some vectors are None")
        return

    # Get figure and axes for the combined plot
    fig, axes, combined_path = combine_steering_plots(plot_dir, layer_idx, strategy)

    # Define consistent colors and markers for each category
    category_styles = {
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

    # Define transformation pairs to visualize
    transformations = [
        {
            "name": "combined",
            "title": "(Hate Yes + Safe No) → (Safe Yes + Hate No)",
            "source_name": "Combined Source",
            "source_data": np.vstack([hate_yes_vectors, safe_no_vectors]),
            "source_types": ["hate_yes"] * len(hate_yes_vectors)
            + ["safe_no"] * len(safe_no_vectors),
            "target_name": "Combined Target",
            "target_data": np.vstack([safe_yes_vectors, hate_no_vectors]),
            "target_types": ["safe_yes"] * len(safe_yes_vectors)
            + ["hate_no"] * len(hate_no_vectors),
            "steering_vector": all_steering_vectors["combined"]["vector"],
            "steering_color": all_steering_vectors["combined"]["color"],
        },
        {
            "name": "hate_yes_to_safe_yes",
            "title": "Hate Yes → Safe Yes",
            "source_type": "hate_yes",
            "source_data": hate_yes_vectors,
            "target_type": "safe_yes",
            "target_data": safe_yes_vectors,
            "steering_vector": all_steering_vectors["hate_yes_to_safe_yes"]["vector"],
            "steering_color": all_steering_vectors["hate_yes_to_safe_yes"]["color"],
        },
        {
            "name": "safe_no_to_hate_no",
            "title": "Safe No → Hate No",
            "source_type": "safe_no",
            "source_data": safe_no_vectors,
            "target_type": "hate_no",
            "target_data": hate_no_vectors,
            "steering_vector": all_steering_vectors["safe_no_to_hate_no"]["vector"],
            "steering_color": all_steering_vectors["safe_no_to_hate_no"]["color"],
        },
        {
            "name": "hate_no_to_hate_yes",
            "title": "Hate No → Hate Yes",
            "source_type": "hate_no",
            "source_data": hate_no_vectors,
            "target_type": "hate_yes",
            "target_data": hate_yes_vectors,
            "steering_vector": all_steering_vectors["hate_yes_to_hate_no"]["vector"]
            * -1,  # Reverse direction
            "steering_color": "#FF9900",  # Orange (same as original)
        },
        {
            "name": "safe_yes_to_safe_no",
            "title": "Safe Yes → Safe No",
            "source_type": "safe_yes",
            "source_data": safe_yes_vectors,
            "target_type": "safe_no",
            "target_data": safe_no_vectors,
            "steering_vector": all_steering_vectors["safe_no_to_safe_yes"]["vector"]
            * -1,  # Reverse direction
            "steering_color": "#00FFCC",  # Teal (same as original)
        },
    ]

    # Create a plot for each transformation
    for i, trans in enumerate(transformations):
        ax = axes[i]
        ax.clear()  # Clear the placeholder
        ax.axis("on")  # Turn axis back on

        # Reshape data if needed
        source_data = trans["source_data"]
        target_data = trans["target_data"]

        if len(source_data.shape) == 3:
            source_data = source_data.reshape(source_data.shape[0], -1)

        if len(target_data.shape) == 3:
            target_data = target_data.reshape(target_data.shape[0], -1)

        # Combine source and target data for PCA
        combined_data = np.vstack([source_data, target_data])

        # Normalize data
        data_mean = np.mean(combined_data, axis=0, keepdims=True)
        data_std = np.std(combined_data, axis=0, keepdims=True) + 1e-10
        normalized_data = (combined_data - data_mean) / data_std

        # Calculate source and target means
        source_mean = np.mean(source_data, axis=0)
        target_mean = np.mean(target_data, axis=0)

        # Normalize means and steering vector
        source_mean_norm = (source_mean - data_mean[0]) / data_std[0]
        target_mean_norm = (target_mean - data_mean[0]) / data_std[0]
        steering_vector_norm = (trans["steering_vector"] - data_mean[0]) / data_std[0]

        # Apply PCA
        pca = PCA(n_components=2, svd_solver="full")
        data_2d = pca.fit_transform(normalized_data)

        # Transform means and steering vector
        source_mean_2d = pca.transform(source_mean_norm.reshape(1, -1))[0]
        target_mean_2d = pca.transform(target_mean_norm.reshape(1, -1))[0]
        steering_vector_2d = pca.transform(steering_vector_norm.reshape(1, -1))[0]

        # Split data back into source and target
        n_source = len(source_data)
        source_2d = data_2d[:n_source]
        target_2d = data_2d[n_source:]

        # Plot points
        if "source_types" in trans:
            # For combined case with multiple types
            for category_type in set(trans["source_types"]):
                indices = [
                    i for i, t in enumerate(trans["source_types"]) if t == category_type
                ]
                style = category_styles[category_type]
                ax.scatter(
                    source_2d[indices, 0],
                    source_2d[indices, 1],
                    color=style["color"],
                    alpha=0.6,
                    s=40,
                    marker=style["marker"],
                    edgecolors="black",
                    linewidths=0.5,
                    label=style["label"],
                    zorder=5,
                )

            for category_type in set(trans["target_types"]):
                indices = [
                    i for i, t in enumerate(trans["target_types"]) if t == category_type
                ]
                style = category_styles[category_type]
                ax.scatter(
                    target_2d[indices, 0],
                    target_2d[indices, 1],
                    color=style["color"],
                    alpha=0.6,
                    s=40,
                    marker=style["marker"],
                    edgecolors="black",
                    linewidths=0.5,
                    label=style["label"],
                    zorder=5,
                )
        else:
            # Simple case with single type for source and target
            source_style = category_styles[trans["source_type"]]
            target_style = category_styles[trans["target_type"]]

            ax.scatter(
                source_2d[:, 0],
                source_2d[:, 1],
                color=source_style["color"],
                alpha=0.6,
                s=40,
                marker=source_style["marker"],
                edgecolors="black",
                linewidths=0.5,
                label=source_style["label"],
                zorder=5,
            )

            ax.scatter(
                target_2d[:, 0],
                target_2d[:, 1],
                color=target_style["color"],
                alpha=0.6,
                s=40,
                marker=target_style["marker"],
                edgecolors="black",
                linewidths=0.5,
                label=target_style["label"],
                zorder=5,
            )

        # Plot mean vectors
        if "source_type" in trans:  # Single source/target types
            source_style = category_styles[trans["source_type"]]
            target_style = category_styles[trans["target_type"]]

            ax.quiver(
                0,
                0,
                source_mean_2d[0],
                source_mean_2d[1],
                color=source_style["color"],
                label=f"{source_style['label']} Mean",
                scale_units="xy",
                scale=1,
                width=0.015,
                headwidth=5,
                headlength=7,
                zorder=50,
            )

            ax.quiver(
                0,
                0,
                target_mean_2d[0],
                target_mean_2d[1],
                color=target_style["color"],
                label=f"{target_style['label']} Mean",
                scale_units="xy",
                scale=1,
                width=0.015,
                headwidth=5,
                headlength=7,
                zorder=50,
            )
        else:  # Combined source/target
            ax.quiver(
                0,
                0,
                source_mean_2d[0],
                source_mean_2d[1],
                color="#880000",  # Dark red for combined source
                label=f"{trans['source_name']} Mean",
                scale_units="xy",
                scale=1,
                width=0.015,
                headwidth=5,
                headlength=7,
                zorder=50,
            )

            ax.quiver(
                0,
                0,
                target_mean_2d[0],
                target_mean_2d[1],
                color="#000088",  # Dark blue for combined target
                label=f"{trans['target_name']} Mean",
                scale_units="xy",
                scale=1,
                width=0.015,
                headwidth=5,
                headlength=7,
                zorder=50,
            )

        # Plot steering vector
        ax.quiver(
            0,
            0,
            steering_vector_2d[0],
            steering_vector_2d[1],
            color=trans["steering_color"],
            label="Steering Vector",
            scale_units="xy",
            scale=1,
            width=0.015,
            headwidth=5,
            headlength=7,
            zorder=50,
        )

        # Set axis limits
        x_min, x_max = np.min(data_2d[:, 0]), np.max(data_2d[:, 0])
        y_min, y_max = np.min(data_2d[:, 1]), np.max(data_2d[:, 1])
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Add labels
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Add descriptive text
    fig.text(
        0.5,
        0.01,
        "Red circles: Hate statements with 'Yes', Blue circles: Hate statements with 'No'\n"
        "Blue triangles: Safe statements with 'Yes', Red triangles: Safe statements with 'No'\n"
        "Arrows show the steering direction between content types.",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Adjust layout to make room for text
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))

    # Save combined figure
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined steering plot to {combined_path}")
    return combined_path


def plot_all_strategies_all_steering_vectors(
    plot_dir,
    layer_idx,
    representations,
    all_steering_vectors_by_strategy,
):
    """
    Create a comprehensive visualization of all steering vectors for all embedding strategies.

    Args:
        plot_dir: Directory to save the plots
        layer_idx: Layer index
        representations: Dictionary of representations for each strategy
        all_steering_vectors_by_strategy: Dictionary of steering vectors for each strategy

    Returns:
        Path to the saved plot or None if unsuccessful
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    # Validate input data with explicit checks
    if not isinstance(representations, dict):
        print(
            f"Error: representations must be a dictionary. Got {type(representations)}"
        )
        return None

    if len(representations) == 0:
        print("Error: representations dictionary is empty")
        return None

    if not isinstance(all_steering_vectors_by_strategy, dict):
        print(
            f"Error: all_steering_vectors_by_strategy must be a dictionary. Got {type(all_steering_vectors_by_strategy)}"
        )
        return None

    if len(all_steering_vectors_by_strategy) == 0:
        print("Error: all_steering_vectors_by_strategy dictionary is empty")
        return None

    # Define strategies and transformations
    strategies = ["last-token", "first-token", "mean"]

    # Filter strategies to only those available
    available_strategies = [s for s in strategies if s in representations]
    if not available_strategies:
        print(
            f"Error: No valid strategies found in representations. Available keys: {list(representations.keys())}"
        )
        return None

    # Validate steering vectors match strategies
    for strategy in available_strategies:
        if strategy not in all_steering_vectors_by_strategy:
            print(
                f"Error: Strategy {strategy} not found in all_steering_vectors_by_strategy"
            )
            return None

    # Print data diagnostics
    print(f"All steering vectors keys: {list(all_steering_vectors_by_strategy.keys())}")
    for strategy in available_strategies:
        print(
            f"Data keys for strategy {strategy}: {list(representations[strategy].keys()) if strategy in representations else 'N/A'}"
        )
        for key in ["hate_yes", "hate_no", "safe_yes", "safe_no"]:
            if strategy in representations and key in representations[strategy]:
                data = representations[strategy][key]
                if isinstance(data, np.ndarray):
                    print(f"{strategy} - {key} shape: {data.shape}")
                else:
                    print(f"{strategy} - {key} type: {type(data)}")

    # Define transformations
    transformations = [
        "combined",
        "hate_yes_to_safe_yes",
        "safe_no_to_hate_no",
        "hate_yes_to_hate_no",
        "safe_no_to_safe_yes",
    ]

    # Filter transformations to only those available for all strategies
    available_transformations = []
    for trans in transformations:
        all_have_trans = True
        for strategy in available_strategies:
            if strategy not in all_steering_vectors_by_strategy:
                all_have_trans = False
                break
            if trans not in all_steering_vectors_by_strategy[strategy]:
                all_have_trans = False
                break
        if all_have_trans:
            available_transformations.append(trans)

    if not available_transformations:
        print("Error: No common transformations available across all strategies")
        return None

    # Create readable titles for transformations
    transformation_titles = {
        "combined": "(Hate Yes + Safe No) → (Safe Yes + Hate No)",
        "hate_yes_to_safe_yes": "Hate Yes → Safe Yes",
        "safe_no_to_hate_no": "Safe No → Hate No",
        "hate_yes_to_hate_no": "Hate Yes → Hate No",
        "safe_no_to_safe_yes": "Safe No → Safe Yes",
    }

    # Define consistent colors and markers for each category
    category_styles = {
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

    # Create figure with subplots - strategies as rows, transformations as columns
    fig, axes = plt.subplots(
        len(available_strategies),
        len(available_transformations),
        figsize=(6 * len(available_transformations), 5 * len(available_strategies)),
    )

    # Handle case with only one subplot
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

        # Validate required data for this strategy
        required_keys = ["hate_yes", "hate_no", "safe_yes", "safe_no"]
        if not all(key in strategy_reps for key in required_keys):
            print(f"Error: Missing required data keys for strategy {strategy}")
            available_keys = (
                list(strategy_reps.keys()) if isinstance(strategy_reps, dict) else "N/A"
            )
            print(f"Available keys: {available_keys}")
            for col_idx in range(len(available_transformations)):
                ax = axes[row_idx][col_idx]
                ax.text(0.5, 0.5, "Missing data", ha="center", va="center")
                ax.set_title(f"{strategy} - Missing data")
            continue

        # Make sure all vector data are numpy arrays
        for key in required_keys:
            if not isinstance(strategy_reps[key], np.ndarray):
                print(
                    f"Error: {strategy} - {key} is not a numpy array. Got {type(strategy_reps[key])}"
                )
                for col_idx in range(len(available_transformations)):
                    ax = axes[row_idx][col_idx]
                    ax.text(
                        0.5, 0.5, f"Invalid {key} data type", ha="center", va="center"
                    )
                    ax.set_title(f"{strategy} - Invalid data")
                continue

        # Extract vectors
        hate_yes_vectors = strategy_reps["hate_yes"]
        hate_no_vectors = strategy_reps["hate_no"]
        safe_yes_vectors = strategy_reps["safe_yes"]
        safe_no_vectors = strategy_reps["safe_no"]

        # Get steering vectors for this strategy
        if strategy not in all_steering_vectors_by_strategy:
            print(f"Error: No steering vectors for strategy {strategy}")
            for col_idx in range(len(available_transformations)):
                ax = axes[row_idx][col_idx]
                ax.text(0.5, 0.5, "No steering vectors", ha="center", va="center")
                ax.set_title(f"{strategy} - No steering")
            continue

        strategy_steering_vectors = all_steering_vectors_by_strategy[strategy]

        # For each transformation
        for col_idx, trans_name in enumerate(available_transformations):
            if trans_name not in strategy_steering_vectors:
                print(
                    f"Error: Transformation {trans_name} not found for strategy {strategy}"
                )
                ax = axes[row_idx][col_idx]
                ax.text(0.5, 0.5, f"Missing {trans_name}", ha="center", va="center")
                ax.set_title(f"{strategy} - Missing transformation")
                continue

            ax = axes[row_idx][col_idx]

            # Set transformation-specific data
            if trans_name == "combined":
                source_data = np.vstack([hate_yes_vectors, safe_no_vectors])
                source_types = ["hate_yes"] * len(hate_yes_vectors) + ["safe_no"] * len(
                    safe_no_vectors
                )
                target_data = np.vstack([safe_yes_vectors, hate_no_vectors])
                target_types = ["safe_yes"] * len(safe_yes_vectors) + ["hate_no"] * len(
                    hate_no_vectors
                )

                if "vector" not in strategy_steering_vectors[trans_name]:
                    print(f"Error: Missing vector in {strategy} - {trans_name}")
                    ax.text(0.5, 0.5, "Missing vector data", ha="center", va="center")
                    ax.set_title(f"{strategy} - Missing vector data")
                    continue

                steering_vector = strategy_steering_vectors[trans_name]["vector"]
                steering_color = strategy_steering_vectors[trans_name].get(
                    "color", "#00FF00"
                )  # Default green
                use_combined = True
            elif trans_name == "hate_yes_to_safe_yes":
                source_data = hate_yes_vectors
                source_type = "hate_yes"
                target_data = safe_yes_vectors
                target_type = "safe_yes"

                if "vector" not in strategy_steering_vectors[trans_name]:
                    print(f"Error: Missing vector in {strategy} - {trans_name}")
                    ax.text(0.5, 0.5, "Missing vector data", ha="center", va="center")
                    ax.set_title(f"{strategy} - Missing vector data")
                    continue

                steering_vector = strategy_steering_vectors[trans_name]["vector"]
                steering_color = strategy_steering_vectors[trans_name].get(
                    "color", "#00FF00"
                )
                use_combined = False
            elif trans_name == "safe_no_to_hate_no":
                source_data = safe_no_vectors
                source_type = "safe_no"
                target_data = hate_no_vectors
                target_type = "hate_no"

                if "vector" not in strategy_steering_vectors[trans_name]:
                    print(f"Error: Missing vector in {strategy} - {trans_name}")
                    ax.text(0.5, 0.5, "Missing vector data", ha="center", va="center")
                    ax.set_title(f"{strategy} - Missing vector data")
                    continue

                steering_vector = strategy_steering_vectors[trans_name]["vector"]
                steering_color = strategy_steering_vectors[trans_name].get(
                    "color", "#00FF00"
                )
                use_combined = False
            elif trans_name == "hate_yes_to_hate_no":
                source_data = hate_yes_vectors
                source_type = "hate_yes"
                target_data = hate_no_vectors
                target_type = "hate_no"

                if (
                    trans_name in strategy_steering_vectors
                    and "vector" in strategy_steering_vectors[trans_name]
                ):
                    steering_vector = strategy_steering_vectors[trans_name]["vector"]
                    steering_color = strategy_steering_vectors[trans_name].get(
                        "color", "#FF9900"
                    )  # Orange
                else:
                    # Fallback if this specific vector isn't available
                    if (
                        "combined" not in strategy_steering_vectors
                        or "vector" not in strategy_steering_vectors["combined"]
                    ):
                        print(
                            f"Error: Missing vector in {strategy} - {trans_name} and no fallback available"
                        )
                        ax.text(
                            0.5, 0.5, "Missing vector data", ha="center", va="center"
                        )
                        ax.set_title(f"{strategy} - Missing vector data")
                        continue

                    steering_vector = strategy_steering_vectors["combined"]["vector"]
                    steering_color = "#FF9900"  # Orange
                use_combined = False
            elif trans_name == "safe_no_to_safe_yes":
                source_data = safe_no_vectors
                source_type = "safe_no"
                target_data = safe_yes_vectors
                target_type = "safe_yes"

                if (
                    trans_name in strategy_steering_vectors
                    and "vector" in strategy_steering_vectors[trans_name]
                ):
                    steering_vector = strategy_steering_vectors[trans_name]["vector"]
                    steering_color = strategy_steering_vectors[trans_name].get(
                        "color", "#00FFCC"
                    )  # Teal
                else:
                    # Fallback if this specific vector isn't available
                    if (
                        "combined" not in strategy_steering_vectors
                        or "vector" not in strategy_steering_vectors["combined"]
                    ):
                        print(
                            f"Error: Missing vector in {strategy} - {trans_name} and no fallback available"
                        )
                        ax.text(
                            0.5, 0.5, "Missing vector data", ha="center", va="center"
                        )
                        ax.set_title(f"{strategy} - Missing vector data")
                        continue

                    steering_vector = strategy_steering_vectors["combined"]["vector"]
                    steering_color = "#00FFCC"  # Teal
                use_combined = False
            else:
                print(f"Error: Unknown transformation: {trans_name}")
                ax.text(0.5, 0.5, f"Unknown {trans_name}", ha="center", va="center")
                ax.set_title(f"{strategy} - Unknown transformation")
                continue

            # Check if steering vector is a numpy array
            if not isinstance(steering_vector, np.ndarray):
                print(
                    f"Error: Steering vector for {strategy} - {trans_name} is not a numpy array. Got {type(steering_vector)}"
                )
                ax.text(0.5, 0.5, "Invalid steering vector", ha="center", va="center")
                ax.set_title(f"{strategy} - Invalid data")
                continue

            # Reshape data if needed
            if len(source_data.shape) == 3:
                source_data = source_data.reshape(source_data.shape[0], -1)
            if len(target_data.shape) == 3:
                target_data = target_data.reshape(target_data.shape[0], -1)
            if len(steering_vector.shape) > 1:
                steering_vector = steering_vector.reshape(-1)

            # Combine source and target data for PCA
            combined_data = np.vstack([source_data, target_data])

            # Check data validity
            if combined_data.size == 0:
                print(f"Error: Empty combined data for {strategy} - {trans_name}")
                ax.text(0.5, 0.5, "Empty data", ha="center", va="center")
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            if np.isnan(combined_data).any():
                print(
                    f"Error: NaN values in combined data for {strategy} - {trans_name}"
                )
                ax.text(0.5, 0.5, "NaN values in data", ha="center", va="center")
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            if np.isinf(combined_data).any():
                print(
                    f"Error: Infinite values in combined data for {strategy} - {trans_name}"
                )
                ax.text(0.5, 0.5, "Infinite values in data", ha="center", va="center")
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            # Normalize data
            data_mean = np.mean(combined_data, axis=0, keepdims=True)
            data_std = np.std(combined_data, axis=0, keepdims=True) + 1e-10
            normalized_data = (combined_data - data_mean) / data_std

            # Calculate source and target means
            source_mean = np.mean(source_data, axis=0)
            target_mean = np.mean(target_data, axis=0)

            # Normalize means and steering vector
            source_mean_norm = (source_mean - data_mean[0]) / data_std[0]
            target_mean_norm = (target_mean - data_mean[0]) / data_std[0]

            # Check if steering vector dimensions match
            if steering_vector.shape[0] != data_mean.shape[1]:
                print(
                    f"Error: Dimension mismatch - steering vector shape: {steering_vector.shape}, required shape: ({data_mean.shape[1]},)"
                )
                ax.text(0.5, 0.5, "Dimension mismatch", ha="center", va="center")
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            steering_vector_norm = (steering_vector - data_mean[0]) / data_std[0]

            # Apply PCA
            # Check if data is suitable for PCA
            if normalized_data.shape[0] < 2 or normalized_data.shape[1] < 2:
                print(f"Error: Insufficient data for PCA: {normalized_data.shape}")
                ax.text(0.5, 0.5, "Insufficient data for PCA", ha="center", va="center")
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            pca = PCA(n_components=2, svd_solver="full")

            # Handle potential PCA errors explicitly
            if np.all(np.std(normalized_data, axis=0) < 1e-10):
                print(f"Error: Zero variance in data for {strategy} - {trans_name}")
                ax.text(0.5, 0.5, "Zero variance in data", ha="center", va="center")
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            data_2d = pca.fit_transform(normalized_data)

            # Transform means and steering vector
            if source_mean_norm.shape != (normalized_data.shape[1],):
                print(
                    f"Error: Source mean shape mismatch: {source_mean_norm.shape} vs expected {(normalized_data.shape[1],)}"
                )
                ax.text(0.5, 0.5, "Source mean shape error", ha="center", va="center")
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            if target_mean_norm.shape != (normalized_data.shape[1],):
                print(
                    f"Error: Target mean shape mismatch: {target_mean_norm.shape} vs expected {(normalized_data.shape[1],)}"
                )
                ax.text(0.5, 0.5, "Target mean shape error", ha="center", va="center")
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            if steering_vector_norm.shape != (normalized_data.shape[1],):
                print(
                    f"Error: Steering vector shape mismatch: {steering_vector_norm.shape} vs expected {(normalized_data.shape[1],)}"
                )
                ax.text(
                    0.5, 0.5, "Steering vector shape error", ha="center", va="center"
                )
                ax.set_title(
                    f"{strategy} - {transformation_titles.get(trans_name, trans_name)}"
                )
                continue

            source_mean_2d = pca.transform(source_mean_norm.reshape(1, -1))[0]
            target_mean_2d = pca.transform(target_mean_norm.reshape(1, -1))[0]
            steering_vector_2d = pca.transform(steering_vector_norm.reshape(1, -1))[0]

            print(f"PCA successful for {strategy}")

            # Split data back into source and target
            n_source = len(source_data)
            source_2d = data_2d[:n_source]
            target_2d = data_2d[n_source:]

            # Plot points
            if use_combined:
                # For combined case with multiple types
                for category_type in set(source_types):
                    indices = [
                        i for i, t in enumerate(source_types) if t == category_type
                    ]
                    style = category_styles[category_type]
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
                    style = category_styles[category_type]
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
                style_source = category_styles[source_type]
                style_target = category_styles[target_type]

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

            # Plot steering vector as arrow
            print(
                f"Plotting {trans_name} at coordinates: ({steering_vector_2d[0]}, {steering_vector_2d[1]})"
            )
            vector_scale = 0.5  # Adjust scale as needed
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

            # Set title and legend for the subplot
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
    """Create a more detailed decision boundary visualization that separates the four data types.

    Args:
        ccs: CCS probe
        hate_yes_vectors: Vectors for hate content with "yes"
        hate_no_vectors: Vectors for hate content with "no"
        safe_yes_vectors: Vectors for safe content with "yes"
        safe_no_vectors: Vectors for safe content with "no"
        steering_vector: The calculated steering vector
        layer_idx: Layer index for title
        log_base: Base path for saving the plot
        strategy: Optional embedding strategy (last-token, first-token, mean)
        steering_coefficient: Optional steering coefficient value
        pair_type: Optional data pair type (e.g., hate_yes_to_safe_yes)
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

    # Ensure arrays are float32
    hate_yes_vectors = hate_yes_vectors.astype(np.float32)
    hate_no_vectors = hate_no_vectors.astype(np.float32)
    safe_yes_vectors = safe_yes_vectors.astype(np.float32)
    safe_no_vectors = safe_no_vectors.astype(np.float32)
    steering_vector = steering_vector.astype(np.float32)

    # Properly flatten vectors if they're 3D
    if hate_yes_vectors is not None and len(hate_yes_vectors.shape) == 3:
        hate_yes_vectors = hate_yes_vectors.reshape(hate_yes_vectors.shape[0], -1)
    if hate_no_vectors is not None and len(hate_no_vectors.shape) == 3:
        hate_no_vectors = hate_no_vectors.reshape(hate_no_vectors.shape[0], -1)
    if safe_yes_vectors is not None and len(safe_yes_vectors.shape) == 3:
        safe_yes_vectors = safe_yes_vectors.reshape(safe_yes_vectors.shape[0], -1)
    if safe_no_vectors is not None and len(safe_no_vectors.shape) == 3:
        safe_no_vectors = safe_no_vectors.reshape(safe_no_vectors.shape[0], -1)

    # Normalize steering vector
    steering_norm = np.linalg.norm(steering_vector)
    if steering_norm > 1e-10:
        steering_vector = steering_vector / steering_norm
    steering_vector_flat = steering_vector.flatten()

    # Combine all vectors for PCA
    X_combined = np.vstack(
        [hate_yes_vectors, hate_no_vectors, safe_yes_vectors, safe_no_vectors]
    )

    # Create labels (0=hate_yes, 1=hate_no, 2=safe_yes, 3=safe_no)
    labels = np.concatenate(
        [
            np.zeros(len(hate_yes_vectors)),
            np.ones(len(hate_no_vectors)),
            np.full(len(safe_yes_vectors), 2),
            np.full(len(safe_no_vectors), 3),
        ]
    )

    # Project data to 2D for visualization
    # First component: steering vector direction
    projection1 = np.array(
        [np.dot(x.flatten(), steering_vector_flat) for x in X_combined]
    )

    # Compute residuals for second direction
    residuals = X_combined - np.outer(projection1, steering_vector_flat)

    # Add small regularization to avoid numerical issues
    epsilon = 1e-8
    residuals += np.random.normal(0, epsilon, residuals.shape)

    # Find second direction (orthogonal to steering vector)
    pca = PCA(n_components=1, svd_solver="arpack")
    pca.fit(residuals)
    second_direction = pca.components_[0]

    # Project onto second direction
    projection2 = np.array(
        [np.dot(x.flatten(), second_direction.flatten()) for x in X_combined]
    )

    # Create 2D projections
    X_2d = np.column_stack([projection1, projection2])

    # Create grid for decision boundary
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Reconstruct grid points in original space
    grid_points = np.array([xx.ravel(), yy.ravel()]).T
    original_space = np.outer(grid_points[:, 0], steering_vector_flat) + np.outer(
        grid_points[:, 1], second_direction.flatten()
    )
    original_space = original_space.astype(np.float32)

    # Predict using CCS probe
    grid_preds, grid_confidences = ccs.predict_from_vectors(original_space)
    grid_preds = grid_preds.reshape(xx.shape)
    grid_confidences = grid_confidences.reshape(xx.shape)

    # Plot decision boundary with higher contrast
    contour_fill = ax.contourf(
        xx, yy, grid_preds, alpha=0.5, cmap="RdBu_r", levels=np.linspace(0, 1, 11)
    )

    # Add explicit decision boundary line
    decision_boundary = ax.contour(
        xx,
        yy,
        grid_confidences,
        levels=[0.5],
        colors="black",
        linestyles="dashed",
        linewidths=2,
    )

    # Add colorbar
    cbar = plt.colorbar(contour_fill, ax=ax, label="Prediction (0=Hate, 1=Safe)")

    # Define custom colors and markers for each category
    category_colors = {
        0: "#FF0000",  # Hate Yes - Red
        1: "#0000FF",  # Hate No - Blue
        2: "#00CCFF",  # Safe Yes - Light Blue
        3: "#FF00CC",  # Safe No - Pink
    }

    category_markers = {
        0: "o",  # Hate Yes - Circle
        1: "s",  # Hate No - Square
        2: "^",  # Safe Yes - Triangle up
        3: "v",  # Safe No - Triangle down
    }

    category_names = {
        0: "Hate Yes",
        1: "Hate No",
        2: "Safe Yes",
        3: "Safe No",
    }

    # Plot each category with distinct color and marker
    for label_id in np.unique(labels):
        idx = labels == label_id
        ax.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            c=category_colors[label_id],
            marker=category_markers[label_id],
            s=70,
            alpha=0.85,
            edgecolor="black",
            linewidths=0.5,
            label=category_names[label_id],
        )

    # Add steering vector direction
    ax.arrow(
        0,
        0,
        -1,
        0,
        color="black",
        width=0.01,
        head_width=0.1,
        head_length=0.1,
        length_includes_head=True,
        label="Steering Direction",
    )

    # Build a more detailed title with all available information
    title_parts = ["Detailed CCS Decision Boundary"]

    title_parts.append(f"Layer {layer_idx}")

    if strategy is not None:
        title_parts.append(f"Strategy: {strategy}")

    if steering_coefficient is not None:
        title_parts.append(f"Coef: {steering_coefficient}")

    if pair_type is not None:
        title_parts.append(f"Pair: {pair_type}")

    # Join all parts with pipes
    title = " | ".join(title_parts)
    plt.title(title, fontsize=14)

    # Add axis labels
    plt.xlabel("Steering Direction", fontsize=12)
    plt.ylabel("Orthogonal Direction", fontsize=12)

    # Add legend
    plt.legend(fontsize=10)

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Add description box explaining the plot
    description = """Description: This plot shows the decision boundary of the CCS probe in the space defined by the steering vector and its orthogonal complement.

    Data Categories:
    - Red points (Hate): Combined category of hate_yes (hate speech with "Yes") and safe_no (safe speech with "No")
    - Blue points (Safe): Combined category of safe_yes (safe speech with "Yes") and hate_no (hate speech with "No")

    Ideal Case:
    - Clear separation between hate (red) and safe (blue) content
    - Decision boundary should be roughly perpendicular to the steering direction
    - Points should cluster into two distinct groups

    Interpretation:
    - The steering vector direction (horizontal axis) shows how content changes when steered
    - The orthogonal direction (vertical axis) shows variations that preserve the steering effect
    - The decision boundary (colored regions) shows where the probe switches between hate and safe predictions
    - A clear boundary indicates the probe can reliably distinguish between content types

    Note: The steering vector points from hate to safe direction because it's calculated as safe_mean - hate_mean.
    When applying positive steering coefficients, content moves in this direction (toward safe classification).
    """

    # Add text box with description
    plt.figtext(
        0.5,
        0.01,
        description,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        wrap=True,
    )

    # Adjust layout for text
    plt.tight_layout(rect=(0, 0.18, 1, 1))

    # Save plot if log_base is provided
    if log_base:
        plt.savefig(
            f"{log_base}_detailed_decision_boundary.png", dpi=300, bbox_inches="tight"
        )

    return fig
