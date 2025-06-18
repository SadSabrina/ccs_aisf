import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def detect_and_handle_nans(data, layer_idx, method="interpolate", verbose=True):
    """
    1. RMSNorm vs LayerNorm - The Main Culprit
    Mistral 7B uses RMSNorm (Root Mean Square Normalization) instead of the traditional LayerNorm used by models like GPT-2, BERT, and older LLaMA versions. Towards Data ScienceMistral
    The Critical Difference:

    LayerNorm: y = (x - mean(x)) / sqrt(var(x) + ε) * γ + β
    RMSNorm: y = x / sqrt(mean(x²) + ε) * γ (no mean subtraction!)

    RMSNorm "does not keep track of or subtract the mean, it only normalizes by the norm. Notice: norm, not standard deviation, because we did not subtract the mean." llm.c/doc/layernorm/layernorm.md at master · karpathy/llm.c
    2. Why This Causes NaN Issues
    Mathematical Instability:

    RMSNorm uses sqrt(mean(x²) + ε) in the denominator Transformer architecture variation: RMSNorm - MartinLwx's Blog
    If the input x has extreme values, x² can become very large
    When mean(x²) approaches the limits of float32 precision, sqrt() can produce NaN
    Without mean centering, extreme outliers aren't normalized away

    LayerNorm's Protection:

    LayerNorm "permanently removes the component of every vector along the uniform vector" through mean subtraction Re-Introducing LayerNorm: Geometric Meaning, Irreversibility and a Comparative Study with RMSNorm
    This mean subtraction provides numerical stability by keeping values centered Because of LayerNorm, Directions in GPT-2 MLP Layers are Monosemantic — LessWrong
    The variance calculation is more stable than raw sum-of-squares


        Detect and handle NaN/infinite values in data with multiple strategies.

        Parameters:
            data: Input data array
            layer_idx: Layer index for logging
            method: Strategy for handling NaNs
                - "interpolate": Replace with interpolated values
                - "zero": Replace with zeros
                - "mean": Replace with layer mean
                - "median": Replace with layer median
                - "remove": Remove samples with NaNs (returns mask)
                - "clip": Clip extreme values before they become NaN
            verbose: Whether to print diagnostic information

        Returns:
            cleaned_data: Data with NaNs handled
            nan_mask: Boolean mask indicating which samples had NaNs (for "remove" method)
            nan_stats: Dictionary with NaN statistics
    """
    original_shape = data.shape
    nan_mask = np.isnan(data).any(axis=1) if data.ndim > 1 else np.isnan(data)
    inf_mask = np.isinf(data).any(axis=1) if data.ndim > 1 else np.isinf(data)
    problem_mask = nan_mask | inf_mask

    nan_stats = {
        "total_samples": data.shape[0],
        "nan_samples": np.sum(nan_mask),
        "inf_samples": np.sum(inf_mask),
        "problem_samples": np.sum(problem_mask),
        "nan_percentage": np.sum(nan_mask) / data.shape[0] * 100,
        "problem_percentage": np.sum(problem_mask) / data.shape[0] * 100,
    }

    if verbose and (nan_stats["problem_samples"] > 0):
        print(f"Layer {layer_idx} NaN/Inf Analysis:")
        print(f"  Total samples: {nan_stats['total_samples']}")
        print(
            f"  NaN samples: {nan_stats['nan_samples']} ({nan_stats['nan_percentage']:.1f}%)"
        )
        print(f"  Inf samples: {nan_stats['inf_samples']}")
        print(
            f"  Total problematic: {nan_stats['problem_samples']} ({nan_stats['problem_percentage']:.1f}%)"
        )

    if nan_stats["problem_samples"] == 0:
        return data, np.ones(data.shape[0], dtype=bool), nan_stats

    # Handle different strategies
    if method == "remove":
        valid_mask = ~problem_mask
        cleaned_data = data[valid_mask]
        if verbose:
            print(
                f"  Strategy: Removed {nan_stats['problem_samples']} problematic samples"
            )
        return cleaned_data, valid_mask, nan_stats

    elif method == "interpolate":
        cleaned_data = data.copy()
        if data.ndim > 1:
            # For each sample with NaN, interpolate from valid samples
            for i in range(data.shape[0]):
                if problem_mask[i]:
                    # Find nearest valid samples
                    valid_indices = np.where(~problem_mask)[0]
                    if len(valid_indices) > 0:
                        # Use mean of valid samples
                        cleaned_data[i] = np.nanmean(data[valid_indices], axis=0)

        # Final cleanup
        cleaned_data = np.nan_to_num(cleaned_data, nan=0.0, posinf=0.0, neginf=0.0)
        if verbose:
            print(
                f"  Strategy: Interpolated {nan_stats['problem_samples']} problematic samples"
            )

    elif method == "mean":
        # Replace with layer mean (ignoring NaN samples)
        layer_mean = np.nanmean(data, axis=0)
        layer_mean = np.nan_to_num(layer_mean, nan=0.0, posinf=0.0, neginf=0.0)

        cleaned_data = data.copy()
        if data.ndim > 1:
            for i in range(data.shape[0]):
                if problem_mask[i]:
                    cleaned_data[i] = layer_mean
        else:
            cleaned_data[problem_mask] = layer_mean

        cleaned_data = np.nan_to_num(cleaned_data, nan=0.0, posinf=0.0, neginf=0.0)
        if verbose:
            print(
                f"  Strategy: Replaced with layer mean for {nan_stats['problem_samples']} samples"
            )

    elif method == "median":
        # Replace with layer median (ignoring NaN samples)
        layer_median = np.nanmedian(data, axis=0)
        layer_median = np.nan_to_num(layer_median, nan=0.0, posinf=0.0, neginf=0.0)

        cleaned_data = data.copy()
        if data.ndim > 1:
            for i in range(data.shape[0]):
                if problem_mask[i]:
                    cleaned_data[i] = layer_median
        else:
            cleaned_data[problem_mask] = layer_median

        cleaned_data = np.nan_to_num(cleaned_data, nan=0.0, posinf=0.0, neginf=0.0)
        if verbose:
            print(
                f"  Strategy: Replaced with layer median for {nan_stats['problem_samples']} samples"
            )

    elif method == "zero":
        cleaned_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if verbose:
            print(
                f"  Strategy: Replaced with zeros for {nan_stats['problem_samples']} samples"
            )

    elif method == "clip":
        # Clip extreme values to reasonable ranges before they become NaN
        cleaned_data = data.copy()

        # Calculate reasonable bounds (e.g., 3 standard deviations from mean)
        valid_data = data[~problem_mask]
        if len(valid_data) > 0:
            if data.ndim > 1:
                means = np.nanmean(valid_data, axis=0)
                stds = np.nanstd(valid_data, axis=0)
                lower_bound = means - 3 * stds
                upper_bound = means + 3 * stds

                cleaned_data = np.clip(cleaned_data, lower_bound, upper_bound)
            else:
                mean_val = np.nanmean(valid_data)
                std_val = np.nanstd(valid_data)
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val

                cleaned_data = np.clip(cleaned_data, lower_bound, upper_bound)

        cleaned_data = np.nan_to_num(cleaned_data, nan=0.0, posinf=0.0, neginf=0.0)
        if verbose:
            print(
                f"  Strategy: Clipped extreme values for {nan_stats['problem_samples']} samples"
            )

    else:
        # Default: replace with zeros
        cleaned_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if verbose:
            print(
                f"  Strategy: Default replacement with zeros for {nan_stats['problem_samples']} samples"
            )

    return cleaned_data, np.ones(data.shape[0], dtype=bool), nan_stats


def robust_pca_with_fallbacks(data, n_components=5, layer_idx=None):
    """
    Robust PCA implementation with multiple fallback strategies.

    Parameters:
        data: Input data for PCA
        n_components: Desired number of components
        layer_idx: Layer index for logging

    Returns:
        X_pca: PCA-transformed data
        success: Whether PCA was successful
        method_used: Which method was successful
    """
    if layer_idx is None:
        layer_idx = "unknown"

    # Strategy 1: Try standard PCA
    try:
        # Check for sufficient variance
        if np.allclose(data, data[0]):
            return np.zeros((data.shape[0], min(n_components, 2))), False, "no_variance"

        # Adjust n_components based on data constraints
        max_components = min(n_components, data.shape[0] - 1, data.shape[1])
        if max_components < 1:
            return np.zeros((data.shape[0], 1)), False, "insufficient_data"

        pca = PCA(n_components=max_components)
        X_pca = pca.fit_transform(data)

        # Check if result contains NaN
        if np.isnan(X_pca).any():
            raise ValueError("PCA produced NaN values")

        return X_pca, True, "standard_pca"

    except Exception as e:
        print(f"Layer {layer_idx}: Standard PCA failed: {e}")

    # Strategy 2: Try PCA with SVD solver
    try:
        max_components = min(n_components, data.shape[0] - 1, data.shape[1])
        pca = PCA(n_components=max_components, svd_solver="full")
        X_pca = pca.fit_transform(data)

        if not np.isnan(X_pca).any():
            return X_pca, True, "full_svd_pca"

    except Exception as e:
        print(f"Layer {layer_idx}: Full SVD PCA failed: {e}")

    # Strategy 3: Try with randomized SVD
    try:
        max_components = min(n_components, data.shape[0] - 1, data.shape[1], 10)
        if max_components > 0:
            pca = PCA(
                n_components=max_components, svd_solver="randomized", random_state=42
            )
            X_pca = pca.fit_transform(data)

            if not np.isnan(X_pca).any():
                return X_pca, True, "randomized_pca"

    except Exception as e:
        print(f"Layer {layer_idx}: Randomized PCA failed: {e}")

    # Strategy 4: Simple standardization without PCA
    try:
        print(f"Layer {layer_idx}: Falling back to standardized coordinates")
        data_std = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        data_std = np.nan_to_num(data_std, nan=0.0)

        # Return first two dimensions as "PCA" result
        if data_std.shape[1] >= 2:
            return data_std[:, :2], True, "standardized_fallback"
        else:
            # Create a second dimension with small random noise
            result = np.zeros((data_std.shape[0], 2))
            result[:, 0] = data_std[:, 0]
            result[:, 1] = np.random.normal(0, 0.1, data_std.shape[0])
            return result, True, "single_dim_fallback"

    except Exception as e:
        print(f"Layer {layer_idx}: Standardization fallback failed: {e}")

    # Strategy 5: Last resort - random projection
    print(f"Layer {layer_idx}: Using random projection as final fallback")
    np.random.seed(42)  # For reproducibility
    n_samples = data.shape[0]
    result = np.random.normal(0, 1, (n_samples, min(n_components, 2)))
    return result, False, "random_fallback"


def plot_pca_or_tsne_layerwise(
    X_pos,
    X_neg,
    hue,
    standardize=True,
    reshape=None,
    n_components=5,
    components=None,
    mode="pca",
    plot_title="pca_or_tsne_layerwise",
    save_path=None,
    nan_strategy="interpolate",  # New parameter
    min_valid_percentage=50,  # New parameter: minimum % of valid samples to proceed
):
    """
    Robust PCA or T-SNE-clustering for each hidden layer plot with comprehensive NaN handling.

    New Parameters:
        nan_strategy: Strategy for handling NaN values
            - "interpolate": Replace NaNs with interpolated values
            - "mean": Replace with layer mean
            - "median": Replace with layer median
            - "remove": Remove samples with NaNs (adjust labels accordingly)
            - "zero": Replace with zeros
            - "clip": Clip extreme values
        min_valid_percentage: Minimum percentage of valid samples required to proceed with layer
    """
    if components is None:
        components = [0, 1]

    if len(X_pos.shape) == 2:
        if reshape is None:
            reshape = [
                int(i)
                for i in input("Get reshape params (len data, n_layers)").split(",")
            ]
        X_pos = X_pos.reshape(reshape[0], reshape[1], -1)
        X_neg = X_neg.reshape(reshape[0], reshape[1], -1)

    n_layers = X_pos.shape[1]
    original_n_samples = X_pos.shape[0]

    # Track statistics across all layers
    layer_stats = {}
    successful_layers = []

    fig, axes = plt.subplots((n_layers - 1) // 6 + 1, 6, figsize=(24, 13))
    axes = axes.flatten()

    print("\nRobust layer-wise analysis starting:")
    print(f"  Total layers: {n_layers}")
    print(f"  NaN handling strategy: {nan_strategy}")
    print(f"  Minimum valid samples: {min_valid_percentage}%")

    for layer_idx in range(1, n_layers):
        ax = axes[layer_idx - 1]

        try:
            X_pos_layer = X_pos[:, layer_idx, :]
            X_neg_layer = X_neg[:, layer_idx, :]

            # Combining hidden states of a model to plot a graph
            states_data = X_pos_layer - X_neg_layer

            # Handle NaN values with chosen strategy
            if nan_strategy == "remove":
                cleaned_data, valid_mask, nan_stats = detect_and_handle_nans(
                    states_data, layer_idx, method=nan_strategy, verbose=True
                )
                # Adjust labels to match cleaned data
                layer_hue = hue[valid_mask] if hasattr(hue, "__getitem__") else hue
            else:
                cleaned_data, valid_mask, nan_stats = detect_and_handle_nans(
                    states_data, layer_idx, method=nan_strategy, verbose=True
                )
                layer_hue = hue

            # Check if we have enough valid samples
            valid_percentage = (cleaned_data.shape[0] / original_n_samples) * 100
            if valid_percentage < min_valid_percentage:
                print(
                    f"Layer {layer_idx}: Only {valid_percentage:.1f}% valid samples, skipping"
                )
                ax.text(
                    0.5,
                    0.5,
                    f"Layer {layer_idx}\nInsufficient valid data\n({valid_percentage:.1f}% valid)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                layer_stats[layer_idx] = {
                    "status": "insufficient_data",
                    "nan_stats": nan_stats,
                }
                continue

            # Store statistics
            layer_stats[layer_idx] = {"status": "processing", "nan_stats": nan_stats}

            # Standardization with robust handling
            if standardize:
                mean_vals = cleaned_data.mean(axis=0)
                std_vals = cleaned_data.std(axis=0)

                # Handle zero/small standard deviations
                std_vals = np.where(std_vals < 1e-8, 1.0, std_vals)

                # Handle NaN in statistics
                mean_vals = np.nan_to_num(mean_vals, nan=0.0)
                std_vals = np.nan_to_num(std_vals, nan=1.0)

                cleaned_data = (cleaned_data - mean_vals) / std_vals
                cleaned_data = np.nan_to_num(
                    cleaned_data, nan=0.0, posinf=0.0, neginf=0.0
                )

            # Apply robust PCA/t-SNE
            if mode == "pca":
                X_proj, pca_success, method_used = robust_pca_with_fallbacks(
                    cleaned_data, n_components, layer_idx
                )
            elif mode == "tsne":
                try:
                    # t-SNE has stricter requirements
                    if cleaned_data.shape[0] < 2:
                        raise ValueError("Insufficient samples for t-SNE")

                    max_components = min(n_components, 3)  # t-SNE typically max 3D
                    perplexity = min(30, cleaned_data.shape[0] - 1)

                    if perplexity < 1:
                        perplexity = 1

                    tsne = TSNE(
                        n_components=max_components,
                        metric="cosine",
                        perplexity=perplexity,
                        random_state=42,
                    )
                    X_proj = tsne.fit_transform(cleaned_data)
                    pca_success = True
                    method_used = "tsne"
                except Exception as e:
                    print(f"Layer {layer_idx}: t-SNE failed, falling back to PCA: {e}")
                    X_proj, pca_success, method_used = robust_pca_with_fallbacks(
                        cleaned_data, n_components, layer_idx
                    )
            else:
                raise ValueError("mode must be 'pca' or 'tsne'")

            # Plotting with error handling
            if pca_success and X_proj.shape[1] >= max(components) + 1:
                # Ensure we have valid data for plotting
                plot_components = [min(c, X_proj.shape[1] - 1) for c in components]

                scatter = ax.scatter(
                    X_proj[:, plot_components[0]],
                    X_proj[:, plot_components[1]],
                    c=layer_hue,
                    cmap="Set1",
                    alpha=0.7,
                    s=30,
                )

                # Success!
                successful_layers.append(layer_idx)
                layer_stats[layer_idx]["status"] = "success"
                layer_stats[layer_idx]["method"] = method_used
                layer_stats[layer_idx]["n_samples_plotted"] = X_proj.shape[0]

            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Layer {layer_idx}\nVisualization failed\nMethod: {method_used}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                layer_stats[layer_idx]["status"] = "failed"
                layer_stats[layer_idx]["method"] = method_used

        except Exception as e:
            print(f"Layer {layer_idx}: Unexpected error: {e}")
            ax.text(
                0.5,
                0.5,
                f"Layer {layer_idx}\nError: {str(e)[:30]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            layer_stats[layer_idx] = {"status": "error", "error": str(e)}

        # Set title with status information
        status = layer_stats[layer_idx]["status"]
        if status == "success":
            title_color = "green"
            method = layer_stats[layer_idx].get("method", "")
            title = f"Layer {layer_idx} ✓\n({method})"
        elif status == "failed":
            title_color = "orange"
            title = f"Layer {layer_idx} ⚠"
        else:
            title_color = "red"
            title = f"Layer {layer_idx} ✗"

        ax.set_title(title, fontsize=10, color=title_color)
        ax.legend().set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis("off")

    # Create legend
    if successful_layers:
        # Use the last successful layer for legend
        last_success_ax = axes[successful_layers[-1] - 1]
        handles, labels = last_success_ax.get_legend_handles_labels()
        if not handles:  # If scatter plot doesn't have handles, create them
            unique_labels = np.unique(hue)
            colors = plt.cm.get_cmap("Set1")(np.linspace(0, 1, len(unique_labels)))
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[i],
                    markersize=8,
                )
                for i in range(len(unique_labels))
            ]
            labels = [f"Label {label}" for label in unique_labels]

        title = hue.name if hasattr(hue, "name") else "Label"
        fig.legend(handles, labels, loc="upper right", fontsize=12, title=title)

    fig.suptitle(
        f"{plot_title}\n(Strategy: {nan_strategy}, {len(successful_layers)}/{n_layers-1} layers successful)",
        fontsize=16,
    )

    # Print summary statistics
    print("\nLayer-wise Analysis Summary:")
    print(f"  Successful layers: {len(successful_layers)}/{n_layers-1}")
    print(f"  Success rate: {len(successful_layers)/(n_layers-1)*100:.1f}%")

    # Count NaN statistics across all layers
    total_nan_samples = sum(
        stats.get("nan_stats", {}).get("nan_samples", 0)
        for stats in layer_stats.values()
    )
    total_samples = sum(
        stats.get("nan_stats", {}).get("total_samples", 0)
        for stats in layer_stats.values()
    )

    if total_samples > 0:
        print(f"  Total NaN samples across all layers: {total_nan_samples}")
        print(f"  Overall NaN rate: {total_nan_samples/total_samples*100:.1f}%")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Robust layer-wise plot saved to: {save_path}")

        # Save detailed statistics
        stats_path = save_path.replace(".png", "_stats.txt")
        with open(stats_path, "w") as f:
            f.write("Layer-wise Analysis Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Strategy used: {nan_strategy}\n")
            f.write(f"Successful layers: {successful_layers}\n")
            f.write(f"Success rate: {len(successful_layers)/(n_layers-1)*100:.1f}%\n\n")

            for layer_idx, stats in layer_stats.items():
                f.write(f"Layer {layer_idx}:\n")
                f.write(f"  Status: {stats['status']}\n")
                if "nan_stats" in stats:
                    ns = stats["nan_stats"]
                    f.write(
                        f"  NaN samples: {ns.get('nan_samples', 0)}/{ns.get('total_samples', 0)} ({ns.get('nan_percentage', 0):.1f}%)\n"
                    )
                if "method" in stats:
                    f.write(f"  Method: {stats['method']}\n")
                if "n_samples_plotted" in stats:
                    f.write(f"  Samples plotted: {stats['n_samples_plotted']}\n")
                f.write("\n")

        print(f"Detailed statistics saved to: {stats_path}")
    else:
        plt.show()

    return fig, layer_stats


def plot_pca_components_matrix(
    X_pos,
    X_neg,
    hue,
    layer_idx,
    standardize=True,
    n_components=5,
    mode="pca",
    plot_title=None,
    save_path=None,
    nan_strategy="interpolate",
    min_valid_percentage=50,
):
    """
    Create triangle/matrix plot showing all PCA components against each other for a single layer.
    Enhanced with robust PCA/t-SNE and comprehensive NaN handling.

    Parameters:
        X_pos (np.ndarray): Positive samples for single layer, shape (n_samples, hidden_dim)
        X_neg (np.ndarray): Negative samples for single layer, shape (n_samples, hidden_dim)
        hue (np.ndarray | pd.Series): y values
        layer_idx (int): Layer index for title
        standardize (bool): If standardization is needed before PCA
        n_components (int): Number of PCA/TSNE components
        mode (str): 'pca' or 'tsne'
        plot_title (str): Plot title
        save_path (str): Path to save the plot
        nan_strategy (str): Strategy for handling NaN values
            - "interpolate": Replace NaNs with interpolated values
            - "mean": Replace with layer mean
            - "median": Replace with layer median
            - "remove": Remove samples with NaNs (adjust labels accordingly)
            - "zero": Replace with zeros
            - "clip": Clip extreme values
        min_valid_percentage (float): Minimum percentage of valid samples required
    """
    try:
        # Combine hidden states
        states_data = X_pos - X_neg

        # Handle NaN values with chosen strategy
        if nan_strategy == "remove":
            cleaned_data, valid_mask, nan_stats = detect_and_handle_nans(
                states_data, layer_idx, method=nan_strategy, verbose=True
            )
            # Adjust labels to match cleaned data
            layer_hue = hue[valid_mask] if hasattr(hue, "__getitem__") else hue
        else:
            cleaned_data, valid_mask, nan_stats = detect_and_handle_nans(
                states_data, layer_idx, method=nan_strategy, verbose=True
            )
            layer_hue = hue

        # Check if we have enough valid samples
        original_n_samples = (
            len(hue) if hasattr(hue, "__len__") else states_data.shape[0]
        )
        valid_percentage = (cleaned_data.shape[0] / original_n_samples) * 100

        if valid_percentage < min_valid_percentage:
            print(
                f"Layer {layer_idx}: Only {valid_percentage:.1f}% valid samples (< {min_valid_percentage}%), skipping"
            )
            return None

        print(
            f"Layer {layer_idx}: Processing {cleaned_data.shape[0]} samples ({valid_percentage:.1f}% valid)"
        )

        # Enhanced standardization with NaN handling
        if standardize:
            mean_vals = cleaned_data.mean(axis=0)
            std_vals = cleaned_data.std(axis=0)

            # Handle zero standard deviation and NaN values
            std_vals = np.where(std_vals < 1e-8, 1.0, std_vals)

            # Handle NaN in mean and std
            mean_vals = np.nan_to_num(mean_vals, nan=0.0, posinf=0.0, neginf=0.0)
            std_vals = np.nan_to_num(std_vals, nan=1.0, posinf=1.0, neginf=1.0)

            cleaned_data = (cleaned_data - mean_vals) / std_vals

            # Final check for NaN values after standardization
            if np.isnan(cleaned_data).any():
                print(
                    f"Warning: Found NaN values in layer {layer_idx} after standardization, cleaning again"
                )
                cleaned_data = np.nan_to_num(
                    cleaned_data, nan=0.0, posinf=0.0, neginf=0.0
                )

        # Apply robust dimensionality reduction
        if mode == "pca":
            X_proj, success, method_used = robust_pca_with_fallbacks(
                cleaned_data, n_components, layer_idx
            )
        elif mode == "tsne":
            try:
                # t-SNE has stricter requirements
                if cleaned_data.shape[0] < 2:
                    raise ValueError("Insufficient samples for t-SNE")

                max_components = min(n_components, 3)  # t-SNE typically max 3D
                perplexity = min(30, cleaned_data.shape[0] - 1)
                if perplexity < 1:
                    perplexity = 1

                tsne = TSNE(
                    n_components=max_components,
                    metric="cosine",
                    perplexity=perplexity,
                    random_state=42,
                )
                X_proj = tsne.fit_transform(cleaned_data)
                success = True
                method_used = "tsne"
            except Exception as e:
                print(
                    f"Layer {layer_idx}: t-SNE failed, falling back to robust PCA: {e}"
                )
                X_proj, success, method_used = robust_pca_with_fallbacks(
                    cleaned_data, n_components, layer_idx
                )
        else:
            raise ValueError("mode must be 'pca' or 'tsne'")

        if not success:
            print(f"Layer {layer_idx}: All dimensionality reduction methods failed")
            return None

        print(
            f"Layer {layer_idx}: Successfully applied {method_used}, got {X_proj.shape[1]} components"
        )

        # Create DataFrame for plotting
        actual_n_components = X_proj.shape[1]
        df = pd.DataFrame(
            X_proj, columns=[f"PC{i}" for i in range(actual_n_components)]
        )

        # Handle hue data carefully
        if hasattr(layer_hue, "__len__") and len(layer_hue) == len(df):
            df["label"] = layer_hue
        else:
            # If hue is scalar or mismatched, create default labels
            df["label"] = 0

        # Ensure we have valid data for matrix plot
        if actual_n_components < 1:
            print(f"Layer {layer_idx}: No valid components for matrix plot")
            return None

        # Create matrix plot (triangle plot) - adjust for actual components
        fig, axes = plt.subplots(
            actual_n_components, actual_n_components, figsize=(15, 15)
        )

        # Ensure axes is always a 2D array for consistent indexing
        if actual_n_components == 1:
            axes = np.array([[axes]])
        elif not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(-1, 1)

        # Create the matrix plot
        for i in range(actual_n_components):
            for j in range(actual_n_components):
                # Get the correct axis
                if actual_n_components == 1:
                    ax = axes[0, 0]
                else:
                    ax = axes[i, j] if actual_n_components > 1 else axes[i]

                try:
                    if i == j:
                        # Diagonal: histogram/density plot
                        unique_labels = df["label"].unique()
                        for label_val in unique_labels:
                            subset = df[df["label"] == label_val]
                            if len(subset) > 0:
                                # Ensure we have valid data for histogram
                                data_to_plot = subset[f"PC{i}"].dropna()
                                if len(data_to_plot) > 0:
                                    ax.hist(
                                        data_to_plot,
                                        alpha=0.6,
                                        label=f"Label {label_val}",
                                        bins=min(20, len(data_to_plot) // 2 + 1),
                                    )

                        ax.set_xlabel(f"PC{i}")
                        ax.set_ylabel("Density")
                        if i == 0 and len(unique_labels) > 1:
                            ax.legend()

                    elif i > j:
                        # Lower triangle: scatter plots
                        # Only plot if we have valid data
                        if (
                            len(df) > 0
                            and not df[f"PC{j}"].isna().all()
                            and not df[f"PC{i}"].isna().all()
                        ):
                            sns.scatterplot(
                                data=df,
                                x=f"PC{j}",
                                y=f"PC{i}",
                                hue="label",
                                ax=ax,
                                alpha=0.7,
                                s=30,
                            )
                            ax.set_xlabel(f"PC{j}")
                            ax.set_ylabel(f"PC{i}")
                            ax.legend().set_visible(False)
                            ax.grid(True, alpha=0.3)
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                "No valid\ndata",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )

                    else:
                        # Upper triangle: leave empty or add info
                        ax.axis("off")
                        if i == 0 and j == actual_n_components - 1:
                            # Add info text in top-right corner
                            info_text = f"Method: {method_used}\nSamples: {len(df)}\nValid: {valid_percentage:.1f}%"
                            ax.text(
                                0.5,
                                0.5,
                                info_text,
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                                fontsize=10,
                                bbox=dict(
                                    boxstyle="round,pad=0.3",
                                    facecolor="lightgray",
                                    alpha=0.8,
                                ),
                            )

                except Exception as e:
                    print(
                        f"Error plotting subplot ({i},{j}) for layer {layer_idx}: {e}"
                    )
                    ax.text(
                        0.5,
                        0.5,
                        f"Plot error\n{str(e)[:20]}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=8,
                    )

        # Set main title
        if plot_title is None:
            plot_title = f"PCA Components Matrix - Layer {layer_idx}"

        # Add method and sample info to title
        plot_title += f"\n(Method: {method_used}, {valid_percentage:.1f}% valid samples, Strategy: {nan_strategy})"

        fig.suptitle(plot_title, fontsize=16, y=0.98)
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"PCA components matrix saved to: {save_path}")
        else:
            plt.show()

        return fig

    except Exception as e:
        print(f"Error creating components matrix for layer {layer_idx}: {e}")
        import traceback

        traceback.print_exc()
        return None


def plot_all_layers_components_matrix(
    X_pos,
    X_neg,
    hue,
    start_layer=0,
    standardize=True,
    n_components=5,
    mode="pca",
    plot_title_prefix="PCA Components Matrix",
    save_dir=None,
    nan_strategy="interpolate",
    min_valid_percentage=50,
    max_layers_to_process=None,  # New parameter to limit processing
):
    """
    Create PCA components matrix plots for all layers starting from start_layer with robust handling.
    Enhanced with comprehensive NaN handling and error recovery.

    Parameters:
        X_pos (np.ndarray): Positive samples, shape (n_samples, n_layers, hidden_dim)
        X_neg (np.ndarray): Negative samples, shape (n_samples, n_layers, hidden_dim)
        hue (np.ndarray | pd.Series): y values
        start_layer (int): Starting layer index
        standardize (bool): If standardization is needed before PCA
        n_components (int): Number of PCA/TSNE components
        mode (str): 'pca' or 'tsne'
        plot_title_prefix (str): Prefix for plot titles
        save_dir (Path): Directory to save plots
        nan_strategy (str): Strategy for handling NaN values
        min_valid_percentage (float): Minimum percentage of valid samples required
        max_layers_to_process (int): Maximum number of layers to process (None = all)
    """
    n_samples, n_layers, hidden_dim = X_pos.shape
    saved_plots = []
    successful_layers = []
    failed_layers = []
    layer_stats = {}

    print(f"\nRobust {mode.upper()} Components Matrix Analysis:")
    print(f"  Total layers available: {n_layers}")
    print(f"  Processing layers: {start_layer} to {n_layers-1}")
    print(f"  NaN handling strategy: {nan_strategy}")
    print(f"  Minimum valid percentage: {min_valid_percentage}%")

    if max_layers_to_process:
        end_layer = min(n_layers, start_layer + max_layers_to_process)
        print(
            f"  Limited to first {max_layers_to_process} layers (up to layer {end_layer-1})"
        )
    else:
        end_layer = n_layers

    # Pre-analyze NaN patterns across layers
    print("\nPre-analyzing NaN patterns...")
    layer_nan_stats = {}
    for layer_idx in range(start_layer, end_layer):
        states_data = X_pos[:, layer_idx, :] - X_neg[:, layer_idx, :]
        nan_mask = np.isnan(states_data).any(axis=1)
        inf_mask = np.isinf(states_data).any(axis=1)
        problem_mask = nan_mask | inf_mask

        layer_nan_stats[layer_idx] = {
            "nan_percentage": np.sum(problem_mask) / len(problem_mask) * 100,
            "problem_samples": np.sum(problem_mask),
        }

    # Sort layers by NaN percentage (process best layers first)
    sorted_layers = sorted(
        layer_nan_stats.items(), key=lambda x: x[1]["nan_percentage"]
    )

    print("Layer NaN analysis:")
    for layer_idx, stats in sorted_layers[:5]:  # Show top 5 best layers
        print(
            f"  Layer {layer_idx}: {stats['nan_percentage']:.1f}% problematic samples"
        )

    # Process layers
    for layer_idx in range(start_layer, end_layer):
        print(f"\nProcessing layer {layer_idx+1}/{end_layer} (Layer {layer_idx})...")

        try:
            # Extract data for this layer
            X_pos_layer = X_pos[:, layer_idx, :]
            X_neg_layer = X_neg[:, layer_idx, :]

            # Pre-check: Skip if layer has too many NaN issues
            nan_pct = layer_nan_stats[layer_idx]["nan_percentage"]
            if nan_pct > 80 and nan_strategy != "remove":
                print(
                    f"  Skipping layer {layer_idx}: {nan_pct:.1f}% NaN samples (too high)"
                )
                failed_layers.append(layer_idx)
                layer_stats[layer_idx] = {
                    "status": "skipped_high_nan",
                    "nan_percentage": nan_pct,
                    "reason": f"{nan_pct:.1f}% NaN samples",
                }
                continue

            # Create plot title
            title = f"{plot_title_prefix} - Layer {layer_idx}"

            # Set save path
            save_path = None
            if save_dir:
                # Ensure save directory exists
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"components_matrix_layer_{layer_idx}.png"

            # Create the plot with robust handling
            fig = plot_pca_components_matrix(
                X_pos=X_pos_layer,
                X_neg=X_neg_layer,
                hue=hue,
                layer_idx=layer_idx,
                standardize=standardize,
                n_components=n_components,
                mode=mode,
                plot_title=title,
                save_path=str(save_path) if save_path else None,
                nan_strategy=nan_strategy,
                min_valid_percentage=min_valid_percentage,
            )

            if fig is not None:
                if save_path:
                    saved_plots.append(save_path)
                successful_layers.append(layer_idx)
                layer_stats[layer_idx] = {
                    "status": "success",
                    "nan_percentage": nan_pct,
                    "plot_path": str(save_path) if save_path else None,
                }
                print(f"  ✓ Layer {layer_idx}: SUCCESS")
            else:
                failed_layers.append(layer_idx)
                layer_stats[layer_idx] = {
                    "status": "failed_processing",
                    "nan_percentage": nan_pct,
                    "reason": "plot_pca_components_matrix returned None",
                }
                print(f"  ✗ Layer {layer_idx}: FAILED (processing error)")

        except Exception as e:
            print(f"  ✗ Layer {layer_idx}: ERROR - {e}")
            failed_layers.append(layer_idx)
            layer_stats[layer_idx] = {
                "status": "error",
                "nan_percentage": layer_nan_stats.get(layer_idx, {}).get(
                    "nan_percentage", 0
                ),
                "error": str(e),
            }
            continue

    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("ROBUST COMPONENTS MATRIX ANALYSIS SUMMARY")
    print("=" * 80)
    print(
        f"Layers processed: {start_layer} to {end_layer-1} ({end_layer - start_layer} total)"
    )
    print(f"Successful layers: {len(successful_layers)}")
    print(f"Failed layers: {len(failed_layers)}")
    print(f"Success rate: {len(successful_layers)/(end_layer - start_layer)*100:.1f}%")

    if successful_layers:
        print(f"\n✓ SUCCESSFUL LAYERS: {successful_layers}")

    if failed_layers:
        print(f"\n✗ FAILED LAYERS: {failed_layers}")

        # Group failures by reason
        failure_reasons = {}
        for layer_idx in failed_layers:
            reason = layer_stats[layer_idx].get(
                "reason", layer_stats[layer_idx].get("status", "unknown")
            )
            if reason not in failure_reasons:
                failure_reasons[reason] = []
            failure_reasons[reason].append(layer_idx)

        print("\nFailure breakdown:")
        for reason, layers in failure_reasons.items():
            print(f"  {reason}: {len(layers)} layers - {layers}")

    # Save detailed summary
    if save_dir and (successful_layers or failed_layers):
        summary_path = save_dir / f"components_matrix_summary_{mode}.txt"
        with open(summary_path, "w") as f:
            f.write("Robust Components Matrix Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write("Analysis Settings:\n")
            f.write(f"  Mode: {mode}\n")
            f.write(f"  NaN Strategy: {nan_strategy}\n")
            f.write(f"  Min Valid Percentage: {min_valid_percentage}%\n")
            f.write(f"  Layers Processed: {start_layer} to {end_layer-1}\n")
            f.write(f"  Components: {n_components}\n")
            f.write(f"  Standardize: {standardize}\n\n")

            f.write("Results:\n")
            f.write(f"  Total layers: {end_layer - start_layer}\n")
            f.write(f"  Successful: {len(successful_layers)}\n")
            f.write(f"  Failed: {len(failed_layers)}\n")
            f.write(
                f"  Success rate: {len(successful_layers)/(end_layer - start_layer)*100:.1f}%\n\n"
            )

            f.write("Detailed Layer Results:\n")
            f.write("-" * 30 + "\n")
            for layer_idx in range(start_layer, end_layer):
                stats = layer_stats.get(layer_idx, {})
                status = stats.get("status", "not_processed")
                nan_pct = stats.get("nan_percentage", 0)
                f.write(f"Layer {layer_idx}: {status} (NaN: {nan_pct:.1f}%)")
                if "reason" in stats:
                    f.write(f" - {stats['reason']}")
                if "error" in stats:
                    f.write(f" - Error: {stats['error']}")
                f.write("\n")

        print(f"\nDetailed summary saved to: {summary_path}")

    print(f"\nPlots saved to: {save_dir}")
    print(f"Total plots created: {len(saved_plots)}")
    print("=" * 80)

    return saved_plots, layer_stats


def get_results_table(ccs_results):
    """
    Enhanced results table function with NaN handling.
    """
    acc_list = []
    agreement_list = []
    agreement_abs_list = []
    s_score = []
    ci_list = []
    im_dist_list = []

    for layer in ccs_results.keys():
        # Handle potential NaN values in results
        acc = ccs_results[layer]["accuracy"]
        acc = 0.5 if np.isnan(acc) else acc  # Default to random performance
        acc_list.append(acc)

        silhouette = ccs_results[layer]["silhouette"]
        silhouette = 0.0 if np.isnan(silhouette) else silhouette
        s_score.append(silhouette)

        agreement = ccs_results[layer]["agreement"]
        agreement_mean = (
            np.nanmean(agreement) if hasattr(agreement, "__len__") else agreement
        )
        agreement_mean = 0.0 if np.isnan(agreement_mean) else agreement_mean
        agreement_list.append(agreement_mean)

        agreement_abs = (
            np.nanmedian(np.abs(agreement))
            if hasattr(agreement, "__len__")
            else abs(agreement)
        )
        agreement_abs = 0.0 if np.isnan(agreement_abs) else agreement_abs
        agreement_abs_list.append(agreement_abs)

        ci = ccs_results[layer]["contradiction idx"]
        ci_mean = np.nanmean(ci) if hasattr(ci, "__len__") else ci
        ci_mean = 0.0 if np.isnan(ci_mean) else ci_mean
        ci_list.append(ci_mean)

        im_dist = ccs_results[layer]["IM dist"]
        im_dist_mean = np.nanmean(im_dist) if hasattr(im_dist, "__len__") else im_dist
        im_dist_mean = 0.0 if np.isnan(im_dist_mean) else im_dist_mean
        im_dist_list.append(im_dist_mean)

    data = pd.DataFrame(
        index=ccs_results.keys(),
        data=np.array(
            [
                acc_list,
                s_score,
                agreement_list,
                agreement_abs_list,
                ci_list,
                im_dist_list,
            ]
        ).T,
        columns=[
            "accuracy",
            "silhouette_score",
            "agreement_score_↓",
            "abs_agreement_score",
            "contradiction_idx_↓",
            "ideal_model_dist_↓",
        ],
    )

    return data


# Additional utility functions for robust analysis
def analyze_nan_patterns(X_pos, X_neg, labels):
    """
    Analyze patterns in NaN occurrences to understand the data better.

    Returns:
        dict: Analysis results including correlations with labels, layer patterns, etc.
    """
    analysis = {
        "layer_nan_counts": {},
        "sample_nan_counts": {},
        "nan_label_correlation": {},
        "recommendations": [],
    }

    n_samples, n_layers, n_features = X_pos.shape

    # Analyze NaN patterns by layer
    for layer_idx in range(n_layers):
        pos_data = X_pos[:, layer_idx, :]
        neg_data = X_neg[:, layer_idx, :]
        diff_data = pos_data - neg_data

        nan_mask = np.isnan(diff_data).any(axis=1)
        analysis["layer_nan_counts"][layer_idx] = {
            "nan_samples": np.sum(nan_mask),
            "percentage": np.sum(nan_mask) / n_samples * 100,
        }

    # Analyze NaN patterns by sample
    for sample_idx in range(n_samples):
        pos_sample = X_pos[sample_idx, :, :]
        neg_sample = X_neg[sample_idx, :, :]
        diff_sample = pos_sample - neg_sample

        nan_layers = np.sum(np.isnan(diff_sample).any(axis=1))
        analysis["sample_nan_counts"][sample_idx] = {
            "nan_layers": nan_layers,
            "percentage": nan_layers / n_layers * 100,
            "label": labels[sample_idx] if hasattr(labels, "__getitem__") else labels,
        }

    # Analyze correlation between NaNs and labels
    sample_nan_percentages = [
        info["percentage"] for info in analysis["sample_nan_counts"].values()
    ]
    sample_labels = [info["label"] for info in analysis["sample_nan_counts"].values()]

    # Group by label
    unique_labels = np.unique(sample_labels)
    for label in unique_labels:
        label_mask = np.array(sample_labels) == label
        label_nan_percentages = np.array(sample_nan_percentages)[label_mask]

        analysis["nan_label_correlation"][label] = {
            "mean_nan_percentage": np.mean(label_nan_percentages),
            "std_nan_percentage": np.std(label_nan_percentages),
            "samples_with_nans": np.sum(label_nan_percentages > 0),
            "total_samples": len(label_nan_percentages),
        }

    # Generate recommendations
    worst_layers = sorted(
        analysis["layer_nan_counts"].items(),
        key=lambda x: x[1]["percentage"],
        reverse=True,
    )[:5]

    if worst_layers[0][1]["percentage"] > 50:
        analysis["recommendations"].append(
            f"Layer {worst_layers[0][0]} has {worst_layers[0][1]['percentage']:.1f}% NaN samples - consider excluding"
        )

    # Check if NaNs are correlated with labels
    label_nan_rates = [
        info["mean_nan_percentage"]
        for info in analysis["nan_label_correlation"].values()
    ]
    if max(label_nan_rates) - min(label_nan_rates) > 20:
        analysis["recommendations"].append(
            "NaN rates vary significantly between labels - may indicate label-specific numerical issues"
        )
