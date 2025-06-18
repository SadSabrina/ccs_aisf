# Fixed steering_analysis.py functions - splitting large plots and handling size issues

# Set matplotlib backend to non-interactive to avoid GUI issues
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")  # Use non-interactive backend


def create_comprehensive_comparison_visualizations(
    comparison_df, best_layer, steering_alpha, plots_dir
):
    """
    Create comprehensive comparison visualization plots.
    CHANGED: Split large plots into smaller, manageable visualizations
    """
    print("Creating comprehensive comparison visualizations...")
    created_plots = []

    # 1. Heatmap comparison (FIXED: Limit size and split if needed)
    heatmap_path = (
        plots_dir / f"heatmap_comparison_layer_{best_layer}_alpha_{steering_alpha}.png"
    )
    _create_metrics_heatmap_fixed(
        comparison_df, best_layer, steering_alpha, heatmap_path
    )
    created_plots.append(heatmap_path)

    # 2. Line comparison (FIXED: Reasonable subplot arrangement)
    line_path = (
        plots_dir / f"line_comparison_layer_{best_layer}_alpha_{steering_alpha}.png"
    )
    _create_metrics_line_plot_fixed(
        comparison_df, best_layer, steering_alpha, line_path
    )
    created_plots.append(line_path)

    # 3. Steering layer bars (FIXED: Split into multiple smaller plots)
    bars_paths = _create_steering_bars_plot_fixed(
        comparison_df, best_layer, steering_alpha, plots_dir
    )
    created_plots.extend(bars_paths)

    # 4. Difference analysis (FIXED: Reasonable size constraints)
    diff_path = (
        plots_dir / f"difference_analysis_layer_{best_layer}_alpha_{steering_alpha}.png"
    )
    _create_difference_analysis_plot_fixed(
        comparison_df, best_layer, steering_alpha, diff_path
    )
    created_plots.append(diff_path)

    # 5. NEW: Layer-wise summary plot
    summary_path = (
        plots_dir / f"layer_summary_layer_{best_layer}_alpha_{steering_alpha}.png"
    )
    _create_layer_summary_plot(comparison_df, best_layer, steering_alpha, summary_path)
    created_plots.append(summary_path)

    print(f"Created {len(created_plots)} comprehensive comparison plots")
    return created_plots


def _create_metrics_heatmap_fixed(comparison_df, best_layer, steering_alpha, save_path):
    """Create heatmap visualization of metrics comparison with size constraints."""
    # CHANGED: Limit the number of metrics and layers to prevent oversized plots

    # Extract base metrics (original vs steered)
    base_metrics = []
    for col in comparison_df.columns:
        if col.endswith("_original"):
            base_metrics.append(col.replace("_original", ""))

    # CHANGED: Limit to first 6 metrics and reasonable number of layers
    max_metrics = 6
    max_layers = min(20, len(comparison_df))  # Limit to 20 layers max

    selected_metrics = base_metrics[:max_metrics]
    selected_layers = comparison_df.index[:max_layers]

    # Create data for heatmap
    heatmap_data = []
    metric_labels = []

    for metric in selected_metrics:
        if (
            f"{metric}_original" in comparison_df.columns
            and f"{metric}_steered" in comparison_df.columns
        ):
            orig_values = comparison_df.loc[
                selected_layers, f"{metric}_original"
            ].values
            steered_values = comparison_df.loc[
                selected_layers, f"{metric}_steered"
            ].values

            heatmap_data.append(orig_values)
            heatmap_data.append(steered_values)
            metric_labels.extend([f"{metric}_orig", f"{metric}_steer"])

    if not heatmap_data:
        print("Warning: No valid metrics found for heatmap")
        return

    heatmap_array = np.array(heatmap_data)

    # CHANGED: Set reasonable figure size
    fig_width = min(max(8, len(selected_layers) * 0.5), 20)  # Width: 8-20 inches
    fig_height = min(max(6, len(metric_labels) * 0.3), 15)  # Height: 6-15 inches

    plt.figure(figsize=(fig_width, fig_height))

    # Create labels
    layer_labels = [f"L{i}" for i in selected_layers]  # Shorter labels

    # Create heatmap with size constraints
    sns.heatmap(
        heatmap_array,
        xticklabels=layer_labels,
        yticklabels=metric_labels,
        annot=False,  # CHANGED: Disable annotations to reduce clutter
        cmap="RdYlBu_r",
        center=0,
        cbar_kws={"label": "Metric Value"},
        fmt=".3f",
    )

    # Highlight steering layer if it's in the selected range
    if best_layer in selected_layers:
        layer_pos = list(selected_layers).index(best_layer)
        plt.axvline(x=layer_pos + 0.5, color="red", linewidth=2, alpha=0.8)

    plt.title(
        f"Metrics Heatmap (Steering Layer {best_layer}, α={steering_alpha})",
        fontsize=12,
    )
    plt.xlabel("Layers", fontsize=10)
    plt.ylabel("Metrics", fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")  # CHANGED: Reduced DPI
    plt.close()
    print(f"Heatmap comparison saved to: {save_path}")


def _create_metrics_line_plot_fixed(
    comparison_df, best_layer, steering_alpha, save_path
):
    """Create line plot visualization with reasonable constraints."""
    # Extract base metrics
    base_metrics = []
    for col in comparison_df.columns:
        if col.endswith("_original"):
            base_metrics.append(col.replace("_original", ""))

    # CHANGED: Create multiple smaller plots instead of one large plot
    max_metrics_per_plot = 4
    selected_metrics = base_metrics[:max_metrics_per_plot]

    # CHANGED: Reasonable figure size
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    layers = comparison_df.index.values

    for i, metric in enumerate(selected_metrics):
        if i >= len(axes):
            break

        ax = axes[i]

        if (
            f"{metric}_original" in comparison_df.columns
            and f"{metric}_steered" in comparison_df.columns
        ):
            orig_values = comparison_df[f"{metric}_original"].values
            steered_values = comparison_df[f"{metric}_steered"].values

            ax.plot(
                layers, orig_values, "b-o", label="Original", linewidth=2, markersize=4
            )
            ax.plot(
                layers,
                steered_values,
                "r-s",
                label="Steered",
                linewidth=2,
                markersize=4,
            )

            # Highlight steering layer
            ax.axvline(
                best_layer,
                color="green",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"Steering L{best_layer}",
            )

            ax.set_xlabel("Layer", fontsize=10)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # CHANGED: Limit x-axis ticks if too many layers
            if len(layers) > 15:
                tick_indices = np.linspace(0, len(layers) - 1, 10, dtype=int)
                ax.set_xticks(layers[tick_indices])

    # Hide unused subplots
    for i in range(len(selected_metrics), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Metrics Comparison (α={steering_alpha})", fontsize=14)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Line comparison saved to: {save_path}")


def _create_steering_bars_plot_fixed(
    comparison_df, best_layer, steering_alpha, plots_dir
):
    """
    Create bar plots showing steering effects - SPLIT INTO MULTIPLE PLOTS.
    CHANGED: This was the function causing the oversized plot error.
    """
    created_plots = []

    # Extract base metrics
    base_metrics = []
    for col in comparison_df.columns:
        if col.endswith("_original"):
            base_metrics.append(col.replace("_original", ""))

    # CHANGED: Split metrics into groups to avoid oversized plots
    metrics_per_plot = 6
    metric_groups = [
        base_metrics[i : i + metrics_per_plot]
        for i in range(0, len(base_metrics), metrics_per_plot)
    ]

    for group_idx, metrics_group in enumerate(metric_groups):
        # Focus on steering layer
        if best_layer not in comparison_df.index:
            print(f"Warning: Best layer {best_layer} not found in comparison data")
            continue

        steering_data = comparison_df.loc[best_layer]

        # Get percent changes for metrics in this group
        percent_changes = []
        valid_metrics = []

        for metric in metrics_group:
            pct_col = f"{metric}_percent_change"
            if pct_col in steering_data.index:
                pct_val = steering_data[pct_col]
                # CHANGED: Handle NaN and infinite values
                if np.isfinite(pct_val):
                    percent_changes.append(pct_val)
                    valid_metrics.append(metric)

        if not valid_metrics:
            continue

        # CHANGED: Reasonable figure size
        plt.figure(figsize=(max(8, len(valid_metrics) * 1.2), 6))

        colors = ["red" if pc < 0 else "green" for pc in percent_changes]
        bars = plt.bar(
            range(len(valid_metrics)),
            percent_changes,
            color=colors,
            alpha=0.7,
            width=0.6,  # CHANGED: Narrower bars
        )

        # Add value labels on bars with better positioning
        for i, (bar, pc) in enumerate(zip(bars, percent_changes)):
            height = bar.get_height()
            label_y = height + (
                0.02 * max(percent_changes)
                if height > 0
                else -0.05 * max(percent_changes)
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f"{pc:.1f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontweight="bold",
                fontsize=9,
            )

        plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
        plt.xlabel("Metrics", fontsize=11)
        plt.ylabel("Percent Change (%)", fontsize=11)
        plt.title(
            f"Steering Effects Layer {best_layer} (α={steering_alpha}) - Group {group_idx+1}",
            fontsize=12,
        )

        # CHANGED: Better x-axis labels
        plt.xticks(
            range(len(valid_metrics)),
            [
                metric.replace("_", "\n").title() for metric in valid_metrics
            ],  # Line breaks for long names
            rotation=0,
            ha="center",
            fontsize=9,
        )
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        # Save this group's plot
        save_path = (
            plots_dir
            / f"steering_layer_bars_layer_{best_layer}_alpha_{steering_alpha}_group{group_idx+1}.png"
        )
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        created_plots.append(save_path)
        print(f"Steering bars plot group {group_idx+1} saved to: {save_path}")

    return created_plots


def _create_difference_analysis_plot_fixed(
    comparison_df, best_layer, steering_alpha, save_path
):
    """Create difference analysis visualization with size constraints."""
    # Extract base metrics
    base_metrics = []
    for col in comparison_df.columns:
        if col.endswith("_original"):
            base_metrics.append(col.replace("_original", ""))

    # CHANGED: Limit number of metrics to prevent overcrowding
    max_metrics = 4
    selected_metrics = base_metrics[:max_metrics]

    # CHANGED: Reasonable figure size
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    layers = comparison_df.index.values

    # Plot 1: Absolute differences for key metrics
    for i, metric in enumerate(selected_metrics[:3]):  # Limit to 3 lines
        diff_col = f"{metric}_diff"
        if diff_col in comparison_df.columns:
            diff_values = comparison_df[diff_col].values
            ax1.plot(
                layers,
                diff_values,
                "o-",
                label=metric.replace("_", " ").title(),
                linewidth=2,
                markersize=4,
                alpha=0.8,
            )

    ax1.axvline(
        best_layer,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Steering L{best_layer}",
    )
    ax1.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Layer", fontsize=10)
    ax1.set_ylabel("Absolute Difference", fontsize=10)
    ax1.set_title("Absolute Differences", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Percent changes for key metrics
    for i, metric in enumerate(selected_metrics[:3]):  # Limit to 3 lines
        pct_col = f"{metric}_percent_change"
        if pct_col in comparison_df.columns:
            pct_values = comparison_df[pct_col].values
            ax2.plot(
                layers,
                pct_values,
                "s-",
                label=metric.replace("_", " ").title(),
                linewidth=2,
                markersize=4,
                alpha=0.8,
            )

    ax2.axvline(
        best_layer,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Steering L{best_layer}",
    )
    ax2.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Layer", fontsize=10)
    ax2.set_ylabel("Percent Change (%)", fontsize=10)
    ax2.set_title("Percent Changes", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Steering layer effects
    if best_layer in comparison_df.index:
        steering_metrics = selected_metrics[:6]  # Limit to 6 metrics
        steering_diffs = []
        valid_steering_metrics = []

        for metric in steering_metrics:
            diff_col = f"{metric}_diff"
            if diff_col in comparison_df.columns:
                diff_val = comparison_df.loc[best_layer, diff_col]
                if np.isfinite(diff_val):
                    steering_diffs.append(diff_val)
                    valid_steering_metrics.append(metric)

        if valid_steering_metrics:
            colors = ["red" if diff < 0 else "green" for diff in steering_diffs]
            bars = ax3.bar(
                range(len(valid_steering_metrics)),
                steering_diffs,
                color=colors,
                alpha=0.7,
                width=0.6,
            )

            ax3.axhline(0, color="black", linestyle="-", linewidth=1)
            ax3.set_xlabel("Metrics", fontsize=10)
            ax3.set_ylabel("Absolute Difference", fontsize=10)
            ax3.set_title(f"Steering Layer {best_layer}", fontsize=11)
            ax3.set_xticks(range(len(valid_steering_metrics)))
            ax3.set_xticklabels(
                [
                    metric.replace("_", "\n").title()
                    for metric in valid_steering_metrics
                ],
                rotation=0,
                ha="center",
                fontsize=8,
            )
            ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Distribution of changes
    all_pct_changes = []
    for metric in selected_metrics:
        pct_col = f"{metric}_percent_change"
        if pct_col in comparison_df.columns:
            pct_values = comparison_df[pct_col].values
            finite_values = pct_values[np.isfinite(pct_values)]  # Filter out NaN/inf
            all_pct_changes.extend(finite_values)

    if all_pct_changes:
        # CHANGED: Limit histogram bins to reasonable number
        n_bins = min(20, max(10, int(np.sqrt(len(all_pct_changes)))))
        ax4.hist(
            all_pct_changes, bins=n_bins, alpha=0.7, color="skyblue", edgecolor="black"
        )
        ax4.axvline(0, color="red", linestyle="--", linewidth=2, label="No Change")
        ax4.set_xlabel("Percent Change (%)", fontsize=10)
        ax4.set_ylabel("Frequency", fontsize=10)
        ax4.set_title("Distribution of Changes", fontsize=11)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Difference Analysis (α={steering_alpha})", fontsize=14)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Difference analysis plot saved to: {save_path}")


def _create_layer_summary_plot(comparison_df, best_layer, steering_alpha, save_path):
    """
    Create a new summary plot showing key insights.
    CHANGED: New function to provide better overview
    """
    # Extract key metrics
    key_metrics = ["accuracy", "silhouette_score"]  # Focus on most important metrics

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    layers = comparison_df.index.values

    # Plot 1: Accuracy comparison
    if (
        "accuracy_original" in comparison_df.columns
        and "accuracy_steered" in comparison_df.columns
    ):
        orig_acc = comparison_df["accuracy_original"].values
        steer_acc = comparison_df["accuracy_steered"].values

        ax1.plot(layers, orig_acc, "b-o", label="Original", linewidth=2, markersize=5)
        ax1.plot(layers, steer_acc, "r-s", label="Steered", linewidth=2, markersize=5)
        ax1.axvline(best_layer, color="green", linestyle="--", linewidth=2, alpha=0.7)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Silhouette score comparison
    if (
        "silhouette_score_original" in comparison_df.columns
        and "silhouette_score_steered" in comparison_df.columns
    ):
        orig_sil = comparison_df["silhouette_score_original"].values
        steer_sil = comparison_df["silhouette_score_steered"].values

        ax2.plot(layers, orig_sil, "b-o", label="Original", linewidth=2, markersize=5)
        ax2.plot(layers, steer_sil, "r-s", label="Steered", linewidth=2, markersize=5)
        ax2.axvline(best_layer, color="green", linestyle="--", linewidth=2, alpha=0.7)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Score Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Combined improvement metric
    if (
        "accuracy_percent_change" in comparison_df.columns
        and "silhouette_score_percent_change" in comparison_df.columns
    ):
        acc_change = comparison_df["accuracy_percent_change"].values
        sil_change = comparison_df["silhouette_score_percent_change"].values

        # Filter finite values
        acc_change = np.where(np.isfinite(acc_change), acc_change, 0)
        sil_change = np.where(np.isfinite(sil_change), sil_change, 0)

        ax3.bar(layers, acc_change, alpha=0.6, label="Accuracy Change %", color="blue")
        ax3_twin = ax3.twinx()
        ax3_twin.bar(
            layers + 0.3,
            sil_change,
            alpha=0.6,
            label="Silhouette Change %",
            color="orange",
        )

        ax3.axvline(best_layer, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax3.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax3_twin.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        ax3.set_xlabel("Layer")
        ax3.set_ylabel("Accuracy Change %", color="blue")
        ax3_twin.set_ylabel("Silhouette Change %", color="orange")
        ax3.set_title("Percent Changes")
        ax3.grid(True, alpha=0.3)

    # Plot 4: Steering effect magnitude
    if best_layer in comparison_df.index:
        # Get all percent changes for steering layer
        steering_row = comparison_df.loc[best_layer]
        pct_changes = []
        metric_names = []

        for col in steering_row.index:
            if col.endswith("_percent_change"):
                val = steering_row[col]
                if np.isfinite(val):
                    pct_changes.append(val)
                    metric_names.append(
                        col.replace("_percent_change", "").replace("_", " ").title()
                    )

        if pct_changes:
            # Limit to top 8 changes by magnitude
            abs_changes = [abs(x) for x in pct_changes]
            sorted_indices = sorted(
                range(len(abs_changes)), key=lambda i: abs_changes[i], reverse=True
            )[:8]

            top_changes = [pct_changes[i] for i in sorted_indices]
            top_names = [metric_names[i] for i in sorted_indices]

            colors = ["red" if x < 0 else "green" for x in top_changes]
            bars = ax4.barh(
                range(len(top_changes)), top_changes, color=colors, alpha=0.7
            )

            ax4.axvline(0, color="black", linestyle="-", linewidth=1)
            ax4.set_xlabel("Percent Change (%)")
            ax4.set_ylabel("Metrics")
            ax4.set_title(f"Top Changes in Layer {best_layer}")
            ax4.set_yticks(range(len(top_names)))
            ax4.set_yticklabels(top_names, fontsize=9)
            ax4.grid(True, alpha=0.3, axis="x")

    plt.suptitle(
        f"Layer Analysis Summary (Steering Layer {best_layer}, α={steering_alpha})",
        fontsize=14,
    )
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Layer summary plot saved to: {save_path}")


# Additional helper function to set matplotlib parameters globally
def setup_matplotlib_for_headless():
    """Set up matplotlib for headless operation and reasonable defaults."""
    matplotlib.use("Agg")  # Non-interactive backend
    plt.rcParams["figure.max_open_warning"] = (
        0  # Disable warning about too many figures
    )
    plt.rcParams["font.size"] = 10  # Reasonable default font size
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9

    # Ensure we close all figures to free memory
    plt.close("all")


# Call setup function when module is imported
setup_matplotlib_for_headless()
