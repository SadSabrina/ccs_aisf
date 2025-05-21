def plot_coefficient_sweep_lines_comparison(results, metrics, save_path=None):
    """Plot multiple metrics across different steering coefficients.

    Args:
        results: List or dictionary containing layer results
        metrics: List of metrics to plot
        save_path: Path to save the plot.
    """
    import matplotlib.pyplot as plt

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define steering coefficients to plot
    steering_coefs = [0.0, 0.5, 1.0, 2.0, 5.0]

    # Create figure with multiple subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    # Extract layer indices and prepare results based on the input type
    if isinstance(results, list):
        # Handle list input
        if results and isinstance(results[0], dict) and "layer_idx" in results[0]:
            # List of dictionaries with layer_idx key
            layers = [r["layer_idx"] for r in results]
            results_list = results
        else:
            # Simple list - use indices as layer indices
            layers = list(range(len(results)))
            results_list = results
    elif isinstance(results, dict):
        # Handle dictionary input
        layers = sorted(list(results.keys()))
        results_list = [results[layer] for layer in layers]
    else:
        raise ValueError(
            f"Unexpected results type: {type(results)}. Expected list or dict."
        )

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Get metrics for each steering coefficient
        for coef in steering_coefs:
            coef_str = f"coef_{coef}"
            metric_values = []

            for r in results_list:
                if isinstance(r, dict) and coef_str in r and metric in r[coef_str]:
                    metric_values.append(r[coef_str][metric])
                else:
                    # Handle missing values
                    metric_values.append(None)

            # Plot this coefficient line
            ax.plot(layers, metric_values, marker="o", label=f"Coef={coef}")

        # Set labels and title for this subplot
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Across Layers")
        ax.legend()
        ax.grid(True)

    # Set x-axis label on bottom subplot
    axes[-1].set_xlabel("Layer")

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
