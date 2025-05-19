def plot_coefficient_sweep_lines_comparison(results, metrics, save_path=None):
    """
    Plot a grid of line/dot plots for selected metrics as subplots in a single figure.
    Each subplot shows metric vs. coefficient for all layers.
    Adds a detailed description block at the bottom explaining each metric in plain language.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    metric_info = {
        "accuracy": {
            "name": "Accuracy",
            "description": "Classification accuracy of the CCS probe. Higher values (closer to 1.0) indicate better separation between hate and safe content.",
            "ideal": "Values close to 1.0 (top of plot) are ideal, indicating perfect classification.",
        },
        "silhouette": {
            "name": "Silhouette Score",
            "description": "Measures how well the representations cluster into two groups (hate and safe). Higher values (above 0.7) indicate strong, well-separated clusters.",
            "ideal": "Values above 0.7 (top of plot) indicate strong cluster separation.",
        },
        "similarity_change": {
            "name": "Similarity Change",
            "description": "Change in cosine similarity between hate and safe representations after steering. Positive values mean steering moved hate content closer to safe content.",
            "ideal": "Positive values (higher on plot) indicate successful steering towards safe content.",
        },
        "path_length": {
            "name": "Path Length",
            "description": "Normalized path length of the steering transformation. Values near 1.0 mean the steering takes a direct, efficient path toward safe representations.",
            "ideal": "Values close to 1.0 (top of plot) indicate efficient steering paths.",
        },
        "semantic_consistency": {
            "name": "Semantic Consistency",
            "description": "How well the semantic meaning is preserved after steering. High values mean the steered content remains similar in meaning to the original.",
            "ideal": "Values above 0.8 (top of plot) indicate good semantic preservation.",
        },
        "steering_alignment": {
            "name": "Steering Alignment",
            "description": "How well the steering vector aligns with the ideal direction from hate to safe. High values mean the steering is moving in the right direction.",
            "ideal": "Values close to 1.0 (top of plot) indicate perfect alignment.",
        },
        "agreement_score": {
            "name": "Agreement Score",
            "description": "Measures how consistently the model predicts the same class for related statement pairs. High agreement means the model is reliable and not random. (Agreement: the extent to which the model's predictions for logically related statements are consistent with each other.)",
            "ideal": "Values above 0.7 (top of plot) indicate strong agreement.",
        },
        "contradiction_index": {
            "name": "Contradiction Index",
            "description": "Measures how often the model makes contradictory predictions for related statement pairs. Low values mean the model is logically consistent. (Contradiction: the extent to which the model's predictions for logically related statements are in conflict with each other.)",
            "ideal": "Values close to 0.0 (bottom of plot) indicate no contradictions.",
        },
        "representation_stability": {
            "name": "Representation Stability",
            "description": "How robust the model's predictions are to small changes (noise) in the input. High stability means predictions don't change much if the input is slightly perturbed. (Stability: the model's resistance to small, irrelevant changes in the input.)",
            "ideal": "Values above 0.9 (top of plot) indicate high stability.",
        },
        "f1": {
            "name": "F1 Score",
            "description": "Harmonic mean of precision and recall. High F1 means the model is both accurate and complete in its predictions.",
            "ideal": "Values close to 1.0 (top of plot) indicate optimal balance.",
        },
    }

    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    n_layers = len(results)
    coefficients = sorted(
        [float(k.split("_")[1]) for k in results[0].keys() if k.startswith("coef_")]
    )

    layer_colors = sns.color_palette("husl", n_layers)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for layer_idx in range(n_layers):
            y = [results[layer_idx][f"coef_{coef}"][metric] for coef in coefficients]
            ax.plot(
                coefficients,
                y,
                marker="o",
                label=f"Layer {layer_idx}",
                color=layer_colors[layer_idx],
                alpha=0.85,
            )
        ax.set_xlabel("Steering Coefficient")
        ax.set_ylabel(metric_info[metric]["name"])
        ax.set_title(metric_info[metric]["name"])
        ax.grid(True)
        if i == 0:
            ax.legend(loc="best", fontsize=10)

    # Hide unused subplots
    for j in range(n_metrics, len(axes)):
        axes[j].axis("off")

    # Build detailed description block
    description = "\n**Metric Explanations:**\n"
    for metric in metrics:
        info = metric_info[metric]
        description += (
            f"\n- **{info['name']}**: {info['description']}\n  Ideal: {info['ideal']}\n"
        )
    description += (
        "\n**How to interpret:**\n"
        "- Each plot shows how the metric changes with steering coefficient for each layer (each line is a layer).\n"
        "- Higher lines are usually better, but check the ideal value for each metric.\n"
        "- Compare across metrics to see which aspects of steering are most effective.\n"
        "- Agreement: Consistency of predictions for logically related statements.\n"
        "- Contradiction: Frequency of conflicting predictions for related statements.\n"
        "- Stability: Resistance to small, irrelevant changes in input.\n"
        "- Consistency: Preservation of semantic meaning after steering.\n"
    )
    fig.text(
        0.5,
        0.01,
        description,
        ha="center",
        va="bottom",
        wrap=True,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    plt.tight_layout(rect=(0, 0.13, 1, 1))
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
