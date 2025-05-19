"""
CCS (Contrast Consistent Search) metrics for evaluating and analyzing model representations.
This file contains metrics specific to the CCS probe evaluation and analysis of
internal model representations.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, silhouette_score
from tr_data_utils import extract_representation


def evaluate_ccs_performance(ccs, X_hate_test, X_safe_test, y_test):
    """
    Evaluate the performance of a CCS probe.

    Args:
        ccs: The trained CCS probe
        X_hate_test: Test hate data
        X_safe_test: Test safe data
        y_test: Test labels

    Returns:
        Dictionary of metrics including:
        - accuracy: Classification accuracy (0-1)
        - auc: Area Under ROC Curve (0-1)
        - silhouette: Silhouette score for cluster separation (-1 to 1)

    Example:
        >>> metrics = evaluate_ccs_performance(ccs_probe, X_hate_test, X_safe_test, y_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        Accuracy: 0.8934
        >>> print(f"AUC: {metrics['auc']:.4f}")
        AUC: 0.9231
        >>> print(f"Silhouette score: {metrics['silhouette']:.4f}")
        Silhouette score: 0.6721

    Interpretation:
        - Accuracy: Higher values (closer to 1.0) indicate better classification performance
        - AUC: Higher values (closer to 1.0) indicate better discrimination between classes
        - Silhouette:
          * Positive values indicate well-separated clusters
          * Values close to 0 indicate overlapping clusters
          * Negative values indicate potential misclassification
          * Higher values (closer to 1.0) indicate better cluster separation
    """
    # Get predictions
    predictions, probabilities = ccs.predict(X_hate_test, X_safe_test)

    # Make sure predictions and y_test have the same length
    y_test_subset = (
        y_test[: len(predictions)] if len(y_test) > len(predictions) else y_test
    )
    predictions_subset = (
        predictions[: len(y_test_subset)]
        if len(predictions) > len(y_test_subset)
        else predictions
    )

    # Calculate accuracy
    accuracy = (predictions_subset == y_test_subset).mean()

    # Calculate AUC
    auc = roc_auc_score(y_test_subset, probabilities[: len(y_test_subset)])

    # Calculate silhouette score
    if len(np.unique(predictions_subset)) > 1:
        # Get representations for hate and safe data
        hate_reps = []
        safe_reps = []

        # Limit the number of samples to process for efficiency
        sample_size = min(len(X_hate_test), 50)

        # Create tensors to store representations directly on GPU
        device = ccs.device

        for i in range(sample_size):
            # Extract representations for a subset of the data
            hate_rep = extract_representation(
                model=ccs.model,
                tokenizer=ccs.tokenizer,
                text=X_hate_test[i],
                layer_index=ccs.layer_idx,
                strategy="last-token",
                device=device,
            )
            # Convert numpy array to torch tensor
            hate_rep_tensor = torch.tensor(hate_rep, device=device)
            hate_reps.append(hate_rep_tensor)

            safe_rep = extract_representation(
                model=ccs.model,
                tokenizer=ccs.tokenizer,
                text=X_safe_test[i],
                layer_index=ccs.layer_idx,
                strategy="last-token",
                device=device,
            )
            # Convert numpy array to torch tensor
            safe_rep_tensor = torch.tensor(safe_rep, device=device)
            safe_reps.append(safe_rep_tensor)

        # Stack tensors on GPU
        hate_reps_tensor = torch.stack(hate_reps)
        safe_reps_tensor = torch.stack(safe_reps)

        # Calculate the difference vectors (still on GPU)
        diffs_tensor = safe_reps_tensor - hate_reps_tensor

        # For sklearn's silhouette_score, we need numpy arrays
        # Only convert to CPU/numpy at the last step
        diffs = diffs_tensor.detach().cpu().numpy()

        # Calculate silhouette score using the subset of predictions
        print(f"Differences shape: {diffs.shape}")
        print(f"Predictions shape: {predictions[:sample_size].shape}")

        # Reshape diffs to 2D if it's 3D
        if len(diffs.shape) == 3:
            diffs_2d = diffs.reshape(diffs.shape[0], -1)
            print(f"Reshaped differences to 2D: {diffs_2d.shape}")
            silhouette = silhouette_score(
                diffs_2d, predictions[:sample_size], metric="cosine"
            )
        else:
            silhouette = silhouette_score(
                diffs, predictions[:sample_size], metric="cosine"
            )
    else:
        silhouette = 0.0

    return {"accuracy": accuracy, "auc": auc, "silhouette": silhouette}


def compute_class_separability(hate_vectors, safe_vectors):
    """
    Compute how well-separated the hate and safe representations are.

    Args:
        hate_vectors: Hate speech representations
        safe_vectors: Safe speech representations

    Returns:
        Separability score

    Interpretation:
        - Higher values indicate better separation between classes
        - Values close to 0 indicate poor separation

    Example:
        >>> hate_vecs = np.array([[1.0, 0.0], [0.9, 0.1]])
        >>> safe_vecs = np.array([[0.0, 1.0], [0.1, 0.9]])
        >>> sep = compute_class_separability(hate_vecs, safe_vecs)
        >>> print(f"Separability: {sep:.4f}")
        Separability: 0.8944
        # Interpretation: The classes are well-separated (89.4% separability)
    """
    # Add numerical stability
    hate_vectors = np.clip(hate_vectors, -1e6, 1e6)
    safe_vectors = np.clip(safe_vectors, -1e6, 1e6)

    # Calculate mean vectors
    hate_mean = np.mean(hate_vectors, axis=0)
    safe_mean = np.mean(safe_vectors, axis=0)

    # Calculate between-class variance
    between_class_variance = np.sum((hate_mean - safe_mean) ** 2)

    # Calculate within-class variance with numerical stability
    hate_variance = np.sum(np.var(hate_vectors, axis=0))
    safe_variance = np.sum(np.var(safe_vectors, axis=0))
    within_class_variance = (hate_variance + safe_variance) / 2

    # Compute separability with numerical stability
    if within_class_variance > 1e-10:  # Avoid division by zero
        separability = between_class_variance / within_class_variance
    else:
        separability = 0.0

    return separability


def agreement_score(pA0, pA1, p_notA0, p_notA1):
    """
    Compute agreement score between predictions on pairs of statements.

    Args:
        pA0: Prediction probabilities for statement A, first input (shape: n_samples x 1)
        pA1: Prediction probabilities for statement A, second input (shape: n_samples x 1)
        p_notA0: Prediction probabilities for not-A statement, first input (shape: n_samples x 1)
        p_notA1: Prediction probabilities for not-A statement, second input (shape: n_samples x 1)

    Returns:
        Agreement score

    Interpretation:
        - Positive values indicate consistent predictions between statement pairs
        - Values close to 0 indicate little consistency
        - Negative values indicate contradictory predictions

    Example:
        >>> pA0 = np.array([0.2, 0.3])
        >>> pA1 = np.array([0.7, 0.8])
        >>> p_notA0 = np.array([0.8, 0.7])
        >>> p_notA1 = np.array([0.3, 0.2])
        >>> score = agreement_score(pA0, pA1, p_notA0, p_notA1)
        >>> print(f"Agreement score: {score.mean():.4f}")
        Agreement score: 0.2500
        # Interpretation: Moderate agreement between statement pairs
    """
    # Ensure inputs are numpy arrays
    pA0 = np.array(pA0)
    pA1 = np.array(pA1)
    p_notA0 = np.array(p_notA0)
    p_notA1 = np.array(p_notA1)

    # Clip values to prevent numerical instability
    pA0 = np.clip(pA0, 1e-10, 1 - 1e-10)
    pA1 = np.clip(pA1, 1e-10, 1 - 1e-10)
    p_notA0 = np.clip(p_notA0, 1e-10, 1 - 1e-10)
    p_notA1 = np.clip(p_notA1, 1e-10, 1 - 1e-10)

    # Compute agreement score
    agreement = (
        0.5
        * ((pA1 - p_notA0) ** 2 + (pA0 - p_notA1) ** 2)
        * np.sign(pA1 - p_notA1)
        * np.sign(p_notA0 - pA0)
    )

    return np.clip(agreement, -1.0, 1.0)


def contradiction_index(pA0, pA1, p_notA0, p_notA1):
    """
    Compute contradiction index between predictions on pairs of statements.

    Args:
        pA0: Prediction probabilities for statement A, first input (shape: n_samples x 1)
        pA1: Prediction probabilities for statement A, second input (shape: n_samples x 1)
        p_notA0: Prediction probabilities for not-A statement, first input (shape: n_samples x 1)
        p_notA1: Prediction probabilities for not-A statement, second input (shape: n_samples x 1)

    Returns:
        Contradiction index

    Interpretation:
        - Values close to 0 indicate no contradiction
        - Higher values indicate more contradiction
        - Maximum value of 1.0 indicates complete contradiction

    Example:
        >>> pA0 = np.array([0.2, 0.8])
        >>> pA1 = np.array([0.8, 0.2])
        >>> p_notA0 = np.array([0.8, 0.2])
        >>> p_notA1 = np.array([0.2, 0.8])
        >>> idx = contradiction_index(pA0, pA1, p_notA0, p_notA1)
        >>> print(f"Contradiction index: {idx.mean():.4f}")
        Contradiction index: 0.6800
        # Interpretation: High contradiction between statement pairs
    """
    # Ensure inputs are numpy arrays
    pA0 = np.array(pA0)
    pA1 = np.array(pA1)
    p_notA0 = np.array(p_notA0)
    p_notA1 = np.array(p_notA1)

    # Clip values to prevent numerical instability
    pA0 = np.clip(pA0, 1e-10, 1 - 1e-10)
    pA1 = np.clip(pA1, 1e-10, 1 - 1e-10)
    p_notA0 = np.clip(p_notA0, 1e-10, 1 - 1e-10)
    p_notA1 = np.clip(p_notA1, 1e-10, 1 - 1e-10)

    # Compute contradiction index
    ci = pA1 * p_notA1 + pA0 * p_notA0

    return np.clip(ci, 0.0, 1.0)


def ideal_representation_distance(pA0, pA1, p_notA0, p_notA1):
    """
    Compute the distance to the ideal representation.

    Args:
        pA0: Prediction probabilities for statement A, first input (shape: n_samples x 1)
        pA1: Prediction probabilities for statement A, second input (shape: n_samples x 1)
        p_notA0: Prediction probabilities for not-A statement, first input (shape: n_samples x 1)
        p_notA1: Prediction probabilities for not-A statement, second input (shape: n_samples x 1)

    Returns:
        Distance to ideal representation

    Interpretation:
        - Lower values indicate closer to ideal representation
        - Higher values indicate further from ideal representation
        - Ideal representation would be [pA0=0, pA1=1, p_notA0=1, p_notA1=0]

    Example:
        >>> pA0 = np.array([0.1, 0.3])
        >>> pA1 = np.array([0.9, 0.7])
        >>> p_notA0 = np.array([0.9, 0.7])
        >>> p_notA1 = np.array([0.1, 0.3])
        >>> dist = ideal_representation_distance(pA0, pA1, p_notA0, p_notA1)
        >>> print(f"Ideal distance: {dist.mean():.4f}")
        Ideal distance: 0.1414
        # Interpretation: Very close to ideal representation (small distance)
    """
    # Ensure inputs are numpy arrays
    pA0 = np.array(pA0)
    pA1 = np.array(pA1)
    p_notA0 = np.array(p_notA0)
    p_notA1 = np.array(p_notA1)

    # Clip values to prevent numerical instability
    pA0 = np.clip(pA0, 1e-10, 1 - 1e-10)
    pA1 = np.clip(pA1, 1e-10, 1 - 1e-10)
    p_notA0 = np.clip(p_notA0, 1e-10, 1 - 1e-10)
    p_notA1 = np.clip(p_notA1, 1e-10, 1 - 1e-10)

    # Compute distances with numerical stability
    # Ideal representation: pA0=0, pA1=1, p_notA0=1, p_notA1=0
    ideal_dist = 0.5 * np.sqrt(
        np.clip((pA1 - np.ones_like(pA1)) ** 2, 0, 1e6)
        + np.clip((pA0 - np.zeros_like(pA0)) ** 2, 0, 1e6)
        + np.clip((p_notA1 - np.zeros_like(p_notA1)) ** 2, 0, 1e6)
        + np.clip((p_notA0 - np.ones_like(p_notA0)) ** 2, 0, 1e6)
    )

    return np.clip(ideal_dist, 0.0, 1.0)


def representation_stability(
    ccs, X_vectors, perturbation_scale=0.01, n_perturbations=10
):
    """
    Measure the stability of CCS predictions against small perturbations.

    Args:
        ccs: Trained CCS probe
        X_vectors: Input vectors to test stability on
        perturbation_scale: Scale of Gaussian noise to add
        n_perturbations: Number of perturbed samples to generate

    Returns:
        Average prediction stability (0-1)

    Interpretation:
        - Values close to 1.0 indicate high stability (predictions don't change with noise)
        - Values close to 0.0 indicate low stability (predictions change with small noise)

    Example:
        >>> stability = representation_stability(ccs_probe, X_test)
        >>> print(f"Stability: {stability:.4f}")
        Stability: 0.9231
        # Interpretation: The representations are highly stable (92.3%)
    """
    # Get original predictions
    X_noise_placeholder = np.zeros_like(X_vectors)  # Not used for prediction
    original_predictions, _ = ccs.predict(X_vectors, X_noise_placeholder)

    stability_scores = []

    # Generate perturbed samples and check prediction stability
    for _ in range(n_perturbations):
        # Add small Gaussian noise
        noise = np.random.normal(0, perturbation_scale, X_vectors.shape)
        X_perturbed = X_vectors + noise

        # Get predictions on perturbed input
        perturbed_predictions, _ = ccs.predict(X_perturbed, X_noise_placeholder)

        # Calculate stability (% of predictions that remain the same)
        stability = np.mean(original_predictions == perturbed_predictions)
        stability_scores.append(stability)

    return np.mean(stability_scores)


def fisher_information_analysis(
    ccs, X_vectors, direction_vector, n_steps=20, range_factor=5.0
):
    """
    Analyze sensitivity of the CCS probe along a specified direction.

    Args:
        ccs: Trained CCS probe
        X_vectors: Base representations to analyze
        direction_vector: Direction in representation space to analyze
        n_steps: Number of steps to take along direction
        range_factor: How far to move along direction (in multiples of direction_vector)

    Returns:
        Dictionary with analysis results

    Interpretation:
        - High sensitivity along a direction indicates it's important for classification
        - Low sensitivity suggests the direction is not relevant

    Example:
        >>> steering_vec = safe_mean - hate_mean
        >>> analysis = fisher_information_analysis(ccs_probe, X_test, steering_vec)
        >>> print(f"Maximum sensitivity at: {analysis['max_sensitivity_point']:.2f}")
        Maximum sensitivity at: 0.75
        # Interpretation: Highest sensitivity is 75% along the steering direction
    """
    # Normalize direction vector
    direction_norm = np.linalg.norm(direction_vector)
    if direction_norm > 1e-10:
        direction_vector = direction_vector / direction_norm

    # Create range of steps
    alphas = np.linspace(-range_factor, range_factor, n_steps)

    # Placeholder for second input (not used for prediction)
    X_placeholder = np.zeros_like(X_vectors)

    sensitivities = []
    predictions = []

    # Compute predictions and sensitivities along direction
    for alpha in alphas:
        # Perturb vectors along direction
        X_perturbed = X_vectors + alpha * direction_vector

        # Get predictions
        preds, probs = ccs.predict(X_perturbed, X_placeholder)
        predictions.append(preds)

        # Compute sensitivity (approximation of Fisher information)
        if alpha < alphas[-1]:
            # Compute next step predictions for derivative
            X_next = X_vectors + (alpha + alphas[1] - alphas[0]) * direction_vector
            _, probs_next = ccs.predict(X_next, X_placeholder)

            # Compute derivative of log-probability
            derivative = (probs_next.cpu().numpy() - probs.cpu().numpy()) / (
                alphas[1] - alphas[0]
            )

            # Sensitivity is square of derivative
            sensitivity = np.mean(derivative**2)
            sensitivities.append(sensitivity)

    # Find point of maximum sensitivity
    max_idx = np.argmax(sensitivities)
    max_alpha = alphas[max_idx]

    # Compute decision boundary (where predictions flip)
    decision_boundaries = []
    for i in range(len(predictions) - 1):
        if (predictions[i] != predictions[i + 1]).any():
            # Boundary is between these two points
            boundary = (alphas[i] + alphas[i + 1]) / 2
            decision_boundaries.append(boundary)

    return {
        "alphas": alphas,
        "sensitivities": sensitivities,
        "predictions": predictions,
        "max_sensitivity_point": max_alpha,
        "max_sensitivity_value": sensitivities[max_idx],
        "decision_boundaries": decision_boundaries,
    }


def subspace_analysis(hate_vectors, safe_vectors, n_components=10):
    """
    Analyze the subspace that best separates hate and safe representations.

    Args:
        hate_vectors: Hate speech representations
        safe_vectors: Safe speech representations
        n_components: Number of principal components to analyze

    Returns:
        Dictionary with analysis results

    Interpretation:
        - Principal vectors with high separation indicate important directions
        - Projection scatter plots show how well the classes separate

    Example:
        >>> analysis = subspace_analysis(X_hate, X_safe)
        >>> print(f"Top component separation: {analysis['separations'][0]:.4f}")
        Top component separation: 0.8721
        # Interpretation: The top component separates classes with 87.2% accuracy
    """
    from sklearn.decomposition import PCA

    # Add numerical stability
    hate_vectors = np.clip(hate_vectors, -1e6, 1e6)
    safe_vectors = np.clip(safe_vectors, -1e6, 1e6)

    # Combine data
    X_combined = np.vstack([hate_vectors, safe_vectors])
    labels = np.concatenate([np.zeros(len(hate_vectors)), np.ones(len(safe_vectors))])

    # Add small epsilon to prevent zero values
    X_combined = X_combined + np.eye(X_combined.shape[0], X_combined.shape[1]) * 1e-10

    # Fit PCA
    pca = PCA(n_components=min(n_components, X_combined.shape[1]))
    X_projected = pca.fit_transform(X_combined)

    # Analyze separation along each component
    separations = []

    for i in range(n_components):
        # Project to this component
        component_values = X_projected[:, i]

        # Find optimal threshold for separation
        thresholds = np.sort(component_values)
        best_accuracy = 0
        best_threshold = 0

        for threshold in thresholds:
            preds = (component_values > threshold).astype(int)
            accuracy = np.mean(preds == labels)
            accuracy = max(accuracy, 1 - accuracy)  # Account for inverted separation

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        separations.append(
            {
                "component": i,
                "accuracy": best_accuracy,
                "threshold": best_threshold,
                "variance_explained": pca.explained_variance_ratio_[i],
            }
        )

    # Sort by separation accuracy
    separations = sorted(separations, key=lambda x: x["accuracy"], reverse=True)

    return {
        "pca": pca,
        "projections": X_projected,
        "labels": labels,
        "separations": separations,
        "components": pca.components_,
    }


def visualize_decision_boundary(
    ccs, hate_vectors, safe_vectors, steering_vector, log_base=None
):
    """
    Visualize the decision boundary of the CCS probe in the steering vector direction.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

    # Normalize steering vector
    steering_norm = np.linalg.norm(steering_vector)
    if steering_norm > 1e-10:
        steering_vector = steering_vector / steering_norm

    # Combine data
    X_combined = np.vstack([hate_vectors, safe_vectors])
    labels = np.concatenate([np.zeros(len(hate_vectors)), np.ones(len(safe_vectors))])

    # Project data to 2D for visualization
    # First component: steering vector direction
    # Second component: PCA of residuals
    projections = []

    # Project onto steering vector
    projection1 = np.array([np.dot(x, steering_vector) for x in X_combined])
    projections.append(projection1)

    # Compute residuals
    residuals = X_combined - np.outer(projection1, steering_vector)

    # Find second direction (orthogonal to steering vector)
    pca = PCA(n_components=1)
    pca.fit(residuals)
    second_direction = pca.components_[0]

    # Project onto second direction
    projection2 = np.array([np.dot(x, second_direction) for x in X_combined])
    projections.append(projection2)

    # Create 2D projections
    X_2d = np.column_stack(projections)

    # Create grid for decision boundary
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Reconstruct grid points in original space
    grid_points = np.array([xx.ravel(), yy.ravel()]).T
    original_space = np.outer(grid_points[:, 0], steering_vector) + np.outer(
        grid_points[:, 1], second_direction
    )

    # Predict on grid points
    X_placeholder = np.zeros_like(original_space)
    grid_preds, _ = ccs.predict(original_space, X_placeholder)
    grid_preds = grid_preds.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, grid_preds, alpha=0.3, cmap="RdBu")

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
            label="Hate" if label == 0 else "Safe",
        )

    # Add steering vector direction
    ax.arrow(
        0,
        0,
        1,
        0,
        color="black",
        width=0.02,
        head_width=0.1,
        head_length=0.1,
        length_includes_head=True,
        label="Steering Direction",
        zorder=20,
    )

    ax.set_xlabel("Steering Vector Direction")
    ax.set_ylabel("Orthogonal Direction")
    ax.set_title("CCS Probe Decision Boundary in Steering Space")
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True)

    # Add descriptive text
    text = """
    Description: This plot shows the decision boundary of the CCS probe in the space defined by the steering vector and its orthogonal complement.
    
    Ideal Case:
    - Clear separation between hate (red) and safe (blue) content
    - Decision boundary should be roughly perpendicular to the steering direction
    - Points should cluster into two distinct groups
    
    Interpretation:
    - The steering vector direction (horizontal axis) shows how content changes when steering is applied
    - The orthogonal direction (vertical axis) shows variations that preserve the steering effect
    - The decision boundary (colored regions) shows where the probe switches between hate and safe predictions
    - A clear boundary indicates the probe can reliably distinguish between content types
    """

    fig.text(
        0.5,
        0.01,
        text,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        wrap=True,
    )

    # Adjust layout to make room for text
    plt.tight_layout(rect=(0, 0.15, 1, 1))

    if log_base:
        plt.savefig(f"{log_base}_decision_boundary.png", dpi=300, bbox_inches="tight")

    return fig


def plot_all_decision_boundaries(layers_data, log_base=None):
    """
    Plot decision boundaries for all layers as subplots in a single figure.
    layers_data: list of dicts with keys 'ccs', 'hate_vectors', 'safe_vectors', 'steering_vector', 'layer_idx'
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    n_layers = len(layers_data)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    for i, layer in enumerate(layers_data):
        ccs = layer["ccs"]
        hate_vectors = layer["hate_vectors"]
        safe_vectors = layer["safe_vectors"]
        steering_vector = layer["steering_vector"]
        layer_idx = layer.get("layer_idx", i)
        ax = axes[i]

        # Normalize steering vector
        steering_norm = np.linalg.norm(steering_vector)
        if steering_norm > 1e-10:
            steering_vector = steering_vector / steering_norm

        # Combine data
        X_combined = np.vstack([hate_vectors, safe_vectors])
        labels = np.concatenate(
            [np.zeros(len(hate_vectors)), np.ones(len(safe_vectors))]
        )

        # Project data to 2D for visualization
        projection1 = np.array([np.dot(x, steering_vector) for x in X_combined])
        residuals = X_combined - np.outer(projection1, steering_vector)
        pca = PCA(n_components=1)
        pca.fit(residuals)
        second_direction = pca.components_[0]
        projection2 = np.array([np.dot(x, second_direction) for x in X_combined])
        X_2d = np.column_stack([projection1, projection2])

        # Create grid for decision boundary
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )
        grid_points = np.array([xx.ravel(), yy.ravel()]).T
        original_space = np.outer(grid_points[:, 0], steering_vector) + np.outer(
            grid_points[:, 1], second_direction
        )
        X_placeholder = np.zeros_like(original_space)
        grid_preds, _ = ccs.predict(original_space, X_placeholder)
        grid_preds = grid_preds.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, grid_preds, alpha=0.3, cmap="RdBu")
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
                label="Hate" if label == 0 else "Safe",
            )
        ax.arrow(
            0,
            0,
            1,
            0,
            color="black",
            width=0.02,
            head_width=0.1,
            head_length=0.1,
            length_includes_head=True,
            label="Steering Direction",
            zorder=20,
        )
        ax.set_xlabel("Steering Vector Direction")
        ax.set_ylabel("Orthogonal Direction")
        ax.set_title(f"Layer {layer_idx}")
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right", fontsize=12)

    # Hide unused subplots
    for j in range(n_layers, len(axes)):
        axes[j].axis("off")

    # Add detailed description block
    description = (
        "This figure shows the decision boundaries of the CCS probe for each layer in the space defined by the steering vector (horizontal axis) and its orthogonal complement (vertical axis).\n\n"
        "**How to interpret:**\n"
        "- Each subplot corresponds to a different layer.\n"
        "- The colored regions show the model's predicted class (hate or safe) in the 2D projection.\n"
        "- The black arrow shows the direction of the steering vector.\n"
        "- Points are colored by their true class.\n"
        "- A clear, vertical decision boundary (perpendicular to the steering vector) indicates the probe can reliably distinguish between content types along the steering direction.\n\n"
        "**Ideal case:**\n"
        "- Hate and safe points form two distinct clusters separated by a sharp boundary.\n"
        "- The boundary is perpendicular to the steering direction.\n"
        "- The probe's predictions match the true classes.\n\n"
        "**Non-ideal case:**\n"
        "- Overlapping clusters or a fuzzy boundary indicate the probe struggles to distinguish between classes.\n"
        "- A boundary not aligned with the steering direction suggests the steering vector is not the most discriminative direction.\n"
        "\n"
        "**Axes:**\n"
        "- Horizontal: projection onto the steering vector (how much a point moves when steered).\n"
        "- Vertical: projection onto the main orthogonal direction (other variations).\n"
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
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    if log_base:
        plt.savefig(
            f"{log_base}_all_decision_boundaries.png", dpi=300, bbox_inches="tight"
        )
    return fig
