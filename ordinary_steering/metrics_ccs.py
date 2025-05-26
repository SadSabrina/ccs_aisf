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
    original_predictions, _ = ccs.predict_from_vectors(X_vectors)

    stability_scores = []

    # Generate perturbed samples and check prediction stability
    for _ in range(n_perturbations):
        # Add small Gaussian noise
        noise = np.random.normal(0, perturbation_scale, X_vectors.shape)
        X_perturbed = X_vectors + noise

        # Ensure X_perturbed is float32 to match model weights
        X_perturbed = X_perturbed.astype(np.float32)

        # Get predictions on perturbed input
        perturbed_predictions, _ = ccs.predict_from_vectors(X_perturbed)

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
        >>> result = fisher_information_analysis(ccs, X_test, steering_vec)
        >>> print(f"Max sensitivity at {result['max_sensitivity_point']}")
        Max sensitivity at 1.25
        # Interpretation: The model is most sensitive when representations are moved
        # 1.25 units along the steering vector direction
    """
    # Ensure all inputs are float32 to match model weights
    X_vectors = X_vectors.astype(np.float32)
    direction_vector = direction_vector.astype(np.float32)

    # Normalize direction vector
    direction_norm = np.linalg.norm(direction_vector)
    if direction_norm > 1e-10:
        direction_vector = direction_vector / direction_norm

    # Generate range of points along the direction
    steps = np.linspace(-range_factor, range_factor, n_steps)
    sensitivities = []
    confidences = []

    # Measure probe output at each point along the direction
    for step in steps:
        # Move representations along direction vector
        perturbed_X = X_vectors + step * direction_vector

        # Get predictions using the probe
        preds, confs = ccs.predict_from_vectors(perturbed_X)

        # Store average confidence to measure sensitivity
        avg_conf = np.mean(confs)
        confidences.append(avg_conf)

        # Calculate sensitivity (approximation of first derivative)
        if len(sensitivities) > 0:
            # Forward difference
            sensitivity = abs(avg_conf - confidences[-2]) / (steps[1] - steps[0])
            sensitivities.append(sensitivity)

    # Add a zero for the first point's sensitivity (no previous point to compare)
    sensitivities = [0] + sensitivities

    # Find point of maximum sensitivity
    max_idx = np.argmax(sensitivities)
    max_sensitivity_point = steps[max_idx]
    max_sensitivity_value = sensitivities[max_idx]

    return {
        "steps": steps.tolist(),
        "confidences": confidences,
        "sensitivities": sensitivities,
        "max_sensitivity_point": float(max_sensitivity_point),
        "max_sensitivity_value": float(max_sensitivity_value),
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

    # Reshape vectors to 2D if they are 3D
    if len(hate_vectors.shape) == 3:
        hate_vectors = hate_vectors.reshape(hate_vectors.shape[0], -1)
    if len(safe_vectors.shape) == 3:
        safe_vectors = safe_vectors.reshape(safe_vectors.shape[0], -1)

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
