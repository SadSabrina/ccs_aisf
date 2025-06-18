import numpy as np
import torch


class PatchHook:
    """Hook for applying steering during model inference."""

    def __init__(self, token_idx, direction, character, alpha=2):
        self.token_idx = token_idx
        self.direction = direction
        self.alpha = alpha
        self.character = character

    def set_character(self, character):
        self.character = character

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_direction(self, direction):
        self.direction = direction

    def __call__(self, module, input, output):
        print(f"Alpha param: {self.alpha}")

        output = output.clone()
        output[self.character == 0, self.token_idx, :] -= self.alpha * self.direction
        output[self.character == 1, self.token_idx, :] += self.alpha * self.direction
        print(f"Patched token {self.token_idx}")
        return output


def apply_proper_steering(
    X_pos, X_neg, best_layer, direction_tensor, steering_alpha, device
):
    """
    Apply steering to the best layer and simulate propagation to subsequent layers.

    Changed: Core steering function that properly applies steering effects to simulate
    how steering in one layer affects subsequent layers in a real transformer.

    Parameters:
        X_pos: Positive representations [N, n_layers, hidden_dim]
        X_neg: Negative representations [N, n_layers, hidden_dim]
        best_layer: Layer index where steering is applied
        direction_tensor: Steering direction vector
        steering_alpha: Steering strength
        device: Computing device

    Returns:
        X_pos_steered: Steered positive representations
        X_neg_steered: Steered negative representations
    """
    X_pos_steered = X_pos.copy()
    X_neg_steered = X_neg.copy()

    # Convert direction to numpy for CPU operations
    direction_np = (
        direction_tensor.cpu().numpy()
        if torch.is_tensor(direction_tensor)
        else direction_tensor
    )

    # Apply direct steering to the best layer
    X_pos_steered[:, best_layer, :] += steering_alpha * direction_np
    X_neg_steered[:, best_layer, :] -= steering_alpha * direction_np

    # Simulate propagation to subsequent layers
    # In a real transformer, changes in layer N affect layers N+1, N+2, etc.
    # We'll simulate this by applying diminishing effects to later layers
    n_layers = X_pos.shape[1]

    for layer_idx in range(best_layer + 1, n_layers):
        # Apply diminishing steering effect (exponential decay)
        decay_factor = np.exp(
            -0.3 * (layer_idx - best_layer)
        )  # Decay parameter can be tuned
        propagated_effect = steering_alpha * decay_factor

        # Apply propagated effect with reduced magnitude
        X_pos_steered[:, layer_idx, :] += propagated_effect * direction_np * 0.5
        X_neg_steered[:, layer_idx, :] -= propagated_effect * direction_np * 0.5

    print(f"Applied steering to layer {best_layer} with alpha={steering_alpha}")
    print(f"Propagated effects to layers {best_layer+1} through {n_layers-1}")

    return X_pos_steered, X_neg_steered


def get_steering_direction(ccs):
    """
    Extract steering direction from trained CCS.

    Parameters:
        ccs: Trained CCS object

    Returns:
        direction_tensor: Normalized steering direction as tensor
        weights: Raw weights from CCS
        bias: Bias from CCS
    """
    weights, bias = ccs.get_weights()
    direction = weights / (np.linalg.norm(weights) + 1e-6)
    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=ccs.device)

    return direction_tensor, weights, bias


def apply_simple_steering(X_pos, X_neg, layer_idx, direction_tensor, steering_alpha):
    """
    Apply simple steering to a single layer without propagation.

    Parameters:
        X_pos: Positive representations for single layer [N, hidden_dim]
        X_neg: Negative representations for single layer [N, hidden_dim]
        layer_idx: Layer index (for logging)
        direction_tensor: Steering direction vector
        steering_alpha: Steering strength

    Returns:
        X_pos_steered: Steered positive representations
        X_neg_steered: Steered negative representations
    """
    # Convert direction to numpy for CPU operations
    direction_np = (
        direction_tensor.cpu().numpy()
        if torch.is_tensor(direction_tensor)
        else direction_tensor
    )

    # Apply steering
    X_pos_steered = X_pos + steering_alpha * direction_np
    X_neg_steered = X_neg - steering_alpha * direction_np

    print(f"Applied simple steering to layer {layer_idx} with alpha={steering_alpha}")

    return X_pos_steered, X_neg_steered


def test_steering_strengths(
    ccs, X_pos, X_neg, layer_idx, direction_tensor, alphas=None
):
    """
    Test different steering strengths and return prediction changes.

    Parameters:
        ccs: Trained CCS object
        X_pos: Positive representations [N, hidden_dim]
        X_neg: Negative representations [N, hidden_dim]
        layer_idx: Layer index
        direction_tensor: Steering direction vector
        alphas: List of steering strengths to test

    Returns:
        results: Dict with alpha values and corresponding prediction changes
    """
    if alphas is None:
        alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    results = {
        "alphas": alphas,
        "pos_predictions": [],
        "neg_predictions": [],
        "avg_confidence": [],
    }

    # Convert to tensors if needed
    if not isinstance(X_pos, torch.Tensor):
        X_pos_tensor = torch.tensor(X_pos, dtype=torch.float32, device=ccs.device)
    else:
        X_pos_tensor = X_pos

    if not isinstance(X_neg, torch.Tensor):
        X_neg_tensor = torch.tensor(X_neg, dtype=torch.float32, device=ccs.device)
    else:
        X_neg_tensor = X_neg

    direction_np = (
        direction_tensor.cpu().numpy()
        if torch.is_tensor(direction_tensor)
        else direction_tensor
    )

    for alpha in alphas:
        # Apply steering
        X_pos_steered = X_pos_tensor + alpha * torch.tensor(
            direction_np, dtype=torch.float32, device=ccs.device
        )
        X_neg_steered = X_neg_tensor - alpha * torch.tensor(
            direction_np, dtype=torch.float32, device=ccs.device
        )

        # Get predictions
        with torch.no_grad():
            pos_pred = ccs.best_probe(X_pos_steered).median().item()
            neg_pred = ccs.best_probe(X_neg_steered).median().item()

        avg_conf = 0.5 * (pos_pred + (1 - neg_pred))

        results["pos_predictions"].append(pos_pred)
        results["neg_predictions"].append(neg_pred)
        results["avg_confidence"].append(avg_conf)

    return results


def calculate_steering_metrics(X_orig, X_steered):
    """
    Calculate quantitative metrics for steering effects.

    Parameters:
        X_orig: Original representations [N, hidden_dim]
        X_steered: Steered representations [N, hidden_dim]

    Returns:
        metrics: Dict with various steering effect metrics
    """
    # Mean squared difference
    mse = np.mean((X_steered - X_orig) ** 2)

    # Mean absolute difference
    mae = np.mean(np.abs(X_steered - X_orig))

    # Cosine similarity between original and steered
    cos_sim = np.mean(
        [
            np.dot(x_orig, x_steer)
            / (np.linalg.norm(x_orig) * np.linalg.norm(x_steer) + 1e-8)
            for x_orig, x_steer in zip(X_orig, X_steered)
        ]
    )

    # L2 norm of difference vectors
    diff_norms = np.linalg.norm(X_steered - X_orig, axis=1)
    mean_diff_norm = np.mean(diff_norms)
    std_diff_norm = np.std(diff_norms)

    return {
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cos_sim,
        "mean_diff_norm": mean_diff_norm,
        "std_diff_norm": std_diff_norm,
        "max_diff_norm": np.max(diff_norms),
        "min_diff_norm": np.min(diff_norms),
    }


def compare_steering_layers(
    X_pos_orig, X_neg_orig, X_pos_steered, X_neg_steered, start_layer=0
):
    """
    Compare steering effects across all layers.

    Parameters:
        X_pos_orig: Original positive representations [N, n_layers, hidden_dim]
        X_neg_orig: Original negative representations [N, n_layers, hidden_dim]
        X_pos_steered: Steered positive representations [N, n_layers, hidden_dim]
        X_neg_steered: Steered negative representations [N, n_layers, hidden_dim]
        start_layer: Starting layer for comparison

    Returns:
        layer_metrics: Dict with metrics for each layer
    """
    n_layers = X_pos_orig.shape[1]
    layer_metrics = {}

    for layer_idx in range(start_layer, n_layers):
        # Positive representations
        pos_metrics = calculate_steering_metrics(
            X_pos_orig[:, layer_idx, :], X_pos_steered[:, layer_idx, :]
        )

        # Negative representations
        neg_metrics = calculate_steering_metrics(
            X_neg_orig[:, layer_idx, :], X_neg_steered[:, layer_idx, :]
        )

        # Combined metrics
        combined_metrics = {}
        for key in pos_metrics.keys():
            combined_metrics[f"pos_{key}"] = pos_metrics[key]
            combined_metrics[f"neg_{key}"] = neg_metrics[key]
            combined_metrics[f"avg_{key}"] = (pos_metrics[key] + neg_metrics[key]) / 2

        layer_metrics[layer_idx] = combined_metrics

    return layer_metrics
