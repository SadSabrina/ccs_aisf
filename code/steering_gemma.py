import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# MAIN STEERING FUNCTIONS FOR GEMMA2


class PatchHook:
    """
    GPU-optimized patch hook for Gemma2 architecture
    CHANGED: Modified for Gemma2 architecture, all operations kept on GPU
    """

    def __init__(self, token_idx, direction, character, alpha=2, device=None):
        self.token_idx = token_idx

        # CHANGED: Ensure direction is on GPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if not isinstance(direction, torch.Tensor):
            self.direction = torch.tensor(direction, dtype=torch.float32, device=device)
        else:
            self.direction = direction.to(device)

        self.alpha = alpha

        # CHANGED: Ensure character tensor is on GPU
        if not isinstance(character, torch.Tensor):
            self.character = torch.tensor(character, dtype=torch.long, device=device)
        else:
            self.character = character.to(device)

        self.device = device

    def set_character(self, character):
        """CHANGED: Ensure character stays on GPU"""
        if not isinstance(character, torch.Tensor):
            self.character = torch.tensor(
                character, dtype=torch.long, device=self.device
            )
        else:
            self.character = character.to(self.device)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_direction(self, direction):
        """CHANGED: Ensure direction stays on GPU"""
        if not isinstance(direction, torch.Tensor):
            self.direction = torch.tensor(
                direction, dtype=torch.float32, device=self.device
            )
        else:
            self.direction = direction.to(self.device)

    def __call__(self, module, input, output):
        print(f"Alpha param: {self.alpha}")

        # CHANGED: Handle Gemma2 output format - output is a tuple (hidden_states,)
        if isinstance(output, tuple):
            hidden_states = output[0].clone()
        else:
            hidden_states = output.clone()

        # CHANGED: Ensure all operations stay on GPU
        hidden_states[self.character == 0, self.token_idx, :] -= (
            self.alpha * self.direction
        )
        hidden_states[self.character == 1, self.token_idx, :] += (
            self.alpha * self.direction
        )

        print(f"Patched token {self.token_idx}")

        # CHANGED: Return in same format as input
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states


class PatchHookTOP:
    """
    GPU-optimized patch hook for multiple tokens in Gemma2
    CHANGED: Modified for Gemma2 architecture, all operations kept on GPU
    """

    def __init__(self, token_idx_list, direction, character, alpha=2, device=None):
        self.token_idx_list = token_idx_list

        # CHANGED: Ensure direction is on GPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if not isinstance(direction, torch.Tensor):
            self.direction = torch.tensor(direction, dtype=torch.float32, device=device)
        else:
            self.direction = direction.to(device)

        self.alpha = alpha

        # CHANGED: Ensure character tensor is on GPU
        if not isinstance(character, torch.Tensor):
            self.character = torch.tensor(character, dtype=torch.long, device=device)
        else:
            self.character = character.to(device)

        self.device = device

    def set_character(self, character):
        """CHANGED: Ensure character stays on GPU"""
        if not isinstance(character, torch.Tensor):
            self.character = torch.tensor(
                character, dtype=torch.long, device=self.device
            )
        else:
            self.character = character.to(self.device)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_direction(self, direction):
        """CHANGED: Ensure direction stays on GPU"""
        if not isinstance(direction, torch.Tensor):
            self.direction = torch.tensor(
                direction, dtype=torch.float32, device=self.device
            )
        else:
            self.direction = direction.to(self.device)

    def set_token_idx_list(self, token_idx_list):
        self.token_idx_list = token_idx_list

    def __call__(self, module, input, output):
        print(f"Alpha param: {self.alpha}")

        # CHANGED: Handle Gemma2 output format
        if isinstance(output, tuple):
            hidden_states = output[0].clone()
        else:
            hidden_states = output.clone()

        # CHANGED: Vectorized operation for better GPU performance
        for token_idx in self.token_idx_list:
            hidden_states[self.character == 0, token_idx, :] -= (
                self.alpha * self.direction
            )
            hidden_states[self.character == 1, token_idx, :] += (
                self.alpha * self.direction
            )
            print(f"Patched token {token_idx}")

        # CHANGED: Return in same format as input
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states


# PLOTTING BEFORE STEERING FUNCTIONS


def plot_steering_power(
    ccs,
    positive_statements,
    negative_statements,
    deltas,
    labels=["POS (statement + ДА)", "NEG (statement + НЕТ)"],
    title="Steering along opinion direction",
    device=None,
):
    """
    GPU-optimized steering power plotting
    CHANGED: All tensor operations kept on GPU, only final results moved for plotting
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights, bias = ccs.get_weights()

    # CHANGED: Use torch operations on GPU instead of numpy
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        weights = weights.to(device)

    direction = weights / (torch.linalg.norm(weights) + 1e-6)

    scores_pos, scores_neg = [], []

    # CHANGED: Ensure statements are GPU tensors
    if not isinstance(positive_statements, torch.Tensor):
        positive_statements = torch.tensor(
            positive_statements, dtype=torch.float32, device=device
        )
    else:
        positive_statements = positive_statements.to(device)

    if not isinstance(negative_statements, torch.Tensor):
        negative_statements = torch.tensor(
            negative_statements, dtype=torch.float32, device=device
        )
    else:
        negative_statements = negative_statements.to(device)

    # CHANGED: All operations on GPU
    for delta in deltas:
        positive_statements_steered = positive_statements + delta * direction
        negative_statements_steered = negative_statements - delta * direction

        score_pos = ccs.best_probe(positive_statements_steered).median().item()
        score_neg = ccs.best_probe(negative_statements_steered).median().item()

        scores_pos.append(score_pos)
        scores_neg.append(score_neg)

    # CHANGED: Only plotting operations use CPU
    plt.plot(deltas, scores_pos, label=labels[0])
    plt.plot(
        deltas, scores_neg, label=labels[1]
    )  # CHANGED: Fixed bug - was labels[0] twice
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Steering delta")
    plt.ylabel("Average CCS result")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_boundary(
    positive_statements,
    negative_statements,
    y_vector,
    ccs,
    n_components,
    components,
    device=None,
):
    """
    GPU-optimized boundary plotting
    CHANGED: Use GPU tensors, minimal CPU transfers only for sklearn PCA, fixed parameter handling
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # CHANGED: Ensure inputs are on GPU and handle different input types
    if not isinstance(positive_statements, torch.Tensor):
        if hasattr(positive_statements, "values"):  # pandas Series/DataFrame
            positive_statements = torch.tensor(
                positive_statements.values, dtype=torch.float32, device=device
            )
        else:  # numpy array or list
            positive_statements = torch.tensor(
                positive_statements, dtype=torch.float32, device=device
            )
    else:
        positive_statements = positive_statements.to(device)

    if not isinstance(negative_statements, torch.Tensor):
        if hasattr(negative_statements, "values"):  # pandas Series/DataFrame
            negative_statements = torch.tensor(
                negative_statements.values, dtype=torch.float32, device=device
            )
        else:  # numpy array or list
            negative_statements = torch.tensor(
                negative_statements, dtype=torch.float32, device=device
            )
    else:
        negative_statements = negative_statements.to(device)

    # CHANGED: Handle y_vector properly
    if hasattr(y_vector, "values"):  # pandas Series
        y_vector_array = y_vector.values
    else:
        y_vector_array = y_vector

    X_all = positive_statements - negative_statements

    # CHANGED: Use sklearn PCA but keep data on GPU as much as possible
    # Note: sklearn PCA requires CPU arrays, so we do minimal CPU transfer
    X_all_cpu = X_all.cpu().numpy()

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_all_cpu)

    w, b = ccs.get_weights()

    # CHANGED: Ensure weights are numpy for PCA operations
    if isinstance(w, torch.Tensor):
        w = w.cpu().numpy()

    w_pca = pca.components_ @ w  # projection w to PCA space

    idx_x, idx_y = components
    x_label = f"PC{idx_x}"
    y_label = f"PC{idx_y}"

    df = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(n_components)])
    df["label"] = y_vector_array

    # CHANGED: Add explicit check for bias value
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    if hasattr(b, "__len__") and len(b) > 0:
        b_val = b[0] if isinstance(b, (list, np.ndarray)) else b
    else:
        b_val = b

    # boundary calculation
    w_x, w_y = w_pca[idx_x], w_pca[idx_y]
    slope = -w_x / (w_y + 1e-8)
    intercept = -b_val / (w_y + 1e-8)

    # plotting
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=df, x=x_label, y=y_label, hue="label", palette="Set1", alpha=0.7
    )

    x_vals = np.linspace(df[x_label].min(), df[x_label].max(), 200)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, "k--", label="Decision boundary")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Decision boundary in PCA space")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# GPU-OPTIMIZED UTILITY FUNCTIONS FOR GEMMA2


def create_steering_direction(ccs, device=None):
    """
    Create normalized steering direction on GPU
    CHANGED: New function to ensure GPU operations
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights, bias = ccs.get_weights()

    # CHANGED: Ensure weights are on GPU and use torch operations
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        weights = weights.to(device)

    # CHANGED: Use torch.linalg.norm instead of numpy
    direction = weights / (torch.linalg.norm(weights) + 1e-6)

    return direction


def prepare_gemma_inputs(texts, tokenizer, device=None):
    """
    Prepare inputs for Gemma2 model on GPU
    CHANGED: New function to ensure all inputs stay on GPU
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(list(texts), return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


def get_gemma_layer_hook_point(model, layer_idx):
    """
    Get the correct hook point for Gemma2 model
    CHANGED: New function to abstract Gemma2 architecture - use model.layers instead of deberta.encoder.layer
    """
    return model.model.layers[layer_idx]


# MAIN STEERING PIPELINE FOR GEMMA2


def run_gemma_steering(
    model, tokenizer, ccs, hate_data, layer_idx=4, alpha=0.049, token_idx=0
):
    """
    Complete GPU-optimized steering pipeline for Gemma2
    CHANGED: New function that encapsulates the entire steering process with correct Gemma2 architecture
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CHANGED: Create steering direction on GPU
    direction = create_steering_direction(ccs, device)

    # CHANGED: Prepare data on GPU
    true = hate_data["is_harmfull_opposition"]
    texts = hate_data["statement"]
    text_yes = texts + " Yes."

    inputs_yes = prepare_gemma_inputs(text_yes, tokenizer, device)
    true_tensor = torch.tensor(true.values, dtype=torch.long, device=device)

    # CHANGED: Ensure model is on GPU
    model = model.to(device)

    # Create and configure hook
    hook_obj = PatchHook(
        token_idx=token_idx, direction=direction, character=true_tensor, alpha=alpha
    )

    print(f"[GEMMA_STEERING] hook_obj id: {id(hook_obj)}")
    print(f"[GEMMA_STEERING] character shape: {hook_obj.character.shape}")
    print(f"[GEMMA_STEERING] direction device: {direction.device}")
    print(f"[GEMMA_STEERING] true_tensor device: {true_tensor.device}")

    # CHANGED: Hook to correct Gemma2 layer - use model.model.layers instead of deberta.encoder.layer
    hook_point = get_gemma_layer_hook_point(model, layer_idx)
    h = hook_point.register_forward_hook(hook_obj)

    # Run inference with patching
    with torch.no_grad():
        outputs_patched_yes = model(**inputs_yes, output_hidden_states=True)

    # Remove hook
    h.remove()

    print("[GEMMA_STEERING] Patching completed successfully!")
    print(f"[GEMMA_STEERING] Output shape: {outputs_patched_yes.logits.shape}")
    print(f"[GEMMA_STEERING] All tensors on device: {device}")

    return outputs_patched_yes, hook_obj
