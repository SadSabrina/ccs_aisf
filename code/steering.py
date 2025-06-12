from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch

# MAIN STEERING FUNCTIONS

class PatchHook:
    
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
        print(f'Alpha param: {self.alpha}')

        output = output.clone()
        output[self.character == 0, self.token_idx, :] -= self.alpha * self.direction
        output[self.character == 1, self.token_idx, :] += self.alpha * self.direction
        print(f"Patched token {self.token_idx}")
        return output


# PLOTTIG BEFORE STEERING FUNCTIONS

def plot_steering_power(ccs, positive_statements, negative_statements, deltas, labels=["POS (statement + ДА)", "NEG (statement + НЕТ)"], title="Steering along opinion direction"):

    weights, bias = ccs.get_weights() 
    direction = weights / (np.linalg.norm(weights) + 1e-6)

    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=ccs.device)

    scores_pos, scores_neg = [], []

    if type(positive_statements) != torch.Tensor:
        positive_statements = torch.Tensor(positive_statements, dtype=torch.float32, device=ccs.device)
    if type(negative_statements) != torch.Tensor:
        negative_statements = torch.Tensor(negative_statements, dtype=torch.float32, device=ccs.device)

    for delta in deltas:
        
        positive_statements_steered = positive_statements + delta * direction_tensor
        negative_statements_steered = negative_statements - delta * direction_tensor

        score_pos = ccs.best_probe(positive_statements_steered).median().item()
        score_neg = ccs.best_probe(negative_statements_steered).median().item()
        
        scores_pos.append(score_pos)
        scores_neg.append(score_neg)

    plt.plot(deltas, scores_pos, label=labels[0])
    plt.plot(deltas, scores_neg, label=labels[0])
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Steering delta")
    plt.ylabel("Average CCS result")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()  


def plot_boundary(positive_statemtns, negative_statemnts, y_vector, ccs, n_components, components):

    X_all = positive_statemtns - negative_statemnts

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_all)

    w, b = ccs.get_weights()
    w_pca = pca.components_ @ w  # projection w to PCA space
    
    idx_x, idx_y = components
    x_label = f"PC{idx_x}"
    y_label = f"PC{idx_y}"

    df = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(n_components)])
    df["label"] = y_vector

    # boundary
    w_x, w_y = w_pca[idx_x], w_pca[idx_y]
    slope = -w_x / (w_y + 1e-8)

    intercept = -b / (w_y + 1e-8)

    # plot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x=x_label, y=y_label, hue="label", palette="Set1", alpha=0.7)

    x_vals = np.linspace(df[x_label].min(), df[x_label].max(), 200)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, 'k--', label="Decision boundary")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Decision boundary in PCA space")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
