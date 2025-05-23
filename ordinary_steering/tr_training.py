import copy
import logging
import os
import random
from datetime import datetime

# Set matplotlib backend to Agg (non-interactive)
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ccs_metrics import (
    agreement_score,
    compute_class_separability,
    contradiction_index,
    fisher_information_analysis,
    ideal_representation_distance,
    representation_stability,
    subspace_analysis,
)
from logger import print_results_summary
from ordinary_steering_metrics import plot_coefficient_sweep_lines_comparison
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from tqdm import tqdm
from tr_data_utils import extract_representation
from tr_plotting import (
    plot_all_decision_boundaries,
    plot_all_layer_vectors,
    plot_all_strategies_all_steering_vectors,
    plot_individual_steering_vectors,
    plot_performance_across_layers,
    plot_vectors_all_strategies,
    visualize_decision_boundary,
    visualize_detailed_decision_boundary,
)

matplotlib.use("Agg")


# Set up logging
def setup_logging(run_dir):
    log_file = os.path.join(
        run_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()


class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        # Ensure output has shape [batch_size, 1]
        if len(o.shape) == 1:
            o = o.unsqueeze(1)
        return torch.sigmoid(o)


class CCS:
    """Contrast Consistent Search (CCS): a probe for analyzing a statement (x1) and the negation of the statement (x0),
    with classification task loss (optional).
    """

    def __init__(
        self,
        model,
        tokenizer,
        layer_idx,
        device="cuda",
        linear=True,
        var_normalize=False,
        lambda_classification=0.0,
        nepochs=1500,
        ntries=1,
        lr=0.015,
        weight_decay=0.01,
    ):
        """Initialize CCS."""
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self.linear = linear
        self.var_normalize = var_normalize
        self.lambda_classification = lambda_classification
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.weight_decay = weight_decay

        # Initialize probe
        self.d = model.config.hidden_size
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device).to(dtype=torch.float32)

    def normalize(self, x):
        x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            x = x / (x.std(axis=0, keepdims=True) + 1e-6)
        return x

    def get_loss(self, p0, p1, y_true=None):
        informative_loss = (torch.min(p0, p1) ** 2).mean()
        consistent_loss = ((p0 - (1 - p1)) ** 2).mean()

        if self.lambda_classification != 0 and y_true is not None:
            avg_pred = 0.5 * (p0 + (1 - p1))
            if not isinstance(y_true, torch.Tensor):
                y_true = torch.tensor(
                    y_true, dtype=torch.float32, device=avg_pred.device
                )
            if y_true.ndim == 1:
                y_true = y_true.view(-1, 1)

            bce = nn.BCELoss()
            classification_loss = bce(avg_pred, y_true)

            return (
                informative_loss
                + consistent_loss
                + self.lambda_classification * classification_loss
            )

        return informative_loss + consistent_loss

    def predict(self, X_hate_test, X_safe_test):
        """Get predictions for test data."""
        with torch.no_grad():
            # Process hate data
            hate_reps = []
            for text in X_hate_test:
                rep = extract_representation(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    text=text,
                    layer_index=self.layer_idx,
                    strategy="last-token",
                    device=self.device,
                    keep_on_gpu=True,
                )
                hate_reps.append(rep)
            hate_reps = torch.tensor(np.stack(hate_reps), device=self.device)

            # Process safe data
            safe_reps = []
            for text in X_safe_test:
                rep = extract_representation(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    text=text,
                    layer_index=self.layer_idx,
                    strategy="last-token",
                    device=self.device,
                    keep_on_gpu=True,
                )
                safe_reps.append(rep)
            safe_reps = torch.tensor(np.stack(safe_reps), device=self.device)

            # Apply variance normalization if configured
            if self.var_normalize:
                hate_reps = self.normalize(hate_reps)
                safe_reps = self.normalize(safe_reps)

            # Predict probabilities
            hate_logits = self.probe(hate_reps.view(hate_reps.size(0), -1))
            safe_logits = self.probe(safe_reps.view(safe_reps.size(0), -1))

            # Get probabilities and predictions
            hate_probs = F.softmax(hate_logits, dim=1)
            safe_probs = F.softmax(safe_logits, dim=1)

            # Stack all predictions
            all_probs = torch.cat([hate_probs, safe_probs], dim=0)
            confidence = torch.max(all_probs, dim=1)[0]
            preds = torch.argmax(all_probs, dim=1)

            return preds.cpu().numpy(), confidence.cpu().numpy()

    def predict_from_vectors(self, vector_data):
        """Predict directly from pre-computed vector representations.

        Args:
            vector_data: tensor or numpy array of shape (batch_size, *) containing vector representations
                        Will be reshaped to (batch_size, -1) for the model

        Returns:
            Tuple of (predictions, confidences) as numpy arrays
        """
        with torch.no_grad():
            # Convert numpy array to tensor if necessary
            if not isinstance(vector_data, torch.Tensor):
                vector_data = torch.tensor(
                    vector_data, device=self.device, dtype=torch.float32
                )
            else:
                # Make sure tensor is on the right device and has the right dtype
                vector_data = vector_data.to(device=self.device, dtype=torch.float32)

            # Apply variance normalization if configured
            if self.var_normalize:
                vector_data = self.normalize(vector_data)

            # Flatten to 2D if needed
            if len(vector_data.shape) > 2:
                vector_data = vector_data.reshape(vector_data.size(0), -1)

            # Get logits and probabilities
            logits = self.probe(vector_data)
            probs = torch.nn.functional.softmax(logits, dim=1)

            # Get predictions and confidence
            confidence = torch.max(probs, dim=1)[0]
            preds = torch.argmax(probs, dim=1)

            # Ensure we return numpy arrays
            preds_np = preds.cpu().numpy()
            confidence_np = confidence.cpu().numpy()

            return preds_np, confidence_np

    def get_acc(self, X_hate_test, X_safe_test, y_test):
        """Get accuracy for test data."""
        predictions, _ = self.predict(X_hate_test, X_safe_test)

        # Match predictions length to y_test length
        if len(predictions) > len(y_test):
            predictions = predictions[: len(y_test)]
        elif len(predictions) < len(y_test):
            y_test = y_test[: len(predictions)]

        acc = float(accuracy_score(y_test, predictions))
        return max(acc, 1 - acc)  # Return the better of the two possible accuracies

    def get_silhouette(self, X_hate_test, X_safe_test, y_test):
        """Get silhouette score for test data."""
        # Get representations
        hate_reps = []
        safe_reps = []

        for text in X_hate_test:
            rep = extract_representation(
                model=self.model,
                tokenizer=self.tokenizer,
                text=text,
                layer_index=self.layer_idx,
                strategy="last-token",
                device=self.device,
                keep_on_gpu=False,
            )
            hate_reps.append(rep)

        for text in X_safe_test:
            rep = extract_representation(
                model=self.model,
                tokenizer=self.tokenizer,
                text=text,
                layer_index=self.layer_idx,
                strategy="last-token",
                device=self.device,
                keep_on_gpu=False,
            )
            safe_reps.append(rep)

        # Stack tensors on GPU
        if hate_reps and safe_reps:
            hate_reps = torch.stack(hate_reps)
            safe_reps = torch.stack(safe_reps)

            # Get predictions
            predictions, _ = self.predict(X_hate_test, X_safe_test)

            # Match predictions length to y_test length
            if len(predictions) > len(y_test):
                predictions = predictions[: len(y_test)]
            elif len(predictions) < len(y_test):
                y_test = y_test[: len(predictions)]

            # Calculate difference vectors on GPU
            diffs = safe_reps - hate_reps

            # Need to move to CPU for silhouette score calculation
            # since sklearn doesn't support GPU computation
            diffs_cpu = diffs.cpu().numpy()

            # Compute silhouette score if we have more than one class in predictions
            if len(np.unique(predictions)) > 1:
                return silhouette_score(diffs_cpu, predictions, metric="cosine")

        return 0.0

    def train(
        self,
        train_dataloader,
        val_dataloader,
        n_epochs=None,
        learning_rate=None,
        batch_size=32,
        save_every=1,
        log_every=1,
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
    ):
        """Train CCS probe for a specific layer."""
        if n_epochs is None:
            n_epochs = self.nepochs
        if learning_rate is None:
            learning_rate = self.lr

        print(f"\nTraining CCS for layer {self.layer_idx}")

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=learning_rate, weight_decay=self.weight_decay
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            self.model.eval()  # Ensure model is in eval mode

            # Training phase
            train_losses = []
            train_preds = []
            train_labels = []
            train_probs = []

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
            for batch in pbar:
                hate_data = batch["hate_data"]
                safe_data = batch["safe_data"]
                labels = batch["labels"]

                # Get representations
                hate_reps = []
                safe_reps = []

                # Process hate data
                for text in hate_data:
                    rep = extract_representation(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        text=text,
                        layer_index=self.layer_idx,
                        strategy="last-token",
                        device=self.device,
                        keep_on_gpu=True,
                    )
                    hate_reps.append(rep)

                # Process safe data
                for text in safe_data:
                    rep = extract_representation(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        text=text,
                        layer_index=self.layer_idx,
                        strategy="last-token",
                        device=self.device,
                        keep_on_gpu=True,
                    )
                    safe_reps.append(rep)

                # Stack tensors directly on GPU
                hate_reps = torch.stack(hate_reps)
                safe_reps = torch.stack(safe_reps)
                hate_reps = self.normalize(hate_reps)
                safe_reps = self.normalize(safe_reps)

                # Get predictions from probe
                p0 = self.probe(hate_reps)
                p1 = self.probe(safe_reps)

                # Compute loss
                loss = self.get_loss(p0, p1, labels)

                # Get predictions and probabilities
                avg_confidence = 0.5 * (p0 + (1 - p1))
                predictions = (avg_confidence.detach().cpu().numpy() > 0.5).astype(int)[
                    :, 0
                ]

                # Store predictions and labels
                train_preds.extend(predictions)
                train_labels.extend(labels.cpu().numpy())
                train_probs.extend(avg_confidence.detach().cpu().numpy()[:, 0])

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                # Update progress bar
                if len(train_labels) > 0 and len(np.unique(train_labels)) > 1:
                    train_acc = accuracy_score(train_labels, train_preds)
                    train_auc = roc_auc_score(train_labels, train_probs)
                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc": f"{train_acc:.4f}",
                            "auc": f"{train_auc:.4f}",
                        }
                    )
                else:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Validation phase
            val_losses = []
            val_preds = []
            val_labels = []
            val_probs = []

            with torch.no_grad():
                for batch in val_dataloader:
                    hate_data = batch["hate_data"]
                    safe_data = batch["safe_data"]
                    labels = batch["labels"]

                    # Get representations
                    hate_reps = []
                    safe_reps = []

                    for text in hate_data:
                        rep = extract_representation(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            text=text,
                            layer_index=self.layer_idx,
                            strategy="last-token",
                            device=self.device,
                            keep_on_gpu=True,
                        )
                        hate_reps.append(rep)

                    for text in safe_data:
                        rep = extract_representation(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            text=text,
                            layer_index=self.layer_idx,
                            strategy="last-token",
                            device=self.device,
                            keep_on_gpu=True,
                        )
                        safe_reps.append(rep)

                    # Stack tensors directly on GPU
                    hate_reps = torch.stack(hate_reps)
                    safe_reps = torch.stack(safe_reps)
                    hate_reps = self.normalize(hate_reps)
                    safe_reps = self.normalize(safe_reps)

                    # Get predictions from probe
                    p0 = self.probe(hate_reps)
                    p1 = self.probe(safe_reps)

                    # Compute loss
                    val_loss = self.get_loss(p0, p1, labels)
                    val_losses.append(val_loss.item())

                    # Get predictions and probabilities
                    avg_confidence = 0.5 * (p0 + (1 - p1))
                    predictions = (avg_confidence.cpu().numpy() > 0.5).astype(int)[:, 0]

                    # Store predictions and labels
                    val_preds.extend(predictions)
                    val_labels.extend(labels.cpu().numpy())
                    val_probs.extend(avg_confidence.cpu().numpy()[:, 0])

            # Compute average losses and metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            # Only calculate metrics if we have sufficient data
            if len(train_labels) > 0 and len(np.unique(train_labels)) > 1:
                train_acc = accuracy_score(train_labels, train_preds)
                train_auc = roc_auc_score(train_labels, train_probs)
            else:
                train_acc = 0
                train_auc = 0

            if len(val_labels) > 0 and len(np.unique(val_labels)) > 1:
                val_acc = accuracy_score(val_labels, val_preds)
                val_auc = roc_auc_score(val_labels, val_probs)
            else:
                val_acc = 0
                val_auc = 0

            # Early stopping check
            if avg_val_loss < best_val_loss - early_stopping_threshold:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_probe = copy.deepcopy(self.probe)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # Log progress
            if (epoch + 1) % log_every == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f} - "
                    f"Val Loss: {avg_val_loss:.4f} - "
                    f"Train Acc: {train_acc:.4f} - "
                    f"Val Acc: {val_acc:.4f} - "
                    f"Train AUC: {train_auc:.4f} - "
                    f"Val AUC: {val_auc:.4f}"
                )

        return {
            "final_metrics": {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_auc": train_auc,
                "val_auc": val_auc,
            },
        }

    def repeated_train(self, train_dataloader, val_dataloader):
        """Train CCS multiple times and keep the best probe."""
        best_loss = float("inf")
        for train_num in range(self.ntries):
            self.initialize_probe()
            result = self.train(
                train_dataloader=train_dataloader, val_dataloader=val_dataloader
            )
            if result["final_metrics"]["val_loss"] < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = result["final_metrics"]["val_loss"]
        return best_loss

    def predict_with_steering(
        self,
        test_dataloader,
        steering_vector,
        steering_coefficient=1.0,
    ):
        """Get predictions with steering applied."""
        predictions = []
        probabilities = []

        # Make sure steering_vector has the right shape for broadcasting
        if len(steering_vector.shape) == 1:
            # Convert 1D vector to match representation shape (add batch and possibly token dims)
            steering_vector = steering_vector.reshape(1, -1)
            if len(steering_vector.shape) == 2 and steering_vector.shape[0] == 1:
                # This is already correctly shaped for 2D case (batch, hidden_dim)
                pass
            else:
                # Reshape for 3D case (batch, token, hidden_dim)
                steering_vector = steering_vector.reshape(1, 1, -1)

        with torch.no_grad():
            for batch in test_dataloader:
                hate_data = batch[
                    "hate_data"
                ]  # This contains the harmful content (hate_yes or safe_no)
                safe_data = batch[
                    "safe_data"
                ]  # This contains the safe content (safe_yes or hate_no)
                data_types = batch.get("data_type", None)

                # Get representations
                hate_reps = []
                safe_reps = []

                for text in hate_data:
                    rep = extract_representation(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        text=text,
                        layer_index=self.layer_idx,
                        strategy="last-token",
                        device=self.device,
                        keep_on_gpu=False,
                    )
                    hate_reps.append(rep)

                for text in safe_data:
                    rep = extract_representation(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        text=text,
                        layer_index=self.layer_idx,
                        strategy="last-token",
                        device=self.device,
                        keep_on_gpu=False,
                    )
                    safe_reps.append(rep)

                # Stack tensors directly on GPU
                hate_reps = torch.stack(hate_reps)
                safe_reps = torch.stack(safe_reps)
                hate_reps = self.normalize(hate_reps)
                safe_reps = self.normalize(safe_reps)

                # Apply steering only to harmful representations (hate_yes and safe_no)
                steering_tensor = torch.tensor(steering_vector, device=self.device)

                # Ensure the steering tensor has the right shape for broadcasting to hate_reps
                if steering_tensor.shape != hate_reps.shape and len(
                    steering_tensor.shape
                ) < len(hate_reps.shape):
                    # If steering tensor has fewer dimensions, expand it to match hate_reps
                    for _ in range(len(hate_reps.shape) - len(steering_tensor.shape)):
                        steering_tensor = steering_tensor.unsqueeze(0)
                    # Expand to batch dimension
                    steering_tensor = steering_tensor.expand_as(hate_reps)

                print(
                    f"steering_tensor shape: {steering_tensor.shape}, hate_reps shape: {hate_reps.shape}, safe_reps shape: {safe_reps.shape}"
                )
                steered_hate_reps = hate_reps + steering_coefficient * steering_tensor

                # Get predictions using probe
                p0 = self.best_probe(steered_hate_reps)
                p1 = self.best_probe(safe_reps)

                # Compute average confidence
                avg_confidence = 0.5 * (p0 + (1 - p1))
                batch_predictions = (avg_confidence.cpu().numpy() > 0.5).astype(int)[
                    :, 0
                ]
                predictions.append(batch_predictions)
                probabilities.append(avg_confidence.cpu().numpy()[:, 0])

        return np.concatenate(predictions), np.concatenate(probabilities)


def get_llm_type(model_cfg) -> str:
    if model_cfg.config.is_encoder_decoder:
        return "encoder-decoder"

    model_type = model_cfg.config.model_type.lower()
    if model_type in [
        "bert",
        "roberta",
        "distilbert",
        "albert",
        "deberta",
        "deberta-v2",
    ]:
        return "encoder"
    elif model_type in [
        "gpt2",
        "gpt",
        "gptj",
        "gpt_neo",
        "gpt_neox",
        "llama",
        "bloom",
        "opt",
        "falcon",
    ]:
        return "decoder"

    return "unknown"


def vectorize_df(
    df_text,
    model,
    tokenizer,
    layer_index=None,
    strategy="last-token",
    model_type=None,
    use_decoder=False,
    get_all_hs=False,
    device=None,
    name_source_data=None,
):
    """Converts text to embedding matrices"""
    n_show = min(3, len(df_text))
    logging.info(
        f"vectorize_df: first {n_show} texts: {[str(t) for t in df_text[:n_show]]}"
    )
    embeddings = []
    for text in tqdm(
        df_text,
        desc=f"Extracting embeddings from {strategy} strategy : {name_source_data}",
    ):
        vec = extract_representation(
            model=model,
            tokenizer=tokenizer,
            text=text,
            layer_index=layer_index,
            strategy=strategy,
            model_type=model_type,
            use_decoder=use_decoder,
            get_all_hs=get_all_hs,
            device=device,
        )
        embeddings.append(vec)
    return np.stack(embeddings)


def compute_steering_vector(hate_representation, safe_representation):
    """Compute steering vector as the difference between hate and safe representations

    Args:
        hate_representation: Tensor or array of hate representations
        safe_representation: Tensor or array of safe representations

    Returns:
        Steering vector in the same format as input (PyTorch tensor or NumPy array)
    """
    # Check input type to determine operation mode
    is_tensor = isinstance(hate_representation, torch.Tensor)
    device = hate_representation.device if is_tensor else None

    # Log original shapes
    logging.info(
        f"compute_steering_vector: hate_representation shape={hate_representation.shape}"
    )
    logging.info(
        f"compute_steering_vector: safe_representation shape={safe_representation.shape}"
    )

    # Reshape if needed (3D -> 2D)
    if len(hate_representation.shape) == 3:
        if is_tensor:
            hate_representation = hate_representation.reshape(
                hate_representation.shape[0], -1
            )
            safe_representation = safe_representation.reshape(
                safe_representation.shape[0], -1
            )
        else:
            hate_representation = hate_representation.reshape(
                hate_representation.shape[0], -1
            )
            safe_representation = safe_representation.reshape(
                safe_representation.shape[0], -1
            )
        logging.info(
            f"compute_steering_vector: reshaped hate_representation shape={hate_representation.shape}"
        )

    # Compute mean representations
    if is_tensor:
        hate_mean = torch.mean(hate_representation, dim=0)
        safe_mean = torch.mean(safe_representation, dim=0)

        # Log statistics
        logging.info(f"compute_steering_vector: hate_mean shape={hate_mean.shape}")
        logging.info(
            f"compute_steering_vector: hate_representation mean={torch.mean(hate_representation).item()}, std={torch.std(hate_representation).item()}"
        )
        logging.info(
            f"compute_steering_vector: safe_representation mean={torch.mean(safe_representation).item()}, std={torch.std(safe_representation).item()}"
        )
    else:
        hate_mean = np.mean(hate_representation, axis=0)
        safe_mean = np.mean(safe_representation, axis=0)

        # Log statistics
        logging.info(f"compute_steering_vector: hate_mean shape={hate_mean.shape}")
        logging.info(
            f"compute_steering_vector: hate_representation mean={np.mean(hate_representation)}, std={np.std(hate_representation)}"
        )
        logging.info(
            f"compute_steering_vector: safe_representation mean={np.mean(safe_representation)}, std={np.std(safe_representation)}"
        )

    # Compute steering vector (difference of means)
    steering_vector = safe_mean - hate_mean
    logging.info(
        f"compute_steering_vector: steering_vector shape={steering_vector.shape}"
    )

    return steering_vector


def apply_steering_vector(representation, steering_vector, coefficient=1.0):
    """Apply steering vector to a representation

    Args:
        representation: Tensor or array to modify
        steering_vector: Steering vector to apply (same type as representation)
        coefficient: Coefficient to scale the steering vector

    Returns:
        Modified representation in the same format as input
    """
    # Ensure coefficient is a scalar in the right format
    if isinstance(representation, torch.Tensor):
        # PyTorch tensors - keep on GPU
        if not isinstance(steering_vector, torch.Tensor):
            steering_vector = torch.tensor(
                steering_vector,
                device=representation.device,
                dtype=representation.dtype,
            )

        # Handle coefficient as tensor
        coef = torch.tensor(
            coefficient, device=representation.device, dtype=representation.dtype
        )

        return representation + coef * steering_vector
    else:
        # NumPy arrays
        return representation + coefficient * steering_vector


def train_ccs_with_steering(
    model,
    tokenizer,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    run_dir,
    n_layers=12,
    n_epochs=10,
    learning_rate=1e-4,
    batch_size=32,
    device="cuda",
    save_every=1,
    log_every=1,
    early_stopping_patience=3,
    early_stopping_threshold=0.001,
    steering_coefficients=None,
):
    """Train CCS probes for each layer of the model and evaluate them with steering.

    ### DETAILED PROCESS EXPLANATION ###

    This function performs the following steps for each layer in the model:

    1. Training Phase:
       - Initialize a CCS probe for the current layer
       - Train the probe using train_dataloader and validate with val_dataloader
       - Apply early stopping based on validation loss

    2. Metrics Calculation (for each layer and steering coefficient):
       - Base metrics: accuracy, AUC, silhouette score (cluster separation)
       - Steering metrics: similarity_change, path_length, semantic_consistency
       - Agreement metrics: agreement_score, contradiction_index
       - Stability metrics: representation_stability
       - Classification metrics: precision, recall, F1

    3. Visualization/Plotting (generated throughout and at the end):
       - Individual layer visualizations:
         * Decision boundaries for each layer
         * Vector representations showing hate/safe clusters
       - Cross-layer visualizations:
         * Performance metrics across all layers
         * Layer vector comparisons
         * Decision boundary comparisons
       - Steering coefficient comparisons:
         * Line plots showing metrics vs. coefficient values
         * Effect of different steering strategies

    4. Results Collection:
       - Store all metrics and visualizations
       - Generate comprehensive summary using print_results_summary
       - Save detailed JSON data and summary text

    The function makes extensive use of the following metrics and plotting functions:
    - Metrics: from ccs_metrics.py (evaluate_ccs_performance, compute_class_separability, etc.)
    - Plotting: from tr_plotting.py (plot_performance_across_layers, plot_all_layer_vectors, etc.)
    - Summary: from logger.py (print_results_summary)

    Args:
        model: The language model to use for extracting representations
        tokenizer: The tokenizer to use for the model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        test_dataloader: DataLoader for test data
        run_dir: Directory to save results
        n_layers: Number of layers to train probes for
        n_epochs: Number of epochs to train for
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        device: Device to use for training
        save_every: Save checkpoint every save_every epochs
        log_every: Log metrics every log_every epochs
        early_stopping_patience: Patience for early stopping
        early_stopping_threshold: Threshold for early stopping
        steering_coefficients: List of steering coefficients to evaluate
    """
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if steering_coefficients is None:
        steering_coefficients = [0.5, 1.0, 2.0, 5.0]

    results = []  # Store results as a list
    all_layer_data = []  # Store data for each layer for later visualization
    all_strategy_data = {}  # Store data for different embedding strategies
    all_steering_vectors = {}  # Store steering vectors for different strategies

    # Train a CCS probe for each layer
    for layer_idx in range(n_layers):
        print(f"\nTraining CCS for layer {layer_idx}")

        ############### TRAINING ###############
        # Initialize CCS
        ccs = CCS(
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            device=device,
        )

        # Train the probe
        training_result = ccs.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            save_every=save_every,
            log_every=log_every,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )

        ############### DATA PREPARATION ###############
        # Get original representations for the layer
        hate_yes_texts = train_dataloader.dataset.get_by_type("hate_yes")
        safe_no_texts = train_dataloader.dataset.get_by_type("safe_no")
        safe_yes_texts = train_dataloader.dataset.get_by_type("safe_yes")
        hate_no_texts = train_dataloader.dataset.get_by_type("hate_no")

        # Validate text data
        if (
            not isinstance(hate_yes_texts, (list, np.ndarray))
            or len(hate_yes_texts) == 0
        ):
            print(
                f"Error: Invalid hate_yes_texts. Type: {type(hate_yes_texts)}, Length: 0"
            )
            continue

        if not isinstance(safe_no_texts, (list, np.ndarray)) or len(safe_no_texts) == 0:
            print(
                f"Error: Invalid safe_no_texts. Type: {type(safe_no_texts)}, Length: 0"
            )
            continue

        if (
            not isinstance(safe_yes_texts, (list, np.ndarray))
            or len(safe_yes_texts) == 0
        ):
            print(
                f"Error: Invalid safe_yes_texts. Type: {type(safe_yes_texts)}, Length: 0"
            )
            continue

        if not isinstance(hate_no_texts, (list, np.ndarray)) or len(hate_no_texts) == 0:
            print(
                f"Error: Invalid hate_no_texts. Type: {type(hate_no_texts)}, Length: 0"
            )
            continue

        print(
            f"Processing layer {layer_idx} - Using {len(hate_yes_texts)} hate_yes texts, "
            f"{len(safe_no_texts)} safe_no texts, {len(safe_yes_texts)} safe_yes texts, "
            f"{len(hate_no_texts)} hate_no texts"
        )

        # Process hate data (hate_yes + safe_no)
        hate_vectors_list = []
        for text in hate_yes_texts + safe_no_texts:
            vector = extract_representation(
                model=model,
                tokenizer=tokenizer,
                text=text,
                layer_index=layer_idx,
                strategy="last-token",
                device=device,
                keep_on_gpu=False,  # Convert to numpy for now
            )
            # Validate vector shape
            if not isinstance(vector, np.ndarray):
                print(f"Error: Invalid vector type: {type(vector)}")
                continue

            hate_vectors_list.append(vector)

        # Verify we got vectors
        if len(hate_vectors_list) == 0:
            print(f"Error: No valid hate vectors extracted for layer {layer_idx}")
            continue

        hate_vectors = np.array(hate_vectors_list)

        # Process safe data (safe_yes + hate_no)
        safe_vectors_list = []
        for text in safe_yes_texts + hate_no_texts:
            vector = extract_representation(
                model=model,
                tokenizer=tokenizer,
                text=text,
                layer_index=layer_idx,
                strategy="last-token",
                device=device,
                keep_on_gpu=False,  # Convert to numpy for now
            )
            # Validate vector shape
            if not isinstance(vector, np.ndarray):
                print(f"Error: Invalid vector type: {type(vector)}")
                continue

            safe_vectors_list.append(vector)

        # Verify we got vectors
        if len(safe_vectors_list) == 0:
            print(f"Error: No valid safe vectors extracted for layer {layer_idx}")
            continue

        safe_vectors = np.array(safe_vectors_list)

        # Calculate steering vector
        steering_vector = compute_steering_vector(hate_vectors, safe_vectors)

        # Validate steering vector
        if not isinstance(steering_vector, np.ndarray):
            print(
                f"Error: Invalid steering vector type: {type(steering_vector)} for layer {layer_idx}"
            )
            continue

        # Store data for visualization with proper validation
        # Verify vectors are valid numpy arrays
        if not isinstance(hate_vectors, np.ndarray) or not isinstance(
            safe_vectors, np.ndarray
        ):
            print(
                f"Layer {layer_idx}: Invalid vector types - skipping for visualization"
            )
            continue

        if hate_vectors.size == 0 or safe_vectors.size == 0:
            print(f"Layer {layer_idx}: Empty vectors - skipping for visualization")
            continue

        # Properly reshape vectors for calculation
        hate_vectors_reshaped = hate_vectors.reshape(hate_vectors.shape[0], -1)
        safe_vectors_reshaped = safe_vectors.reshape(safe_vectors.shape[0], -1)

        # Check for NaN or Inf values - skip layer if found
        if (
            np.isnan(hate_vectors_reshaped).any()
            or np.isnan(safe_vectors_reshaped).any()
        ):
            print(
                f"Layer {layer_idx}: NaN values detected - skipping for visualization"
            )
            continue

        if (
            np.isinf(hate_vectors_reshaped).any()
            or np.isinf(safe_vectors_reshaped).any()
        ):
            print(
                f"Layer {layer_idx}: Infinite values detected - skipping for visualization"
            )
            continue

        # Calculate mean vectors
        hate_mean_vector = np.mean(hate_vectors_reshaped, axis=0)
        safe_mean_vector = np.mean(safe_vectors_reshaped, axis=0)

        # Validate calculated means
        if np.isnan(hate_mean_vector).any() or np.isnan(safe_mean_vector).any():
            print(
                f"Layer {layer_idx}: NaN values in mean vectors - skipping for visualization"
            )
            continue

        if np.isinf(hate_mean_vector).any() or np.isinf(safe_mean_vector).any():
            print(
                f"Layer {layer_idx}: Infinite values in mean vectors - skipping for visualization"
            )
            continue

        # Debug information
        print(
            f"Layer {layer_idx} - Hate mean shape: {hate_mean_vector.shape}, Safe mean shape: {safe_mean_vector.shape}"
        )
        print(f"Layer {layer_idx} - Steering vector shape: {steering_vector.shape}")

        # Store valid data
        layer_data = {
            "ccs": ccs,
            "layer_idx": layer_idx,
            "hate_vectors": hate_vectors,
            "safe_vectors": safe_vectors,
            "steering_vector": steering_vector,
            "hate_mean_vector": hate_mean_vector,
            "safe_mean_vector": safe_mean_vector,
        }
        all_layer_data.append(layer_data)

        ############### METRICS CALCULATION ###############
        # Initialize layer result dictionary
        layer_result: dict[str, any] = {"layer_idx": layer_idx}

        # Include training metrics from the training result
        if (
            training_result
            and isinstance(training_result, dict)
            and "final_metrics" in training_result
        ):
            layer_result["final_metrics"] = training_result["final_metrics"]
        else:
            layer_result["final_metrics"] = {"base_metrics": {}}

        # Calculate class separability
        class_sep = float(compute_class_separability(hate_vectors, safe_vectors))
        layer_result["class_separability"] = class_sep
        print(f"Layer {layer_idx} - Class separability: {class_sep:.4f}")

        # Calculate subspace analysis
        subspace_results = subspace_analysis(hate_vectors, safe_vectors)
        layer_result["subspace_analysis"] = {
            "top_component_separation": float(
                subspace_results["separations"][0]["accuracy"]
            ),
            "variance_explained": float(
                subspace_results["separations"][0]["variance_explained"]
            ),
        }
        print(
            f"Layer {layer_idx} - Subspace separation: {layer_result['subspace_analysis']['top_component_separation']:.4f}"
        )

        ############### LAYER-SPECIFIC PLOTTING ###############
        # Visualize the decision boundary for the first layer
        visualize_decision_boundary(
            ccs=ccs,
            hate_vectors=hate_vectors,
            safe_vectors=safe_vectors,
            steering_vector=steering_vector,
            log_base=os.path.join(plot_dir, f"layer_{layer_idx}"),
            layer_idx=layer_idx,
            strategy="last-token",  # Default strategy used for extraction
            pair_type="combined",  # Default pair type
        )
        print(f"Generated decision boundary visualization for layer {layer_idx}")

        # We'll create detailed decision boundary visualization after layer_strategy_data is populated

        # Extract different types of representations for all embedding strategies
        all_strategies = ["last-token", "first-token", "mean"]
        layer_strategy_data = {}

        for strategy in all_strategies:
            # Get representations for each data type
            hate_yes_vecs = []
            hate_no_vecs = []
            safe_yes_vecs = []
            safe_no_vecs = []

            print(f"Extracting {strategy} embeddings for layer {layer_idx}...")

            # Extract hate_yes representations
            for text in hate_yes_texts:
                rep = extract_representation(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    keep_on_gpu=False,
                )
                # Validate representation
                if not isinstance(rep, np.ndarray):
                    print(
                        f"Error: Invalid {strategy} representation for hate_yes. Type: {type(rep)}"
                    )
                    continue
                hate_yes_vecs.append(rep)

            # Extract hate_no representations
            for text in hate_no_texts:
                rep = extract_representation(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    keep_on_gpu=False,
                )
                # Validate representation
                if not isinstance(rep, np.ndarray):
                    print(
                        f"Error: Invalid {strategy} representation for hate_no. Type: {type(rep)}"
                    )
                    continue
                hate_no_vecs.append(rep)

            # Extract safe_yes representations
            for text in safe_yes_texts:
                rep = extract_representation(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    keep_on_gpu=False,
                )
                # Validate representation
                if not isinstance(rep, np.ndarray):
                    print(
                        f"Error: Invalid {strategy} representation for safe_yes. Type: {type(rep)}"
                    )
                    continue
                safe_yes_vecs.append(rep)

            # Extract safe_no representations
            for text in safe_no_texts:
                rep = extract_representation(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    keep_on_gpu=False,
                )
                # Validate representation
                if not isinstance(rep, np.ndarray):
                    print(
                        f"Error: Invalid {strategy} representation for safe_no. Type: {type(rep)}"
                    )
                    continue
                safe_no_vecs.append(rep)

            # Check if we have sufficient data
            if (
                len(hate_yes_vecs) == 0
                or len(hate_no_vecs) == 0
                or len(safe_yes_vecs) == 0
                or len(safe_no_vecs) == 0
            ):
                print(f"Error: Insufficient data for strategy {strategy}. Skipping.")
                continue

            # Convert to numpy arrays
            hate_yes_vecs = np.array(hate_yes_vecs)
            hate_no_vecs = np.array(hate_no_vecs)
            safe_yes_vecs = np.array(safe_yes_vecs)
            safe_no_vecs = np.array(safe_no_vecs)

            # Check data shapes
            if (
                hate_yes_vecs.ndim < 2
                or hate_no_vecs.ndim < 2
                or safe_yes_vecs.ndim < 2
                or safe_no_vecs.ndim < 2
            ):
                print(f"Error: Invalid dimensions in vectors for strategy {strategy}")
                print(
                    f"Shapes: hate_yes={hate_yes_vecs.shape}, hate_no={hate_no_vecs.shape}, "
                    f"safe_yes={safe_yes_vecs.shape}, safe_no={safe_no_vecs.shape}"
                )
                continue

            # Store in layer strategy data
            layer_strategy_data[strategy] = {
                "hate_yes": hate_yes_vecs,
                "hate_no": hate_no_vecs,
                "safe_yes": safe_yes_vecs,
                "safe_no": safe_no_vecs,
                "hate": np.vstack([hate_yes_vecs, safe_no_vecs]),
                "safe": np.vstack([safe_yes_vecs, hate_no_vecs]),
            }

            # Calculate steering vectors for each strategy
            hate_vecs = np.vstack([hate_yes_vecs, safe_no_vecs])
            safe_vecs = np.vstack([safe_yes_vecs, hate_no_vecs])

            # Combined steering vector
            combined_steering = compute_steering_vector(hate_vecs, safe_vecs)

            # Validate steering vectors
            if not isinstance(combined_steering, np.ndarray):
                print(
                    f"Error: Invalid combined steering vector for strategy {strategy}"
                )
                continue

            # Specific steering vectors
            hate_yes_to_safe_yes = compute_steering_vector(hate_yes_vecs, safe_yes_vecs)
            if not isinstance(hate_yes_to_safe_yes, np.ndarray):
                print(
                    f"Error: Invalid hate_yes_to_safe_yes steering vector for strategy {strategy}"
                )
                continue

            safe_no_to_hate_no = compute_steering_vector(safe_no_vecs, hate_no_vecs)
            if not isinstance(safe_no_to_hate_no, np.ndarray):
                print(
                    f"Error: Invalid safe_no_to_hate_no steering vector for strategy {strategy}"
                )
                continue

            hate_yes_to_hate_no = compute_steering_vector(hate_yes_vecs, hate_no_vecs)
            if not isinstance(hate_yes_to_hate_no, np.ndarray):
                print(
                    f"Error: Invalid hate_yes_to_hate_no steering vector for strategy {strategy}"
                )
                continue

            safe_no_to_safe_yes = compute_steering_vector(safe_no_vecs, safe_yes_vecs)
            if not isinstance(safe_no_to_safe_yes, np.ndarray):
                print(
                    f"Error: Invalid safe_no_to_safe_yes steering vector for strategy {strategy}"
                )
                continue

            # Store steering vectors
            if strategy not in all_steering_vectors:
                all_steering_vectors[strategy] = {}

            all_steering_vectors[strategy]["combined"] = {
                "vector": combined_steering,
                "color": "#00FF00",  # Green
                "label": "Combined Steering Vector",
            }

            all_steering_vectors[strategy]["hate_yes_to_safe_yes"] = {
                "vector": hate_yes_to_safe_yes,
                "color": "#FF00FF",  # Purple
                "label": "Hate Yes → Safe Yes",
            }

            all_steering_vectors[strategy]["safe_no_to_hate_no"] = {
                "vector": safe_no_to_hate_no,
                "color": "#FFFF00",  # Yellow
                "label": "Safe No → Hate No",
            }

            all_steering_vectors[strategy]["hate_yes_to_hate_no"] = {
                "vector": hate_yes_to_hate_no,
                "color": "#FF9900",  # Orange
                "label": "Hate Yes → Hate No",
            }

            all_steering_vectors[strategy]["safe_no_to_safe_yes"] = {
                "vector": safe_no_to_safe_yes,
                "color": "#00FFCC",  # Teal
                "label": "Safe No → Safe Yes",
            }

        # Store strategy data for this layer
        if len(layer_strategy_data) == 0:
            print(
                f"Warning: No valid strategy data for layer {layer_idx}. Skipping visualizations."
            )
            continue

        all_strategy_data[layer_idx] = layer_strategy_data

        # Create detailed decision boundary visualization with separated data types
        if layer_idx == 0 or layer_idx == n_layers - 1:  # For first and last layer
            strategy = "last-token"

            # Validate strategy data exists
            if strategy not in layer_strategy_data:
                print(
                    f"Warning: 'last-token' strategy missing for layer {layer_idx}. Skipping detailed visualization."
                )
            else:
                hate_yes_vectors = layer_strategy_data[strategy]["hate_yes"]
                hate_no_vectors = layer_strategy_data[strategy]["hate_no"]
                safe_yes_vectors = layer_strategy_data[strategy]["safe_yes"]
                safe_no_vectors = layer_strategy_data[strategy]["safe_no"]

                # Validate vectors
                if (
                    isinstance(hate_yes_vectors, np.ndarray)
                    and isinstance(hate_no_vectors, np.ndarray)
                    and isinstance(safe_yes_vectors, np.ndarray)
                    and isinstance(safe_no_vectors, np.ndarray)
                    and hate_yes_vectors.size > 0
                    and hate_no_vectors.size > 0
                    and safe_yes_vectors.size > 0
                    and safe_no_vectors.size > 0
                ):
                    visualize_detailed_decision_boundary(
                        ccs=ccs,
                        hate_yes_vectors=hate_yes_vectors,
                        hate_no_vectors=hate_no_vectors,
                        safe_yes_vectors=safe_yes_vectors,
                        safe_no_vectors=safe_no_vectors,
                        steering_vector=steering_vector,
                        layer_idx=layer_idx,
                        log_base=os.path.join(plot_dir, f"layer_{layer_idx}_detailed"),
                        strategy=strategy,
                        pair_type="combined",
                    )
                    print(
                        f"Generated detailed decision boundary visualization for layer {layer_idx}"
                    )
                else:
                    print(
                        f"Warning: Invalid vector data for layer {layer_idx} detailed visualization"
                    )

        # Generate visualizations for layer-specific strategies
        # Plot vectors for all strategies
        if "last-token" in all_steering_vectors:
            plot_vectors_all_strategies(
                layer_idx=layer_idx,
                all_strategy_data=layer_strategy_data,
                current_strategy="last-token",
                save_path=os.path.join(
                    plot_dir, f"layer_{layer_idx}_all_strategies_vectors.png"
                ),
                all_steering_vectors=all_steering_vectors["last-token"],
            )
        else:
            print(
                f"Warning: 'last-token' strategy missing in all_steering_vectors for layer {layer_idx}"
            )

        # Plot individual steering vectors for each strategy
        for strategy in all_strategies:
            # Validate strategy exists in both data structures
            if strategy not in layer_strategy_data:
                print(
                    f"Warning: Strategy {strategy} missing in layer_strategy_data for layer {layer_idx}"
                )
                continue

            if strategy not in all_steering_vectors:
                print(
                    f"Warning: Strategy {strategy} missing in all_steering_vectors for layer {layer_idx}"
                )
                continue

            # Validate required data exists
            strategy_data = layer_strategy_data[strategy]
            if not all(
                key in strategy_data
                for key in ["hate_yes", "hate_no", "safe_yes", "safe_no"]
            ):
                print(
                    f"Warning: Missing required keys in strategy_data for {strategy}, layer {layer_idx}"
                )
                continue

            # Validate all required vectors
            if (
                not isinstance(strategy_data["hate_yes"], np.ndarray)
                or not isinstance(strategy_data["hate_no"], np.ndarray)
                or not isinstance(strategy_data["safe_yes"], np.ndarray)
                or not isinstance(strategy_data["safe_no"], np.ndarray)
            ):
                print(
                    f"Warning: Invalid vector types in strategy_data for {strategy}, layer {layer_idx}"
                )
                continue

            plot_individual_steering_vectors(
                plot_dir=plot_dir,
                layer_idx=layer_idx,
                all_steering_vectors=all_steering_vectors[strategy],
                hate_yes_vectors=strategy_data["hate_yes"],
                safe_yes_vectors=strategy_data["safe_yes"],
                hate_no_vectors=strategy_data["hate_no"],
                safe_no_vectors=strategy_data["safe_no"],
                strategy=strategy,
            )

        # Plot all strategies and steering vectors in a comprehensive visualization
        if all_steering_vectors:
            plot_all_strategies_all_steering_vectors(
                plot_dir=plot_dir,
                layer_idx=layer_idx,
                representations=layer_strategy_data,
                all_steering_vectors_by_strategy=all_steering_vectors,
            )
        else:
            print(f"Warning: No steering vectors for layer {layer_idx}")

        ############### ADDITIONAL METRICS CALCULATION ###############
        print(f"\nCalculating additional metrics for layer {layer_idx}...")

        # Calculate agreement score
        # Note: We'd need predictions for pairs of statements to compute this properly
        # This is a simplified example
        hate_test_indices = np.random.choice(
            len(hate_vectors), min(10, len(hate_vectors)), replace=False
        )
        safe_test_indices = np.random.choice(
            len(safe_vectors), min(10, len(safe_vectors)), replace=False
        )

        hate_test = hate_vectors[hate_test_indices]
        safe_test = safe_vectors[safe_test_indices]

        # Get predictions
        with torch.no_grad():
            hate_preds = ccs.probe(torch.tensor(hate_test, device=device)).cpu().numpy()
            safe_preds = ccs.probe(torch.tensor(safe_test, device=device)).cpu().numpy()

            # Calculate metrics
            agreement = float(
                np.mean(agreement_score(hate_preds, safe_preds, safe_preds, hate_preds))
            )
            contradiction = float(
                np.mean(
                    contradiction_index(hate_preds, safe_preds, safe_preds, hate_preds)
                )
            )
            ideal_distance = float(
                np.mean(
                    ideal_representation_distance(
                        hate_preds, safe_preds, safe_preds, hate_preds
                    )
                )
            )

            # Representation stability
            stability = representation_stability(
                ccs, hate_vectors, perturbation_scale=0.01, n_perturbations=5
            )

            # Fisher information analysis
            fisher_info = fisher_information_analysis(
                ccs, hate_vectors, steering_vector
            )

            # Store additional metrics
            layer_result["agreement_score"] = agreement
            layer_result["contradiction_index"] = contradiction
            layer_result["ideal_distance"] = ideal_distance
            layer_result["representation_stability"] = stability
            layer_result["max_sensitivity_point"] = fisher_info["max_sensitivity_point"]
            layer_result["max_sensitivity_value"] = fisher_info["max_sensitivity_value"]

            print(f"Layer {layer_idx} - Agreement score: {agreement:.4f}")
            print(f"Layer {layer_idx} - Contradiction index: {contradiction:.4f}")
            print(f"Layer {layer_idx} - Ideal distance: {ideal_distance:.4f}")
            print(f"Layer {layer_idx} - Representation stability: {stability:.4f}")
            print(
                f"Layer {layer_idx} - Max sensitivity at point: {fisher_info['max_sensitivity_point']:.4f}"
            )

        # Calculate metrics for different steering coefficient values
        print(
            f"\nCalculating metrics for different steering coefficients for layer {layer_idx}..."
        )
        for coef in steering_coefficients:
            coef_key = f"coef_{coef}"

            # Special case for coefficient 0.0 - use baseline metrics
            if coef == 0.0:
                # Copy baseline metrics to coefficient data structure
                layer_result[coef_key] = {
                    "agreement_score": agreement,
                    "contradiction_index": contradiction,
                    "ideal_distance": ideal_distance,
                    "representation_stability": stability,
                }

                # Add metrics from final_metrics too
                if "final_metrics" in layer_result:
                    final_metrics = layer_result["final_metrics"]
                    if "base_metrics" in final_metrics:
                        base_metrics = final_metrics["base_metrics"]
                        for metric_name, metric_value in base_metrics.items():
                            layer_result[coef_key][metric_name] = metric_value
                    else:
                        # Copy any metrics from final_metrics directly
                        for metric_name, metric_value in final_metrics.items():
                            if metric_name != "base_metrics":
                                layer_result[coef_key][metric_name] = metric_value

                print("  Coef=0.0 (baseline) - Using existing metrics")
                continue

            # Apply steering with this coefficient
            steered_hate_vectors = apply_steering_vector(
                representation=hate_vectors,
                steering_vector=steering_vector,
                coefficient=coef,
            )

            # Generate decision boundary plot for each coefficient (for key layers)
            if (
                layer_idx == 0
                or layer_idx == n_layers - 1
                or layer_idx == n_layers // 2
            ):
                visualize_decision_boundary(
                    ccs=ccs,
                    hate_vectors=steered_hate_vectors,  # Use steered vectors
                    safe_vectors=safe_vectors,
                    steering_vector=steering_vector,
                    log_base=os.path.join(plot_dir, f"layer_{layer_idx}_coef_{coef}"),
                    layer_idx=layer_idx,
                    strategy="last-token",
                    steering_coefficient=coef,
                    pair_type="combined",
                )
                print(
                    f"  Generated decision boundary for coefficient {coef}, layer {layer_idx}"
                )

            # Convert to tensor for predictions
            steered_hate_tensor = torch.tensor(steered_hate_vectors, device=device)
            safe_tensor = torch.tensor(safe_vectors, device=device)

            # Get predictions
            with torch.no_grad():
                steered_hate_preds = ccs.probe(steered_hate_tensor).cpu().numpy()
                safe_preds = ccs.probe(safe_tensor).cpu().numpy()

                # Calculate metrics with steered representations
                steered_agreement = float(
                    np.mean(
                        agreement_score(
                            steered_hate_preds,
                            safe_preds,
                            safe_preds,
                            steered_hate_preds,
                        )
                    )
                )
                steered_contradiction = float(
                    np.mean(
                        contradiction_index(
                            steered_hate_preds,
                            safe_preds,
                            safe_preds,
                            steered_hate_preds,
                        )
                    )
                )
                steered_ideal_distance = float(
                    np.mean(
                        ideal_representation_distance(
                            steered_hate_preds,
                            safe_preds,
                            safe_preds,
                            steered_hate_preds,
                        )
                    )
                )

                # Calculate stability on steered vectors
                steered_stability = representation_stability(
                    ccs,
                    steered_hate_vectors,
                    perturbation_scale=0.01,
                    n_perturbations=5,
                )

                # Calculate accuracy and other standard metrics
                # Create merged dataset with steered hate vectors and original safe vectors
                steered_X = np.vstack([steered_hate_vectors, safe_vectors])
                steered_y = np.concatenate(
                    [np.zeros(len(steered_hate_vectors)), np.ones(len(safe_vectors))]
                )

                # Get predictions on steered data
                steered_probs = (
                    ccs.probe(torch.tensor(steered_X, device=device)).cpu().numpy()
                )
                steered_preds = (steered_probs > 0.5).astype(int)

                # Calculate accuracy
                steered_accuracy = float(np.mean(steered_preds.flatten() == steered_y))

                # Calculate silhouette score if more than one class is present
                if len(np.unique(steered_preds)) > 1:
                    from sklearn.metrics import silhouette_score

                    steered_silhouette = float(
                        silhouette_score(steered_X, steered_preds)
                    )
                else:
                    steered_silhouette = 0.0

                # Store metrics for this coefficient
                layer_result[coef_key] = {
                    "agreement_score": steered_agreement,
                    "contradiction_index": steered_contradiction,
                    "ideal_distance": steered_ideal_distance,
                    "representation_stability": steered_stability,
                    "accuracy": steered_accuracy,
                    "silhouette": steered_silhouette,
                }

                # Print key metrics
                print(
                    f"  Coef={coef} - Agreement: {steered_agreement:.4f}, Accuracy: {steered_accuracy:.4f}"
                )

        # Store results
        results.append(layer_result)

    ############### CROSS-LAYER PLOTTING ###############
    print("\nGenerating visualizations for all layers...")

    # Validate results and all_layer_data before plotting
    if not results:
        print(
            "Warning: No results collected. Cannot generate cross-layer visualizations."
        )
        return [], [], {}, {}

    if not all_layer_data:
        print(
            "Warning: No layer data collected. Cannot generate layer-based visualizations."
        )
    else:
        # 1. Plot all decision boundaries in a grid
        plot_all_decision_boundaries(
            layers_data=all_layer_data,
            log_base=os.path.join(plot_dir, "all_decision_boundaries"),
        )
        print(
            f"Saved all decision boundaries plot to {os.path.join(plot_dir, 'all_decision_boundaries.png')}"
        )

        # 2. Generate individual decision boundary plots for each layer
        for layer_idx, layer_data in enumerate(all_layer_data):
            # Validate required keys
            required_keys = ["ccs", "hate_vectors", "safe_vectors", "steering_vector"]
            if not all(key in layer_data for key in required_keys):
                print(
                    f"Warning: Missing required keys for decision boundary plot for layer {layer_idx}"
                )
                continue

            # Validate data types
            if not isinstance(layer_data["hate_vectors"], np.ndarray) or not isinstance(
                layer_data["safe_vectors"], np.ndarray
            ):
                print(
                    f"Warning: Invalid vector types for decision boundary plot for layer {layer_idx}"
                )
                continue

            if layer_idx > 0:  # We already did layer 0 during training
                decision_boundary_path = os.path.join(plot_dir, f"layer_{layer_idx}")
                visualize_decision_boundary(
                    ccs=layer_data["ccs"],
                    hate_vectors=layer_data["hate_vectors"],
                    safe_vectors=layer_data["safe_vectors"],
                    steering_vector=layer_data["steering_vector"],
                    log_base=decision_boundary_path,
                    layer_idx=layer_idx,
                    strategy="last-token",
                )
                print(f"Saved decision boundary plot for layer {layer_idx}")

        # 3. Plot vectors across all layers
        # Validate layer data has required keys for vector plotting
        valid_vector_layers = []
        for layer_idx, layer_data in enumerate(all_layer_data):
            required_keys = ["hate_mean_vector", "safe_mean_vector", "steering_vector"]
            if not all(key in layer_data for key in required_keys):
                print(
                    f"Warning: Missing required keys for vector plot for layer {layer_idx}"
                )
                continue

            if (
                not isinstance(layer_data["hate_mean_vector"], np.ndarray)
                or not isinstance(layer_data["safe_mean_vector"], np.ndarray)
                or not isinstance(layer_data["steering_vector"], np.ndarray)
            ):
                print(
                    f"Warning: Invalid vector types for vector plot for layer {layer_idx}"
                )
                continue

            valid_vector_layers.append(layer_data)

        if valid_vector_layers:
            plot_all_layer_vectors(results=valid_vector_layers, save_dir=plot_dir)
            print(
                f"Saved all layer vectors plot to {os.path.join(plot_dir, 'all_layer_vectors.png')}"
            )
        else:
            print("Warning: No valid layer data for vector plotting.")

    # 4. Plot performance metrics across layers
    # Check if we have accuracy metrics in any of the coefficient data structures
    has_accuracy = False

    # First try checking coefficient data
    if results:
        # Look for accuracy in coef_0.0, coef_1.0, etc.
        coef_keys = [k for k in results[0].keys() if k.startswith("coef_")]
        if coef_keys:
            # Check if any layer has accuracy metrics in coefficient data
            has_accuracy = any(
                "accuracy" in results[0][coef_key]
                for coef_key in coef_keys
                if isinstance(results[0][coef_key], dict)
            )

        # If not found in coefficient data, check the traditional location
        if not has_accuracy:
            has_accuracy = any(
                "accuracy"
                in layer_result.get("final_metrics", {}).get("base_metrics", {})
                for layer_result in results
            )

    # Always generate the plot - our fixed plotting function will handle missing data gracefully
    metrics_plot_path = os.path.join(plot_dir, "performance_accuracy.png")
    plot_performance_across_layers(
        results=results, metric="accuracy", save_path=metrics_plot_path
    )
    print(f"Saved performance metrics plot to {metrics_plot_path}")

    # 5. Plot coefficient sweep comparison for multiple metrics
    metrics_to_plot = [
        "accuracy",
        "silhouette",
        "agreement_score",
        "contradiction_index",
        "representation_stability",
        "ideal_distance",
    ]

    # Validate at least some metrics exist
    available_metrics = []
    for metric in metrics_to_plot:
        if any(metric in layer_result for layer_result in results):
            available_metrics.append(metric)

    if available_metrics:
        coef_sweep_path = os.path.join(plot_dir, "coefficient_sweep_comparison.png")
        plot_coefficient_sweep_lines_comparison(
            results=results, metrics=available_metrics, save_path=coef_sweep_path
        )
        print(f"Saved coefficient sweep comparison plot to {coef_sweep_path}")
    else:
        print("Warning: No metrics available for coefficient sweep plot.")

    ############### RESULTS SUMMARY ###############
    # Generate comprehensive results summary
    print("\nGenerating comprehensive results summary...")
    summary_result = print_results_summary(
        results=results,
        steering_coefficients=steering_coefficients,
        model_name=model.__class__.__name__,
        model_family=model.config.model_type,
        model_variant=model.config.name_or_path.split("/")[-1],
        run_dir=run_dir,
        layer_data=all_layer_data,
        all_strategy_data=all_strategy_data,
        all_steering_vectors=all_steering_vectors,
    )

    print(f"All visualizations have been saved to {plot_dir}")

    return results, all_layer_data, all_strategy_data, all_steering_vectors
