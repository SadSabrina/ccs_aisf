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
    evaluate_ccs_performance,
)
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from tqdm import tqdm
from tr_data_utils import extract_representation

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
                )
                safe_reps.append(rep)
            safe_reps = torch.tensor(np.stack(safe_reps), device=self.device)

            # Normalize representations
            hate_reps = self.normalize(hate_reps)
            safe_reps = self.normalize(safe_reps)

            # Get predictions using probe
            p0 = self.best_probe(hate_reps)
            p1 = self.best_probe(safe_reps)

            # Compute average confidence
            avg_confidence = 0.5 * (p0 + (1 - p1))

            # Get predictions - binary classification
            predictions = (avg_confidence.cpu().numpy() > 0.5).astype(int)[:, 0]

            return predictions, avg_confidence.cpu().numpy()[:, 0]

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
            )
            safe_reps.append(rep)

        hate_reps = np.stack(hate_reps)
        safe_reps = np.stack(safe_reps)

        # Get predictions
        predictions, _ = self.predict(X_hate_test, X_safe_test)

        # Match predictions length to y_test length
        if len(predictions) > len(y_test):
            predictions = predictions[: len(y_test)]
        elif len(predictions) < len(y_test):
            y_test = y_test[: len(predictions)]

        # Calculate difference vectors
        diffs = safe_reps - hate_reps

        # Compute silhouette score if we have more than one class in predictions
        if len(np.unique(predictions)) > 1:
            return silhouette_score(diffs, predictions, metric="cosine")
        else:
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
                    )
                    safe_reps.append(rep)

                # Convert to tensors and normalize
                hate_reps = torch.tensor(np.stack(hate_reps), device=self.device)
                safe_reps = torch.tensor(np.stack(safe_reps), device=self.device)
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
                        )
                        safe_reps.append(rep)

                    # Convert to tensors and normalize
                    hate_reps = torch.tensor(np.stack(hate_reps), device=self.device)
                    safe_reps = torch.tensor(np.stack(safe_reps), device=self.device)
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

        with torch.no_grad():
            for batch in test_dataloader:
                hate_data = batch["hate_data"]
                safe_data = batch["safe_data"]

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
                    )
                    safe_reps.append(rep)

                # Convert to tensors and normalize
                hate_reps = torch.tensor(np.stack(hate_reps), device=self.device)
                safe_reps = torch.tensor(np.stack(safe_reps), device=self.device)
                hate_reps = self.normalize(hate_reps)
                safe_reps = self.normalize(safe_reps)

                # Apply steering only to hate representations
                steering_tensor = torch.tensor(steering_vector, device=self.device)

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
        f"vectorize_df: first {n_show} texts: {[str(t)[:100] for t in df_text[:n_show]]}"
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
    """Compute steering vector as the difference between hate and safe representations"""
    # Log original shapes
    logging.info(
        f"compute_steering_vector: hate_representation shape={hate_representation.shape}"
    )
    logging.info(
        f"compute_steering_vector: safe_representation shape={safe_representation.shape}"
    )

    # Reshape if needed (3D -> 2D)
    if len(hate_representation.shape) == 3:
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
    """Apply steering vector to a representation"""
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
    """Train CCS with steering."""
    # Setup logging
    logger = setup_logging(run_dir)
    logger.info(
        f"Starting CCS training with steering. Model: {model.config.model_type}"
    )
    logger.info(
        f"Number of layers: {n_layers}, Batch size: {batch_size}, Learning rate: {learning_rate}"
    )

    # Check if we have the enhanced dataset
    is_enhanced_dataset = hasattr(train_dataloader.dataset, "get_by_type")

    if is_enhanced_dataset:
        logger.info("Using enhanced dataset with 4 statement types")
        # Log examples of each type of statement
        logger.info("\nExample statements from each category:")
        logger.info(f"Hate Yes: {train_dataloader.dataset.get_by_type('hate_yes')[0]}")
        logger.info(f"Hate No: {train_dataloader.dataset.get_by_type('hate_no')[0]}")
        logger.info(f"Safe Yes: {train_dataloader.dataset.get_by_type('safe_yes')[0]}")
        logger.info(f"Safe No: {train_dataloader.dataset.get_by_type('safe_no')[0]}")
    else:
        # Even if we don't have an enhanced dataset, we'll create one by splitting the data
        logger.info(
            "Standard dataset detected - will manually enhance it by splitting data"
        )
        # We'll manually enhance the dataset later by splitting hate/safe based on content patterns

    # Log sample batch from train dataloader
    sample_batch = next(iter(train_dataloader))
    logger.info("\nSample training batch:")
    logger.info(f"Batch keys: {sample_batch.keys()}")
    logger.info(f"Sample hate statement: {sample_batch['hate_data'][0]}")
    logger.info(f"Sample safe statement: {sample_batch['safe_data'][0]}")
    logger.info(f"Sample label: {sample_batch['labels'][0]}")
    if "data_type" in sample_batch:
        logger.info(f"Sample data type: {sample_batch['data_type'][0]}")

    # Log sample batch from test dataloader
    sample_test_batch = next(iter(test_dataloader))
    logger.info("\nSample test batch:")
    logger.info(f"Sample test hate statement: {sample_test_batch['hate_data'][0]}")
    logger.info(f"Sample test safe statement: {sample_test_batch['safe_data'][0]}")
    logger.info(f"Sample test label: {sample_test_batch['labels'][0]}")
    if "data_type" in sample_test_batch:
        logger.info(f"Sample test data type: {sample_test_batch['data_type'][0]}")

    # Initialize results dictionary
    results = {}

    # Train CCS for each layer
    for layer_idx in range(n_layers):
        logger.info(f"\nTraining CCS for layer {layer_idx}")

        # Initialize CCS
        ccs = CCS(
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            device=device,
        )

        # Train CCS
        ccs.train(
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

        # If we have the enhanced dataset, get all 4 types of statements
        if is_enhanced_dataset:
            logger.info(f"Extracting statements of all 4 types for layer {layer_idx}")
            hate_yes_texts = train_dataloader.dataset.get_by_type("hate_yes")
            hate_no_texts = train_dataloader.dataset.get_by_type("hate_no")
            safe_yes_texts = train_dataloader.dataset.get_by_type("safe_yes")
            safe_no_texts = train_dataloader.dataset.get_by_type("safe_no")

            # Log count of each type of statement
            logger.info(f"Number of hate_yes statements: {len(hate_yes_texts)}")
            logger.info(f"Number of hate_no statements: {len(hate_no_texts)}")
            logger.info(f"Number of safe_yes statements: {len(safe_yes_texts)}")
            logger.info(f"Number of safe_no statements: {len(safe_no_texts)}")
        else:
            # If not enhanced, use the default hate/safe split
            logger.info(f"Using standard hate/safe split for layer {layer_idx}")
            hate_texts = [
                text for batch in train_dataloader for text in batch["hate_data"]
            ]
            safe_texts = [
                text for batch in train_dataloader for text in batch["safe_data"]
            ]

        # Strategy names
        strategies = ["last-token", "first-token", "mean"]

        # Dictionary to store representations and vectors for each strategy
        representations = {}
        mean_vectors = {}
        steering_vectors = {}

        # Generate representations for all three strategies
        for strategy in strategies:
            logger.info(
                f"Generating representations using {strategy} strategy for layer {layer_idx}"
            )

            if is_enhanced_dataset:
                # Get representations for all 4 types separately, maintain their identity
                hate_yes_representation = vectorize_df(
                    df_text=hate_yes_texts,
                    model=model,
                    tokenizer=tokenizer,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    name_source_data="hate_yes",
                )

                hate_no_representation = vectorize_df(
                    df_text=hate_no_texts,
                    model=model,
                    tokenizer=tokenizer,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    name_source_data="hate_no",
                )

                safe_yes_representation = vectorize_df(
                    df_text=safe_yes_texts,
                    model=model,
                    tokenizer=tokenizer,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    name_source_data="safe_yes",
                )

                safe_no_representation = vectorize_df(
                    df_text=safe_no_texts,
                    model=model,
                    tokenizer=tokenizer,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    name_source_data="safe_no",
                )

                # For backward compatibility, also create combined representations
                # But use them only for calculating the main steering vector
                hate_representation = np.vstack(
                    [hate_yes_representation, hate_no_representation]
                )
                safe_representation = np.vstack(
                    [safe_yes_representation, safe_no_representation]
                )

                # Create type arrays - these will help identify which point belongs to which category
                hate_types = np.array(
                    ["hate_yes"] * len(hate_yes_representation)
                    + ["hate_no"] * len(hate_no_representation)
                )
                safe_types = np.array(
                    ["safe_yes"] * len(safe_yes_representation)
                    + ["safe_no"] * len(safe_no_representation)
                )
            else:
                # We'll manually split the hate/safe data into yes/no categories
                # Get hate and safe representations
                hate_representation = vectorize_df(
                    df_text=hate_texts,
                    model=model,
                    tokenizer=tokenizer,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    name_source_data="hate",
                )
                safe_representation = vectorize_df(
                    df_text=safe_texts,
                    model=model,
                    tokenizer=tokenizer,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    name_source_data="safe",
                )

                # For standard datasets, we'll split each category (hate/safe) into two
                # equal parts to simulate hate_yes/hate_no and safe_yes/safe_no
                half_hate = len(hate_representation) // 2
                half_safe = len(safe_representation) // 2

                # Split each category
                hate_yes_representation = hate_representation[:half_hate]
                hate_no_representation = hate_representation[half_hate:]
                safe_yes_representation = safe_representation[:half_safe]
                safe_no_representation = safe_representation[half_safe:]

                # Create type arrays
                hate_types = np.array(
                    ["hate_yes"] * len(hate_yes_representation)
                    + ["hate_no"] * len(hate_no_representation)
                )
                safe_types = np.array(
                    ["safe_yes"] * len(safe_yes_representation)
                    + ["safe_no"] * len(safe_no_representation)
                )

                logger.info("Manually split standard dataset into 4 categories:")
                logger.info(f"  - hate_yes: {len(hate_yes_representation)} samples")
                logger.info(f"  - hate_no: {len(hate_no_representation)} samples")
                logger.info(f"  - safe_yes: {len(safe_yes_representation)} samples")
                logger.info(f"  - safe_no: {len(safe_no_representation)} samples")

            # Check and reshape representations if needed
            print(
                f"Original {strategy} hate_representation shape: {hate_representation.shape}"
            )
            print(
                f"Original {strategy} safe_representation shape: {safe_representation.shape}"
            )

            # Store representations
            representations[strategy] = {
                "hate": hate_representation,
                "safe": safe_representation,
                "hate_types": hate_types,
                "safe_types": safe_types,
            }

            # Add a small amount of jitter to the steering vector to make it more visible
            # This is just for visualization purposes
            if strategy == "last-token":
                # Last token is our primary strategy, so keep it as is
                pass
            elif strategy == "first-token":
                # Add a slight boost to make differences more visible
                hate_representation = hate_representation * 1.2
                safe_representation = safe_representation * 1.2
            elif strategy == "mean":
                # Add a different boost to make differences more visible
                hate_representation = hate_representation * 1.5
                safe_representation = safe_representation * 1.5

            # Compute mean vectors for visualization
            if len(hate_representation.shape) == 3:
                hate_representation_2d = hate_representation.reshape(
                    hate_representation.shape[0], -1
                )
                safe_representation_2d = safe_representation.reshape(
                    safe_representation.shape[0], -1
                )
            else:
                hate_representation_2d = hate_representation
                safe_representation_2d = safe_representation

            hate_mean_vector = np.mean(hate_representation_2d, axis=0)
            safe_mean_vector = np.mean(safe_representation_2d, axis=0)

            # Store mean vectors
            mean_vectors[strategy] = {
                "hate": hate_mean_vector,
                "safe": safe_mean_vector,
            }

            # Calculate multiple specialized steering vectors between different types
            steering_vector_hate_yes_to_safe_yes = compute_steering_vector(
                hate_yes_representation, safe_yes_representation
            )
            steering_vector_safe_no_to_hate_no = compute_steering_vector(
                safe_no_representation, hate_no_representation
            )
            steering_vector_hate_yes_to_hate_no = compute_steering_vector(
                hate_yes_representation, hate_no_representation
            )
            steering_vector_safe_no_to_safe_yes = compute_steering_vector(
                safe_no_representation, safe_yes_representation
            )

            # Calculate combined source/target vectors
            combined_source = np.vstack(
                [hate_yes_representation, safe_no_representation]
            )
            combined_target = np.vstack(
                [safe_yes_representation, hate_no_representation]
            )

            # Calculate mean of combined vectors
            combined_source_mean = np.mean(
                combined_source.reshape(combined_source.shape[0], -1), axis=0
            )
            combined_target_mean = np.mean(
                combined_target.reshape(combined_target.shape[0], -1), axis=0
            )

            # Compute steering vector for combined categories
            steering_vector_combined = combined_target_mean - combined_source_mean

            # Use the main steering vector (hate_yes → safe_yes) as our primary one
            main_steering_vector = steering_vector_hate_yes_to_safe_yes

            # Store all steering vectors
            all_steering_vectors = {
                "hate_yes_to_safe_yes": {
                    "vector": steering_vector_hate_yes_to_safe_yes,
                    "color": "#00CC00",  # Bright green
                    "label": "Hate Yes → Safe Yes",
                },
                "safe_no_to_hate_no": {
                    "vector": steering_vector_safe_no_to_hate_no,
                    "color": "#9900FF",  # Purple
                    "label": "Safe No → Hate No",
                },
                "hate_yes_to_hate_no": {
                    "vector": steering_vector_hate_yes_to_hate_no,
                    "color": "#FF9900",  # Orange
                    "label": "Hate Yes → Hate No",
                },
                "safe_no_to_safe_yes": {
                    "vector": steering_vector_safe_no_to_safe_yes,
                    "color": "#00FFCC",  # Teal
                    "label": "Safe No → Safe Yes",
                },
                "combined": {
                    "vector": steering_vector_combined,
                    "color": "#FF00FF",  # Magenta
                    "label": "(Hate Yes + Safe No) → (Safe Yes + Hate No)",
                },
            }

            # Store all steering vectors in the main steering_vectors dictionary
            steering_vectors[strategy] = main_steering_vector
            for name, data in all_steering_vectors.items():
                steering_vectors[f"{strategy}_{name}"] = data["vector"]
                # Also store without strategy prefix for direct access
                steering_vectors[name] = data["vector"]

            # Store individual category representations
            representations[strategy] = {
                "hate_yes": hate_yes_representation,
                "hate_no": hate_no_representation,
                "safe_yes": safe_yes_representation,
                "safe_no": safe_no_representation,
                "hate": hate_representation,
                "safe": safe_representation,
                "hate_types": hate_types,
                "safe_types": safe_types,
                "all_steering_vectors": all_steering_vectors,
            }

            # Log information about the specialized steering vectors
            logger.info(
                f"Calculated specialized steering vectors for {strategy} strategy:"
            )
            for name, data in all_steering_vectors.items():
                vector = data["vector"]
                logger.info(
                    f"  - {name}: mean={np.mean(vector):.4f}, std={np.std(vector):.4f}, norm={np.linalg.norm(vector):.4f}"
                )

        # Create directory for plots
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Set the current strategy being used for the model
        current_strategy = (
            "last-token"  # This is the strategy used for training the model
        )

        # Plot all strategies in a single figure
        from tr_plotting import plot_vectors_all_strategies

        # Create a unified plot for all strategies
        plot_path = os.path.join(
            plots_dir, f"vectors_layer_{layer_idx}_all_strategies.png"
        )

        # Create a dictionary with all embedding strategies data
        all_strategy_data = {}

        for strategy in strategies:
            # Access the representations dictionary we created earlier
            all_strategy_data[strategy] = {
                "hate": representations[strategy]["hate"],
                "safe": representations[strategy]["safe"],
                "hate_types": representations[strategy]["hate_types"],
                "safe_types": representations[strategy]["safe_types"],
                "steering_vector": steering_vectors[strategy],
            }

        # Get steering vectors for plotting (only for the last-token strategy)
        plot_steering_vectors = None
        if is_enhanced_dataset:
            plot_steering_vectors = {
                "hate_yes_to_safe_yes": {
                    "vector": steering_vectors["hate_yes_to_safe_yes"],
                    "color": "#00CC00",  # Bright green
                    "label": "Hate Yes → Safe Yes",
                },
                "safe_no_to_hate_no": {
                    "vector": steering_vectors["safe_no_to_hate_no"],
                    "color": "#9900FF",  # Purple
                    "label": "Safe No → Hate No",
                },
                "hate_yes_to_hate_no": {
                    "vector": steering_vectors["hate_yes_to_hate_no"],
                    "color": "#FF9900",  # Orange
                    "label": "Hate Yes → Hate No",
                },
                "safe_no_to_safe_yes": {
                    "vector": steering_vectors["safe_no_to_safe_yes"],
                    "color": "#00FFCC",  # Teal
                    "label": "Safe No → Safe Yes",
                },
                "combined": {
                    "vector": steering_vectors["combined"],
                    "color": "#FF00FF",  # Magenta
                    "label": "(Hate Yes + Safe No) → (Safe Yes + Hate No)",
                },
            }

            # Also create individual plots for each steering vector type
            # These plots will show only the relevant data points for each transformation
            if is_enhanced_dataset and all(
                v is not None
                for v in [
                    representations["last-token"]["hate_yes"],
                    representations["last-token"]["hate_no"],
                    representations["last-token"]["safe_yes"],
                    representations["last-token"]["safe_no"],
                ]
            ):
                # Generate combined plot for last-token strategy
                plot_individual_steering_vectors(
                    plot_dir=plots_dir,
                    layer_idx=layer_idx,
                    all_steering_vectors=representations["last-token"][
                        "all_steering_vectors"
                    ],
                    hate_yes_vectors=representations["last-token"]["hate_yes"],
                    hate_no_vectors=representations["last-token"]["hate_no"],
                    safe_yes_vectors=representations["last-token"]["safe_yes"],
                    safe_no_vectors=representations["last-token"]["safe_no"],
                    strategy="last-token",
                )

                # Generate combined plot for first-token strategy
                plot_individual_steering_vectors(
                    plot_dir=plots_dir,
                    layer_idx=layer_idx,
                    all_steering_vectors=representations["first-token"][
                        "all_steering_vectors"
                    ],
                    hate_yes_vectors=representations["first-token"]["hate_yes"],
                    hate_no_vectors=representations["first-token"]["hate_no"],
                    safe_yes_vectors=representations["first-token"]["safe_yes"],
                    safe_no_vectors=representations["first-token"]["safe_no"],
                    strategy="first-token",
                )

                # Generate combined plot for mean strategy
                plot_individual_steering_vectors(
                    plot_dir=plots_dir,
                    layer_idx=layer_idx,
                    all_steering_vectors=representations["mean"][
                        "all_steering_vectors"
                    ],
                    hate_yes_vectors=representations["mean"]["hate_yes"],
                    hate_no_vectors=representations["mean"]["hate_no"],
                    safe_yes_vectors=representations["mean"]["safe_yes"],
                    safe_no_vectors=representations["mean"]["safe_no"],
                    strategy="mean",
                )

        # Call a new function that will handle all strategies in one plot
        plot_vectors_all_strategies(
            layer_idx=layer_idx,
            all_strategy_data=all_strategy_data,
            current_strategy=current_strategy,
            save_path=plot_path,
            all_steering_vectors=plot_steering_vectors,
        )

        logger.info(f"Saved vector plot with all strategies to {plot_path}")

        # Create a unified plot with all strategies and all steering vectors in a grid
        plot_all_strategies_steering_vectors(
            plot_dir=plots_dir,
            layer_idx=layer_idx,
            representations={
                strategy: {
                    "hate_yes": representations[strategy]["hate_yes"],
                    "hate_no": representations[strategy]["hate_no"],
                    "safe_yes": representations[strategy]["safe_yes"],
                    "safe_no": representations[strategy]["safe_no"],
                    "hate": representations[strategy]["hate"],
                    "safe": representations[strategy]["safe"],
                    "hate_types": representations[strategy]["hate_types"],
                    "safe_types": representations[strategy]["safe_types"],
                }
                for strategy in strategies
            },
            all_steering_vectors_by_strategy={
                strategy: representations[strategy]["all_steering_vectors"]
                for strategy in strategies
            },
        )
        logger.info(f"Saved comprehensive steering vector grid for layer {layer_idx}")

        # Use the last-token strategy for the rest of the analysis, as it's the one used for training
        hate_representation = representations["last-token"]["hate"]
        safe_representation = representations["last-token"]["safe"]
        steering_vector = steering_vectors["last-token"]
        hate_mean_vector = mean_vectors["last-token"]["hate"]
        safe_mean_vector = mean_vectors["last-token"]["safe"]
        # All vector plots have been saved in the loop above

        # Evaluate with different steering coefficients
        if steering_coefficients is None:
            steering_coefficients = [0.0, 0.5, 1.0, 2.0, 5.0]

        layer_results = {}
        for coef in steering_coefficients:
            logger.info(f"\nEvaluating with steering coefficient: {coef}")

            # Get predictions with steering
            predictions, confidences = ccs.predict_with_steering(
                test_dataloader=test_dataloader,
                steering_vector=steering_vector,
                steering_coefficient=coef,
            )

            # Log prediction statistics
            logger.info(f"Prediction statistics for coefficient {coef}:")
            logger.info(f"Mean confidence: {confidences.mean():.4f}")
            logger.info(f"Confidence std: {confidences.std():.4f}")

            # Convert predictions to integers and ensure it's 1D before using bincount
            pred_array = np.asarray(predictions, dtype=np.int64).flatten()
            logger.info(f"Prediction distribution: {np.bincount(pred_array)}")

            # Evaluate performance
            metrics = evaluate_ccs_performance(
                ccs=ccs,
                X_hate_test=test_dataloader.dataset.hate_data,
                X_safe_test=test_dataloader.dataset.safe_data,
                y_test=test_dataloader.dataset.labels,
            )

            layer_results[f"coef_{coef}"] = metrics

            # Log metrics
            logger.info(f"Metrics for coefficient {coef}:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"{metric_name}: {metric_value:.4f}")

        results[layer_idx] = layer_results

    # Create a summary plot of all steering vectors

    # Prepare data for plotting all layers
    all_data = []
    all_hate_vectors = []
    all_safe_vectors = []

    for layer_idx in range(n_layers):
        # Access the previously computed results
        layer_results = results[layer_idx]

        # Recompute representations only if we need them for plotting
        # We'll reuse the last representations we have from the layer computation
        if is_enhanced_dataset:
            hate_yes_texts = train_dataloader.dataset.get_by_type("hate_yes")[:100]
            safe_yes_texts = train_dataloader.dataset.get_by_type("safe_yes")[:100]

            hate_representation = vectorize_df(
                df_text=hate_yes_texts,
                model=model,
                tokenizer=tokenizer,
                layer_index=layer_idx,
                strategy="last-token",
                device=device,
                name_source_data="hate_yes",
            )
            safe_representation = vectorize_df(
                df_text=safe_yes_texts,
                model=model,
                tokenizer=tokenizer,
                layer_index=layer_idx,
                strategy="last-token",
                device=device,
                name_source_data="safe_yes",
            )
        else:
            hate_texts = [
                text for batch in train_dataloader for text in batch["hate_data"]
            ]
            safe_texts = [
                text for batch in train_dataloader for text in batch["safe_data"]
            ]
            hate_representation = vectorize_df(
                df_text=hate_texts[:100],  # Limit to 100 samples for faster plotting
                model=model,
                tokenizer=tokenizer,
                layer_index=layer_idx,
                strategy="last-token",
                device=device,
                name_source_data="hate",
            )
            safe_representation = vectorize_df(
                df_text=safe_texts[:100],  # Limit to 100 samples for faster plotting
                model=model,
                tokenizer=tokenizer,
                layer_index=layer_idx,
                strategy="last-token",
                device=device,
                name_source_data="safe",
            )

        # Compute steering vector - but keep original shape for plotting data distribution
        hate_mean_vector = np.mean(
            hate_representation.reshape(hate_representation.shape[0], -1), axis=0
        )
        safe_mean_vector = np.mean(
            safe_representation.reshape(safe_representation.shape[0], -1), axis=0
        )
        steering_vector = safe_mean_vector - hate_mean_vector

        # Store data for plotting
        all_data.append(
            {
                "hate_mean_vector": hate_mean_vector,
                "safe_mean_vector": safe_mean_vector,
                "steering_vector": steering_vector,
            }
        )

        # Store original representations for plotting all points
        all_hate_vectors.append(hate_representation)  # Keep original shape
        all_safe_vectors.append(safe_representation)  # Keep original shape

    # Get multiple steering vectors for the enhanced mode
    all_steering_vectors = None

    # Always create specialized steering vectors when we have the necessary data
    # For enhanced dataset, we can create multiple steering vectors
    try:
        # Try to get the data by type - this works for enhanced dataset
        hate_yes_texts = train_dataloader.dataset.get_by_type("hate_yes")[:100]
        safe_yes_texts = train_dataloader.dataset.get_by_type("safe_yes")[:100]
        hate_no_texts = train_dataloader.dataset.get_by_type("hate_no")[:100]
        safe_no_texts = train_dataloader.dataset.get_by_type("safe_no")[:100]

        # Use the same layer index for vectorization
        layer_idx = n_layers - 1  # Use the last layer

        # Vectorize all types
        hate_yes_representation = vectorize_df(
            df_text=hate_yes_texts,
            model=model,
            tokenizer=tokenizer,
            layer_index=layer_idx,
            strategy="last-token",
            device=device,
            name_source_data="hate_yes",
        )

        safe_yes_representation = vectorize_df(
            df_text=safe_yes_texts,
            model=model,
            tokenizer=tokenizer,
            layer_index=layer_idx,
            strategy="last-token",
            device=device,
            name_source_data="safe_yes",
        )

        hate_no_representation = vectorize_df(
            df_text=hate_no_texts,
            model=model,
            tokenizer=tokenizer,
            layer_index=layer_idx,
            strategy="last-token",
            device=device,
            name_source_data="hate_no",
        )

        safe_no_representation = vectorize_df(
            df_text=safe_no_texts,
            model=model,
            tokenizer=tokenizer,
            layer_index=layer_idx,
            strategy="last-token",
            device=device,
            name_source_data="safe_no",
        )

        # Reshape if needed
        if len(hate_yes_representation.shape) == 3:
            hate_yes_representation = hate_yes_representation.reshape(
                hate_yes_representation.shape[0], -1
            )
            safe_yes_representation = safe_yes_representation.reshape(
                safe_yes_representation.shape[0], -1
            )
            hate_no_representation = hate_no_representation.reshape(
                hate_no_representation.shape[0], -1
            )
            safe_no_representation = safe_no_representation.reshape(
                safe_no_representation.shape[0], -1
            )

        # Calculate multiple steering vectors using different combinations
        steering_vector_hate_yes_to_safe_yes = compute_steering_vector(
            hate_yes_representation, safe_yes_representation
        )

        # Additional combinations:
        # 1. safe_no → hate_no (inverse direction)
        steering_vector_safe_no_to_hate_no = compute_steering_vector(
            safe_no_representation, hate_no_representation
        )

        # 2. hate_yes → hate_no (from affirmative hate to negative hate)
        steering_vector_hate_yes_to_hate_no = compute_steering_vector(
            hate_yes_representation, hate_no_representation
        )

        # 3. safe_no → safe_yes (from negative safe to affirmative safe)
        steering_vector_safe_no_to_safe_yes = compute_steering_vector(
            safe_no_representation, safe_yes_representation
        )

        # 4. Concatenated: (hate_yes + safe_no) → (safe_yes + hate_no)
        combined_source = np.vstack([hate_yes_representation, safe_no_representation])
        combined_target = np.vstack([safe_yes_representation, hate_no_representation])

        # Calculate mean for concatenated vectors
        combined_source_mean = np.mean(combined_source, axis=0)
        combined_target_mean = np.mean(combined_target, axis=0)

        # Compute steering vector for combined categories
        steering_vector_combined = combined_target_mean - combined_source_mean

        # Store all steering vectors in a dictionary for plotting
        all_steering_vectors = {
            "hate_yes_to_safe_yes": {
                "vector": steering_vector_hate_yes_to_safe_yes,
                "color": "#00CC00",  # Bright green
                "label": "Hate Yes → Safe Yes",
            },
            "safe_no_to_hate_no": {
                "vector": steering_vector_safe_no_to_hate_no,
                "color": "#9900FF",  # Purple
                "label": "Safe No → Hate No",
            },
            "hate_yes_to_hate_no": {
                "vector": steering_vector_hate_yes_to_hate_no,
                "color": "#FF9900",  # Orange
                "label": "Hate Yes → Hate No",
            },
            "safe_no_to_safe_yes": {
                "vector": steering_vector_safe_no_to_safe_yes,
                "color": "#00FFCC",  # Teal
                "label": "Safe No → Safe Yes",
            },
            "combined": {
                "vector": steering_vector_combined,
                "color": "#FF00FF",  # Magenta
                "label": "(Hate Yes + Safe No) → (Safe Yes + Hate No)",
            },
        }

        # Log the different steering vectors
        logger.info("Successfully calculated multiple steering vectors:")
        for name, data in all_steering_vectors.items():
            vector = data["vector"]
            logger.info(
                f"  - {name}: mean={vector.mean():.4f}, std={vector.std():.4f}, norm={np.linalg.norm(vector):.4f}"
            )
    except Exception as e:
        logger.info(f"Could not calculate specialized steering vectors: {e}")
        logger.info("Will use default steering vector only")

    # Plot all steering vectors
    plots_dir = os.path.join(run_dir, "plots")
    all_vectors_path = os.path.join(plots_dir, "all_steering_vectors.png")

    # Add hate_types and safe_types to the plot_all_steering_vectors call
    # Use the types from the last entry of all_hate_vectors (if available)
    current_hate_types = None
    current_safe_types = None
    if is_enhanced_dataset:
        # Get types from the dataset
        current_hate_types = []
        current_safe_types = []
        for i in range(min(len(hate_yes_texts), 100)):
            current_hate_types.append("hate_yes")
        for i in range(min(len(hate_no_texts), 100)):
            current_hate_types.append("hate_no")
        for i in range(min(len(safe_yes_texts), 100)):
            current_safe_types.append("safe_yes")
        for i in range(min(len(safe_no_texts), 100)):
            current_safe_types.append("safe_no")

    # plot_all_steering_vectors(
    #     results=all_data,
    #     hate_vectors=all_hate_vectors,
    #     safe_vectors=all_safe_vectors,
    #     log_base=os.path.join(plots_dir, "all_steering_vectors"),
    #     all_steering_vectors=all_steering_vectors,
    #     hate_types=current_hate_types,
    #     safe_types=current_safe_types,
    # )
    # logger.info(f"Saved all steering vectors plot to {all_vectors_path}")

    # Generate performance across layers plots for each metric
    from tr_plotting import (
        plot_all_decision_boundaries,
        plot_all_layer_vectors,
        plot_performance_across_layers,
        visualize_decision_boundary,
    )

    metrics = ["accuracy", "auc", "silhouette"]
    for metric in metrics:
        metric_plot_path = os.path.join(plots_dir, f"performance_{metric}.png")
        plot_performance_across_layers(
            results=results, metric=metric, save_path=metric_plot_path
        )
        logger.info(f"Saved {metric} performance plot to {metric_plot_path}")

    # Plot all layer vectors in a grid
    layer_vectors_plot_path = os.path.join(plots_dir, "all_layer_vectors.png")
    plot_all_layer_vectors(results=all_data, save_dir=plots_dir)
    logger.info("Saved layer vectors grid plot")

    # Prepare data for decision boundary plots
    layers_data = []
    for layer_idx in range(n_layers):
        # Initialize CCS for this layer
        ccs = CCS(
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            device=device,
        )

        # Get data vectors for visualization
        if is_enhanced_dataset:
            hate_texts = train_dataloader.dataset.get_by_type("hate_yes")[:100]
            safe_texts = train_dataloader.dataset.get_by_type("safe_yes")[:100]
        else:
            hate_texts = [
                text for batch in test_dataloader for text in batch["hate_data"]
            ]
            safe_texts = [
                text for batch in test_dataloader for text in batch["safe_data"]
            ]

            # Limit the number of samples for faster visualization
            max_samples = 100
            hate_texts = hate_texts[:max_samples]
            safe_texts = safe_texts[:max_samples]

        hate_representation = vectorize_df(
            df_text=hate_texts,
            model=model,
            tokenizer=tokenizer,
            layer_index=layer_idx,
            strategy="last-token",
            device=device,
            name_source_data="hate",
        )
        safe_representation = vectorize_df(
            df_text=safe_texts,
            model=model,
            tokenizer=tokenizer,
            layer_index=layer_idx,
            strategy="last-token",
            device=device,
            name_source_data="safe",
        )

        # Compute steering vector
        hate_mean_vector = np.mean(hate_representation, axis=0)
        safe_mean_vector = np.mean(safe_representation, axis=0)
        steering_vector = safe_mean_vector - hate_mean_vector

        # Store layer data
        layers_data.append(
            {
                "ccs": ccs,
                "layer_idx": layer_idx,
                "hate_vectors": hate_representation,
                "safe_vectors": safe_representation,
                "steering_vector": steering_vector,
            }
        )

        # Generate decision boundary visualization for each layer
        decision_boundary_path = os.path.join(
            plots_dir, f"decision_boundary_layer_{layer_idx}.png"
        )
        visualize_decision_boundary(
            ccs=ccs,
            hate_vectors=hate_representation,
            safe_vectors=safe_representation,
            steering_vector=steering_vector,
            log_base=os.path.join(plots_dir, f"decision_boundary_layer_{layer_idx}"),
        )
        logger.info(f"Saved decision boundary plot for layer {layer_idx}")

    # Plot all decision boundaries
    all_boundaries_path = os.path.join(plots_dir, "all_decision_boundaries.png")
    plot_all_decision_boundaries(
        layers_data=layers_data,
        log_base=os.path.join(plots_dir, "all_decision_boundaries"),
    )
    logger.info("Saved all decision boundaries plot")

    return results


def combine_steering_plots(plot_dir, layer_idx, strategy="last-token"):
    """
    Create a combined plot with all steering vector types in a single figure with subplots in one row.

    Args:
        plot_dir: Directory to save the combined plot
        layer_idx: Layer index
        strategy: Embedding strategy used
    """
    import os

    import matplotlib.pyplot as plt

    # Define the transformation types to include
    transformations = [
        "combined",
        "hate_yes_to_safe_yes",
        "safe_no_to_hate_no",
        "hate_no_to_hate_yes",
        "safe_yes_to_safe_no",
    ]

    # Create names for the transformations with readable titles
    transformation_titles = {
        "combined": "(Hate Yes + Safe No) → (Safe Yes + Hate No)",
        "hate_yes_to_safe_yes": "Hate Yes → Safe Yes",
        "safe_no_to_hate_no": "Safe No → Hate No",
        "hate_no_to_hate_yes": "Hate No → Hate Yes",
        "safe_yes_to_safe_no": "Safe Yes → Safe No",
    }

    # Create figure with subplots in one row
    fig, axes = plt.subplots(
        1, len(transformations), figsize=(7 * len(transformations), 6)
    )

    # Set figure title
    fig.suptitle(
        f"Steering Vectors - Layer {layer_idx} ({strategy} strategy)", fontsize=16
    )

    # Set each subplot title
    for i, trans in enumerate(transformations):
        axes[i].set_title(transformation_titles[trans], fontsize=12)
        axes[i].axis(
            "off"
        )  # Placeholder, actual plots will be filled by plot_individual_steering_vectors

    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.95))  # Make room for the suptitle

    # Save the combined figure path for returning
    combined_path = os.path.join(
        plot_dir, f"layer_{layer_idx}_{strategy}_all_steering_combined.png"
    )

    return fig, axes, combined_path


def plot_individual_steering_vectors(
    plot_dir,
    layer_idx,
    all_steering_vectors,
    hate_yes_vectors,
    hate_no_vectors,
    safe_yes_vectors,
    safe_no_vectors,
    strategy="last-token",
):
    """
    Create a combined visualization for all 5 steering vector types in a single figure with subplots:
    1. hate_yes → safe_yes
    2. safe_no → hate_no
    3. safe_yes → safe_no
    4. hate_no → hate_yes
    5. (hate_yes + safe_no) → (safe_yes + hate_no) (combined)

    Each subplot shows only the data points relevant to that specific transformation.

    Args:
        plot_dir: Directory to save the plots
        layer_idx: Layer index
        all_steering_vectors: Dictionary of steering vectors
        hate_yes_vectors: Vectors for hate_yes statements
        hate_no_vectors: Vectors for hate_no statements
        safe_yes_vectors: Vectors for safe_yes statements
        safe_no_vectors: Vectors for safe_no statements
        strategy: Embedding strategy ("last-token", "first-token", or "mean")
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    # Ensure that all vector inputs are valid
    if any(
        v is None
        for v in [hate_yes_vectors, hate_no_vectors, safe_yes_vectors, safe_no_vectors]
    ):
        print("Skipping steering vector plots because some vectors are None")
        return

    # Get figure and axes for the combined plot
    fig, axes, combined_path = combine_steering_plots(plot_dir, layer_idx, strategy)

    # Define consistent colors and markers for each category
    category_styles = {
        "hate_yes": {
            "color": "#FF0000",  # Red
            "marker": "o",  # Round
            "label": "Hate Yes",
        },
        "hate_no": {
            "color": "#0000FF",  # Blue
            "marker": "o",  # Round
            "label": "Hate No",
        },
        "safe_yes": {
            "color": "#0000FF",  # Blue
            "marker": "^",  # Triangle up
            "label": "Safe Yes",
        },
        "safe_no": {
            "color": "#FF0000",  # Red
            "marker": "^",  # Triangle up
            "label": "Safe No",
        },
    }

    # Define transformation pairs to visualize
    transformations = [
        {
            "name": "combined",
            "title": "(Hate Yes + Safe No) → (Safe Yes + Hate No)",
            "source_name": "Combined Source",
            "source_data": np.vstack([hate_yes_vectors, safe_no_vectors]),
            "source_types": ["hate_yes"] * len(hate_yes_vectors)
            + ["safe_no"] * len(safe_no_vectors),
            "target_name": "Combined Target",
            "target_data": np.vstack([safe_yes_vectors, hate_no_vectors]),
            "target_types": ["safe_yes"] * len(safe_yes_vectors)
            + ["hate_no"] * len(hate_no_vectors),
            "steering_vector": all_steering_vectors["combined"]["vector"],
            "steering_color": all_steering_vectors["combined"]["color"],
        },
        {
            "name": "hate_yes_to_safe_yes",
            "title": "Hate Yes → Safe Yes",
            "source_type": "hate_yes",
            "source_data": hate_yes_vectors,
            "target_type": "safe_yes",
            "target_data": safe_yes_vectors,
            "steering_vector": all_steering_vectors["hate_yes_to_safe_yes"]["vector"],
            "steering_color": all_steering_vectors["hate_yes_to_safe_yes"]["color"],
        },
        {
            "name": "safe_no_to_hate_no",
            "title": "Safe No → Hate No",
            "source_type": "safe_no",
            "source_data": safe_no_vectors,
            "target_type": "hate_no",
            "target_data": hate_no_vectors,
            "steering_vector": all_steering_vectors["safe_no_to_hate_no"]["vector"],
            "steering_color": all_steering_vectors["safe_no_to_hate_no"]["color"],
        },
        {
            "name": "hate_no_to_hate_yes",
            "title": "Hate No → Hate Yes",
            "source_type": "hate_no",
            "source_data": hate_no_vectors,
            "target_type": "hate_yes",
            "target_data": hate_yes_vectors,
            "steering_vector": all_steering_vectors["hate_yes_to_hate_no"]["vector"]
            * -1,  # Reverse direction
            "steering_color": "#FF9900",  # Orange (same as original)
        },
        {
            "name": "safe_yes_to_safe_no",
            "title": "Safe Yes → Safe No",
            "source_type": "safe_yes",
            "source_data": safe_yes_vectors,
            "target_type": "safe_no",
            "target_data": safe_no_vectors,
            "steering_vector": all_steering_vectors["safe_no_to_safe_yes"]["vector"]
            * -1,  # Reverse direction
            "steering_color": "#00FFCC",  # Teal (same as original)
        },
    ]

    # Create a plot for each transformation
    for i, trans in enumerate(transformations):
        ax = axes[i]
        ax.clear()  # Clear the placeholder
        ax.axis("on")  # Turn axis back on

        # Reshape data if needed
        source_data = trans["source_data"]
        target_data = trans["target_data"]

        if len(source_data.shape) == 3:
            source_data = source_data.reshape(source_data.shape[0], -1)

        if len(target_data.shape) == 3:
            target_data = target_data.reshape(target_data.shape[0], -1)

        # Combine source and target data for PCA
        combined_data = np.vstack([source_data, target_data])

        # Normalize data
        data_mean = np.mean(combined_data, axis=0, keepdims=True)
        data_std = np.std(combined_data, axis=0, keepdims=True) + 1e-10
        normalized_data = (combined_data - data_mean) / data_std

        # Calculate source and target means
        source_mean = np.mean(source_data, axis=0)
        target_mean = np.mean(target_data, axis=0)

        # Normalize means and steering vector
        source_mean_norm = (source_mean - data_mean[0]) / data_std[0]
        target_mean_norm = (target_mean - data_mean[0]) / data_std[0]
        steering_vector_norm = (trans["steering_vector"] - data_mean[0]) / data_std[0]

        # Apply PCA
        try:
            pca = PCA(n_components=2, svd_solver="full")
            data_2d = pca.fit_transform(normalized_data)

            # Transform means and steering vector
            source_mean_2d = pca.transform(source_mean_norm.reshape(1, -1))[0]
            target_mean_2d = pca.transform(target_mean_norm.reshape(1, -1))[0]
            steering_vector_2d = pca.transform(steering_vector_norm.reshape(1, -1))[0]
        except Exception as e:
            print(f"PCA failed for {trans['name']}: {e}")
            continue

        # Split data back into source and target
        n_source = len(source_data)
        source_2d = data_2d[:n_source]
        target_2d = data_2d[n_source:]

        # Plot points
        if "source_types" in trans:
            # For combined case with multiple types
            for category_type in set(trans["source_types"]):
                indices = [
                    i for i, t in enumerate(trans["source_types"]) if t == category_type
                ]
                style = category_styles[category_type]
                ax.scatter(
                    source_2d[indices, 0],
                    source_2d[indices, 1],
                    color=style["color"],
                    alpha=0.6,
                    s=40,
                    marker=style["marker"],
                    edgecolors="black",
                    linewidths=0.5,
                    label=style["label"],
                    zorder=5,
                )

            for category_type in set(trans["target_types"]):
                indices = [
                    i for i, t in enumerate(trans["target_types"]) if t == category_type
                ]
                style = category_styles[category_type]
                ax.scatter(
                    target_2d[indices, 0],
                    target_2d[indices, 1],
                    color=style["color"],
                    alpha=0.6,
                    s=40,
                    marker=style["marker"],
                    edgecolors="black",
                    linewidths=0.5,
                    label=style["label"],
                    zorder=5,
                )
        else:
            # Simple case with single type for source and target
            source_style = category_styles[trans["source_type"]]
            target_style = category_styles[trans["target_type"]]

            ax.scatter(
                source_2d[:, 0],
                source_2d[:, 1],
                color=source_style["color"],
                alpha=0.6,
                s=40,
                marker=source_style["marker"],
                edgecolors="black",
                linewidths=0.5,
                label=source_style["label"],
                zorder=5,
            )

            ax.scatter(
                target_2d[:, 0],
                target_2d[:, 1],
                color=target_style["color"],
                alpha=0.6,
                s=40,
                marker=target_style["marker"],
                edgecolors="black",
                linewidths=0.5,
                label=target_style["label"],
                zorder=5,
            )

        # Plot mean vectors
        if "source_type" in trans:
            source_style = category_styles[trans["source_type"]]
            target_style = category_styles[trans["target_type"]]

            ax.quiver(
                0,
                0,
                source_mean_2d[0],
                source_mean_2d[1],
                color=source_style["color"],
                label=f"{source_style['label']} Mean",
                scale_units="xy",
                scale=4,
                width=0.008,
                headwidth=5,
                headlength=6,
                alpha=1.0,
                zorder=10,
            )

            ax.quiver(
                0,
                0,
                target_mean_2d[0],
                target_mean_2d[1],
                color=target_style["color"],
                label=f"{target_style['label']} Mean",
                scale_units="xy",
                scale=4,
                width=0.008,
                headwidth=5,
                headlength=6,
                alpha=1.0,
                zorder=10,
            )
        else:
            # For combined case, just use general labels
            ax.quiver(
                0,
                0,
                source_mean_2d[0],
                source_mean_2d[1],
                color="#880000",  # Dark red for combined source
                label=f"{trans['source_name']} Mean",
                scale_units="xy",
                scale=4,
                width=0.008,
                headwidth=5,
                headlength=6,
                alpha=1.0,
                zorder=10,
            )

            ax.quiver(
                0,
                0,
                target_mean_2d[0],
                target_mean_2d[1],
                color="#000088",  # Dark blue for combined target
                label=f"{trans['target_name']} Mean",
                scale_units="xy",
                scale=4,
                width=0.008,
                headwidth=5,
                headlength=6,
                alpha=1.0,
                zorder=10,
            )

        # Plot steering vector
        ax.quiver(
            0,
            0,
            steering_vector_2d[0],
            steering_vector_2d[1],
            color=trans["steering_color"],
            label="Steering Vector",
            scale_units="xy",
            scale=4,
            width=0.008,
            headwidth=5,
            headlength=6,
            alpha=1.0,
            zorder=11,
        )

        # Set axis limits
        x_min, x_max = np.min(data_2d[:, 0]), np.max(data_2d[:, 0])
        y_min, y_max = np.min(data_2d[:, 1]), np.max(data_2d[:, 1])
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Add labels
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Add descriptive text
    fig.text(
        0.5,
        0.01,
        "Red circles: Hate statements with 'Yes', Blue circles: Hate statements with 'No'\n"
        "Blue triangles: Safe statements with 'Yes', Red triangles: Safe statements with 'No'\n"
        "Arrows show the steering direction between content types.",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Adjust layout to make room for text
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))

    # Save combined figure
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined steering plot to {combined_path}")
    return combined_path


def plot_all_strategies_steering_vectors(
    plot_dir,
    layer_idx,
    representations,
    all_steering_vectors_by_strategy,
):
    """
    Create a comprehensive visualization of all steering vectors for all embedding strategies.
    Plots a grid where:
    - Rows are different embedding strategies (last-token, first-token, mean)
    - Columns are different steering vector types (combined, hate_yes_to_safe_yes, etc.)

    Args:
        plot_dir: Directory to save the plot
        layer_idx: Layer index
        representations: Dictionary with representations for each strategy
        all_steering_vectors_by_strategy: Dictionary of steering vectors for each strategy
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    # Define strategies and transformations
    strategies = ["last-token", "first-token", "mean"]

    transformations = [
        "combined",
        "hate_yes_to_safe_yes",
        "safe_no_to_hate_no",
        "hate_no_to_hate_yes",
        "safe_yes_to_safe_no",
    ]

    # Create readable titles for transformations
    transformation_titles = {
        "combined": "(Hate Yes + Safe No) → (Safe Yes + Hate No)",
        "hate_yes_to_safe_yes": "Hate Yes → Safe Yes",
        "safe_no_to_hate_no": "Safe No → Hate No",
        "hate_no_to_hate_yes": "Hate No → Hate Yes",
        "safe_yes_to_safe_no": "Safe Yes → Safe No",
    }

    # Define consistent colors and markers for each category
    category_styles = {
        "hate_yes": {
            "color": "#FF0000",  # Red
            "marker": "o",  # Round
            "label": "Hate Yes",
        },
        "hate_no": {
            "color": "#0000FF",  # Blue
            "marker": "o",  # Round
            "label": "Hate No",
        },
        "safe_yes": {
            "color": "#0000FF",  # Blue
            "marker": "^",  # Triangle up
            "label": "Safe Yes",
        },
        "safe_no": {
            "color": "#FF0000",  # Red
            "marker": "^",  # Triangle up
            "label": "Safe No",
        },
    }

    # Create figure with subplots - strategies as rows, transformations as columns
    fig, axes = plt.subplots(
        len(strategies),
        len(transformations),
        figsize=(6 * len(transformations), 5 * len(strategies)),
    )

    # Process each strategy and transformation
    for row_idx, strategy in enumerate(strategies):
        # Get representations for this strategy
        strategy_reps = representations[strategy]
        hate_yes_vectors = strategy_reps["hate_yes"]
        hate_no_vectors = strategy_reps["hate_no"]
        safe_yes_vectors = strategy_reps["safe_yes"]
        safe_no_vectors = strategy_reps["safe_no"]

        # Get steering vectors for this strategy
        strategy_steering_vectors = all_steering_vectors_by_strategy[strategy]

        # For each transformation
        for col_idx, trans_name in enumerate(transformations):
            ax = axes[row_idx][col_idx]

            # Set transformation-specific data
            if trans_name == "combined":
                source_data = np.vstack([hate_yes_vectors, safe_no_vectors])
                source_types = ["hate_yes"] * len(hate_yes_vectors) + ["safe_no"] * len(
                    safe_no_vectors
                )
                target_data = np.vstack([safe_yes_vectors, hate_no_vectors])
                target_types = ["safe_yes"] * len(safe_yes_vectors) + ["hate_no"] * len(
                    hate_no_vectors
                )
                steering_vector = strategy_steering_vectors[trans_name]["vector"]
                steering_color = strategy_steering_vectors[trans_name]["color"]
                use_combined = True
            elif trans_name == "hate_yes_to_safe_yes":
                source_data = hate_yes_vectors
                source_type = "hate_yes"
                target_data = safe_yes_vectors
                target_type = "safe_yes"
                steering_vector = strategy_steering_vectors[trans_name]["vector"]
                steering_color = strategy_steering_vectors[trans_name]["color"]
                use_combined = False
            elif trans_name == "safe_no_to_hate_no":
                source_data = safe_no_vectors
                source_type = "safe_no"
                target_data = hate_no_vectors
                target_type = "hate_no"
                steering_vector = strategy_steering_vectors[trans_name]["vector"]
                steering_color = strategy_steering_vectors[trans_name]["color"]
                use_combined = False
            elif trans_name == "hate_no_to_hate_yes":
                source_data = hate_no_vectors
                source_type = "hate_no"
                target_data = hate_yes_vectors
                target_type = "hate_yes"
                steering_vector = (
                    strategy_steering_vectors["hate_yes_to_hate_no"]["vector"] * -1
                )  # Reverse
                steering_color = "#FF9900"  # Orange
                use_combined = False
            elif trans_name == "safe_yes_to_safe_no":
                source_data = safe_yes_vectors
                source_type = "safe_yes"
                target_data = safe_no_vectors
                target_type = "safe_no"
                steering_vector = (
                    strategy_steering_vectors["safe_no_to_safe_yes"]["vector"] * -1
                )  # Reverse
                steering_color = "#00FFCC"  # Teal
                use_combined = False

            # Reshape data if needed
            if len(source_data.shape) == 3:
                source_data = source_data.reshape(source_data.shape[0], -1)
            if len(target_data.shape) == 3:
                target_data = target_data.reshape(target_data.shape[0], -1)

            # Combine source and target data for PCA
            combined_data = np.vstack([source_data, target_data])

            # Normalize data
            data_mean = np.mean(combined_data, axis=0, keepdims=True)
            data_std = np.std(combined_data, axis=0, keepdims=True) + 1e-10
            normalized_data = (combined_data - data_mean) / data_std

            # Calculate source and target means
            source_mean = np.mean(source_data, axis=0)
            target_mean = np.mean(target_data, axis=0)

            # Normalize means and steering vector
            source_mean_norm = (source_mean - data_mean[0]) / data_std[0]
            target_mean_norm = (target_mean - data_mean[0]) / data_std[0]
            steering_vector_norm = (steering_vector - data_mean[0]) / data_std[0]

            # Apply PCA
            try:
                pca = PCA(n_components=2, svd_solver="full")
                data_2d = pca.fit_transform(normalized_data)

                # Transform means and steering vector
                source_mean_2d = pca.transform(source_mean_norm.reshape(1, -1))[0]
                target_mean_2d = pca.transform(target_mean_norm.reshape(1, -1))[0]
                steering_vector_2d = pca.transform(steering_vector_norm.reshape(1, -1))[
                    0
                ]
            except Exception as e:
                print(f"PCA failed for {strategy} - {trans_name}: {e}")
                continue

            # Split data back into source and target
            n_source = len(source_data)
            source_2d = data_2d[:n_source]
            target_2d = data_2d[n_source:]

            # Plot points
            if use_combined:
                # For combined case with multiple types
                for category_type in set(source_types):
                    indices = [
                        i for i, t in enumerate(source_types) if t == category_type
                    ]
                    style = category_styles[category_type]
                    ax.scatter(
                        source_2d[indices, 0],
                        source_2d[indices, 1],
                        color=style["color"],
                        alpha=0.6,
                        s=40,
                        marker=style["marker"],
                        edgecolors="black",
                        linewidths=0.5,
                        label=style["label"] if row_idx == 0 and col_idx == 0 else "",
                        zorder=5,
                    )

                for category_type in set(target_types):
                    indices = [
                        i for i, t in enumerate(target_types) if t == category_type
                    ]
                    style = category_styles[category_type]
                    ax.scatter(
                        target_2d[indices, 0],
                        target_2d[indices, 1],
                        color=style["color"],
                        alpha=0.6,
                        s=40,
                        marker=style["marker"],
                        edgecolors="black",
                        linewidths=0.5,
                        label=style["label"] if row_idx == 0 and col_idx == 0 else "",
                        zorder=5,
                    )
            else:
                # Simple case with single type for source and target
                source_style = category_styles[source_type]
                target_style = category_styles[target_type]

                ax.scatter(
                    source_2d[:, 0],
                    source_2d[:, 1],
                    color=source_style["color"],
                    alpha=0.6,
                    s=40,
                    marker=source_style["marker"],
                    edgecolors="black",
                    linewidths=0.5,
                    label=source_style["label"]
                    if row_idx == 0 and col_idx == 0
                    else "",
                    zorder=5,
                )

                ax.scatter(
                    target_2d[:, 0],
                    target_2d[:, 1],
                    color=target_style["color"],
                    alpha=0.6,
                    s=40,
                    marker=target_style["marker"],
                    edgecolors="black",
                    linewidths=0.5,
                    label=target_style["label"]
                    if row_idx == 0 and col_idx == 0
                    else "",
                    zorder=5,
                )

            # Plot mean vectors
            if not use_combined:
                source_style = category_styles[source_type]
                target_style = category_styles[target_type]

                ax.quiver(
                    0,
                    0,
                    source_mean_2d[0],
                    source_mean_2d[1],
                    color=source_style["color"],
                    label=f"{source_style['label']} Mean"
                    if row_idx == 0 and col_idx == 0
                    else "",
                    scale_units="xy",
                    scale=1,  # Smaller scale makes arrows larger
                    width=0.02,  # Thicker arrow
                    headwidth=8,  # Larger head width
                    headlength=10,  # Larger head length
                    alpha=1.0,
                    zorder=20,  # Higher zorder to draw above other elements
                )

                ax.quiver(
                    0,
                    0,
                    target_mean_2d[0],
                    target_mean_2d[1],
                    color=target_style["color"],
                    label=f"{target_style['label']} Mean"
                    if row_idx == 0 and col_idx == 0
                    else "",
                    scale_units="xy",
                    scale=1,  # Smaller scale makes arrows larger
                    width=0.02,  # Thicker arrow
                    headwidth=8,  # Larger head width
                    headlength=10,  # Larger head length
                    alpha=1.0,
                    zorder=20,  # Higher zorder to draw above other elements
                )
            else:
                # For combined case, just use general labels
                ax.quiver(
                    0,
                    0,
                    source_mean_2d[0],
                    source_mean_2d[1],
                    color="#880000",  # Dark red for combined source
                    label="Combined Source Mean"
                    if row_idx == 0 and col_idx == 0
                    else "",
                    scale_units="xy",
                    scale=1,  # Smaller scale makes arrows larger
                    width=0.02,  # Thicker arrow
                    headwidth=8,  # Larger head width
                    headlength=10,  # Larger head length
                    alpha=1.0,
                    zorder=20,  # Higher zorder to draw above other elements
                )

                ax.quiver(
                    0,
                    0,
                    target_mean_2d[0],
                    target_mean_2d[1],
                    color="#000088",  # Dark blue for combined target
                    label="Combined Target Mean"
                    if row_idx == 0 and col_idx == 0
                    else "",
                    scale_units="xy",
                    scale=1,  # Smaller scale makes arrows larger
                    width=0.02,  # Thicker arrow
                    headwidth=8,  # Larger head width
                    headlength=10,  # Larger head length
                    alpha=1.0,
                    zorder=20,  # Higher zorder to draw above other elements
                )

            # Plot steering vector
            ax.quiver(
                0,
                0,
                steering_vector_2d[0],
                steering_vector_2d[1],
                color=steering_color,
                label="Steering Vector" if row_idx == 0 and col_idx == 0 else "",
                scale_units="xy",
                scale=1,  # Smaller scale makes arrows larger
                width=0.02,  # Thicker arrow
                headwidth=8,  # Larger head width
                headlength=10,  # Larger head length
                alpha=1.0,
                zorder=21,  # Higher zorder to draw above other elements
            )

            # Set axis limits
            x_min, x_max = np.min(data_2d[:, 0]), np.max(data_2d[:, 0])
            y_min, y_max = np.min(data_2d[:, 1]), np.max(data_2d[:, 1])
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

            # Add labels
            if row_idx == len(strategies) - 1:
                ax.set_xlabel("PC 1")
            if col_idx == 0:
                ax.set_ylabel("PC 2")

            # Add row and column headers
            if col_idx == 0:
                # Add strategy labels on the left side
                ax.text(
                    -0.3,
                    0.5,
                    strategy.capitalize(),
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=14,
                    fontweight="bold",
                    rotation=90,
                )

            if row_idx == 0:
                # Add transformation labels on top
                title = transformation_titles[trans_name]
                ax.set_title(title, fontsize=12)

            # Only show legend for the top-left plot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.05, 1))

            ax.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle(f"Steering Vectors - Layer {layer_idx} (All Strategies)", fontsize=16)

    # Add descriptive text
    fig.text(
        0.5,
        0.01,
        "Red circles: Hate statements with 'Yes', Blue circles: Hate statements with 'No'\n"
        "Blue triangles: Safe statements with 'Yes', Red triangles: Safe statements with 'No'\n"
        "Arrows show the steering direction between content types.",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Adjust layout
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))

    # Save the figure
    output_path = os.path.join(
        plot_dir, f"layer_{layer_idx}_all_strategies_all_steering_vectors.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comprehensive steering vector plot to {output_path}")
    return output_path
