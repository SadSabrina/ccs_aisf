# Suppress specific warnings from scikit-learn and numerical operations
# import warnings
# warnings.filterwarnings("ignore", message="invalid value encountered in matmul")
# warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import Dataset
from tqdm import tqdm

# Global log counter for extract_representation - CHANGED: Removed frequent logging
_extract_representation_log_count = 0


class LogCounter:
    def __init__(self):
        self.count = 0
        self.max_logs = 1  # CHANGED: Reduced from 3 to 1

    def should_log(self):
        if self.count < self.max_logs:
            self.count += 1
            return True
        return False


# Create a single instance of the counter
_log_counter = LogCounter()


class HateSafeDataset(Dataset):
    def __init__(self, hate_data, safe_data, labels=None):
        """
        Dataset for paired hate and safe statements.

        Args:
            hate_data: List of hate statements
            safe_data: List of safe statements
            labels: Binary labels (0 for hate, 1 for safe). If None, creates labels assuming first half is hate, second half is safe.
        """
        # Ensure hate_data and safe_data have the same length
        min_len = min(len(hate_data), len(safe_data))
        self.hate_data = hate_data[:min_len]
        self.safe_data = safe_data[:min_len]

        # Create labels if not provided
        if labels is None:
            # Create labels: 0 for hate (first half), 1 for safe (second half)
            self.labels = np.concatenate(
                [
                    np.zeros(min_len),  # hate
                    np.ones(min_len),  # safe
                ]
            )
        else:
            # If labels are provided, ensure they match the data length
            self.labels = labels[
                : min_len * 2
            ]  # Assuming labels are for both hate and safe

        # Total samples is 2*min_len to account for both hate and safe versions
        self.n_samples = min_len * 2

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Determine if this is a hate (first half) or safe (second half) sample
        is_safe = idx >= len(self.hate_data)

        # Get the correct index within the respective array
        if is_safe:
            data_idx = idx - len(self.hate_data)
            return {
                "hate_data": self.safe_data[
                    data_idx
                ],  # Use safe as both for "safe" samples
                "safe_data": self.safe_data[data_idx],
                "labels": self.labels[idx],
            }
        else:
            return {
                "hate_data": self.hate_data[idx],  # Use hate as both for "hate" samples
                "safe_data": self.hate_data[idx],
                "labels": self.labels[idx],
            }


def load_data(data_dir="../data/yes_no", dataset_type="hate_vs_antagonist"):
    """Load and prepare the dataset with paired hate and safe statements, accounting for all 4 statement types."""
    # Load yes/no pairs
    yes_file = os.path.join(data_dir, f"{dataset_type}_yes.csv")
    no_file = os.path.join(data_dir, f"{dataset_type}_no.csv")

    yes_df = pd.read_csv(yes_file)
    no_df = pd.read_csv(no_file)

    # Sort both dataframes by index to ensure correct pairing
    yes_df = yes_df.sort_index().reset_index(drop=True)
    no_df = no_df.sort_index().reset_index(drop=True)

    # Extract the 4 types of statements
    # Type 1: Hate speech with "Yes" (is_harmfull_opposition=0 in yes file)
    hate_yes_statements = yes_df[yes_df["is_harmfull_opposition"] == 0][
        "statement"
    ].values
    # Type 2: Safe speech with "Yes" (is_harmfull_opposition=1 in yes file)
    safe_yes_statements = yes_df[yes_df["is_harmfull_opposition"] == 1][
        "statement"
    ].values
    # Type 3: Hate speech with "No" (is_harmfull_opposition=0 in no file)
    hate_no_statements = no_df[no_df["is_harmfull_opposition"] == 0]["statement"].values
    # Type 4: Safe speech with "No" (is_harmfull_opposition=1 in no file)
    safe_no_statements = no_df[no_df["is_harmfull_opposition"] == 1]["statement"].values

    # CHANGED: Only log once at the beginning
    print("\n📊 DATASET LOADING SUMMARY")
    print("-" * 40)
    print(f"Type 1 (Hate Yes): {len(hate_yes_statements)} samples")
    print(f"Type 2 (Safe Yes): {len(safe_yes_statements)} samples")
    print(f"Type 3 (Hate No): {len(hate_no_statements)} samples")
    print(f"Type 4 (Safe No): {len(safe_no_statements)} samples")

    # Calculate minimum length to ensure balanced datasets
    min_len = min(
        len(hate_yes_statements),
        len(safe_yes_statements),
        len(hate_no_statements),
        len(safe_no_statements),
    )

    # Trim all statement arrays to the same length
    hate_yes_statements = hate_yes_statements[:min_len]
    safe_yes_statements = safe_yes_statements[:min_len]
    hate_no_statements = hate_no_statements[:min_len]
    safe_no_statements = safe_no_statements[:min_len]

    # Create paired indices
    n_samples = min_len
    indices = np.random.permutation(n_samples)

    # Split indices for train/val/test
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create datasets with enhanced typing information
    train_dataset = EnhancedHateSafeDataset(
        hate_yes_data=hate_yes_statements[train_indices],
        safe_yes_data=safe_yes_statements[train_indices],
        hate_no_data=hate_no_statements[train_indices],
        safe_no_data=safe_no_statements[train_indices],
    )

    val_dataset = EnhancedHateSafeDataset(
        hate_yes_data=hate_yes_statements[val_indices],
        safe_yes_data=safe_yes_statements[val_indices],
        hate_no_data=hate_no_statements[val_indices],
        safe_no_data=safe_no_statements[val_indices],
    )

    test_dataset = EnhancedHateSafeDataset(
        hate_yes_data=hate_yes_statements[test_indices],
        safe_yes_data=safe_yes_statements[test_indices],
        hate_no_data=hate_no_statements[test_indices],
        safe_no_data=safe_no_statements[test_indices],
    )

    print(
        f"✅ Created datasets: {train_size} train, {val_size} val, {len(test_indices)} test samples per category"
    )
    return train_dataset, val_dataset, test_dataset


def extract_representation(
    model,
    tokenizer,
    text,
    layer_index=None,
    get_all_hs=False,
    strategy="first-token",
    model_type=None,
    use_decoder=False,
    device=None,
    keep_on_gpu=True,
):
    """Extract representation from model with minimal logging."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(text, (str, list)):
        raise ValueError(f"Text must be string or list, got {type(text)}")

    if strategy not in ["first-token", "last-token", "mean"]:
        raise ValueError(
            f"Strategy must be one of ['first-token', 'last-token', 'mean'], got {strategy}"
        )

    # CHANGED: Only log for the very first call
    if _log_counter.should_log():
        print(f"🔧 Extracting representations using strategy: {strategy}")

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get hidden states
    if use_decoder:
        hidden_states = outputs.decoder_hidden_states
    else:
        hidden_states = outputs.hidden_states

    if get_all_hs:
        return hidden_states

    # Get representation based on strategy
    if layer_index is None:
        layer_index = -1

    hidden_state = hidden_states[layer_index]

    if strategy == "first-token":
        representation = hidden_state[:, 0, :]
    elif strategy == "last-token":
        representation = hidden_state[:, -1, :]
    elif strategy == "mean":
        representation = hidden_state.mean(dim=1)

    # Return representation on GPU or as numpy array on CPU
    if keep_on_gpu:
        return representation
    else:
        return representation.cpu().numpy()


def extract_all_representations(
    model,
    tokenizer,
    dataset,
    layer_index,
    strategy="first-token",
    device=None,
    keep_on_gpu=True,
):
    """Extract representations for all four data types from the dataset.

    Args:
        model: The model to extract representations from
        tokenizer: The tokenizer for the model
        dataset: EnhancedHateSafeDataset containing all four data types
        layer_index: Index of the layer to extract from
        strategy: Strategy to use for extraction ('first-token', 'last-token', or 'mean')
        device: Device to use for computation
        keep_on_gpu: Whether to keep representations on GPU

    Returns:
        Dictionary with keys: hate_yes, hate_no, safe_yes, safe_no
        Each containing numpy arrays of representations
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate dataset type
    if not isinstance(dataset, EnhancedHateSafeDataset):
        raise ValueError(
            f"Dataset must be EnhancedHateSafeDataset, got {type(dataset)}"
        )

    representations = {}

    # Extract representations for each data type
    for data_type in ["hate_yes", "safe_yes", "hate_no", "safe_no"]:
        texts = dataset.get_by_type(data_type)

        if len(texts) == 0:
            raise ValueError(f"No data found for type {data_type}")

        # Extract representations for all texts of this type
        type_representations = []
        model.eval()

        with torch.no_grad():
            for text in texts:
                representation = extract_representation(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    layer_index=layer_index,
                    strategy=strategy,
                    device=device,
                    keep_on_gpu=keep_on_gpu,
                )

                if keep_on_gpu:
                    type_representations.append(representation.cpu().numpy())
                else:
                    type_representations.append(representation)

        # Stack all representations for this type
        representations[data_type] = np.vstack(type_representations)

    return representations


def calculate_steering_vectors_domain_preserving(representations):
    """Calculate steering vectors with domain-preserving normalization.

    CHANGED: Instead of CCS normalization that removes domain differences,
    we use a method that removes only Yes/No token bias while preserving
    semantic differences between hate and safe content.

    Args:
        representations: Dictionary with keys hate_yes, hate_no, safe_yes, safe_no

    Returns:
        Dictionary of steering vectors with metadata
    """
    # Validate input
    required_keys = ["hate_yes", "hate_no", "safe_yes", "safe_no"]
    for key in required_keys:
        if key not in representations:
            raise ValueError(f"Missing required key: {key}")
        if not isinstance(representations[key], np.ndarray):
            raise ValueError(
                f"Representations[{key}] must be numpy array, got {type(representations[key])}"
            )
        if representations[key].size == 0:
            raise ValueError(f"Representations[{key}] is empty")

    # CHANGED: Removed verbose logging during normalization
    # Calculate the Yes/No bias vector
    hate_yes_no_diff = representations["hate_yes"] - representations["hate_no"]
    safe_yes_no_diff = representations["safe_yes"] - representations["safe_no"]
    yes_no_bias_vector = np.mean(
        np.vstack([hate_yes_no_diff, safe_yes_no_diff]), axis=0
    )

    # Project out the bias
    yes_no_bias_norm = np.linalg.norm(yes_no_bias_vector)
    if yes_no_bias_norm > 1e-10:
        yes_no_unit_vector = yes_no_bias_vector / yes_no_bias_norm

        normalized_reps = {}
        for key in required_keys:
            projections = np.dot(representations[key], yes_no_unit_vector)
            bias_component = (
                projections[:, np.newaxis] * yes_no_unit_vector[np.newaxis, :]
            )
            normalized_reps[key] = representations[key] - bias_component
    else:
        normalized_reps = representations.copy()

    # Calculate mean vectors for each type using normalized data
    means = {}
    for key in required_keys:
        means[key] = np.mean(normalized_reps[key], axis=0)

    # Calculate steering vectors using normalized representations
    steering_vectors = {}

    # Main steering vector: Hate content -> Safe content
    hate_combined_mean = np.mean(
        np.vstack([normalized_reps["hate_yes"], normalized_reps["hate_no"]]), axis=0
    )
    safe_combined_mean = np.mean(
        np.vstack([normalized_reps["safe_yes"], normalized_reps["safe_no"]]), axis=0
    )

    steering_vectors["hate_to_safe"] = {
        "vector": safe_combined_mean - hate_combined_mean,
        "color": "#00FF00",
        "label": "Hate → Safe",
    }

    steering_vectors["combined"] = {
        "vector": safe_combined_mean - hate_combined_mean,
        "color": "#00FF00",
        "label": "Combined Steering Vector",
    }

    # Answer-specific steering vectors
    steering_vectors["hate_yes_to_safe_yes"] = {
        "vector": means["safe_yes"] - means["hate_yes"],
        "color": "#FF00FF",
        "label": "Hate Yes → Safe Yes",
    }

    steering_vectors["hate_no_to_safe_no"] = {
        "vector": means["safe_no"] - means["hate_no"],
        "color": "#FFFF00",
        "label": "Hate No → Safe No",
    }

    steering_vectors["hate_yes_to_hate_no"] = {
        "vector": means["hate_no"] - means["hate_yes"],
        "color": "#FF9900",
        "label": "Hate Yes → Hate No",
    }

    steering_vectors["safe_yes_to_safe_no"] = {
        "vector": means["safe_no"] - means["safe_yes"],
        "color": "#00FFCC",
        "label": "Safe Yes → Safe No",
    }

    # Normalize all steering vectors
    for name, data in steering_vectors.items():
        vector = data["vector"]
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            data["vector"] = vector / norm
        else:
            # Add small epsilon to prevent zero norm error
            epsilon = 1e-8
            random_vec = np.random.randn(*vector.shape)
            random_vec = random_vec / np.linalg.norm(random_vec) * epsilon
            vector = vector + random_vec
            norm = np.linalg.norm(vector)
            data["vector"] = vector / norm

    return steering_vectors


class EnhancedHateSafeDataset(Dataset):
    def __init__(self, hate_yes_data, safe_yes_data, hate_no_data, safe_no_data):
        """
        Enhanced dataset for all four types of statements.

        Args:
            hate_yes_data: Hate statements with "Yes"
            safe_yes_data: Safe statements with "Yes"
            hate_no_data: Hate statements with "No"
            safe_no_data: Safe statements with "No"
        """
        # Validate inputs
        if (
            len(hate_yes_data) == 0
            or len(safe_yes_data) == 0
            or len(hate_no_data) == 0
            or len(safe_no_data) == 0
        ):
            raise ValueError("All data types must have at least one sample")

        # Store all types of data
        self.hate_yes_data = hate_yes_data
        self.safe_yes_data = safe_yes_data
        self.hate_no_data = hate_no_data
        self.safe_no_data = safe_no_data

        # Store length of dataset (each sample will provide one statement of each type)
        self.n_per_type = len(hate_yes_data)

        # Group hate and safe data following the logical combinations:
        # - hate_data = hate_yes + safe_no (considered harmful content)
        # - safe_data = safe_yes + hate_no (considered safe content)
        self.hate_data_combined = np.concatenate(
            [hate_yes_data, safe_no_data]
        )  # Harmful content
        self.safe_data_combined = np.concatenate(
            [safe_yes_data, hate_no_data]
        )  # Safe content

        # Create labels: 0 for hate, 1 for safe
        self.labels = np.concatenate(
            [
                np.zeros(len(self.hate_data_combined)),  # hate (hate_yes + safe_no)
                np.ones(len(self.safe_data_combined)),  # safe (safe_yes + hate_no)
            ]
        )

        # Calculate total samples
        self.n_samples = len(self.labels)

        # Store type information for each sample
        self.data_types = []
        for i in range(len(hate_yes_data)):
            self.data_types.append("hate_yes")
        for i in range(len(safe_no_data)):
            self.data_types.append("safe_no")
        for i in range(len(safe_yes_data)):
            self.data_types.append("safe_yes")
        for i in range(len(hate_no_data)):
            self.data_types.append("hate_no")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return data with the proper hate/safe combinations
        if idx < len(self.hate_data_combined):
            # This is a harmful sample (either hate_yes or safe_no)
            return {
                "hate_data": self.hate_data_combined[idx],
                "safe_data": self.hate_data_combined[
                    idx
                ],  # Same for training compatibility
                "labels": self.labels[idx],  # 0 for hate
                "data_type": self.data_types[idx],
            }
        else:
            # This is a safe sample (either safe_yes or hate_no)
            safe_idx = idx - len(self.hate_data_combined)
            return {
                "hate_data": self.safe_data_combined[safe_idx],  # Using safe data
                "safe_data": self.safe_data_combined[
                    safe_idx
                ],  # Same for training compatibility
                "labels": self.labels[idx],  # 1 for safe
                "data_type": self.data_types[idx],
            }

    def get_by_type(self, data_type):
        """Get all statements of a specific type"""
        if data_type == "hate_yes":
            return self.hate_yes_data
        elif data_type == "safe_yes":
            return self.safe_yes_data
        elif data_type == "hate_no":
            return self.hate_no_data
        elif data_type == "safe_no":
            return self.safe_no_data
        else:
            raise ValueError(f"Unknown data type: {data_type}")


def create_ccs_contrast_pairs_domain_preserving(representations):
    """
    Create contrast pairs for CCS training with domain-preserving normalization.

    CHANGED: Apply the domain-preserving normalization before creating contrast pairs.

    Args:
        representations: Dict with keys ['hate_yes', 'hate_no', 'safe_yes', 'safe_no']

    Returns:
        tuple: (correct_representations, incorrect_representations)
    """
    # Apply domain-preserving normalization
    hate_yes_no_diff = representations["hate_yes"] - representations["hate_no"]
    safe_yes_no_diff = representations["safe_yes"] - representations["safe_no"]
    yes_no_bias_vector = np.mean(
        np.vstack([hate_yes_no_diff, safe_yes_no_diff]), axis=0
    )

    # Project out the bias
    yes_no_bias_norm = np.linalg.norm(yes_no_bias_vector)
    if yes_no_bias_norm > 1e-10:
        yes_no_unit_vector = yes_no_bias_vector / yes_no_bias_norm

        normalized_reps = {}
        for key in ["hate_yes", "hate_no", "safe_yes", "safe_no"]:
            projections = np.dot(representations[key], yes_no_unit_vector)
            bias_component = (
                projections[:, np.newaxis] * yes_no_unit_vector[np.newaxis, :]
            )
            normalized_reps[key] = representations[key] - bias_component
    else:
        normalized_reps = representations.copy()

    # CORRECT ANSWERS (truthful responses):
    # - hate_yes: "Is this hate speech? Yes" when it IS hate speech ✓
    # - safe_no: "Is this hate speech? No" when it's NOT hate speech ✓
    correct_answers = np.vstack(
        [
            normalized_reps["hate_yes"],  # Correctly identifying hate as hate
            normalized_reps["safe_no"],  # Correctly identifying safe as safe
        ]
    )

    # INCORRECT ANSWERS (untruthful responses):
    # - hate_no: "Is this hate speech? No" when it IS hate speech ✗
    # - safe_yes: "Is this hate speech? Yes" when it's NOT hate speech ✗
    incorrect_answers = np.vstack(
        [
            normalized_reps["hate_no"],  # Incorrectly saying hate is not hate
            normalized_reps["safe_yes"],  # Incorrectly saying safe is hate
        ]
    )

    return correct_answers, incorrect_answers


def apply_steering_to_representations(
    representations, steering_vector, steering_strength
):
    """
    Apply steering to representations.

    CHANGED: Only steer hate content toward safe content, leave safe content unchanged.

    Args:
        representations: Dict with normalized representations
        steering_vector: Normalized steering vector (hate -> safe direction)
        steering_strength: Float, how much to steer

    Returns:
        Dict with steered representations
    """
    steered_reps = representations.copy()

    # Apply steering to hate content only
    steered_reps["hate_yes"] = (
        representations["hate_yes"] + steering_strength * steering_vector
    )
    steered_reps["hate_no"] = (
        representations["hate_no"] + steering_strength * steering_vector
    )

    # Keep safe content unchanged
    steered_reps["safe_yes"] = representations["safe_yes"]
    steered_reps["safe_no"] = representations["safe_no"]

    return steered_reps


class CCSTruthProbe(nn.Module):
    """
    CCS probe for learning truth/correctness direction.

    CHANGED: Implements the actual CCS loss from the paper.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.probe = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.probe(x)

    def predict_from_vectors(self, vectors):
        """Predict probabilities and classifications from vectors."""
        if isinstance(vectors, np.ndarray):
            vectors = torch.tensor(
                vectors, dtype=torch.float32, device=next(self.parameters()).device
            )

        with torch.no_grad():
            logits = self.forward(vectors)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

        return preds, probs


def train_ccs_probe(
    correct_representations,
    incorrect_representations,
    n_epochs=1000,
    learning_rate=1e-3,
    device="cuda",
):
    """
    Train CCS probe using the consistency loss from the original paper.

    CHANGED: Added tqdm progress bar for training epochs.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    input_dim = correct_representations.shape[1]

    # Create CCS probe
    ccs_probe = CCSTruthProbe(input_dim).to(device)
    optimizer = optim.Adam(ccs_probe.parameters(), lr=learning_rate)

    # Convert to tensors
    correct_tensor = torch.tensor(
        correct_representations, dtype=torch.float32, device=device
    )
    incorrect_tensor = torch.tensor(
        incorrect_representations, dtype=torch.float32, device=device
    )

    # CHANGED: Minimal logging during training
    # Training loop with progress bar but fewer updates
    with tqdm(
        range(n_epochs),
        desc="Training CCS probe",
        unit="epoch",
        # CHANGED: Only update progress bar every 10% of epochs to reduce output
        mininterval=1.0,
        maxinterval=10.0,
    ) as pbar:
        for epoch in pbar:
            optimizer.zero_grad()

            # Get probabilities for correct and incorrect answers
            correct_logits = ccs_probe(correct_tensor)
            incorrect_logits = ccs_probe(incorrect_tensor)

            correct_probs = torch.sigmoid(correct_logits)
            incorrect_probs = torch.sigmoid(incorrect_logits)

            # CCS consistency loss
            consistency_loss = torch.mean((correct_probs + incorrect_probs - 1) ** 2)

            # CCS confidence loss
            confidence_loss = torch.mean(
                torch.min(correct_probs, 1 - incorrect_probs) ** 2
            )

            # Total CCS loss
            total_loss = consistency_loss + confidence_loss

            total_loss.backward()
            optimizer.step()

            # CHANGED: Only update progress bar, no printing during training
            if epoch % max(1, n_epochs // 10) == 0:  # Update every 10% of epochs
                pbar.set_postfix(
                    {
                        "total_loss": f"{total_loss.item():.4f}",
                        "consistency": f"{consistency_loss.item():.4f}",
                        "confidence": f"{confidence_loss.item():.4f}",
                    }
                )

    return ccs_probe


def evaluate_ccs_probe(ccs_probe, correct_representations, incorrect_representations):
    """
    Evaluate CCS probe accuracy.

    Args:
        ccs_probe: trained CCS probe
        correct_representations: correct answer representations
        incorrect_representations: incorrect answer representations

    Returns:
        accuracy score
    """
    # Create labels: 1 for correct, 0 for incorrect
    all_representations = np.vstack(
        [correct_representations, incorrect_representations]
    )
    true_labels = np.concatenate(
        [
            np.ones(len(correct_representations)),
            np.zeros(len(incorrect_representations)),
        ]
    )

    # Get predictions
    predictions, probabilities = ccs_probe.predict_from_vectors(all_representations)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)

    # Calculate AUC
    try:
        auc = roc_auc_score(true_labels, probabilities)
    except:
        auc = 0.5

    return {
        "accuracy": accuracy,
        "auc": auc,
        "predictions": predictions,
        "probabilities": probabilities,
        "true_labels": true_labels,
    }
