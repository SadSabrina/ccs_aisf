import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import Dataset
from tqdm import tqdm

# Global log counter for extract_representation
_extract_representation_log_count = 0


class LogCounter:
    def __init__(self):
        self.count = 0
        self.max_logs = 3

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

    # Log sample statements for each type
    logging.info(
        f"Type 1 (Hate Yes) example: {hate_yes_statements[0] if len(hate_yes_statements) > 0 else 'No examples'}"
    )
    logging.info(
        f"Type 2 (Safe Yes) example: {safe_yes_statements[0] if len(safe_yes_statements) > 0 else 'No examples'}"
    )
    logging.info(
        f"Type 3 (Hate No) example: {hate_no_statements[0] if len(hate_no_statements) > 0 else 'No examples'}"
    )
    logging.info(
        f"Type 4 (Safe No) example: {safe_no_statements[0] if len(safe_no_statements) > 0 else 'No examples'}"
    )

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

    logging.info(
        f"Created datasets with {train_size} training samples, {val_size} validation samples, and {len(test_indices)} test samples for each category"
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
    """Extract representation from model.

    Args:
        model: The model to extract representations from
        tokenizer: The tokenizer for the model
        text: The text to extract representation for
        layer_index: Index of the layer to extract from (None for last layer)
        get_all_hs: Whether to return all hidden states
        strategy: Strategy to use for extraction ('first-token', 'last-token', or 'mean')
        model_type: Type of model (unused)
        use_decoder: Whether to use decoder hidden states (for encoder-decoder models)
        device: Device to use for computation
        keep_on_gpu: Whether to keep the representation on GPU (True) or move to CPU (False)

    Returns:
        Representation tensor (on GPU if keep_on_gpu=True, otherwise on CPU as numpy array)
    """
    # Validate inputs
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(text, (str, list)):
        raise ValueError(f"Text must be string or list, got {type(text)}")

    if strategy not in ["first-token", "last-token", "mean"]:
        raise ValueError(
            f"Strategy must be one of ['first-token', 'last-token', 'mean'], got {strategy}"
        )

    # Log input text and strategy only for the first 3 calls
    if _log_counter.should_log():
        logging.info(
            f"extract_representation: type(text)={type(text)}, text={str(text)}"
        )
    logging.debug(f"Extracting representation for text: {text}...")
    logging.debug(f"Using strategy: {strategy}, layer: {layer_index}")

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

    # Log representation statistics only for the first 3 calls
    if _log_counter.should_log():
        logging.info(
            f"Representation mean: {representation.mean().item():.4f}, std: {representation.std().item():.4f}"
        )
    logging.debug(f"Representation shape: {representation.shape}")
    logging.debug(f"Representation mean: {representation.mean().item():.4f}")
    logging.debug(f"Representation std: {representation.std().item():.4f}")

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
        logging.info(
            f"Extracted {len(type_representations)} representations for {data_type}, shape: {representations[data_type].shape}"
        )

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

    print("Applying domain-preserving normalization...")

    # APPROACH 1: Remove the Yes/No bias while preserving content differences
    # Calculate the pure Yes/No direction (independent of content)

    # Get paired differences for the same content type
    # For hate content: hate_yes - hate_no (pure Yes/No effect on hate)
    # For safe content: safe_yes - safe_no (pure Yes/No effect on safe)
    hate_yes_no_diff = representations["hate_yes"] - representations["hate_no"]
    safe_yes_no_diff = representations["safe_yes"] - representations["safe_no"]

    # The average difference represents the pure Yes/No token effect
    # (should be similar for both hate and safe if it's just token bias)
    yes_no_bias_vector = np.mean(
        np.vstack([hate_yes_no_diff, safe_yes_no_diff]), axis=0
    )

    print(f"Yes/No bias vector norm: {np.linalg.norm(yes_no_bias_vector):.6f}")

    # APPROACH 2: Project out the Yes/No bias from all representations
    # Normalize the bias vector
    yes_no_bias_norm = np.linalg.norm(yes_no_bias_vector)
    if yes_no_bias_norm > 1e-10:
        yes_no_unit_vector = yes_no_bias_vector / yes_no_bias_norm

        # Remove the Yes/No component from each representation
        normalized_reps = {}
        for key in required_keys:
            # Project out the Yes/No bias: x_clean = x - (x · bias_unit) * bias_unit
            projections = np.dot(representations[key], yes_no_unit_vector)
            bias_component = (
                projections[:, np.newaxis] * yes_no_unit_vector[np.newaxis, :]
            )
            normalized_reps[key] = representations[key] - bias_component

            print(
                f"Removed Yes/No bias from {key}: "
                f"mean projection = {np.mean(projections):.6f}"
            )
    else:
        print(
            "Warning: Yes/No bias vector has near-zero norm, using original representations"
        )
        normalized_reps = representations.copy()

    # Verify the bias removal worked
    # Check that Yes/No differences are now minimized
    new_hate_diff = np.mean(
        normalized_reps["hate_yes"] - normalized_reps["hate_no"], axis=0
    )
    new_safe_diff = np.mean(
        normalized_reps["safe_yes"] - normalized_reps["safe_no"], axis=0
    )

    print("After normalization:")
    print(f"  Hate Yes-No difference norm: {np.linalg.norm(new_hate_diff):.6f}")
    print(f"  Safe Yes-No difference norm: {np.linalg.norm(new_safe_diff):.6f}")

    # Verify semantic differences are preserved
    hate_mean = np.mean(
        np.vstack([normalized_reps["hate_yes"], normalized_reps["hate_no"]]), axis=0
    )
    safe_mean = np.mean(
        np.vstack([normalized_reps["safe_yes"], normalized_reps["safe_no"]]), axis=0
    )
    semantic_diff = safe_mean - hate_mean

    print(f"  Preserved semantic difference norm: {np.linalg.norm(semantic_diff):.6f}")

    # Calculate mean vectors for each type using normalized data
    means = {}
    for key in required_keys:
        means[key] = np.mean(normalized_reps[key], axis=0)
        logging.debug(
            f"Normalized mean vector for {key}: shape {means[key].shape}, norm {np.linalg.norm(means[key]):.6f}"
        )

    # Calculate steering vectors using normalized representations
    steering_vectors = {}

    # 1. Main steering vector: Hate content -> Safe content
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

    # 2. Combined steering vector (original logic for compatibility)
    steering_vectors["combined"] = {
        "vector": safe_combined_mean - hate_combined_mean,
        "color": "#00FF00",
        "label": "Combined Steering Vector",
    }

    # 3. Answer-specific steering vectors
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

    # 4. Truth-direction steering vectors (for analysis)
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
            logging.info(
                f"Calculated steering vector {name}: norm={norm:.6f} (after domain-preserving normalization)"
            )
        else:
            # If still near zero, there may not be meaningful differences
            logging.warning(
                f"Steering vector {name} has near-zero norm ({norm:.6f}) even after normalization"
            )

            # Add small epsilon to prevent zero norm error
            epsilon = 1e-8
            logging.warning(f"Adding epsilon {epsilon} to prevent zero norm error")
            random_vec = np.random.randn(*vector.shape)
            random_vec = random_vec / np.linalg.norm(random_vec) * epsilon
            vector = vector + random_vec
            norm = np.linalg.norm(vector)
            data["vector"] = vector / norm
            logging.info(f"New norm after adding epsilon: {norm:.6f}")

        logging.info(
            f"Final steering vector {name}: norm={np.linalg.norm(data['vector']):.6f}"
        )

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
    # First apply domain-preserving normalization
    print("Applying domain-preserving normalization before CCS training...")

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

    print("CCS training data:")
    print(f"  Correct answers: {len(correct_answers)} samples")
    print(f"  Incorrect answers: {len(incorrect_answers)} samples")
    print(
        f"  Total contrast pairs: {min(len(correct_answers), len(incorrect_answers))}"
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

    print(f"Applied steering with strength {steering_strength}")
    print(
        f"  Hate content moved by: {steering_strength * np.linalg.norm(steering_vector):.6f}"
    )
    print("  Safe content unchanged")

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

    print(f"Training CCS probe for {n_epochs} epochs...")

    # Training loop with progress bar
    with tqdm(range(n_epochs), desc="Training CCS probe", unit="epoch") as pbar:
        for epoch in pbar:
            optimizer.zero_grad()

            # Get probabilities for correct and incorrect answers
            correct_logits = ccs_probe(correct_tensor)
            incorrect_logits = ccs_probe(incorrect_tensor)

            correct_probs = torch.sigmoid(correct_logits)
            incorrect_probs = torch.sigmoid(incorrect_logits)

            # CCS consistency loss: p(correct) + p(incorrect) should equal 1
            consistency_loss = torch.mean((correct_probs + incorrect_probs - 1) ** 2)

            # CCS confidence loss: encourage confident predictions
            confidence_loss = torch.mean(
                torch.min(correct_probs, 1 - incorrect_probs) ** 2
            )

            # Total CCS loss
            total_loss = consistency_loss + confidence_loss

            total_loss.backward()
            optimizer.step()

            # Update progress bar with loss info
            pbar.set_postfix(
                {
                    "total_loss": f"{total_loss.item():.4f}",
                    "consistency": f"{consistency_loss.item():.4f}",
                    "confidence": f"{confidence_loss.item():.4f}",
                }
            )

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: consistency_loss={consistency_loss.item():.4f}, "
                    f"confidence_loss={confidence_loss.item():.4f}, total_loss={total_loss.item():.4f}"
                )

    print(f"CCS training completed. Final loss: {total_loss.item():.4f}")
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

    print(f"CCS Evaluation: Accuracy={accuracy:.3f}, AUC={auc:.3f}")

    return {
        "accuracy": accuracy,
        "auc": auc,
        "predictions": predictions,
        "probabilities": probabilities,
        "true_labels": true_labels,
    }
