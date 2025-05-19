import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
):
    """Extract representation from model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log input text and strategy only for the first 3 calls
    if _log_counter.should_log():
        logging.info(
            f"extract_representation: type(text)={type(text)}, text={str(text)[:100]}"
        )
    logging.debug(f"Extracting representation for text: {text[:100]}...")
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
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Log representation statistics only for the first 3 calls
    if _log_counter.should_log():
        logging.info(
            f"Representation mean: {representation.mean().item():.4f}, std: {representation.std().item():.4f}"
        )
    logging.debug(f"Representation shape: {representation.shape}")
    logging.debug(f"Representation mean: {representation.mean().item():.4f}")
    logging.debug(f"Representation std: {representation.std().item():.4f}")

    return representation.cpu().numpy()


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
        # Store all types of data
        self.hate_yes_data = hate_yes_data
        self.safe_yes_data = safe_yes_data
        self.hate_no_data = hate_no_data
        self.safe_no_data = safe_no_data

        # Store length of dataset (each sample will provide one statement of each type)
        self.n_per_type = len(hate_yes_data)

        # Traditional hate/safe split for compatibility with existing code
        self.hate_data = np.concatenate([hate_yes_data, hate_no_data])
        self.safe_data = np.concatenate([safe_yes_data, safe_no_data])

        # Create labels: 0 for hate, 1 for safe
        self.labels = np.concatenate(
            [
                np.zeros(len(self.hate_data)),  # hate
                np.ones(len(self.safe_data)),  # safe
            ]
        )

        # Calculate total samples
        self.n_samples = len(self.labels)

        # Store type information for each sample
        self.data_types = []
        for i in range(len(hate_yes_data)):
            self.data_types.append("hate_yes")
        for i in range(len(hate_no_data)):
            self.data_types.append("hate_no")
        for i in range(len(safe_yes_data)):
            self.data_types.append("safe_yes")
        for i in range(len(safe_no_data)):
            self.data_types.append("safe_no")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # For compatibility with existing code, maintain the same structure
        # but add data_type information
        if idx < len(self.hate_data):
            # This is a hate sample (could be hate_yes or hate_no)
            return {
                "hate_data": self.hate_data[idx],
                "safe_data": self.hate_data[idx],  # Same as in original code
                "labels": self.labels[idx],
                "data_type": self.data_types[idx],
            }
        else:
            # This is a safe sample (could be safe_yes or safe_no)
            safe_idx = idx - len(self.hate_data)
            return {
                "hate_data": self.safe_data[safe_idx],  # Same as in original code
                "safe_data": self.safe_data[safe_idx],
                "labels": self.labels[idx],
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
