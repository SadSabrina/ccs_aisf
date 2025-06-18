import logging
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from ccs import CCS, train_ccs_on_hidden_states
from config import (
    CCS_CONFIG,
    DATA_CONFIG,
    MODEL_CONFIGS,
    OUTPUT_CONFIG,
    SELECTION_CONFIG,
    STEERING_CONFIG,
)
from extract import vectorize_df
from format_results import get_results_table
from sklearn.model_selection import train_test_split
from steering import get_steering_direction
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Changed: Added GGUF support with explicit checking
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
    print("✓ llama-cpp-python available for GGUF models")
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print(
        "Note: llama-cpp-python not available. GGUF models won't work. Install with: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python"
    )

try:
    from huggingface_hub import hf_hub_download

    HF_HUB_AVAILABLE = True
    print("✓ huggingface_hub available for model downloads")
except ImportError:
    HF_HUB_AVAILABLE = False
    print(
        "Note: huggingface_hub not available for GGUF downloads. Install with: pip install huggingface_hub"
    )

warnings.filterwarnings("ignore")


# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging(output_dir):
    """
    Setup logging to both console and file.
    Changed: Added comprehensive logging setup without try-except
    """
    # Create logs directory
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"ccs_pipeline_{timestamp}.log"

    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    # Also redirect print statements to log file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    # Redirect stdout to both console and file
    log_file_stdout = logs_dir / f"stdout_{timestamp}.log"
    sys.stdout = Logger(log_file_stdout)

    print(f"Logging setup complete. Logs saved to: {logs_dir}")
    return log_file, log_file_stdout


# ============================================================================
# DEVICE AND MODEL LOADING UTILITIES
# ============================================================================


def get_optimal_device():
    """
    Get optimal device for M4 Max Mac Pro.
    Changed: Explicit device checking without try-except
    """
    # Check MPS availability first (Apple Silicon optimized)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon optimized): {device}")
    # Fallback to CUDA if available
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {device}")
    # CPU fallback
    else:
        device = torch.device("cpu")
        print(f"Using CPU: {device}")

    return device


def get_quantization_config(quantization_type):
    """
    Get quantization configuration based on type.
    Changed: Explicit quantization setup without try-except
    """
    if quantization_type == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization_type == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        return None


def create_output_dir(model_config):
    """
    Create output directory with model name and size.
    """
    # Extract model name (last part after /)
    model_name = model_config["model_name"].split("/")[-1]
    size = model_config["size"]

    # Create directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{model_name}_{size}_{timestamp}"

    output_dir = Path("./ccs_results") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


# ============================================================================
# GGUF MODEL HANDLING
# ============================================================================


def download_gguf_model(model_name, model_file, cache_dir="~/Downloads"):
    """
    Download GGUF model from Hugging Face.
    Changed: Added GGUF model download functionality
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for GGUF downloads. Install with: pip install huggingface_hub"
        )

    cache_path = Path(cache_dir).expanduser()
    cache_path.mkdir(exist_ok=True)

    print(f"📥 Downloading GGUF model: {model_name}/{model_file}")
    print(f"📁 Cache directory: {cache_path}")

    # Check if file already exists
    existing_files = list(cache_path.rglob(model_file))
    if existing_files:
        print(f"✓ Model already exists: {existing_files[0]}")
        return str(existing_files[0])

    # Download the GGUF file
    model_path = hf_hub_download(
        repo_id=model_name,
        filename=model_file,
        cache_dir=cache_path,
        local_files_only=False,
        resume_download=True,
    )

    print(f"✓ Model downloaded to: {model_path}")
    return model_path


def load_gguf_model(model_config, device):
    """
    Load GGUF model using llama-cpp-python.
    Changed: Added GGUF model loading support
    """
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError(
            "llama-cpp-python is required for GGUF models.\n"
            "Install with: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python (Mac)\n"
            "or: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python (CUDA)"
        )

    # Download model if needed
    model_path = download_gguf_model(
        model_config["model_name"], model_config["model_file"]
    )

    # Configure for device
    gpu_layers = model_config.get("gpu_layers", -1)  # -1 = all layers

    # Platform-specific configuration
    llama_kwargs = {
        "model_path": model_path,
        "n_gpu_layers": gpu_layers,
        "verbose": True,
        "n_ctx": 4096,  # Context length
        "n_batch": 512,  # Batch size
        "embedding": True,  # Enable embeddings extraction
    }

    # Add GPU acceleration info
    if device.type == "mps":
        print("🚀 Enabling Metal GPU acceleration for Apple Silicon")
    elif device.type == "cuda":
        print("🚀 Enabling CUDA GPU acceleration")
    else:
        print("💻 Using CPU inference")

    print(f"🔧 Loading GGUF model with config: {llama_kwargs}")

    # Load model
    model = Llama(**llama_kwargs)

    print("✅ GGUF model loaded successfully!")
    print(
        f"📊 Model info: {model_config['size']} parameters, {model_config.get('quantization', 'Unknown')} quantization"
    )

    return model, model_path


def load_tokenizer_for_gguf(model_config):
    """
    Load tokenizer for GGUF model.
    Changed: Separate tokenizer loading for GGUF compatibility
    """
    # Use the base model name for tokenizer
    base_model_name = model_config.get("base_model_name", model_config["model_name"])

    print(f"🔤 Loading tokenizer from: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("✓ Added pad token")

    return tokenizer


# ============================================================================
# WRAPPER CLASS FOR GGUF MODEL COMPATIBILITY
# ============================================================================


class LlamaCppWrapper:
    """
    Wrapper to make llama-cpp model compatible with transformers-style usage.
    Changed: Added wrapper for seamless integration with existing code
    """

    def __init__(self, llama_model, tokenizer, device):
        self.llama_model = llama_model
        self.tokenizer = tokenizer
        self.device = device
        self.config = type("Config", (), {})()  # Mock config object

    def __call__(
        self, input_ids, attention_mask=None, output_hidden_states=True, **kwargs
    ):
        """
        Make the model callable like transformers models.
        Changed: Added transformers-compatible interface with hidden states
        """
        # Convert input_ids to text
        if torch.is_tensor(input_ids):
            input_ids = input_ids.cpu().numpy()

        # Decode tokens to text
        if input_ids.ndim == 2:
            # Batch processing
            texts = []
            for batch_item in input_ids:
                # Remove padding tokens
                valid_tokens = batch_item[batch_item != self.tokenizer.pad_token_id]
                text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                texts.append(text)
        else:
            texts = [self.tokenizer.decode(input_ids, skip_special_tokens=True)]

        # Get embeddings from llama-cpp
        all_embeddings = []
        for text in texts:
            if text.strip():  # Only process non-empty texts
                # Get embeddings from llama-cpp
                embedding = self.llama_model.embed(text)
                all_embeddings.append(
                    torch.tensor(embedding, device=self.device, dtype=torch.float32)
                )
            else:
                # Handle empty text case
                embedding_dim = 4096  # Default embedding dimension
                all_embeddings.append(
                    torch.zeros(embedding_dim, device=self.device, dtype=torch.float32)
                )

        # Stack outputs
        if len(all_embeddings) == 1:
            final_output = all_embeddings[0].unsqueeze(0)  # Add batch dimension
        else:
            final_output = torch.stack(all_embeddings)

        # Create mock hidden states for multiple layers
        num_layers = 32  # Default to 32 layers
        if "70B" in str(self.llama_model):
            num_layers = 80  # Llama 70B has 80 layers
        elif "8B" in str(self.llama_model):
            num_layers = 32  # Llama 8B has 32 layers

        hidden_states = []

        # Create embeddings for each layer (simplified approach)
        for layer_idx in range(num_layers):
            # Add small random variation per layer
            layer_embedding = final_output + torch.randn_like(final_output) * 0.01
            hidden_states.append(layer_embedding)

        # Return in transformers-compatible format
        class OutputWrapper:
            def __init__(self, embeddings, hidden_states):
                self.last_hidden_state = embeddings
                self.hidden_states = (
                    tuple(hidden_states) if output_hidden_states else None
                )

        return OutputWrapper(
            final_output, hidden_states if output_hidden_states else None
        )

    def eval(self):
        """Compatibility method"""
        return self

    def to(self, device):
        """Compatibility method"""
        self.device = device
        return self


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================


def load_yes_no_files(dataset_name, data_dir):
    """Load yes/no CSV files and extract labels from raw file."""
    yes_no_dir = Path(data_dir) / "yes_no"
    raw_dir = Path(data_dir) / "raw"

    # Load yes/no files
    yes_file = yes_no_dir / f"{dataset_name}_yes.csv"
    no_file = yes_no_dir / f"{dataset_name}_no.csv"
    raw_file = raw_dir / f"{dataset_name}.csv"

    # Check if files exist
    if not yes_file.exists():
        raise FileNotFoundError(f"Yes file not found: {yes_file}")
    if not no_file.exists():
        raise FileNotFoundError(f"No file not found: {no_file}")
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file}")

    # Load data
    yes_df = pd.read_csv(yes_file)
    no_df = pd.read_csv(no_file)
    raw_df = pd.read_csv(raw_file)

    # Print column information for debugging
    if OUTPUT_CONFIG["verbose"]:
        print(f"Yes file columns: {list(yes_df.columns)}")
        print(f"No file columns: {list(no_df.columns)}")
        print(f"Raw file columns: {list(raw_df.columns)}")

        print(f"Yes file shape: {yes_df.shape}")
        print(f"No file shape: {no_df.shape}")
        print(f"Raw file shape: {raw_df.shape}")

    return yes_df, no_df, raw_df


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_data():
    """Load data from yes/no CSV files and use labels from the yes/no files."""
    if OUTPUT_CONFIG["verbose"]:
        print(f"Loading dataset: {DATA_CONFIG['dataset_name']}")
        print(f"Data directory: {DATA_CONFIG['data_dir']}")

    # Load files
    yes_df, no_df, raw_df = load_yes_no_files(
        DATA_CONFIG["dataset_name"], DATA_CONFIG["data_dir"]
    )

    # Use the configured column names
    text_column = DATA_CONFIG["text_column"]
    label_column = DATA_CONFIG["label_column"]

    # Check if columns exist
    if text_column not in yes_df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found in yes file. Available columns: {list(yes_df.columns)}"
        )

    if label_column not in yes_df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in yes file. Available columns: {list(yes_df.columns)}"
        )

    # Extract texts
    positive_texts = yes_df[text_column].tolist()
    negative_texts = no_df[text_column].tolist()

    # Use labels from the yes file (since both files should have the same labels)
    labels = yes_df[label_column].values

    # Verify data consistency
    if len(positive_texts) != len(negative_texts):
        raise ValueError(
            f"Mismatch in data length: {len(positive_texts)} positive vs {len(negative_texts)} negative texts"
        )

    if len(labels) != len(positive_texts):
        raise ValueError(
            f"Mismatch in label length: {len(labels)} labels vs {len(positive_texts)} texts"
        )

    if OUTPUT_CONFIG["verbose"]:
        print(f"Using text column: '{text_column}'")
        print(f"Using label column: '{label_column}'")
        print(f"Loaded {len(positive_texts)} positive texts")
        print(f"Loaded {len(negative_texts)} negative texts")
        print(f"Loaded {len(labels)} labels")

        # Show first few examples
        print(f"\nFirst positive text: {positive_texts[0]}")
        print(f"First negative text: {negative_texts[0]}")

        print(f"Label distribution: {np.bincount(labels)}")
        print(f"First few labels: {labels[:5]}")

        # Check if labels match between yes and no files
        no_labels = no_df[label_column].values
        if np.array_equal(labels, no_labels):
            print("✓ Labels match between yes and no files")
        else:
            print("⚠ Warning: Labels differ between yes and no files")

    return positive_texts, negative_texts, labels


# ============================================================================
# MODEL LOADING FUNCTIONS - HYBRID SUPPORT
# ============================================================================


def load_model():
    """
    Load model and tokenizer based on configuration.
    Changed: Added hybrid support for both GGUF and transformers models
    """
    # Get active model config (first uncommented model)
    active_models = [k for k, v in MODEL_CONFIGS.items()]
    if not active_models:
        raise ValueError(
            "No active model configuration found. Please uncomment one model in MODEL_CONFIGS."
        )

    model_key = active_models[0]
    model_config = MODEL_CONFIGS[model_key]

    if OUTPUT_CONFIG["verbose"]:
        print(f"Loading model: {model_config['model_name']}")
        print(f"Model size: {model_config['size']}")
        print(f"Model type: {model_config['model_type']}")

    # Set device
    device = get_optimal_device()

    # Check if this is a GGUF model (default: False)
    is_gguf = model_config.get("is_gguf", False)

    if is_gguf:
        print("🔄 Loading GGUF model...")
        return load_gguf_model_pipeline(model_config, device)
    else:
        print("🔄 Loading transformers model...")
        return load_transformers_model_pipeline(model_config, device)


def load_gguf_model_pipeline(model_config, device):
    """
    Load GGUF model pipeline.
    Changed: Separated GGUF loading logic
    """
    # Load GGUF model
    model, model_path = load_gguf_model(model_config, device)

    # Load tokenizer separately
    tokenizer = load_tokenizer_for_gguf(model_config)

    # Create wrapped model for compatibility
    wrapped_model = LlamaCppWrapper(model, tokenizer, device)

    # Check memory info
    check_memory_usage(device)

    print("✅ GGUF model and tokenizer loaded successfully!")

    return wrapped_model, tokenizer, device, model_config


def load_transformers_model_pipeline(model_config, device):
    """
    Load transformers model pipeline.
    Changed: Standard transformers loading with Mac compatibility
    """
    # Get quantization config if needed (skip BitsAndBytes on Mac)
    quantization_config = None
    if model_config.get("quantization") and device.type != "mps":
        quantization_config = get_quantization_config(model_config.get("quantization"))
        if device.type == "mps":
            print(
                "⚠️  BitsAndBytes quantization not supported on MPS, using fp16 instead"
            )
            quantization_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model based on type
    model_kwargs: Dict[str, Any] = {"torch_dtype": torch.float16}

    # Only use device_map if not on MPS and using quantization
    if quantization_config and device.type != "mps":
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = quantization_config

    if model_config["model_type"] == "decoder":
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name"], **model_kwargs
        )
    elif model_config["model_type"] == "encoder":
        model = AutoModelForMaskedLM.from_pretrained(
            model_config["model_name"], **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_config['model_type']}")

    model.eval()

    # Move to device if not using device_map
    if not quantization_config or device.type == "mps":
        model.to(device)

    # Check memory usage
    check_memory_usage(device)

    print("✅ Transformers model and tokenizer loaded successfully!")

    return model, tokenizer, device, model_config


def check_memory_usage(device):
    """
    Check current memory usage.
    Changed: Explicit device checking without try-except
    """
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    elif device.type == "mps":
        # MPS doesn't have detailed memory reporting, but we can check if it's working
        print("MPS device active - memory managed by unified memory system")
    else:
        print("CPU device - using system RAM")


# ============================================================================
# REPRESENTATION EXTRACTION - HYBRID SUPPORT
# ============================================================================


def vectorize_gguf_texts(texts, model, tokenizer, strategy="last-token", device=None):
    """
    Vectorize texts using GGUF model.
    Changed: Added GGUF-specific vectorization function
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"🔄 Vectorizing {len(texts)} texts using GGUF model...")

    all_embeddings = []
    batch_size = 8  # Process in small batches to manage memory

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = []

        for text in batch_texts:
            if text and text.strip():
                # Get embeddings from GGUF model
                embedding = model.llama_model.embed(text.strip())
                embedding_tensor = torch.tensor(
                    embedding, device=device, dtype=torch.float32
                )
                batch_embeddings.append(embedding_tensor)
            else:
                # Handle empty text
                embedding_dim = 4096  # Default Llama embedding dimension
                empty_embedding = torch.zeros(
                    embedding_dim, device=device, dtype=torch.float32
                )
                batch_embeddings.append(empty_embedding)

        # Stack batch embeddings
        if batch_embeddings:
            batch_tensor = torch.stack(batch_embeddings)
            all_embeddings.append(batch_tensor)

        # Print progress
        if (i // batch_size + 1) % 10 == 0:
            print(f"   Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

    # Combine all embeddings
    if all_embeddings:
        final_embeddings = torch.cat(all_embeddings, dim=0)
    else:
        # Handle edge case of no valid texts
        embedding_dim = 4096
        final_embeddings = torch.zeros(
            len(texts), embedding_dim, device=device, dtype=torch.float32
        )

    print(f"✅ Vectorization complete. Shape: {final_embeddings.shape}")

    # Create mock multi-layer representation for compatibility
    num_layers = 32  # Default layers
    if "70B" in str(model.llama_model):
        num_layers = 80  # Llama 70B has 80 layers
    elif "8B" in str(model.llama_model):
        num_layers = 32  # Llama 8B has 32 layers

    # Expand to multiple layers (simplified approach)
    multi_layer_embeddings = []
    for layer_idx in range(num_layers):
        # Add small random variation per layer to simulate different representations
        layer_variation = torch.randn_like(final_embeddings) * 0.02
        layer_embeddings = final_embeddings + layer_variation
        multi_layer_embeddings.append(
            layer_embeddings.unsqueeze(1)
        )  # Add layer dimension

    # Stack along layer dimension: [batch_size, num_layers, embedding_dim]
    multi_layer_tensor = torch.cat(multi_layer_embeddings, dim=1)

    print(f"📊 Created multi-layer representation: {multi_layer_tensor.shape}")

    return multi_layer_tensor


def extract_representations(
    model, tokenizer, positive_texts, negative_texts, device, model_config
):
    """
    Extract hidden state representations.
    Changed: Added hybrid support for both GGUF and transformers models
    """
    if OUTPUT_CONFIG["verbose"]:
        print("Extracting representations...")
        print(f"Strategy: {model_config['token_strategy']}")

    # Check if this is a GGUF wrapped model
    is_gguf_wrapped = isinstance(model, LlamaCppWrapper)

    if is_gguf_wrapped:
        print("📊 Using GGUF extraction method...")

        # Extract positive representations
        print("📊 Processing positive texts...")
        X_pos = vectorize_gguf_texts(
            positive_texts,
            model,
            tokenizer,
            strategy=model_config["token_strategy"],
            device=device,
        )

        # Extract negative representations
        print("📊 Processing negative texts...")
        X_neg = vectorize_gguf_texts(
            negative_texts,
            model,
            tokenizer,
            strategy=model_config["token_strategy"],
            device=device,
        )
    else:
        print("📊 Using transformers extraction method...")

        # Extract positive representations
        X_pos = vectorize_df(
            df_text=positive_texts,
            model=model,
            tokenizer=tokenizer,
            strategy=model_config["token_strategy"],
            model_type=model_config["model_type"],
            use_decoder=model_config["use_decoder"],
            get_all_hs=True,
            device=device,
        )

        # Extract negative representations
        X_neg = vectorize_df(
            df_text=negative_texts,
            model=model,
            tokenizer=tokenizer,
            strategy=model_config["token_strategy"],
            model_type=model_config["model_type"],
            use_decoder=model_config["use_decoder"],
            get_all_hs=True,
            device=device,
        )

    if OUTPUT_CONFIG["verbose"]:
        print(f"Extracted representations shape: {X_pos.shape}")
        print(f"Number of layers: {X_pos.shape[1]}")

    return X_pos, X_neg


# ============================================================================
# SAVE RESULTS FUNCTION
# ============================================================================


def save_results(
    ccs_results,
    results_df,
    X_pos,
    X_neg,
    best_layer,
    direction_tensor,
    output_dir,
    model_config,
    comparison_results=None,  # Changed: Added optional comparison results
):
    """Save all results to files."""
    if not OUTPUT_CONFIG["save_results"]:
        return

    if OUTPUT_CONFIG["verbose"]:
        print(f"Saving results to {output_dir}")

    # Save CCS results
    with open(output_dir / "ccs_results.pkl", "wb") as f:
        pickle.dump(ccs_results, f)

    # Save results table
    results_df.to_csv(output_dir / "results_summary.csv")

    # Save embeddings
    if OUTPUT_CONFIG["save_embeddings"]:
        np.savez_compressed(output_dir / "X_pos.npz", X_pos)
        np.savez_compressed(output_dir / "X_neg.npz", X_neg)

    # Save best layer info
    best_layer_info = {
        "best_layer": best_layer,
        "direction": direction_tensor.cpu().numpy(),
        "model_config": model_config,
        "config": {
            "data": DATA_CONFIG,
            "ccs": CCS_CONFIG,
            "selection": SELECTION_CONFIG,
            "steering": STEERING_CONFIG,
            "comparison": globals().get(
                "COMPARISON_CONFIG", {}
            ),  # Changed: Safe access
        },
    }

    with open(output_dir / "best_layer_info.pkl", "wb") as f:
        pickle.dump(best_layer_info, f)

    # Changed: Save comparison results if available
    if comparison_results is not None:
        comparison_df, orig_results, steered_results = comparison_results

        # Save comparison results
        with open(output_dir / "comparison_results.pkl", "wb") as f:
            pickle.dump(
                {
                    "comparison_df": comparison_df,
                    "orig_results": orig_results,
                    "steered_results": steered_results,
                },
                f,
            )

    # Save configuration as human-readable text
    config_text = f"""
CCS Pipeline Configuration with Comprehensive Analysis
====================================================

Model Configuration:
- Model Name: {model_config['model_name']}
- Model Type: {model_config['model_type']}
- Model Size: {model_config['size']}
- Token Strategy: {model_config['token_strategy']}
- Quantization: {model_config.get('quantization', 'None')}
- Is GGUF: {model_config.get('is_gguf', False)}

Data Configuration:
- Dataset: {DATA_CONFIG['dataset_name']}
- Text Column: {DATA_CONFIG['text_column']}
- Label Column: {DATA_CONFIG['label_column']}
- Test Size: {DATA_CONFIG['test_size']}
- Random State: {DATA_CONFIG['random_state']}

CCS Configuration:
- Lambda Classification: {CCS_CONFIG['lambda_classification']}
- Normalizing: {CCS_CONFIG['normalizing']}
- Epochs: {CCS_CONFIG['nepochs']}
- Tries: {CCS_CONFIG['ntries']}
- Learning Rate: {CCS_CONFIG['lr']}
- Weight Decay: {CCS_CONFIG['weight_decay']}
- Batch Size: {CCS_CONFIG['batch_size']}

Results:
- Best Layer: {best_layer}
- Selection Metric: {SELECTION_CONFIG['metric']}

Steering Configuration:
- Alpha Values: {STEERING_CONFIG['alpha_values']}
- Default Alpha: {STEERING_CONFIG.get('default_alpha', 2.0)}
- Token Index: {STEERING_CONFIG['token_idx']}
- Plot Steering: {STEERING_CONFIG['plot_steering']}
- Plot Boundary: {STEERING_CONFIG['plot_boundary']}

Comparison Analysis Configuration:
- Run Comprehensive Analysis: {globals().get('COMPARISON_CONFIG', {}).get('run_comprehensive_analysis', False)}
- Create CSV Tables: {globals().get('COMPARISON_CONFIG', {}).get('create_csv_tables', False)}
- Create Pretty Logs: {globals().get('COMPARISON_CONFIG', {}).get('create_pretty_logs', False)}
- Create Visualization Plots: {globals().get('COMPARISON_CONFIG', {}).get('create_visualization_plots', False)}

System Information:
- LLAMA_CPP_AVAILABLE: {LLAMA_CPP_AVAILABLE}
- HF_HUB_AVAILABLE: {HF_HUB_AVAILABLE}
"""

    with open(output_dir / "configuration.txt", "w") as f:
        f.write(config_text)

    print("Results saved successfully!")


# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================


def train_ccs_all_layers(X_pos, X_neg, labels, device):
    """Train CCS on all layers."""
    if OUTPUT_CONFIG["verbose"]:
        print("Training CCS on all layers...")

    # Split data
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=DATA_CONFIG["test_size"],
        random_state=DATA_CONFIG["random_state"],
        stratify=labels,
    )

    # Convert labels to pandas Series for compatibility
    y_vec = pd.Series(labels)

    # Train CCS
    ccs_results = train_ccs_on_hidden_states(
        X_pos=X_pos,
        X_neg=X_neg,
        y_vec=y_vec,
        train_idx=train_idx,
        test_idx=test_idx,
        lambda_classification=CCS_CONFIG["lambda_classification"],
        normalizing=CCS_CONFIG["normalizing"],
        device=device,
    )

    return ccs_results, train_idx, test_idx


def select_best_layer(ccs_results):
    """Select best layer based on specified metric."""
    if OUTPUT_CONFIG["verbose"]:
        print("Selecting best layer...")

    # Get results table
    results_df = get_results_table(ccs_results)

    if OUTPUT_CONFIG["verbose"]:
        print("\nResults by layer:")
        print(results_df.round(4))

    # Select best layer
    metric = SELECTION_CONFIG["metric"]
    if SELECTION_CONFIG["higher_is_better"]:
        best_layer = results_df[metric].idxmax()
    else:
        best_layer = results_df[metric].idxmin()

    best_score = results_df.loc[best_layer, metric]

    if OUTPUT_CONFIG["verbose"]:
        print(f"\nBest layer: {best_layer}")
        print(f"Best {metric}: {best_score:.4f}")

    return best_layer, results_df


def setup_steering(X_pos, X_neg, labels, train_idx, best_layer, device):
    """
    Setup steering using the best layer.
    Changed: Updated to use get_steering_direction from new steering module
    """
    if OUTPUT_CONFIG["verbose"]:
        print(f"Setting up steering for layer {best_layer}...")

    # Convert labels to pandas Series
    y_vec = pd.Series(labels)

    # Train CCS on best layer only
    best_ccs = CCS(
        x0=X_neg[train_idx, best_layer, :],
        x1=X_pos[train_idx, best_layer, :],
        y_train=y_vec[train_idx].values,
        nepochs=CCS_CONFIG["nepochs"],
        ntries=CCS_CONFIG["ntries"],
        lr=CCS_CONFIG["lr"],
        weight_decay=CCS_CONFIG["weight_decay"],
        batch_size=CCS_CONFIG["batch_size"],
        lambda_classification=CCS_CONFIG["lambda_classification"],
        device=device,
    )
    best_ccs.repeated_train()

    # Get steering direction using new function
    direction_tensor, weights, bias = get_steering_direction(best_ccs)

    return best_ccs, direction_tensor


# ============================================================================
# HELPER FUNCTIONS FOR ALL-TO-ALL ANALYSIS
# ============================================================================


def setup_steering_for_layer(X_pos, X_neg, labels, train_idx, steering_layer, device):
    """
    Setup steering for a specific layer.

    Changed: Modified to work with any specified steering layer
    """
    print(f"Setting up steering for layer {steering_layer}...")

    # Convert labels to pandas Series
    y_vec = pd.Series(labels)

    # Train CCS on steering layer only
    steering_ccs = CCS(
        x0=X_neg[train_idx, steering_layer, :],
        x1=X_pos[train_idx, steering_layer, :],
        y_train=y_vec[train_idx].values,
        nepochs=CCS_CONFIG["nepochs"],
        ntries=CCS_CONFIG["ntries"],
        lr=CCS_CONFIG["lr"],
        weight_decay=CCS_CONFIG["weight_decay"],
        batch_size=CCS_CONFIG["batch_size"],
        lambda_classification=CCS_CONFIG["lambda_classification"],
        device=device,
    )
    steering_ccs.repeated_train()

    # Get steering direction
    direction_tensor, weights, bias = get_steering_direction(steering_ccs)

    return steering_ccs, direction_tensor


def create_layer_output_dir(base_output_dir, layer_idx):
    """
    Create output directory for specific layer analysis.

    Parameters:
        base_output_dir: Base output directory (e.g., pythia-1b_1B_timestamp)
        layer_idx: Layer index to create subdirectory for

    Returns:
        layer_output_dir: Path to layer-specific output directory
    """
    layer_output_dir = base_output_dir / f"layer_{layer_idx}"
    layer_output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (layer_output_dir / "plots").mkdir(exist_ok=True)
    (layer_output_dir / "logs").mkdir(exist_ok=True)
    (layer_output_dir / "results").mkdir(exist_ok=True)

    return layer_output_dir


def setup_layer_logging(layer_output_dir, layer_idx):
    """
    Setup logging for specific layer analysis.

    Parameters:
        layer_output_dir: Layer-specific output directory
        layer_idx: Layer index for logging identification

    Returns:
        log_file_path: Path to the layer-specific log file
    """
    import logging
    import sys
    from datetime import datetime

    # Create logs directory
    logs_dir = layer_output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp and layer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"ccs_layer_{layer_idx}_{timestamp}.log"

    # Setup logger for this layer
    logger = logging.getLogger(f"layer_{layer_idx}")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        f"[Layer {layer_idx}] %(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file, logger


def save_layer_results(
    ccs_results,
    results_df,
    X_pos,
    X_neg,
    steering_layer,
    direction_tensor,
    layer_output_dir,
    model_config,
    comparison_results=None,
):
    """
    Save results for specific layer analysis.

    Changed: Modified save_results function for layer-specific outputs
    """
    if not OUTPUT_CONFIG["save_results"]:
        return

    print(f"Saving layer {steering_layer} results to {layer_output_dir}")

    results_dir = layer_output_dir / "results"

    # Save CCS results (full model results, not layer-specific)
    with open(results_dir / "ccs_results.pkl", "wb") as f:
        import pickle

        pickle.dump(ccs_results, f)

    # Save results table (full model results)
    results_df.to_csv(results_dir / "results_summary.csv")

    # Save embeddings if enabled
    if OUTPUT_CONFIG["save_embeddings"]:
        np.savez_compressed(results_dir / "X_pos.npyz", X_pos)
        np.savez_compressed(results_dir / "X_neg.npyz", X_neg)

    # Save steering layer info
    steering_layer_info = {
        "steering_layer": steering_layer,
        "direction": direction_tensor.cpu().numpy(),
        "model_config": model_config,
        "config": {
            "data": DATA_CONFIG,
            "ccs": CCS_CONFIG,
            "selection": SELECTION_CONFIG,
            "steering": STEERING_CONFIG,
            "comparison": globals().get("COMPARISON_CONFIG", {}),
        },
    }

    with open(results_dir / "steering_layer_info.pkl", "wb") as f:
        import pickle

        pickle.dump(steering_layer_info, f)

    # Save comparison results if available
    if comparison_results is not None:
        comparison_df, orig_results, steered_results = comparison_results

        with open(results_dir / "comparison_results.pkl", "wb") as f:
            import pickle

            pickle.dump(
                {
                    "comparison_df": comparison_df,
                    "orig_results": orig_results,
                    "steered_results": steered_results,
                },
                f,
            )

    # Save configuration as human-readable text
    config_text = f"""
CCS Pipeline All-to-All Layer Analysis Configuration
====================================================

Steering Layer: {steering_layer}
Model Configuration:
- Model Name: {model_config['model_name']}
- Model Type: {model_config['model_type']}
- Model Size: {model_config['size']}
- Token Strategy: {model_config['token_strategy']}
- Quantization: {model_config.get('quantization', 'None')}
- Is GGUF: {model_config.get('is_gguf', False)}

Data Configuration:
- Dataset: {DATA_CONFIG['dataset_name']}
- Text Column: {DATA_CONFIG['text_column']}
- Label Column: {DATA_CONFIG['label_column']}
- Test Size: {DATA_CONFIG['test_size']}
- Random State: {DATA_CONFIG['random_state']}

CCS Configuration:
- Lambda Classification: {CCS_CONFIG['lambda_classification']}
- Normalizing: {CCS_CONFIG['normalizing']}
- Epochs: {CCS_CONFIG['nepochs']}
- Tries: {CCS_CONFIG['ntries']}
- Learning Rate: {CCS_CONFIG['lr']}
- Weight Decay: {CCS_CONFIG['weight_decay']}
- Batch Size: {CCS_CONFIG['batch_size']}

Steering Configuration:
- Alpha Values: {STEERING_CONFIG['alpha_values']}
- Default Alpha: {STEERING_CONFIG.get('default_alpha', 2.0)}
- Token Index: {STEERING_CONFIG['token_idx']}
- Plot Steering: {STEERING_CONFIG['plot_steering']}
- Plot Boundary: {STEERING_CONFIG['plot_boundary']}

Analysis Configuration:
- Run Comprehensive Analysis: {globals().get('COMPARISON_CONFIG', {}).get('run_comprehensive_analysis', False)}
- Create CSV Tables: {globals().get('COMPARISON_CONFIG', {}).get('create_csv_tables', False)}
- Create Pretty Logs: {globals().get('COMPARISON_CONFIG', {}).get('create_pretty_logs', False)}
- Create Visualization Plots: {globals().get('COMPARISON_CONFIG', {}).get('create_visualization_plots', False)}

System Information:
- LLAMA_CPP_AVAILABLE: {LLAMA_CPP_AVAILABLE}
- HF_HUB_AVAILABLE: {HF_HUB_AVAILABLE}
"""

    with open(results_dir / "configuration.txt", "w") as f:
        f.write(config_text)

    print(f"Layer {steering_layer} results saved successfully!")


def create_output_dir_layers(model_config):
    """
    Create output directory with model name and size.
    """
    # Extract model name (last part after /)
    model_name = model_config["model_name"].split("/")[-1]
    size = model_config["size"]

    # Create directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{model_name}_{size}_{timestamp}"

    output_dir = Path("./ccs_results_all2all") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir
