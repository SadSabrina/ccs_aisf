import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# MODEL CONFIGURATIONS SECTION - CHOOSE ONE AT A TIME
# ============================================================================

MODEL_CONFIGS = {
    # ===== SMALL MODELS (2-3B) =====
    # "gpt2_small": {
    #     "model_name": "gpt2",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "124M",
    #     "quantization": None,
    # },
    # "gpt2_medium": {
    #     "model_name": "gpt2-medium",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "355M",
    #     "quantization": None,
    # },
    # "distilbert_base": {
    #     "model_name": "distilbert-base-uncased",
    #     "model_type": "encoder",
    #     "token_strategy": "first-token",
    #     "use_decoder": False,
    #     "size": "66M",
    #     "quantization": None,
    # },
    # "pythia_1b": {
    #     "model_name": "EleutherAI/pythia-1b",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "1B",
    #     "quantization": None,
    # },
    # "pythia_2.8b": {
    #     "model_name": "EleutherAI/pythia-2.8b",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "2.8B",
    #     "quantization": None,
    # },
    # ===== MEDIUM MODELS (6-8B) =====
    # "pythia_6.9b": {
    #     "model_name": "EleutherAI/pythia-6.9b",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "6.9B",
    #     "quantization": None,
    # },
    # "llama3_8b_base": {
    #     "model_name": "meta-llama/Meta-Llama-3-8B",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "8B",
    #     "quantization": None,
    # },
    # "llama3_8b_instruct": {
    #     "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "8B",
    #     "is_gguf": False,
    #     "quantization": None,
    # },
    # "mistral_7b_base": {
    #     "model_name": "mistralai/Mistral-7B-v0.1",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "7B",
    #     "quantization": None,
    # },
    # "mistral_7b_instruct": {
    #     "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "7B",
    #     "quantization": None,
    # },
    # ===== LARGE MODELS (13-15B) =====
    # "llama2_13b_base": {
    #     "model_name": "meta-llama/Llama-2-13b-hf",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "13B",
    #     "quantization": None,
    # },
    # "llama2_13b_chat": {
    #     "model_name": "meta-llama/Llama-2-13b-chat-hf",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "13B",
    #     "quantization": None,
    # },
    # "pythia_12b": {
    #     "model_name": "EleutherAI/pythia-12b",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "12B",
    #     "quantization": None,
    # },
    # ===== HUGE MODELS (60-72B) WITH QUANTIZATION =====
    # ===== LLAMA 3 70B 4-BIT (Q4_K_M) - RECOMMENDED =====
    # "llama3_70b_q4": {
    #     "model_name": "NousResearch/Meta-Llama-3-70B-GGUF",
    #     "model_file": "Meta-Llama-3-70B-Q4_K_M.gguf",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "70B",
    #     "quantization": "Q4_K_M",
    #     "gpu_layers": -1,  # -1 means use all available GPU layers on Mac
    #     "memory_required_gb": 43,
    #     "is_gguf": True,
    #     "base_model_name": "meta-llama/Meta-Llama-3-70B",  # For tokenizer
    # },
    # ===== LLAMA 3 70B 8-BIT (Q8_0) =====
    # "llama3_70b_q8": {
    #     "model_name": "NousResearch/Meta-Llama-3-70B-GGUF",
    #   "model_file": "Meta-Llama-3-70B-Q8_0.gguf",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "70B",
    #     "quantization": "Q8_0",
    #     "gpu_layers": -1,  # -1 means use all available GPU layers
    #     "memory_required_gb": 75,
    #     "is_gguf": True,
    #     "base_model_name": "meta-llama/Meta-Llama-3-70B",
    # },
    # ===== LLAMA 3 70B INSTRUCT 4-BIT =====
    # "llama3_70b_instruct_q4": {
    #     "model_name": "bartowski/Meta-Llama-3-70B-Instruct-GGUF",
    #     "model_file": "Meta-Llama-3-70B-Instruct-Q4_K_M.gguf",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "70B",
    #     "quantization": "Q4_K_M",
    #     "gpu_layers": -1,
    #     "memory_required_gb": 43,
    #     "is_gguf": True,
    #     "base_model_name": "meta-llama/Meta-Llama-3-70B-Instruct",
    # },
    # ===== LLAMA 3 70B INSTRUCT 8-BIT =====
    # "llama3_70b_instruct_q8": {
    #     "model_name": "bartowski/Meta-Llama-3-70B-Instruct-GGUF",
    #     "model_file": "Meta-Llama-3-70B-Instruct-Q8_0.gguf",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "70B",
    #     "quantization": "Q8_0",
    #     "gpu_layers": -1,
    #     "memory_required_gb": 75,
    #     "is_gguf": True,
    #     "base_model_name": "meta-llama/Meta-Llama-3-70B-Instruct",
    # },
    # ===== ENCODER MODELS =====
    # "bert_base": {
    #     "model_name": "google-bert/bert-base-uncased",
    #     "model_type": "encoder",
    #     "token_strategy": "first-token",
    #     "use_decoder": False,
    #     "size": "110M",
    #     "quantization": None,
    # },
    # "bert_large": {
    #     "model_name": "google-bert/bert-large-uncased",
    #     "model_type": "encoder",
    #     "token_strategy": "first-token",
    #     "use_decoder": False,
    #     "size": "340M",
    #     "quantization": None,
    # },
    # "roberta_base": {
    #     "model_name": "FacebookAI/roberta-base",
    #     "model_type": "encoder",
    #     "token_strategy": "first-token",
    #     "use_decoder": False,
    #     "size": "125M",
    #     "quantization": None,
    # },
    # "roberta_large": {
    #     "model_name": "FacebookAI/roberta-large",
    #     "model_type": "encoder",
    #     "token_strategy": "first-token",
    #     "use_decoder": False,
    #     "size": "355M",
    #     "quantization": None,
    # },
    # "deberta_base": {
    #     "model_name": "microsoft/deberta-base",
    #     "model_type": "encoder",
    #     "token_strategy": "first-token",
    #     "use_decoder": False,
    #     "size": "100M",
    #     "quantization": None,
    # },
    # "deberta_large": {
    #     "model_name": "microsoft/deberta-large",
    #     "model_type": "encoder",
    #     "token_strategy": "first-token",
    #     "use_decoder": False,
    #     "size": "400M",
    #     "quantization": None,
    # },
    # ===== TOXICITY/SAFETY FINE-TUNED MODELS =====
    # "bert_hate_speech": {
    #     "model_name": "unitary/toxic-bert",
    #     "model_type": "encoder",
    #     "token_strategy": "first-token",
    #     "use_decoder": False,
    #     "size": "110M",
    #     "quantization": None,
    # },
    # "roberta_hate_speech": {
    #     "model_name": "SkolkovoInstitute/roberta_toxicity_classifier",
    #     "model_type": "encoder",
    #     "token_strategy": "first-token",
    #     "use_decoder": False,
    #     "size": "125M",
    #     "quantization": None,
    # },
    # "gpt2_detox": {
    #     "model_name": "ybelkada/gpt-neo-125m-detox",
    #     "model_type": "decoder",
    #     "token_strategy": "last-token",
    #     "use_decoder": False,
    #     "size": "125M",
    #     "quantization": None,
    # },
    # ACTIVE MODEL - UNCOMMENT ONE AT A TIME
    "pythia_1b": {
        "model_name": "EleutherAI/pythia-1b",
        "model_type": "decoder",
        "token_strategy": "last-token",
        "use_decoder": False,
        "size": "1B",
        "quantization": None,
    },
}

# ============================================================================
# OTHER CONFIGURATIONS
# ============================================================================

# Data Configuration
DATA_CONFIG = {
    "dataset_name": "hate_vs_antagonist",  # Options: 'hate_vs_antagonist', 'real_vs_ideal_world'
    "data_dir": "../data",  # Base data directory
    "text_column": "statement",  # Column name for text data
    "label_column": "is_harmfull_opposition",  # Column name for labels
    "test_size": 0.2,  # Train/test split ratio
    "random_state": 42,  # Random seed for reproducibility
}

# CCS Training Configuration
CCS_CONFIG = {
    "lambda_classification": 0.0,  # 0.0 for vanilla CCS, >0 for supervised CCS
    "normalizing": "mean",  # Options: 'mean', 'median', 'l2', 'raw', None
    "nepochs": 1500,  # Number of training epochs
    "ntries": 10,  # Number of random initializations
    "lr": 0.015,  # Learning rate
    "weight_decay": 0.01,  # L2 regularization
    "batch_size": -1,  # -1 for full batch
}

# Best Layer Selection Configuration
SELECTION_CONFIG = {
    "metric": "accuracy",  # Options: 'accuracy', 'silhouette_score', 'abs_agreement_score'
    "higher_is_better": True,  # True for accuracy/silhouette, False for agreement/contradiction
}

# Steering Configuration
STEERING_CONFIG = {
    "alpha_values": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # Steering strengths to test
    "token_idx": -1,  # Token position to steer (-1 for last token)
    "plot_steering": True,  # Whether to plot steering effects
    "plot_boundary": True,  # Whether to plot decision boundary
    "default_alpha": 2.0,  # Changed: Default alpha for comprehensive analysis
}

# Output Configuration
OUTPUT_CONFIG = {
    "save_embeddings": True,  # Whether to save extracted embeddings
    "save_results": True,  # Whether to save CCS results
    "save_plots": True,  # Whether to save plots
    "verbose": True,  # Print progress information
}

# Changed: Added comparison analysis configuration
COMPARISON_CONFIG = {
    "run_comprehensive_analysis": True,  # Whether to run comprehensive comparison
    "create_csv_tables": True,  # Whether to create CSV comparison tables
    "create_pretty_logs": True,  # Whether to create pretty-printed log tables
    "create_visualization_plots": True,  # Whether to create comparison plots
}
