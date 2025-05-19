import argparse
import os
import random
from datetime import datetime

# Set matplotlib backend to Agg (non-interactive)
import matplotlib
import numpy as np
import torch
from logger import print_results_summary
from torch.utils.data import DataLoader
from tr_data_utils import load_data
from tr_training import train_ccs_with_steering
from transformers import AutoModel, AutoTokenizer

matplotlib.use("Agg")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ordinary Steering")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        choices=[
            # BERT models
            "bert-base-uncased",  # 110M parameters
            "bert-large-uncased",  # 340M parameters
            "bert-base-cased",
            "bert-large-cased",
            # Finetuned BERT models
            "bert-base-uncased-finetuned-sst2",
            "bert-base-uncased-finetuned-mrpc",
            "bert-base-uncased-finetuned-qqp",
            # RoBERTa models
            "roberta-base",  # 125M parameters
            "roberta-large",  # 355M parameters
            # Finetuned RoBERTa models
            "roberta-base-finetuned-sst2",
            "roberta-base-finetuned-mrpc",
            "roberta-base-finetuned-qqp",
            # DistilBERT models
            "distilbert-base-uncased",  # 66M parameters
            "distilbert-base-cased",
            # Finetuned DistilBERT models
            "distilbert-base-uncased-finetuned-sst2",
            "distilbert-base-uncased-finetuned-mrpc",
            # GPT-2 models
            "gpt2",  # 117M parameters
            "gpt2-medium",  # 345M parameters
            "gpt2-large",  # 774M parameters
            "gpt2-xl",  # 1.5B parameters
            # Finetuned GPT-2 models
            "gpt2-finetuned-sst2",
            "gpt2-finetuned-mrpc",
            # LLaMA models
            "meta-llama/Llama-2-7b-hf",  # 7B parameters
            "meta-llama/Llama-2-13b-hf",  # 13B parameters
            "meta-llama/Llama-2-70b-hf",  # 70B parameters
            # Finetuned LLaMA models
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            # Other LLaMA variants
            "huggyllama/llama-7b",  # 7B parameters
            "huggyllama/llama-13b",  # 13B parameters
            "huggyllama/llama-30b",  # 30B parameters
            "huggyllama/llama-65b",  # 65B parameters
            # Finetuned HuggyLLaMA models
            "huggyllama/llama-7b-chat",
            "huggyllama/llama-13b-chat",
            "huggyllama/llama-30b-chat",
            "huggyllama/llama-65b-chat",
        ],
        help="Name of the model to use. Model sizes are indicated in comments.",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=f"ordinary_steering_logs/run_distilbert_base_uncased_{time}",
        help="Directory to save results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Get number of layers from model config
    n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} layers")

    # Load data
    print("Loading data...")
    train_dataset, val_dataset, test_dataset = load_data()

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    print("Starting CCS training with steering...")
    results = train_ccs_with_steering(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        run_dir=args.run_dir,
        n_layers=n_layers,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=str(device),
    )

    # Print results summary
    print_results_summary(
        results=results,
        steering_coefficients=[0.0, 0.5, 1.0, 2.0, 5.0],
        model_name=args.model_name,
        model_family=args.model_name.split("/")[0]
        if "/" in args.model_name
        else args.model_name.split("-")[0],
        model_variant=args.model_name.split("/")[-1]
        if "/" in args.model_name
        else args.model_name.split("-")[1]
        if "-" in args.model_name
        else "base",
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
