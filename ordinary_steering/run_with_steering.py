#!/usr/bin/env python
"""
Run Coefficient Controlled Steering with pre-steering approach

This script:
1. Loads model and tokenizer
2. Prepares data
3. Runs CCS training with steering applied before training
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch

# Import plot combination functions
from combine_plots import (
    get_plot_files,
    group_plots_by_directory,
    group_plots_by_layer,
    group_plots_by_metric,
    group_plots_by_strategy,
    process_groups,
)
from torch.utils.data import DataLoader
from tr_data_utils import load_data
from tr_training_w_steering import (
    generate_visualizations,
    train_ccs_with_steering_strategies,
)
from transformers import AutoModel, AutoTokenizer


# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CCS with steering applied before training"
    )

    # Model and data arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Model name or path (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/yes_no",
        help="Directory containing the dataset (default: ../data/yes_no)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )

    # Training arguments
    parser.add_argument(
        "--n_layers",
        type=int,
        default=6,
        help="Number of layers to analyze (default: 6)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=15,
        help="Number of epochs to train each CCS (default: 15)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for training (default: 1e-3)",
    )

    # Steering arguments
    parser.add_argument(
        "--steering_coefficients",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 5.0],
        help="List of steering coefficients to use",
    )
    parser.add_argument(
        "--embedding_strategies",
        type=str,
        nargs="+",
        default=["last-token", "first-token", "mean"],
        help="List of embedding strategies to use",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: logs/run_TIMESTAMP)",
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            "ordinary_steering_logs",
            f"run_with_steering_{args.model_name.split('/')[-1]}_{timestamp}",
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    # # Fall back to standard model loading (for encoder models)
    model = AutoModel.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Move model to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Print model information
    print(f"Model type: {model.config.model_type}")
    print(f"Number of layers: {args.n_layers}")

    # Load data
    print(f"Loading data from: {args.data_dir}")
    train_dataset, val_dataset, test_dataset = load_data(
        data_dir=args.data_dir, dataset_type="hate_vs_antagonist"
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Print dataset information
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Run CCS with steering
    print("Starting CCS training with steering applied before training...")
    all_results = train_ccs_with_steering_strategies(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        run_dir=args.output_dir,
        n_layers=args.n_layers,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        device=str(device),
        steering_coefficients=args.steering_coefficients,
        embedding_strategies=args.embedding_strategies,
    )

    # Print completion message
    print(f"Training complete! Results saved to {args.output_dir}")

    # Generate visualizations separately
    print("\nGenerating visualizations...")
    plot_dir = os.path.join(args.output_dir, "plots")
    generate_visualizations(
        all_results,
        plot_dir,
        args.embedding_strategies,
        [
            ("combined", "Hate + Safe No → Safe + Hate No"),
            ("hate_yes_to_safe_yes", "Hate Yes → Safe Yes"),
            ("safe_no_to_hate_no", "Safe No → Hate No"),
            ("hate_yes_to_hate_no", "Hate Yes → Hate No"),
            ("safe_yes_to_safe_no", "Safe Yes → Safe No"),
        ],
        args.steering_coefficients,
        args.n_layers,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        device=device,
    )

    # Print metrics summary
    print("\nMetrics Summary:")
    for strategy in args.embedding_strategies:
        print(f"\nStrategy: {strategy}")
        for layer_idx in range(args.n_layers):
            print(f"  Layer {layer_idx}:")
            for coef in args.steering_coefficients:
                avg_acc = np.mean(
                    [
                        all_results[strategy][pair_name][layer_idx][coef][
                            "metrics"
                        ].get("accuracy", 0.0)
                        for pair_name in all_results[strategy]
                        if layer_idx in all_results[strategy][pair_name]
                        and coef in all_results[strategy][pair_name][layer_idx]
                        and "metrics"
                        in all_results[strategy][pair_name][layer_idx][coef]
                        and "accuracy"
                        in all_results[strategy][pair_name][layer_idx][coef]["metrics"]
                    ]
                )
                print(f"    Coef {coef}: Avg Accuracy = {avg_acc:.4f}")

    # Automatically combine plots
    print("\nCombining plots for easier visualization...")
    plot_dir = os.path.join(args.output_dir, "plots")

    if os.path.exists(plot_dir):
        # Create output directories for combined plots
        combined_dirs = {
            "directory": os.path.join(args.output_dir, "combined_plots_by_directory"),
            "metric": os.path.join(args.output_dir, "combined_plots_by_metric"),
            "strategy": os.path.join(args.output_dir, "combined_plots_by_strategy"),
            "layer": os.path.join(args.output_dir, "combined_plots_by_layer"),
        }

        for dir_path in combined_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # Get all plot files
        plot_files = get_plot_files(plot_dir)
        print(f"Found {len(plot_files)} plot files to combine")

        if plot_files:
            # Group and process plots by directory
            print("Grouping plots by directory...")
            groups = group_plots_by_directory(plot_files)
            process_groups(groups, combined_dirs["directory"], max_per_figure=9)

            # Group and process plots by metric
            print("Grouping plots by metric...")
            groups = group_plots_by_metric(plot_files)
            process_groups(groups, combined_dirs["metric"], max_per_figure=9)

            # Group and process plots by strategy
            print("Grouping plots by strategy...")
            groups = group_plots_by_strategy(plot_files)
            process_groups(groups, combined_dirs["strategy"], max_per_figure=9)

            # Group and process plots by layer
            print("Grouping plots by layer...")
            groups = group_plots_by_layer(plot_files)
            process_groups(groups, combined_dirs["layer"], max_per_figure=9)

            print("Combined plots have been saved to:")
            for group_by, dir_path in combined_dirs.items():
                print(f"  - {dir_path}")
        else:
            print("No plot files found to combine.")
    else:
        print(f"Plot directory not found: {plot_dir}")

    print("\nAll processing complete!")


if __name__ == "__main__":
    main()
