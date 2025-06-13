#!/usr/bin/env python
"""
Run Coefficient Controlled Steering with pre-steering approach

CHANGED: Updated to use new separated logger and results_analyzer modules.
No more plotting in logger - all analysis is properly separated.

This script:
1. Loads model and tokenizer
2. Prepares data
3. Runs CCS training with steering applied before training
4. Uses separated modules for logging vs analysis/plotting
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch

# CHANGED: Import separated modules instead of using logger for analysis
from logger import log_results_summary, setup_logger

# Import plot combination functions
from logger_combine_plots import (
    get_plot_files,
    group_plots_by_directory,
    group_plots_by_layer,
    group_plots_by_metric,
    group_plots_by_strategy,
    process_groups,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from tr_data_utils import load_data
from tr_training_w_steering import (
    analyze_steering_experiment_results,
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

    print("🚀 Starting CCS + Steering Experiment")
    print("=" * 50)
    print("📋 Experiment Configuration:")
    print(f"   Model: {args.model_name}")
    print(f"   Layers: {args.n_layers}")
    print(f"   Epochs per CCS: {args.n_epochs}")
    print(f"   Steering coefficients: {args.steering_coefficients}")
    print(f"   Embedding strategies: {args.embedding_strategies}")
    print(f"   Output directory: {args.output_dir}")
    print("=" * 50)

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            "ordinary_steering_logs",
            f"run_with_steering_{args.model_name.split('/')[-1]}_{timestamp}",
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # CHANGED: Set up logger using separated module (no plotting in logger)
    logger, run_dir, log_base = setup_logger(
        model_name=args.model_name,
        model_family="transformer",
        model_variant=args.model_name.split("/")[-1],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        test_size=0.3,
        random_state=42,
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Load model and tokenizer
    print(f"📦 Loading model: {args.model_name}")
    with tqdm(desc="Loading model", unit="step") as pbar:
        model = AutoModel.from_pretrained(args.model_name)
        pbar.update(1)
        pbar.set_description("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        pbar.update(1)

    # Move model to device
    print(f"🔄 Moving model to {device}...")
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Print model information
    print("🔍 Model Information:")
    print(f"   Type: {model.config.model_type}")
    print(f"   Layers to analyze: {args.n_layers}")

    # Load data
    print(f"📚 Loading data from: {args.data_dir}")
    with tqdm(desc="Loading datasets", unit="dataset") as pbar:
        train_dataset, val_dataset, test_dataset = load_data(
            data_dir=args.data_dir, dataset_type="hate_vs_antagonist"
        )
        pbar.update(1)

    # Create DataLoaders
    print("🔄 Creating data loaders...")
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Print dataset information
    print("📊 Dataset Information:")
    print(f"   Train dataset size: {len(train_dataset)}")
    print(f"   Validation dataset size: {len(val_dataset)}")
    print(f"   Test dataset size: {len(test_dataset)}")

    # Run CCS with steering - CHANGED: Now returns both results and analysis
    print("\n🧪 Starting CCS training with steering applied before training...")
    all_results, analysis_results = train_ccs_with_steering_strategies(
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
    print(f"\n🎉 Training complete! Results saved to {args.output_dir}")

    # CHANGED: Use separated logger for basic logging (no plotting)
    print("\n📝 Generating basic results summary...")
    summary_str, summary_file, json_file = log_results_summary(
        results=all_results,
        steering_coefficients=args.steering_coefficients,
        model_name=args.model_name,
        model_family="transformer",
        model_variant=args.model_name.split("/")[-1],
        run_dir=args.output_dir,
    )

    print(f"📄 Summary saved to: {summary_file}")
    print(f"📊 Detailed JSON saved to: {json_file}")

    # Analyze steering experiment results - CHANGED: Now uses simple analysis
    print("\n" + "=" * 60)
    print("ANALYZING STEERING EXPERIMENT RESULTS")
    print("=" * 60)

    analyze_steering_experiment_results(
        all_results=all_results,
        steering_coefficients=args.steering_coefficients,
        n_layers=args.n_layers,
    )

    print("\nKey Insights:")
    print("- Look for layers where steering preserves high CCS accuracy")
    print("- Compare how different steering strengths affect truth detection")
    print("- Check if hate content becomes more similar to safe content")
    print("- This experiment tests: Can we change semantics without breaking logic?")

    # Print metrics summary
    print("\n📊 Quick Metrics Summary:")
    for strategy in args.embedding_strategies:
        print(f"\nStrategy: {strategy}")
        for layer_idx in range(args.n_layers):
            print(f"  Layer {layer_idx}:")
            for coef in args.steering_coefficients:
                # CHANGED: Added safety checks to prevent crashes
                strategy_results = all_results.get(strategy, {})
                if not strategy_results:
                    print(f"    Coef {coef}: No data available")
                    continue

                accuracy_values = []
                for pair_name in strategy_results:
                    layer_results = strategy_results.get(pair_name, {})
                    if layer_idx in layer_results and coef in layer_results[layer_idx]:
                        coef_data = layer_results[layer_idx][coef]
                        if (
                            "metrics" in coef_data
                            and "accuracy" in coef_data["metrics"]
                        ):
                            accuracy_values.append(coef_data["metrics"]["accuracy"])

                if accuracy_values:
                    avg_acc = np.mean(accuracy_values)
                    print(f"    Coef {coef}: Avg Accuracy = {avg_acc:.4f}")
                else:
                    print(f"    Coef {coef}: No accuracy data available")

    # Automatically combine plots
    print("\n🖼️  Combining plots for easier visualization...")
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
            with tqdm(total=4, desc="Combining plots", unit="group") as pbar:
                # Group and process plots by directory
                print("Grouping plots by directory...")
                groups = group_plots_by_directory(plot_files)
                process_groups(groups, combined_dirs["directory"], max_per_figure=9)
                pbar.update(1)

                # Group and process plots by metric
                print("Grouping plots by metric...")
                groups = group_plots_by_metric(plot_files)
                process_groups(groups, combined_dirs["metric"], max_per_figure=9)
                pbar.update(1)

                # Group and process plots by strategy
                print("Grouping plots by strategy...")
                groups = group_plots_by_strategy(plot_files)
                process_groups(groups, combined_dirs["strategy"], max_per_figure=9)
                pbar.update(1)

                # Group and process plots by layer
                print("Grouping plots by layer...")
                groups = group_plots_by_layer(plot_files)
                process_groups(groups, combined_dirs["layer"], max_per_figure=9)
                pbar.update(1)

            print("Combined plots have been saved to:")
            for group_by, dir_path in combined_dirs.items():
                print(f"  - {dir_path}")
        else:
            print("No plot files found to combine.")
    else:
        print(f"Plot directory not found: {plot_dir}")

    # CHANGED: Show comprehensive analysis results
    if analysis_results and "plot_paths" in analysis_results:
        print(f"\n🎨 Generated {len(analysis_results['plot_paths'])} analysis plots")
        print("📊 Comprehensive analysis completed!")

        # Show which analysis plots were generated
        print("\n📊 Analysis Plots Generated:")
        for plot_name, plot_path in analysis_results["plot_paths"].items():
            rel_path = os.path.relpath(plot_path, args.output_dir)
            print(f"  - {plot_name}: {rel_path}")

    print(f"\n📁 All results organized in: {args.output_dir}")
    print("   ├── plots/ (individual plots)")
    print("   ├── combined_plots_*/ (combined visualizations)")
    print("   ├── models/ (trained CCS models)")
    print("   ├── *.txt (summary files)")
    print("   └── *.json (detailed results)")

    # Final summary of what was accomplished
    print("\n🎯 EXPERIMENT SUMMARY:")
    print(f"   • Analyzed {args.n_layers} layers")
    print(f"   • Tested {len(args.steering_coefficients)} steering coefficients")
    print(f"   • Used {len(args.embedding_strategies)} embedding strategies")

    total_experiments = (
        args.n_layers * len(args.steering_coefficients) * len(args.embedding_strategies)
    )
    print(f"   • Conducted {total_experiments} total CCS training experiments")

    if analysis_results:
        effectiveness_results = analysis_results.get("effectiveness_analysis", {})
        if effectiveness_results:
            print("   • Generated effectiveness analysis for all layers")

        plot_paths = analysis_results.get("plot_paths", {})
        if plot_paths:
            print(f"   • Created {len(plot_paths)} comprehensive analysis plots")

    print("\n✅ All processing complete!")
    print("🔍 Check the output directory for comprehensive results and visualizations!")


if __name__ == "__main__":
    main()
