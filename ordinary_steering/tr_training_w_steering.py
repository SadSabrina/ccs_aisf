import logging
import os
from datetime import datetime

# Suppress specific warnings from scikit-learn PCA
# import warnings
# warnings.filterwarnings("ignore", message="invalid value encountered in matmul")
# warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
import matplotlib
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# Set matplotlib backend to Agg (non-interactive)
matplotlib.use("Agg")

# Import modules - CHANGED: Added results_analyzer import
from metrics_analyser import comprehensive_results_analysis
from tr_data_utils import (
    CCSTruthProbe,
    apply_steering_to_representations,
    calculate_steering_vectors_domain_preserving,
    evaluate_ccs_probe,
    extract_all_representations,
    train_ccs_probe,
)


# Custom class to handle plot data structure without type mismatch errors
class PlotDataPoint:
    def __init__(self, layer_idx):
        self.data = {"layer_idx": layer_idx}

    def __setitem__(self, key, value):
        self.data[key] = value

    def to_dict(self):
        return self.data


# Set up logging
def setup_logging(run_dir):
    """Setup logging with reduced verbosity."""
    log_file = os.path.join(
        run_dir, f'training_w_steering_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    # CHANGED: Only log WARNING and above to reduce console output
    logging.basicConfig(
        level=logging.WARNING,  # Changed from INFO to WARNING
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()


def extract_representations_for_all_strategies(
    model,
    tokenizer,
    train_dataloader,
    n_layers=12,
    embedding_strategies=None,
    device="cuda",
):
    """Extract representations for all strategies and layers with minimal logging."""
    if embedding_strategies is None:
        embedding_strategies = ["last-token", "first-token", "mean"]

    print("\n🔧 EXTRACTING REPRESENTATIONS")
    print("-" * 50)

    # Get dataset
    dataset = train_dataloader.dataset
    if not hasattr(dataset, "get_by_type"):
        raise ValueError("Dataset must have get_by_type method")

    # Test data availability
    required_types = ["hate_yes", "hate_no", "safe_yes", "safe_no"]
    for data_type in required_types:
        test_data = dataset.get_by_type(data_type)
        if len(test_data) == 0:
            raise ValueError(f"No data found for type {data_type}")

    print(f"📊 Strategies: {', '.join(embedding_strategies)}")
    print(f"🏗️  Layers: {n_layers}")
    print(f"📝 Data types: {len(required_types)}")

    representations = {}
    total_extractions = len(embedding_strategies) * n_layers

    # CHANGED: Single progress bar with cleaner output
    with tqdm(
        total=total_extractions,
        desc="🔄 Extracting",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        for strategy_idx, strategy in enumerate(embedding_strategies):
            representations[strategy] = {}

            for layer_idx in range(n_layers):
                # CHANGED: Less frequent postfix updates to reduce output
                if layer_idx % max(1, n_layers // 4) == 0:
                    pbar.set_postfix_str(f"{strategy} L{layer_idx}")

                # Extract representations
                layer_representations = extract_all_representations(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    keep_on_gpu=False,
                )

                representations[strategy][layer_idx] = layer_representations
                pbar.update(1)

    print("✅ Representation extraction completed")
    return representations


def calculate_steering_vectors_for_all_strategies(representations):
    """Calculate steering vectors for all strategies and layers with minimal logging."""
    print("\n📐 CALCULATING STEERING VECTORS")
    print("-" * 50)

    steering_vectors = {}
    total_calculations = sum(
        len(representations[strategy]) for strategy in representations
    )

    with tqdm(
        total=total_calculations,
        desc="🧮 Computing",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        for strategy in representations:
            steering_vectors[strategy] = {}

            for layer_idx in representations[strategy]:
                # CHANGED: Less frequent postfix updates
                if layer_idx % max(1, len(representations[strategy]) // 4) == 0:
                    pbar.set_postfix_str(f"{strategy} L{layer_idx}")

                layer_representations = representations[strategy][layer_idx]
                layer_steering_vectors = calculate_steering_vectors_domain_preserving(
                    layer_representations
                )
                steering_vectors[strategy][layer_idx] = layer_steering_vectors

                pbar.update(1)

    print("✅ Steering vector calculation completed")
    return steering_vectors


def train_single_ccs_probe(
    representations,
    pair_type,
    steering_coefficient=0.0,
    n_epochs=1000,
    learning_rate=1e-3,
    device="cuda",
    batch_silent=False,
):
    """Train a single CCS probe for a specific steering coefficient and pair type."""
    # Extract representations for the pair type
    correct_representations = representations[pair_type]["correct"]
    incorrect_representations = representations[pair_type]["incorrect"]

    # Apply steering to representations
    if steering_coefficient > 0:
        steering_vector = representations[pair_type]["steering_vector"]
        correct_representations = apply_steering_to_representations(
            correct_representations, steering_vector, steering_coefficient
        )
        incorrect_representations = apply_steering_to_representations(
            incorrect_representations, steering_vector, steering_coefficient
        )

    # Train CCS probe
    # CHANGED: Pass silent parameter to control individual progress bars
    if batch_silent:
        # Train silently without progress bar
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

        # Silent training loop without progress bar
        for epoch in range(n_epochs):
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
                (1 - correct_probs) ** 2 + (incorrect_probs) ** 2
            )

            # Total loss
            total_loss = consistency_loss + confidence_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()

        # Evaluate the trained probe
        metrics = evaluate_ccs_probe(
            ccs_probe, correct_representations, incorrect_representations
        )
    else:
        # Use standard training with progress bar
        ccs_probe, metrics = train_ccs_probe(
            correct_representations,
            incorrect_representations,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            device=device,
        )

    return {
        "probe": ccs_probe,
        "metrics": metrics,
        "steering_coefficient": steering_coefficient,
    }


def train_ccs_with_steering_strategies(
    model,
    tokenizer,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    run_dir,
    n_layers=12,
    n_epochs=10,
    learning_rate=1e-4,
    device="cuda",
    steering_coefficients=None,
    embedding_strategies=None,
):
    """Train CCS probes with steering applied before training."""
    if steering_coefficients is None:
        steering_coefficients = [0.0, 0.5, 1.0, 2.0, 5.0]
    if embedding_strategies is None:
        embedding_strategies = ["last-token", "first-token", "mean"]

    print("\n" + "=" * 80)
    print("🚀 STARTING CCS TRAINING WITH STEERING")
    print("=" * 80)
    print("📊 Configuration:")
    print(f"   • Strategies: {', '.join(embedding_strategies)}")
    print(f"   • Layers: {n_layers}")
    print(f"   • Steering coefficients: {steering_coefficients}")
    print(f"   • Epochs per probe: {n_epochs}")
    print("   • Data pairs: 5")

    # Step 1: Extract representations from model
    print("\n🚀 PHASE 1: Extracting representations from model...")
    representations = extract_representations_for_all_strategies(
        model, tokenizer, train_dataloader, n_layers, embedding_strategies, device
    )

    # Step 2: Calculate steering vectors
    print("\n📐 PHASE 2: Calculating steering vectors...")
    print("\n📐 CALCULATING STEERING VECTORS")
    print("-" * 50)
    steering_vectors = calculate_steering_vectors_for_all_strategies(representations)
    print("✅ Steering vector calculation completed")

    # Step 3: Train CCS probes for all combinations
    print("\n🧠 PHASE 3: Training CCS probes for all combinations...")

    # Define pair types to train on
    pair_types = [
        "hate_yes_to_hate_no",
        "hate_yes_to_safe_yes",
        "safe_yes_to_safe_no",
        "safe_no_to_hate_no",
        "overall",
    ]

    # Calculate total number of training tasks
    total_tasks = (
        len(embedding_strategies)
        * n_layers
        * len(steering_coefficients)
        * len(pair_types)
    )
    print(f"📈 Total training tasks: {total_tasks}")

    # CHANGED: Use a single progress bar for all training tasks
    all_results = {}
    task_count = 0

    # Create progress bar for overall training
    with tqdm(total=total_tasks, desc="🔄 Training probes", unit="task") as main_pbar:
        for strategy in embedding_strategies:
            if strategy not in all_results:
                all_results[strategy] = {}

            for pair_type in pair_types:
                if pair_type not in all_results[strategy]:
                    all_results[strategy][pair_type] = {}

                for layer_idx in range(n_layers):
                    if layer_idx not in all_results[strategy][pair_type]:
                        all_results[strategy][pair_type][layer_idx] = {}

                    for coef in steering_coefficients:
                        # CHANGED: Show progress milestones at 25%, 50%, 75%, 100%
                        task_count += 1
                        if task_count % (total_tasks // 4) == 0:
                            percentage = int(100 * task_count / total_tasks)
                            print(
                                f"📊 Progress milestone: {percentage}% complete ({task_count}/{total_tasks} tasks)"
                            )

                        # CHANGED: Use batch_silent=True to suppress individual progress bars
                        result = train_single_ccs_probe(
                            representations=representations[strategy][layer_idx],
                            pair_type=pair_type,
                            steering_coefficient=coef,
                            n_epochs=n_epochs,
                            learning_rate=learning_rate,
                            device=device,
                            batch_silent=True,  # Suppress individual progress bars
                        )

                        all_results[strategy][pair_type][layer_idx][coef] = result
                        main_pbar.update(1)

    print("\n✅ All CCS probe training completed!")

    # PHASE 4: Analysis and visualization
    print(
        "\n📊 PHASE 4: Running comprehensive analysis and generating visualizations..."
    )
    analysis_results = comprehensive_results_analysis(
        results=all_results,
        steering_coefficients=steering_coefficients,
        model_name=model.__class__.__name__,
        model_family="transformer",
        model_variant=getattr(model.config, "model_type", "unknown"),
        run_dir=run_dir,
        layer_data=None,
        all_strategy_data=representations,
        all_steering_vectors=steering_vectors,
    )

    print("\n" + "=" * 80)
    print("✅ TRAINING AND ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📁 Results saved to: {run_dir}")
    print(f"📊 Total experiments conducted: {total_tasks}")
    print(f"🎯 Analysis plots generated: {len(analysis_results.get('plot_paths', {}))}")

    return all_results, analysis_results


def analyze_steering_experiment_results(all_results, steering_coefficients, n_layers):
    """Analyze the results of the steering experiment with minimal output."""
    print("\n" + "=" * 60)
    print("📈 BASIC STEERING EXPERIMENT ANALYSIS")
    print("=" * 60)

    # Only show summary for first few layers to avoid spam
    layers_to_show = min(3, n_layers)

    for layer_idx in range(layers_to_show):
        print(f"\n🏗️  Layer {layer_idx} Summary:")
        print("-" * 30)

        # Get baseline accuracy
        baseline_acc = None
        steering_accs = {}

        # Extract accuracies for this layer
        for strategy in all_results:
            for pair_name in all_results[strategy]:
                if layer_idx in all_results[strategy][pair_name]:
                    for coef in steering_coefficients:
                        if coef in all_results[strategy][pair_name][layer_idx]:
                            acc = all_results[strategy][pair_name][layer_idx][coef][
                                "metrics"
                            ].get("accuracy", 0)

                            if coef == 0.0:
                                baseline_acc = (
                                    acc
                                    if baseline_acc is None
                                    else (baseline_acc + acc) / 2
                                )
                            else:
                                if coef not in steering_accs:
                                    steering_accs[coef] = []
                                steering_accs[coef].append(acc)

        # Print concise analysis
        if baseline_acc is not None:
            print(f"   Baseline (no steering): {baseline_acc:.3f}")

            best_improvement = 0
            best_coef = 0

            for coef in sorted(steering_accs.keys()):
                if steering_accs[coef]:
                    avg_acc = np.mean(steering_accs[coef])
                    change = avg_acc - baseline_acc

                    if change > best_improvement:
                        best_improvement = change
                        best_coef = coef

                    status = "🟢" if change > 0.02 else "🟡" if change > -0.02 else "🔴"
                    print(
                        f"   Coef {coef:4.1f}: {avg_acc:.3f} ({change:+.3f}) {status}"
                    )

            if best_improvement > 0.02:
                print(f"   🎯 Best steering: {best_coef} (+{best_improvement:.3f})")
        else:
            print("   ❌ No data available")

    if n_layers > layers_to_show:
        print(f"\n   ... (showing {layers_to_show}/{n_layers} layers for brevity)")

    print("\n📋 Legend: 🟢 Helpful  🟡 Neutral  🔴 Harmful")
    print("📊 For detailed analysis, check the comprehensive results and plots!")


if __name__ == "__main__":
    print(
        "This module should be imported and used with the appropriate dataloader and model."
    )
    print("Example usage:")
    print("from tr_training_w_steering import train_ccs_with_steering_strategies")
    print(
        "results = train_ccs_with_steering_strategies(model, tokenizer, train_dataloader, val_dataloader, test_dataloader, run_dir)"
    )
