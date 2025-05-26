import logging
import os
import warnings
from datetime import datetime

# Suppress specific warnings from scikit-learn PCA
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

# Set matplotlib backend to Agg (non-interactive)
matplotlib.use("Agg")

# Import modules
from metrics_ordinary_steering import (
    plot_coefficient_sweep_lines_comparison,
)
from tr_data_utils import (
    apply_steering_to_representations,
    calculate_steering_vectors_domain_preserving,
    create_ccs_contrast_pairs_domain_preserving,
    evaluate_ccs_probe,
    extract_all_representations,
    train_ccs_probe,
)
from tr_plotting import (
    plot_all_decision_boundaries,
    plot_all_layer_vectors,
    plot_all_strategies_all_steering_vectors,
    plot_performance_across_layers,
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
    log_file = os.path.join(
        run_dir, f'training_w_steering_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
    """Extract representations for all strategies and layers with progress tracking.

    CHANGED: Added comprehensive tqdm progress bars to track extraction progress.
    """
    if embedding_strategies is None:
        embedding_strategies = ["last-token", "first-token", "mean"]

    logger = logging.getLogger(__name__)
    logger.info("Starting representation extraction for all strategies and layers")

    # Get dataset
    dataset = train_dataloader.dataset
    if not hasattr(dataset, "get_by_type"):
        raise ValueError("Dataset must have get_by_type method")

    # Test that we can get data for all required types
    required_types = ["hate_yes", "hate_no", "safe_yes", "safe_no"]
    for data_type in required_types:
        test_data = dataset.get_by_type(data_type)
        if len(test_data) == 0:
            raise ValueError(f"No data found for type {data_type}")
        logger.info(f"Found {len(test_data)} samples for {data_type}")

    representations = {}

    # Overall progress: strategies × layers
    total_extractions = len(embedding_strategies) * n_layers

    with tqdm(
        total=total_extractions,
        desc="Extracting representations",
        unit="strategy-layer",
    ) as pbar:
        # Extract for each strategy
        for strategy_idx, strategy in enumerate(embedding_strategies):
            logger.info(f"Extracting representations for strategy: {strategy}")
            representations[strategy] = {}

            # Extract for each layer with progress
            for layer_idx in range(n_layers):
                pbar.set_postfix(
                    {
                        "strategy": strategy,
                        "layer": f"{layer_idx}/{n_layers-1}",
                        "current": f"{strategy_idx+1}/{len(embedding_strategies)}",
                    }
                )

                logger.info(f"Processing layer {layer_idx}")

                # Extract representations using the existing function
                layer_representations = extract_all_representations(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    layer_index=layer_idx,
                    strategy=strategy,
                    device=device,
                    keep_on_gpu=False,  # Return as numpy arrays
                )

                # Store the layer representations
                representations[strategy][layer_idx] = layer_representations

                # Log shapes for verification
                for data_type, data in layer_representations.items():
                    logger.debug(
                        f"Strategy {strategy}, Layer {layer_idx}, Type {data_type}: shape {data.shape}"
                    )

                pbar.update(1)

    logger.info("Successfully extracted all representations")
    return representations


def calculate_steering_vectors_for_all_strategies(representations):
    """Calculate steering vectors for all strategies and layers with progress tracking.

    CHANGED: Added tqdm progress bars to track steering vector calculation.
    """
    logger = logging.getLogger(__name__)
    logger.info("Calculating steering vectors for all strategies and layers")

    steering_vectors = {}

    # Count total operations
    total_calculations = sum(
        len(representations[strategy]) for strategy in representations
    )

    with tqdm(
        total=total_calculations, desc="Calculating steering vectors", unit="layer"
    ) as pbar:
        for strategy in representations:
            logger.info(f"Calculating steering vectors for strategy: {strategy}")
            steering_vectors[strategy] = {}

            for layer_idx in representations[strategy]:
                pbar.set_postfix({"strategy": strategy, "layer": layer_idx})

                logger.info(f"Processing layer {layer_idx}")

                # Get layer representations - this is Dict[data_type] -> numpy array
                layer_representations = representations[strategy][layer_idx]

                # Calculate steering vectors using domain-preserving normalization
                layer_steering_vectors = calculate_steering_vectors_domain_preserving(
                    layer_representations
                )
                steering_vectors[strategy][layer_idx] = layer_steering_vectors

                # Log calculated vectors
                for vector_type in layer_steering_vectors:
                    vector_norm = np.linalg.norm(
                        layer_steering_vectors[vector_type]["vector"]
                    )
                    logger.debug(
                        f"Strategy {strategy}, Layer {layer_idx}, Vector {vector_type}: norm {vector_norm:.6f}"
                    )

                pbar.update(1)

    logger.info("Successfully calculated all steering vectors")
    return steering_vectors


def train_single_ccs_probe(
    representations,
    pair_type,
    steering_coefficient=0.0,
    n_epochs=1000,
    learning_rate=1e-3,
    device="cuda",
):
    """Train a single CCS probe for given representations and pair type.

    CHANGED: Now implements proper CCS training with correct contrast pairs
    and applies steering before training.

    Args:
        representations: Dict[data_type] -> numpy array for a single layer
        pair_type: Type of data pair (not used in new implementation)
        steering_coefficient: Steering coefficient to apply
        n_epochs: Number of training epochs
        learning_rate: Learning rate for training
        device: Device to use for training

    Returns:
        Dictionary with trained CCS model and metrics
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Training CCS probe with steering coefficient {steering_coefficient}")

    # Check we have all required data types
    required_types = ["hate_yes", "hate_no", "safe_yes", "safe_no"]
    for req_type in required_types:
        if req_type not in representations:
            raise ValueError(f"Missing required representation type: {req_type}")
        if not isinstance(representations[req_type], np.ndarray):
            raise ValueError(
                f"representations[{req_type}] must be numpy array, got {type(representations[req_type])}"
            )
        if representations[req_type].size == 0:
            raise ValueError(f"representations[{req_type}] is empty")

    # Apply steering if coefficient > 0
    if steering_coefficient > 0.0:
        # Calculate steering vector (hate -> safe)
        hate_mean = np.mean(
            np.vstack([representations["hate_yes"], representations["hate_no"]]), axis=0
        )
        safe_mean = np.mean(
            np.vstack([representations["safe_yes"], representations["safe_no"]]), axis=0
        )
        steering_vector = safe_mean - hate_mean
        steering_norm = np.linalg.norm(steering_vector)

        if steering_norm > 1e-10:
            steering_vector = steering_vector / steering_norm
            # Apply steering to representations
            steered_representations = apply_steering_to_representations(
                representations, steering_vector, steering_coefficient
            )
        else:
            logger.warning(
                f"Steering vector has near-zero norm ({steering_norm:.6f}), using original representations"
            )
            steered_representations = representations
    else:
        steered_representations = representations

    # Create CCS contrast pairs with domain-preserving normalization
    correct_answers, incorrect_answers = create_ccs_contrast_pairs_domain_preserving(
        steered_representations
    )

    # Train CCS probe
    ccs_probe = train_ccs_probe(
        correct_answers,
        incorrect_answers,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device,
    )

    # Evaluate CCS probe
    metrics = evaluate_ccs_probe(ccs_probe, correct_answers, incorrect_answers)

    # Add steering-specific metrics
    if steering_coefficient > 0.0:
        # Calculate how much hate content has moved toward safe content
        original_hate_mean = np.mean(
            np.vstack([representations["hate_yes"], representations["hate_no"]]), axis=0
        )
        steered_hate_mean = np.mean(
            np.vstack(
                [
                    steered_representations["hate_yes"],
                    steered_representations["hate_no"],
                ]
            ),
            axis=0,
        )
        safe_mean = np.mean(
            np.vstack([representations["safe_yes"], representations["safe_no"]]), axis=0
        )

        # Similarity metrics
        original_hate_safe_sim = np.dot(original_hate_mean, safe_mean) / (
            np.linalg.norm(original_hate_mean) * np.linalg.norm(safe_mean)
        )
        steered_hate_safe_sim = np.dot(steered_hate_mean, safe_mean) / (
            np.linalg.norm(steered_hate_mean) * np.linalg.norm(safe_mean)
        )

        metrics["steering_effect"] = {
            "original_hate_safe_similarity": original_hate_safe_sim,
            "steered_hate_safe_similarity": steered_hate_safe_sim,
            "similarity_increase": steered_hate_safe_sim - original_hate_safe_sim,
            "steering_coefficient": steering_coefficient,
        }

    result = {
        "ccs": ccs_probe,
        "metrics": metrics,
        "pair_type": pair_type,
        "steering_coefficient": steering_coefficient,
    }

    logger.debug(
        f"CCS training completed - Accuracy: {metrics['accuracy']:.3f}, AUC: {metrics['auc']:.3f}"
    )
    return result


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
    """Train CCS probes with steering applied before training.

    CHANGED: Added comprehensive progress tracking for the entire experiment.
    """
    logger = setup_logging(run_dir)
    logger.info("Starting training CCS probes with steering strategies")

    # Default values
    if steering_coefficients is None:
        steering_coefficients = [0.0, 0.5, 1.0, 2.0, 5.0]

    if embedding_strategies is None:
        embedding_strategies = ["last-token", "first-token", "mean"]

    # Create directory structure
    plot_dir = os.path.join(run_dir, "plots")
    table_dir = os.path.join(run_dir, "tables")
    model_dir = os.path.join(run_dir, "models")

    for directory in [plot_dir, table_dir, model_dir]:
        os.makedirs(directory, exist_ok=True)

    # Define the data pair types to analyze
    data_pair_types = [
        ("combined", "Hate + Safe No → Safe + Hate No"),
        ("hate_yes_to_safe_yes", "Hate Yes → Safe Yes"),
        ("safe_no_to_hate_no", "Safe No → Hate No"),
        ("hate_yes_to_hate_no", "Hate Yes → Hate No"),
        ("safe_yes_to_safe_no", "Safe Yes → Safe No"),
    ]

    print("\n🚀 PHASE 1: Extracting representations from model...")
    # Extract all representations from the model
    representations = extract_representations_for_all_strategies(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        n_layers=n_layers,
        embedding_strategies=embedding_strategies,
        device=device,
    )

    print("\n📐 PHASE 2: Calculating steering vectors...")
    # Calculate all steering vectors
    steering_vectors = calculate_steering_vectors_for_all_strategies(representations)

    print("\n🧠 PHASE 3: Training CCS probes for all combinations...")
    # Train CCS probes for all combinations
    all_results = {}
    ccs_models = {}  # Store trained CCS models for visualization

    # Calculate total training tasks
    total_tasks = (
        len(embedding_strategies)
        * len(data_pair_types)
        * n_layers
        * len(steering_coefficients)
    )

    with tqdm(total=total_tasks, desc="Training CCS probes", unit="task") as pbar:
        for strategy in embedding_strategies:
            logger.info(f"Training CCS probes for strategy: {strategy}")
            all_results[strategy] = {}
            ccs_models[strategy] = {}

            for pair_name, pair_description in data_pair_types:
                logger.info(f"Processing pair type: {pair_name}")
                all_results[strategy][pair_name] = {}

                for layer_idx in range(n_layers):
                    logger.info(f"Processing layer {layer_idx}")
                    all_results[strategy][pair_name][layer_idx] = {}

                    # Get representations for this strategy and layer
                    layer_representations = representations[strategy][layer_idx]

                    for coef in steering_coefficients:
                        pbar.set_postfix(
                            {
                                "strategy": strategy,
                                "pair": pair_name[:15] + "..."
                                if len(pair_name) > 15
                                else pair_name,
                                "layer": layer_idx,
                                "coef": coef,
                            }
                        )

                        logger.debug(f"Training with coefficient {coef}")

                        # Train CCS probe for this combination
                        ccs_result = train_single_ccs_probe(
                            representations=layer_representations,
                            pair_type=pair_name,
                            steering_coefficient=coef,
                            n_epochs=n_epochs,
                            learning_rate=learning_rate,
                            device=device,
                        )

                        all_results[strategy][pair_name][layer_idx][coef] = {
                            "metrics": ccs_result["metrics"],
                            "ccs": ccs_result["ccs"],
                        }

                        # Store CCS model for visualization (use coefficient 0.0 as baseline)
                        if coef == 0.0:
                            ccs_models[strategy][layer_idx] = ccs_result["ccs"]

                        logger.debug(
                            f"Completed training for {strategy}/{pair_name}/layer_{layer_idx}/coef_{coef}"
                        )

                        pbar.update(1)

    print("\n🎨 PHASE 4: Generating visualizations...")
    # Generate visualizations with the extracted data
    generate_visualizations(
        representations=representations,
        steering_vectors=steering_vectors,
        ccs_models=ccs_models,
        all_results=all_results,
        plot_dir=plot_dir,
        embedding_strategies=embedding_strategies,
        data_pair_types=data_pair_types,
        steering_coefficients=steering_coefficients,
        n_layers=n_layers,
    )

    print("\n📊 PHASE 5: Generating result tables...")
    # Generate tables with the results
    generate_result_tables(
        all_results,
        table_dir,
        embedding_strategies,
        data_pair_types,
        steering_coefficients,
        n_layers,
    )

    print("\n✅ Training and analysis completed successfully!")
    logger.info("Training and analysis completed successfully")
    return all_results


def generate_visualizations(
    representations,
    steering_vectors,
    ccs_models,
    all_results,
    plot_dir,
    embedding_strategies,
    data_pair_types,
    steering_coefficients,
    n_layers,
):
    """Create visualizations from the extracted data.

    CHANGED: Removed all validation, only use proper data structure.

    Args:
        representations: Dict[strategy][layer_idx] -> Dict[data_type] -> numpy array
        steering_vectors: Dict[strategy][layer_idx] -> Dict[vector_type] -> {"vector": array, "color": str, "label": str}
        ccs_models: Dict[strategy][layer_idx] -> CCS model
        all_results: Dictionary of results
        plot_dir: Directory to save plots
        embedding_strategies: List of embedding strategies
        data_pair_types: List of data pair types
        steering_coefficients: List of steering coefficients
        n_layers: Number of layers
    """
    logger = logging.getLogger(__name__)

    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Generate coefficient sweep comparison
    logger.info("Generating coefficient sweep comparison plot...")
    metrics_for_sweep = ["accuracy", "silhouette", "auc", "class_separability"]

    # Convert the nested all_results structure into a flat list for the plotting function
    flattened_results = []
    for layer_idx in range(n_layers):
        # Create a data point using our custom class
        layer_result = PlotDataPoint(layer_idx)

        # Use the first available strategy and pair
        for strategy in embedding_strategies:
            if strategy in all_results:
                for pair_name, _ in data_pair_types:
                    if (
                        pair_name in all_results[strategy]
                        and layer_idx in all_results[strategy][pair_name]
                    ):
                        for coef in steering_coefficients:
                            if coef in all_results[strategy][pair_name][layer_idx]:
                                metrics = all_results[strategy][pair_name][layer_idx][
                                    coef
                                ].get("metrics", {})
                                layer_result[f"coef_{coef}"] = metrics

                        # Only need data from one strategy/pair combination
                        if any(
                            f"coef_{coef}" in layer_result.data
                            for coef in steering_coefficients
                        ):
                            break
                if any(
                    f"coef_{coef}" in layer_result.data
                    for coef in steering_coefficients
                ):
                    break

        flattened_results.append(layer_result.to_dict())

    sweep_path = os.path.join(plot_dir, "coefficient_sweep_comparison.png")
    plot_coefficient_sweep_lines_comparison(
        results=flattened_results, metrics=metrics_for_sweep, save_path=sweep_path
    )
    logger.info(f"Saved coefficient sweep comparison plot to {sweep_path}")

    # 2. Generate performance plots for each strategy and pair type
    logger.info("Generating individual performance plots...")
    metrics_to_plot = ["accuracy", "silhouette", "auc", "class_separability"]

    for strategy in embedding_strategies:
        if strategy not in all_results:
            continue

        for pair_name, pair_description in data_pair_types:
            if pair_name not in all_results[strategy]:
                continue

            # Create a directory for this combination
            strategy_pair_dir = os.path.join(plot_dir, f"{strategy}_{pair_name}")
            os.makedirs(strategy_pair_dir, exist_ok=True)

            for metric in metrics_to_plot:
                # Format data for plotting
                plot_data = []
                for layer_idx in range(n_layers):
                    if layer_idx not in all_results[strategy][pair_name]:
                        continue

                    # Create a data point using our custom class
                    data_point = PlotDataPoint(layer_idx)

                    # Add baseline data (coef=0.0)
                    if 0.0 in all_results[strategy][pair_name][layer_idx]:
                        metrics_dict = all_results[strategy][pair_name][layer_idx][
                            0.0
                        ].get("metrics", {})
                        if metric in metrics_dict:
                            data_point["final_metrics"] = {
                                "base_metrics": {metric: metrics_dict[metric]}
                            }

                    # Add data for each coefficient
                    for coef in steering_coefficients:
                        if coef in all_results[strategy][pair_name][layer_idx]:
                            metrics_dict = all_results[strategy][pair_name][layer_idx][
                                coef
                            ].get("metrics", {})
                            if metric in metrics_dict:
                                data_point[f"coef_{coef}"] = {
                                    metric: metrics_dict[metric]
                                }

                    plot_data.append(data_point.to_dict())

                # Create plot
                save_path = os.path.join(strategy_pair_dir, f"performance_{metric}.png")
                plot_performance_across_layers(
                    results=plot_data, metric=metric, save_path=save_path
                )
                logger.info(f"Saved performance plot: {save_path}")

    # 3. Generate all strategies all steering vectors plots for each layer
    # logger.info("Generating all strategies all steering vectors plots...")
    # for layer_idx in range(n_layers):
    #     # Prepare data for this layer - get data from actual structure
    #     layer_representations = {}
    #     layer_steering_vectors = {}

    #     for strategy in embedding_strategies:
    #         if strategy in representations and layer_idx in representations[strategy]:
    #             layer_representations[strategy] = representations[strategy][layer_idx]

    #         if strategy in steering_vectors and layer_idx in steering_vectors[strategy]:
    #             layer_steering_vectors[strategy] = steering_vectors[strategy][layer_idx]

    #     # Only plot if we have data for at least one strategy
    #     if layer_representations and layer_steering_vectors:
    #         plot_path = plot_all_strategies_all_steering_vectors(
    #             plot_dir=plot_dir,
    #             layer_idx=layer_idx,
    #             representations=layer_representations,
    #             all_steering_vectors_by_strategy=layer_steering_vectors,
    #         )

    #         if plot_path:
    #             logger.info(
    #                 f"Saved all strategies steering vectors plot for layer {layer_idx}: {plot_path}"
    #             )
    #         else:
    #             logger.warning(
    #                 f"Failed to create steering vectors plot for layer {layer_idx}"
    #             )
    #     else:
    #         logger.warning(f"No data for layer {layer_idx} steering vectors plot")

    # 4. Generate layer vectors plot
    logger.info("Generating layer vectors plot...")
    # Create layer data compatible with plot_all_layer_vectors
    layer_vectors_data = []

    for layer_idx in range(n_layers):
        # Use first available strategy
        first_strategy = embedding_strategies[0]

        if (
            first_strategy in representations
            and layer_idx in representations[first_strategy]
            and first_strategy in steering_vectors
            and layer_idx in steering_vectors[first_strategy]
        ):
            layer_reps = representations[first_strategy][layer_idx]
            layer_steering = steering_vectors[first_strategy][layer_idx]

            # Calculate mean vectors for hate and safe
            hate_combined = np.vstack([layer_reps["hate_yes"], layer_reps["safe_no"]])
            safe_combined = np.vstack([layer_reps["safe_yes"], layer_reps["hate_no"]])

            hate_mean = np.mean(hate_combined, axis=0)
            safe_mean = np.mean(safe_combined, axis=0)

            # Use combined steering vector
            if "combined" in layer_steering:
                steering_vector = layer_steering["combined"]["vector"]
            else:
                # Use first available steering vector
                steering_vector = list(layer_steering.values())[0]["vector"]

            layer_data = {
                "hate_mean_vector": hate_mean,
                "safe_mean_vector": safe_mean,
                "steering_vector": steering_vector,
                "layer_idx": layer_idx,
            }

            layer_vectors_data.append(layer_data)

    if layer_vectors_data:
        vectors_path = plot_all_layer_vectors(
            results=layer_vectors_data, save_dir=plot_dir
        )
        if vectors_path:
            logger.info(f"Saved layer vectors plot: {vectors_path}")
        else:
            logger.warning("Failed to create layer vectors plot")
    else:
        logger.warning("No data available for layer vectors plot")

    # 5. Generate COMPREHENSIVE decision boundaries and data visualizations
    logger.info("Generating comprehensive visualizations with decision boundaries...")
    # Create directories for different visualization types
    decision_boundaries_dir = os.path.join(plot_dir, "all_decision_boundaries")
    steering_vectors_dir = os.path.join(plot_dir, "all_steering_vectors")
    os.makedirs(decision_boundaries_dir, exist_ok=True)
    os.makedirs(steering_vectors_dir, exist_ok=True)

    # Process each layer
    for layer_idx in range(n_layers):
        logger.info(f"Creating comprehensive visualizations for layer {layer_idx}")

        # Collect data for this layer
        layer_representations = {}
        layer_steering_vectors = {}
        layer_ccs_models = {}

        for strategy in embedding_strategies:
            # Check if we have all required data
            if (
                strategy in representations
                and layer_idx in representations[strategy]
                and strategy in steering_vectors
                and layer_idx in steering_vectors[strategy]
                and strategy in ccs_models
                and layer_idx in ccs_models[strategy]
            ):
                # Add to the dictionaries
                layer_representations[strategy] = representations[strategy][layer_idx]
                layer_steering_vectors[strategy] = steering_vectors[strategy][layer_idx]
                layer_ccs_models[strategy] = {
                    layer_idx: ccs_models[strategy][layer_idx]
                }

        # 1. Create decision boundaries visualization (if we have CCS models)
        if layer_representations and layer_steering_vectors and layer_ccs_models:
            layer_dir = decision_boundaries_dir
            os.makedirs(layer_dir, exist_ok=True)

            plot_all_decision_boundaries(
                representations=layer_representations,
                steering_vectors=layer_steering_vectors,
                ccs_models=layer_ccs_models,
                save_dir=layer_dir,
                strategies=list(layer_representations.keys()),
                max_samples_per_type=50,
            )
            logger.info(f"Created decision boundaries plot for layer {layer_idx}")

        # 2. Create comprehensive steering vectors visualization
        if layer_representations and layer_steering_vectors:
            plot_path = plot_all_strategies_all_steering_vectors(
                plot_dir=steering_vectors_dir,
                layer_idx=layer_idx,
                representations=layer_representations,
                all_steering_vectors_by_strategy=layer_steering_vectors,
            )

            if plot_path:
                logger.info(
                    f"Created steering vectors visualization for layer {layer_idx}"
                )
            else:
                logger.warning(
                    f"Failed to create steering vectors visualization for layer {layer_idx}"
                )

    logger.info("Visualization generation completed")


def generate_result_tables(
    all_results,
    table_dir,
    embedding_strategies,
    data_pair_types,
    steering_coefficients,
    n_layers,
):
    """Generate tables summarizing the results."""
    logger = logging.getLogger(__name__)
    os.makedirs(table_dir, exist_ok=True)

    # Create a comprehensive CSV
    import csv

    csv_path = os.path.join(table_dir, "all_results.csv")

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "Strategy",
            "Pair Type",
            "Layer",
            "Coefficient",
            "Accuracy",
            "Silhouette",
            "AUC",
            "Class Separability",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for strategy in embedding_strategies:
            if strategy not in all_results:
                continue

            for pair_name, pair_description in data_pair_types:
                if pair_name not in all_results[strategy]:
                    continue

                for layer_idx in range(n_layers):
                    if layer_idx not in all_results[strategy][pair_name]:
                        continue

                    for coef in steering_coefficients:
                        if coef not in all_results[strategy][pair_name][layer_idx]:
                            continue

                        metrics = all_results[strategy][pair_name][layer_idx][coef].get(
                            "metrics", {}
                        )

                        row_dict = {
                            "Strategy": str(strategy),
                            "Pair Type": str(pair_name),
                            "Layer": str(layer_idx),
                            "Coefficient": str(coef),
                            "Accuracy": str(metrics.get("accuracy", "N/A")),
                            "Silhouette": str(metrics.get("silhouette", "N/A")),
                            "AUC": str(metrics.get("auc", "N/A")),
                            "Class Separability": str(
                                metrics.get("class_separability", "N/A")
                            ),
                        }

                        writer.writerow(row_dict)

    logger.info(f"Saved comprehensive results to {csv_path}")

    logger.info("Table generation completed")


def analyze_steering_experiment_results(all_results, steering_coefficients, n_layers):
    """
    Analyze the results of the steering experiment.

    Args:
        all_results: Results from train_ccs_with_steering_strategies
        steering_coefficients: List of steering coefficients used
        n_layers: Number of layers analyzed
    """
    print("\n" + "=" * 80)
    print("STEERING EXPERIMENT ANALYSIS")
    print("=" * 80)

    # Analyze each layer
    for layer_idx in range(n_layers):
        print(f"\nLayer {layer_idx} Analysis:")
        print("-" * 40)

        # Get baseline accuracy (steering_coefficient = 0.0)
        baseline_acc = None
        steering_accs = {}

        # Extract accuracies for this layer across strategies and pairs
        for strategy in all_results:
            for pair_name in all_results[strategy]:
                if layer_idx in all_results[strategy][pair_name]:
                    for coef in steering_coefficients:
                        if coef in all_results[strategy][pair_name][layer_idx]:
                            acc = all_results[strategy][pair_name][layer_idx][coef][
                                "metrics"
                            ].get("accuracy", 0)

                            if coef == 0.0:
                                if baseline_acc is None:
                                    baseline_acc = acc
                                else:
                                    baseline_acc = (
                                        baseline_acc + acc
                                    ) / 2  # Average across strategies/pairs
                            else:
                                if coef not in steering_accs:
                                    steering_accs[coef] = []
                                steering_accs[coef].append(acc)

        # Print analysis for this layer
        if baseline_acc is not None:
            print(f"  Baseline accuracy (no steering): {baseline_acc:.3f}")

            for coef in sorted(steering_accs.keys()):
                if steering_accs[coef]:
                    avg_acc = np.mean(steering_accs[coef])
                    accuracy_change = avg_acc - baseline_acc
                    print(
                        f"  Steering {coef:4.1f}: {avg_acc:.3f} (change: {accuracy_change:+.3f})"
                    )

                    # Interpretation
                    if accuracy_change > 0.05:
                        print("    → Steering IMPROVES truth detection!")
                    elif accuracy_change > -0.05:
                        print("    → Steering preserves truth detection")
                    elif accuracy_change > -0.15:
                        print("    → Steering somewhat disrupts truth detection")
                    else:
                        print("    → Steering severely disrupts truth detection")
        else:
            print(f"  No data available for layer {layer_idx}")

    print("\n" + "=" * 80)
    print("OVERALL CONCLUSIONS:")
    print("=" * 80)
    print("• If accuracy stays high: Steering preserves truthfulness structure")
    print("• If accuracy drops: Steering disrupts logical reasoning")
    print("• If accuracy improves: Steering actually helps (very interesting!)")
    print("• Compare across layers to see where steering helps/hurts most")


if __name__ == "__main__":
    print(
        "This module should be imported and used with the appropriate dataloader and model."
    )
    print("Example usage:")
    print("from tr_training_w_steering import train_ccs_with_steering_strategies")
    print(
        "results = train_ccs_with_steering_strategies(model, tokenizer, train_dataloader, val_dataloader, test_dataloader, run_dir)"
    )
