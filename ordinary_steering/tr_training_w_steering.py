import logging
import os
from datetime import datetime

import matplotlib
import numpy as np
import torch

# Set matplotlib backend to Agg (non-interactive)
matplotlib.use("Agg")

# Import modules
from ordinary_steering_metrics import plot_coefficient_sweep_lines_comparison
from tr_plotting import (
    plot_all_decision_boundaries,
    plot_all_layer_vectors,
    plot_all_strategies_all_steering_vectors,
    plot_performance_across_layers,
    visualize_decision_boundary,
)


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

    This function differs from the original train_ccs_with_steering by:
    1. Applying steering BEFORE training each CCS probe (not just during evaluation)
    2. Training separate probes for each (layer, coefficient, strategy, data_pair) combination
    3. Generating more detailed metrics and visualizations

    Args:
        model: Model to extract representations
        tokenizer: Tokenizer for model
        train_dataloader: DataLoader for training
        val_dataloader: DataLoader for validation
        test_dataloader: DataLoader for testing
        run_dir: Directory to save results
        n_layers: Number of layers to process
        n_epochs: Number of epochs to train each CCS
        learning_rate: Learning rate for training
        device: Device to use
        steering_coefficients: List of steering coefficients to use
        embedding_strategies: List of embedding strategies to use

    Returns:
        Dictionary of results for all combinations
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

    # Create data structures to store results
    all_results = {}
    for strategy in embedding_strategies:
        all_results[strategy] = {}
        for pair_name, _ in data_pair_types:
            all_results[strategy][pair_name] = {}
            for layer_idx in range(n_layers):
                all_results[strategy][pair_name][layer_idx] = {}
                for coef in steering_coefficients:
                    all_results[strategy][pair_name][layer_idx][coef] = {
                        "metrics": {},
                        "ccs": None,
                    }

    # Training and evaluation code goes here
    # ...

    # Generate tables with the results
    generate_result_tables(
        all_results,
        table_dir,
        embedding_strategies,
        data_pair_types,
        steering_coefficients,
        n_layers,
    )

    return all_results


def generate_visualizations(
    all_results,
    plot_dir,
    embedding_strategies,
    data_pair_types,
    steering_coefficients,
    n_layers,
    model=None,
    tokenizer=None,
    train_dataloader=None,
    device=None,
):
    """Create visualizations from the results.

    Args:
        all_results: Dictionary of results with structure all_results[strategy][pair_name][layer_idx][coef]
        plot_dir: Directory to save plots
        embedding_strategies: List of embedding strategies
        data_pair_types: List of data pair types as (name, description) tuples
        steering_coefficients: List of steering coefficients
        n_layers: Number of layers
        model: The model to extract representations from
        tokenizer: The tokenizer for the model
        train_dataloader: DataLoader containing the training data
        device: Device to use for computation
    """
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Plot metrics across layers for each strategy and pair type
    metrics_to_plot = ["accuracy", "silhouette", "auc", "class_separability"]

    # Initialize data structures
    all_layer_data = []
    all_strategy_data = {}
    all_steering_vectors = {}

    # Generate individual plots for each strategy/pair/metric combination
    for strategy in embedding_strategies:
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

                    # Create layer data dictionary with proper typing
                    layer_data = {}
                    layer_data["layer_idx"] = layer_idx

                    # Add baseline data (coef=0.0)
                    if 0.0 in all_results[strategy][pair_name][layer_idx]:
                        if (
                            "metrics"
                            in all_results[strategy][pair_name][layer_idx][0.0]
                        ):
                            metrics_dict = all_results[strategy][pair_name][layer_idx][
                                0.0
                            ]["metrics"]
                            if metric in metrics_dict:
                                metric_value = metrics_dict[metric]

                                # Create the nested structure explicitly
                                base_metrics = {}
                                base_metrics[metric] = metric_value

                                final_metrics = {}
                                final_metrics["base_metrics"] = base_metrics

                                layer_data["final_metrics"] = final_metrics

                    # Add data for each coefficient
                    for coef in steering_coefficients:
                        if coef in all_results[strategy][pair_name][layer_idx]:
                            if (
                                "metrics"
                                in all_results[strategy][pair_name][layer_idx][coef]
                            ):
                                metrics_dict = all_results[strategy][pair_name][
                                    layer_idx
                                ][coef]["metrics"]
                                if metric in metrics_dict:
                                    metric_value = metrics_dict[metric]

                                    # Create coefficient data explicitly
                                    coef_data = {}
                                    coef_data[metric] = metric_value

                                    layer_data[f"coef_{coef}"] = coef_data

                    plot_data.append(layer_data)

                # Create plot
                save_path = os.path.join(strategy_pair_dir, f"performance_{metric}.png")
                plot_performance_across_layers(
                    results=plot_data, metric=metric, save_path=save_path
                )

    # Create comparison plots for different strategies
    for pair_name, pair_description in data_pair_types:
        pair_dir = os.path.join(plot_dir, f"comparison_{pair_name}")
        os.makedirs(pair_dir, exist_ok=True)

        for metric in metrics_to_plot:
            for coef in steering_coefficients:
                # Prepare data for comparison plot
                strategy_comparison = []

                for strategy in embedding_strategies:
                    if pair_name not in all_results[strategy]:
                        continue

                    strategy_data = {"name": strategy, "values": []}

                    for layer_idx in range(n_layers):
                        # Find metrics for this layer and coefficient
                        value = None
                        if (
                            layer_idx in all_results[strategy][pair_name]
                            and coef in all_results[strategy][pair_name][layer_idx]
                            and "metrics"
                            in all_results[strategy][pair_name][layer_idx][coef]
                            and metric
                            in all_results[strategy][pair_name][layer_idx][coef][
                                "metrics"
                            ]
                        ):
                            value = all_results[strategy][pair_name][layer_idx][coef][
                                "metrics"
                            ][metric]

                        strategy_data["values"].append(value)

                    strategy_comparison.append(strategy_data)

                # Create comparison plot
                save_path = os.path.join(
                    pair_dir, f"strategy_comparison_{metric}_coef_{coef}.png"
                )

                # Create a simple matplotlib plot with arrays
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 6))

                x = list(range(n_layers))
                for strategy_data in strategy_comparison:
                    values = strategy_data["values"]
                    plt.plot(x, values, marker="o", label=strategy_data["name"])

                plt.title(
                    f"{pair_description} - {metric.capitalize()} Comparison (Coef={coef})"
                )
                plt.xlabel("Layer")
                plt.ylabel(metric.capitalize())
                plt.legend()
                plt.grid(True)
                plt.savefig(save_path)
                plt.close()

                print(f"Saved strategy comparison plot to {save_path}")

    # Collect data for comprehensive visualizations
    print("Collecting data for comprehensive visualizations...")

    # Initialize data structures
    for layer_idx in range(n_layers):
        # Initialize layer data structures
        all_strategy_data[layer_idx] = {}
        all_steering_vectors[layer_idx] = {}

        for strategy in embedding_strategies:
            # Initialize strategy data structures
            all_strategy_data[layer_idx][strategy] = {
                "hate_yes": None,
                "hate_no": None,
                "safe_yes": None,
                "safe_no": None,
                "hate": None,
                "safe": None,
            }

            all_steering_vectors[layer_idx][strategy] = {}

            # Extract data from all_results
            for pair_name, _ in data_pair_types:
                if (
                    pair_name in all_results[strategy]
                    and layer_idx in all_results[strategy][pair_name]
                    and 0.0 in all_results[strategy][pair_name][layer_idx]
                ):
                    # Get steering vector if available
                    if (
                        "steering_vector"
                        in all_results[strategy][pair_name][layer_idx][0.0]
                    ):
                        all_steering_vectors[layer_idx][strategy][pair_name] = {
                            "vector": all_results[strategy][pair_name][layer_idx][0.0][
                                "steering_vector"
                            ],
                            "color": {
                                "combined": "#00FF00",  # Green
                                "hate_yes_to_safe_yes": "#FF00FF",  # Purple
                                "safe_no_to_hate_no": "#FFFF00",  # Yellow
                                "hate_yes_to_hate_no": "#FF9900",  # Orange
                                "safe_yes_to_safe_no": "#00FFCC",  # Teal
                            }.get(pair_name, "#00FF00"),
                            "label": {
                                "combined": "Combined Steering Vector",
                                "hate_yes_to_safe_yes": "Hate Yes → Safe Yes",
                                "safe_no_to_hate_no": "Safe No → Hate No",
                                "hate_yes_to_hate_no": "Hate Yes → Hate No",
                                "safe_yes_to_safe_no": "Safe Yes → Safe No",
                            }.get(pair_name, pair_name),
                        }

                    # Get source and target vectors based on pair type
                    if pair_name == "hate_yes_to_safe_yes":
                        if (
                            "source_vectors"
                            in all_results[strategy][pair_name][layer_idx][0.0]
                        ):
                            all_strategy_data[layer_idx][strategy]["hate_yes"] = (
                                all_results[
                                    strategy
                                ][pair_name][layer_idx][0.0]["source_vectors"]
                            )
                        if (
                            "target_vectors"
                            in all_results[strategy][pair_name][layer_idx][0.0]
                        ):
                            all_strategy_data[layer_idx][strategy]["safe_yes"] = (
                                all_results[
                                    strategy
                                ][pair_name][layer_idx][0.0]["target_vectors"]
                            )

                    elif pair_name == "safe_no_to_hate_no":
                        if (
                            "source_vectors"
                            in all_results[strategy][pair_name][layer_idx][0.0]
                        ):
                            all_strategy_data[layer_idx][strategy]["safe_no"] = (
                                all_results[
                                    strategy
                                ][pair_name][layer_idx][0.0]["source_vectors"]
                            )
                        if (
                            "target_vectors"
                            in all_results[strategy][pair_name][layer_idx][0.0]
                        ):
                            all_strategy_data[layer_idx][strategy]["hate_no"] = (
                                all_results[
                                    strategy
                                ][pair_name][layer_idx][0.0]["target_vectors"]
                            )

            # Check if we have all required data and create dummy data if needed
            for key in ["hate_yes", "hate_no", "safe_yes", "safe_no"]:
                if all_strategy_data[layer_idx][strategy][key] is None:
                    print(
                        f"Warning: {key} vectors missing for strategy {strategy} at layer {layer_idx}, extracting from model"
                    )

                    # Extract real representations from model if available
                    if (
                        model is not None
                        and tokenizer is not None
                        and train_dataloader is not None
                        and device is not None
                    ):
                        # Get text samples for the specific category
                        if key == "hate_yes":
                            texts = train_dataloader.dataset.get_by_type("hate_yes")
                        elif key == "hate_no":
                            texts = train_dataloader.dataset.get_by_type("hate_no")
                        elif key == "safe_yes":
                            texts = train_dataloader.dataset.get_by_type("safe_yes")
                        elif key == "safe_no":
                            texts = train_dataloader.dataset.get_by_type("safe_no")
                        else:
                            texts = []

                        # Limit to a reasonable number of samples to avoid memory issues
                        texts = texts[:100] if len(texts) > 100 else texts

                        if len(texts) > 0:
                            # Extract representations
                            vectors = []
                            model.eval()
                            with torch.no_grad():
                                for text in texts:
                                    inputs = tokenizer(
                                        text,
                                        return_tensors="pt",
                                        truncation=True,
                                        padding=True,
                                    )
                                    inputs = {
                                        k: v.to(device) for k, v in inputs.items()
                                    }
                                    outputs = model(**inputs, output_hidden_states=True)
                                    hidden_states = outputs.hidden_states

                                    # Extract representation based on strategy
                                    if strategy == "last-token":
                                        # Get last token representation for each sequence
                                        token_embeddings = hidden_states[layer_idx + 1]
                                        last_token_idx = (
                                            inputs["attention_mask"].sum(dim=1) - 1
                                        )
                                        batch_size = token_embeddings.shape[0]
                                        vector = torch.stack(
                                            [
                                                token_embeddings[
                                                    i, last_token_idx[i], :
                                                ]
                                                for i in range(batch_size)
                                            ]
                                        )
                                    elif strategy == "first-token":
                                        # Get first token (CLS) representation
                                        vector = hidden_states[layer_idx + 1][:, 0, :]
                                    elif strategy == "mean":
                                        # Get mean of all token representations
                                        token_embeddings = hidden_states[layer_idx + 1]
                                        mask = (
                                            inputs["attention_mask"]
                                            .unsqueeze(-1)
                                            .expand(token_embeddings.size())
                                            .float()
                                        )
                                        vector = torch.sum(
                                            token_embeddings * mask, 1
                                        ) / torch.clamp(mask.sum(1), min=1e-9)
                                    else:
                                        # Default to mean pooling
                                        token_embeddings = hidden_states[layer_idx + 1]
                                        mask = (
                                            inputs["attention_mask"]
                                            .unsqueeze(-1)
                                            .expand(token_embeddings.size())
                                            .float()
                                        )
                                        vector = torch.sum(
                                            token_embeddings * mask, 1
                                        ) / torch.clamp(mask.sum(1), min=1e-9)

                                    vectors.append(vector.cpu().numpy())

                            # Concatenate all vectors
                            if vectors:
                                all_strategy_data[layer_idx][strategy][key] = np.vstack(
                                    vectors
                                )
                                continue

                    # Fall back to random vectors if extraction failed
                    print(
                        f"Warning: Could not extract real representations for {key}, using random data"
                    )
                    all_strategy_data[layer_idx][strategy][key] = np.random.randn(
                        100, 768
                    )

            # Create combined hate and safe vectors
            all_strategy_data[layer_idx][strategy]["hate"] = np.vstack(
                [
                    all_strategy_data[layer_idx][strategy]["hate_yes"],
                    all_strategy_data[layer_idx][strategy]["safe_no"],
                ]
            )

            all_strategy_data[layer_idx][strategy]["safe"] = np.vstack(
                [
                    all_strategy_data[layer_idx][strategy]["safe_yes"],
                    all_strategy_data[layer_idx][strategy]["hate_no"],
                ]
            )

            # Ensure we have all required steering vectors
            for pair_name, _ in data_pair_types:
                if pair_name not in all_steering_vectors[layer_idx][strategy]:
                    print(
                        f"Warning: steering vector for {pair_name} missing for strategy {strategy} at layer {layer_idx}, calculating from data"
                    )

                    # Calculate steering vector from data if possible
                    if (
                        pair_name == "hate_yes_to_safe_yes"
                        and "hate_yes" in all_strategy_data[layer_idx][strategy]
                        and "safe_yes" in all_strategy_data[layer_idx][strategy]
                    ):
                        hate_yes_mean = np.mean(
                            all_strategy_data[layer_idx][strategy]["hate_yes"], axis=0
                        )
                        safe_yes_mean = np.mean(
                            all_strategy_data[layer_idx][strategy]["safe_yes"], axis=0
                        )
                        steering_vector = safe_yes_mean - hate_yes_mean
                    elif (
                        pair_name == "safe_no_to_hate_no"
                        and "safe_no" in all_strategy_data[layer_idx][strategy]
                        and "hate_no" in all_strategy_data[layer_idx][strategy]
                    ):
                        safe_no_mean = np.mean(
                            all_strategy_data[layer_idx][strategy]["safe_no"], axis=0
                        )
                        hate_no_mean = np.mean(
                            all_strategy_data[layer_idx][strategy]["hate_no"], axis=0
                        )
                        steering_vector = hate_no_mean - safe_no_mean
                    elif (
                        pair_name == "hate_yes_to_hate_no"
                        and "hate_yes" in all_strategy_data[layer_idx][strategy]
                        and "hate_no" in all_strategy_data[layer_idx][strategy]
                    ):
                        hate_yes_mean = np.mean(
                            all_strategy_data[layer_idx][strategy]["hate_yes"], axis=0
                        )
                        hate_no_mean = np.mean(
                            all_strategy_data[layer_idx][strategy]["hate_no"], axis=0
                        )
                        steering_vector = hate_no_mean - hate_yes_mean
                    elif (
                        pair_name == "safe_yes_to_safe_no"
                        and "safe_yes" in all_strategy_data[layer_idx][strategy]
                        and "safe_no" in all_strategy_data[layer_idx][strategy]
                    ):
                        safe_yes_mean = np.mean(
                            all_strategy_data[layer_idx][strategy]["safe_yes"], axis=0
                        )
                        safe_no_mean = np.mean(
                            all_strategy_data[layer_idx][strategy]["safe_no"], axis=0
                        )
                        steering_vector = safe_no_mean - safe_yes_mean
                    elif pair_name == "combined":
                        # Combined vector is the average of all other vectors
                        combined_vectors = []
                        for p in ["hate_yes_to_safe_yes", "safe_no_to_hate_no"]:
                            if p in all_steering_vectors[layer_idx][strategy]:
                                combined_vectors.append(
                                    all_steering_vectors[layer_idx][strategy][p][
                                        "vector"
                                    ]
                                )

                        if combined_vectors:
                            steering_vector = np.mean(combined_vectors, axis=0)
                        else:
                            # Calculate from hate and safe means
                            if (
                                "hate" in all_strategy_data[layer_idx][strategy]
                                and "safe" in all_strategy_data[layer_idx][strategy]
                            ):
                                hate_mean = np.mean(
                                    all_strategy_data[layer_idx][strategy]["hate"],
                                    axis=0,
                                )
                                safe_mean = np.mean(
                                    all_strategy_data[layer_idx][strategy]["safe"],
                                    axis=0,
                                )
                                steering_vector = safe_mean - hate_mean
                            else:
                                # Fall back to random vector
                                steering_vector = np.random.randn(768)
                    else:
                        # Fall back to random vector
                        steering_vector = np.random.randn(768)

                    # Normalize the steering vector
                    norm = np.linalg.norm(steering_vector)
                    if norm > 1e-10:
                        steering_vector = steering_vector / norm

                    all_steering_vectors[layer_idx][strategy][pair_name] = {
                        "vector": steering_vector,
                        "color": {
                            "combined": "#00FF00",  # Green
                            "hate_yes_to_safe_yes": "#FF00FF",  # Purple
                            "safe_no_to_hate_no": "#FFFF00",  # Yellow
                            "hate_yes_to_hate_no": "#FF9900",  # Orange
                            "safe_yes_to_safe_no": "#00FFCC",  # Teal
                        }.get(pair_name, "#00FF00"),
                        "label": {
                            "combined": "Combined Steering Vector",
                            "hate_yes_to_safe_yes": "Hate Yes → Safe Yes",
                            "safe_no_to_hate_no": "Safe No → Hate No",
                            "hate_yes_to_hate_no": "Hate Yes → Hate No",
                            "safe_yes_to_safe_no": "Safe Yes → Safe No",
                        }.get(pair_name, pair_name),
                    }

    # Populate all_layer_data for decision boundary plots
    for layer_idx in range(n_layers):
        # Use the first strategy as the default
        if embedding_strategies and layer_idx in all_strategy_data:
            default_strategy = embedding_strategies[0]
            if default_strategy in all_strategy_data[layer_idx]:
                strategy_data = all_strategy_data[layer_idx][default_strategy]

                # Create layer data
                layer_data = {
                    "layer_idx": layer_idx,
                    "hate_vectors": strategy_data["hate"],
                    "safe_vectors": strategy_data["safe"],
                    "hate_mean_vector": np.mean(strategy_data["hate"], axis=0),
                    "safe_mean_vector": np.mean(strategy_data["safe"], axis=0),
                }

                # Add steering vector if available
                if (
                    default_strategy in all_steering_vectors[layer_idx]
                    and "combined" in all_steering_vectors[layer_idx][default_strategy]
                ):
                    layer_data["steering_vector"] = all_steering_vectors[layer_idx][
                        default_strategy
                    ]["combined"]["vector"]
                else:
                    # Create a dummy steering vector as the difference between hate and safe means
                    layer_data["steering_vector"] = (
                        layer_data["safe_mean_vector"] - layer_data["hate_mean_vector"]
                    )

                all_layer_data.append(layer_data)

    # Generate comprehensive plots
    # 1. Generate coefficient sweep comparison
    print("Generating coefficient sweep comparison plot...")
    metrics_for_sweep = ["accuracy", "silhouette", "auc", "class_separability"]
    available_metrics = []

    # Check which metrics are available in the results
    for strategy in embedding_strategies:
        for pair_name in all_results[strategy]:
            for layer_idx in range(n_layers):
                if layer_idx in all_results[strategy][pair_name]:
                    for coef in steering_coefficients:
                        if coef in all_results[strategy][pair_name][layer_idx]:
                            metrics_dict = all_results[strategy][pair_name][layer_idx][
                                coef
                            ].get("metrics", {})
                            for metric in metrics_for_sweep:
                                if (
                                    metric in metrics_dict
                                    and metric not in available_metrics
                                ):
                                    available_metrics.append(metric)

    if available_metrics:
        # Convert the nested all_results structure into a flat list for the plotting function
        flattened_results = []

        for layer_idx in range(n_layers):
            layer_result = {"layer_idx": layer_idx}

            # Use the first available strategy and pair
            for strategy in embedding_strategies:
                for pair_name in all_results[strategy]:
                    if layer_idx in all_results[strategy][pair_name]:
                        for coef in steering_coefficients:
                            if coef in all_results[strategy][pair_name][layer_idx]:
                                metrics = all_results[strategy][pair_name][layer_idx][
                                    coef
                                ].get("metrics", {})
                                layer_result[f"coef_{coef}"] = metrics

                        # Only need data from one strategy/pair combination
                        if any(
                            f"coef_{coef}" in layer_result
                            for coef in steering_coefficients
                        ):
                            break

                # Break if we found data for this layer
                if any(
                    f"coef_{coef}" in layer_result for coef in steering_coefficients
                ):
                    break

            flattened_results.append(layer_result)

        sweep_path = os.path.join(plot_dir, "coefficient_sweep_comparison.png")
        plot_coefficient_sweep_lines_comparison(
            results=flattened_results, metrics=available_metrics, save_path=sweep_path
        )
        print(f"Saved coefficient sweep comparison plot to {sweep_path}")

    # 2. Generate all layer vectors plot
    print("Generating all layer vectors plot...")
    if all_layer_data:
        vectors_path = plot_all_layer_vectors(results=all_layer_data, save_dir=plot_dir)
        if vectors_path:
            print(f"Saved all layer vectors plot to {vectors_path}")
        else:
            print("Failed to create all layer vectors plot")
    else:
        print("Skipping all layer vectors plot due to missing data")

    # 3. Generate all decision boundaries plot
    print("Generating all decision boundaries plot...")
    if all_layer_data:
        boundaries_path = plot_all_decision_boundaries(
            layers_data=all_layer_data,
            log_base=os.path.join(plot_dir, "all_decision_boundaries"),
        )
        if boundaries_path:
            print(f"Saved all decision boundaries plot to {boundaries_path}")
        else:
            print("Failed to create all decision boundaries plot")

        # 3.1 Generate individual layer decision boundary plots
        for i, layer_data in enumerate(all_layer_data):
            if (
                "hate_vectors" in layer_data
                and "safe_vectors" in layer_data
                and "steering_vector" in layer_data
            ):
                # Create a simple CCS-like object for visualization
                class DummyCCS:
                    def predict_from_vectors(self, vectors):
                        # Simple linear classifier using steering vector
                        steering = layer_data["steering_vector"]
                        # Project vectors onto steering direction
                        projections = np.dot(vectors, steering)
                        # Use sigmoid to get probabilities
                        probs = 1 / (1 + np.exp(-projections))
                        # Convert to binary predictions
                        preds = (probs > 0.5).astype(int)
                        return preds, probs

                dummy_ccs = DummyCCS()

                # Generate decision boundary plot for this layer
                layer_boundary_path = os.path.join(
                    plot_dir, f"layer_{i}_decision_boundary.png"
                )
                visualize_decision_boundary(
                    ccs=dummy_ccs,
                    hate_vectors=layer_data["hate_vectors"],
                    safe_vectors=layer_data["safe_vectors"],
                    steering_vector=layer_data["steering_vector"],
                    log_base=layer_boundary_path[:-4],  # Remove .png extension
                    layer_idx=i,
                    strategy="last-token",  # Using default strategy
                )
                print(
                    f"Saved decision boundary plot for layer {i} to {layer_boundary_path}"
                )
    else:
        print("Skipping all decision boundaries plot due to missing data")

    # 4. Generate all strategies all steering vectors plots for each layer
    for layer_idx in range(n_layers):
        # Check if we have enough data for this layer
        print(
            f"Generating all strategies all steering vectors plot for layer {layer_idx}..."
        )

        # Verify we have data for this layer before plotting
        has_data_for_layer = True
        for strategy in embedding_strategies:
            if (
                layer_idx not in all_strategy_data
                or strategy not in all_strategy_data[layer_idx]
                or layer_idx not in all_steering_vectors
                or strategy not in all_steering_vectors[layer_idx]
            ):
                has_data_for_layer = False
                print(
                    f"Warning: Missing data for strategy {strategy} at layer {layer_idx}, skipping plot"
                )
                break

            # Also check if we have the necessary data in the strategy data
            if (
                "hate" not in all_strategy_data[layer_idx][strategy]
                or "safe" not in all_strategy_data[layer_idx][strategy]
            ):
                has_data_for_layer = False
                print(
                    f"Warning: Missing hate/safe data for strategy {strategy} at layer {layer_idx}, skipping plot"
                )
                break

        if has_data_for_layer:
            # Debug information
            print(
                f"All steering vectors keys: {list(all_steering_vectors[layer_idx].keys())}"
            )
            for strategy in all_steering_vectors[layer_idx]:
                print(
                    f"Data keys for strategy {strategy}: {list(all_strategy_data[layer_idx][strategy].keys())}"
                )
                for key in ["hate_yes", "hate_no", "safe_yes", "safe_no"]:
                    print(
                        f"{strategy} - {key} shape: {all_strategy_data[layer_idx][strategy][key].shape}"
                    )

            plot_path = plot_all_strategies_all_steering_vectors(
                plot_dir=plot_dir,
                layer_idx=layer_idx,
                representations=all_strategy_data[layer_idx],
                all_steering_vectors_by_strategy=all_steering_vectors[layer_idx],
            )

            if plot_path:
                print(
                    f"Saved all strategies all steering vectors plot for layer {layer_idx} to {plot_path}"
                )
            else:
                print(
                    f"Failed to create all strategies all steering vectors plot for layer {layer_idx}"
                )
        else:
            print(
                f"Skipping all strategies all steering vectors plot for layer {layer_idx} due to missing data"
            )


def generate_result_tables(
    all_results,
    table_dir,
    embedding_strategies,
    data_pair_types,
    steering_coefficients,
    n_layers,
):
    """Generate tables summarizing the results."""
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
            for pair_name, pair_description in data_pair_types:
                if pair_name in all_results[strategy]:
                    for layer_idx in range(n_layers):
                        if layer_idx in all_results[strategy][pair_name]:
                            for coef in steering_coefficients:
                                if coef in all_results[strategy][pair_name][layer_idx]:
                                    metrics = all_results[strategy][pair_name][
                                        layer_idx
                                    ][coef].get("metrics", {})

                                    # Create a new dictionary for each row with proper string values
                                    row_dict = {}
                                    row_dict["Strategy"] = str(strategy)
                                    row_dict["Pair Type"] = str(pair_name)
                                    row_dict["Layer"] = str(layer_idx)
                                    row_dict["Coefficient"] = str(coef)

                                    # Add metrics with proper conversion
                                    row_dict["Accuracy"] = str(
                                        metrics.get("accuracy", "N/A")
                                    )
                                    row_dict["Silhouette"] = str(
                                        metrics.get("silhouette", "N/A")
                                    )
                                    row_dict["AUC"] = str(metrics.get("auc", "N/A"))
                                    row_dict["Class Separability"] = str(
                                        metrics.get("class_separability", "N/A")
                                    )

                                    writer.writerow(row_dict)

    print(f"Saved comprehensive results to {csv_path}")

    # Create summary tables for each strategy/pair type
    for strategy in embedding_strategies:
        for pair_name, pair_description in data_pair_types:
            if pair_name in all_results[strategy]:
                # Create a summary table comparing coefficients
                summary_path = os.path.join(
                    table_dir, f"summary_{strategy}_{pair_name}.csv"
                )

                with open(summary_path, "w", newline="") as csvfile:
                    fieldnames = ["Layer"]
                    for coef in steering_coefficients:
                        fieldnames.append(f"Accuracy (Coef={coef})")

                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for layer_idx in range(n_layers):
                        if layer_idx in all_results[strategy][pair_name]:
                            # Create a new dictionary for each row
                            row_dict = {}
                            row_dict["Layer"] = str(layer_idx)

                            for coef in steering_coefficients:
                                if (
                                    coef in all_results[strategy][pair_name][layer_idx]
                                    and "metrics"
                                    in all_results[strategy][pair_name][layer_idx][coef]
                                    and "accuracy"
                                    in all_results[strategy][pair_name][layer_idx][
                                        coef
                                    ]["metrics"]
                                ):
                                    accuracy = all_results[strategy][pair_name][
                                        layer_idx
                                    ][coef]["metrics"]["accuracy"]
                                    row_dict[f"Accuracy (Coef={coef})"] = str(accuracy)
                                else:
                                    row_dict[f"Accuracy (Coef={coef})"] = "N/A"

                            writer.writerow(row_dict)

                print(f"Saved summary for {strategy}/{pair_name} to {summary_path}")


if __name__ == "__main__":
    print(
        "This module should be imported and used with the appropriate dataloader and model."
    )
    print("Example usage:")
    print("from tr_training_w_steering import train_ccs_with_steering_strategies")
    print(
        "results = train_ccs_with_steering_strategies(model, tokenizer, train_dataloader, val_dataloader, test_dataloader, run_dir)"
    )
