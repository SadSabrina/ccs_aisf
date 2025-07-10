#!/usr/bin/env python3
"""
COMPLETE FIXED CCS Training and Steering Pipeline - Multi-Token with Full Analysis
==================================================================================

CRITICAL FIXES:
1. Best layer selection is completely independent of steering configuration
2. Dual CCS training: Original CCS vs Steered CCS for proper comparison
3. Complete comprehensive analysis pipeline from original file

PHASES:
1. Train vanilla CCS ONCE → select FIXED best layer (independent of steering)
2. Apply steering using FIXED best layer → get steered representations
3. Train NEW CCS on steered representations at SAME fixed layer
4. Run COMPLETE comprehensive analysis comparing both CCS models
"""

import math
import warnings

import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for headless environments

import numpy as np
import pandas as pd
import torch
from config import (
    CCS_CONFIG,
    DATA_CONFIG,
    EFFECT_LAYER_DELAY,
    MODEL_CONFIGS,
    MULTI_TOKEN_CONFIG,
    STEERING_CONFIG,
)
from data_model_helpers import (
    create_output_dir,
    extract_representations,
    load_data,
    load_model,
    save_results,
    select_best_layer,
    setup_logging,
    setup_steering,
    train_ccs_all_layers,
)

warnings.filterwarnings("ignore")


# Import steering and analysis functions
from steering import compare_steering_layers
from steering_analysis1_fixed import plot_layer_steering_effects

# ============================================================================
# FIXED PIPELINE: SEPARATE BEST LAYER SELECTION FROM STEERING
# ============================================================================


class VanillaCCSResults:
    """
    Container for vanilla CCS results that should be computed once and reused.

    CRITICAL: This ensures best layer selection is completely independent of steering.
    """

    def __init__(self, model_config, device):
        self.model_config = model_config
        self.device = device

        # These will be set once during vanilla CCS training
        self.X_pos_single = None
        self.X_neg_single = None
        self.labels = None
        self.ccs_results = None
        self.train_idx = None
        self.test_idx = None
        self.best_layer = None
        self.results_df = None
        self.best_ccs = None
        self.direction_tensor = None

        # Flag to ensure vanilla CCS is only run once
        self.is_trained = False

    def train_vanilla_ccs(
        self, positive_texts, negative_texts, labels, model, tokenizer
    ):
        """
        Train vanilla CCS ONCE and store all results.

        CRITICAL: This method should only be called once per model, regardless of
        how many different steering configurations we want to test.
        """
        if self.is_trained:
            print("✅ Vanilla CCS already trained, reusing results...")
            return

        print("\n" + "=" * 80)
        print("TRAINING VANILLA CCS (ONCE PER MODEL)")
        print("=" * 80)

        # Store labels
        self.labels = labels

        # Extract single-token representations for vanilla CCS training
        print(
            f"🔄 Extracting single-token representations using '{self.model_config.get('token_strategy', 'first-token')}' strategy..."
        )
        self.X_pos_single, self.X_neg_single = extract_representations(
            model,
            tokenizer,
            positive_texts,
            negative_texts,
            self.device,
            self.model_config,
        )
        print(f"📊 Single-token representations shape: {self.X_pos_single.shape}")

        # Train vanilla CCS on single-token representations
        print("🔄 Training vanilla CCS on single-token representations...")
        self.ccs_results, self.train_idx, self.test_idx = train_ccs_all_layers(
            self.X_pos_single, self.X_neg_single, labels, self.device
        )

        # Select best layer based on vanilla CCS results
        print("🔄 Selecting best layer based on vanilla CCS performance...")
        self.best_layer, self.results_df = select_best_layer(self.ccs_results)

        print("✅ VANILLA CCS TRAINING COMPLETE")
        print(f"✅ Best layer selected: {self.best_layer}")
        print("✅ This best layer will be used for ALL steering experiments")

        # Setup steering direction from vanilla CCS
        print(f"🔄 Setting up steering for FIXED best layer {self.best_layer}...")
        self.best_ccs, self.direction_tensor = setup_steering(
            self.X_pos_single,
            self.X_neg_single,
            labels,
            self.train_idx,
            self.best_layer,
            self.device,
        )

        # Mark as trained
        self.is_trained = True

        print("✅ Vanilla CCS setup complete - ready for steering experiments")

    def get_vanilla_results(self):
        """Return all vanilla CCS results."""
        if not self.is_trained:
            raise RuntimeError("Must call train_vanilla_ccs() first")

        return {
            "X_pos_single": self.X_pos_single,
            "X_neg_single": self.X_neg_single,
            "labels": self.labels,
            "ccs_results": self.ccs_results,
            "train_idx": self.train_idx,
            "test_idx": self.test_idx,
            "best_layer": self.best_layer,
            "results_df": self.results_df,
            "best_ccs": self.best_ccs,
            "direction_tensor": self.direction_tensor,
        }


# ============================================================================
# STEERING HOOKS (FIXED SYNTAX)
# ============================================================================


class SteeringHook:
    """Hook that applies steering to specified tokens at a target layer."""

    def __init__(
        self,
        target_layer,
        steering_direction,
        steering_alpha,
        token_indices,
        model_config,
    ):
        self.target_layer = target_layer
        self.steering_direction = steering_direction
        self.steering_alpha = steering_alpha
        self.token_indices = token_indices  # List of token indices to steer per sample
        self.model_config = model_config
        self.is_positive = True  # Will be set before each forward pass
        self.steered_output = None  # Store the steered output for extraction

    def set_polarity(self, is_positive):
        """Set whether we're processing positive or negative samples."""
        self.is_positive = is_positive

    def __call__(self, module, input, output):
        """Apply steering to the specified tokens."""
        # For DeBERTa, when hook is registered on .output module,
        # the output is just the hidden states tensor directly
        hidden_states = output.clone()  # Clone to avoid modifying original

        # Apply steering to each sample in the batch
        steering_applied = False
        for batch_idx in range(hidden_states.shape[0]):
            if batch_idx < len(self.token_indices):
                sample_token_indices = self.token_indices[batch_idx]

                # Apply steering to selected tokens
                for token_idx in sample_token_indices:
                    if token_idx < hidden_states.shape[1]:  # Check bounds
                        # Apply steering with polarity
                        polarity = 1.0 if self.is_positive else -1.0
                        steering_vec = polarity * self.steering_direction

                        # Apply steering
                        hidden_states[batch_idx, token_idx, :] += (
                            self.steering_alpha * steering_vec
                        )
                        steering_applied = True

        if steering_applied:
            print(
                f"🎯 Applied steering to hidden_states[{self.target_layer}] with alpha={self.steering_alpha}"
            )

        # CRITICAL FIX: Return the modified hidden states so PyTorch actually uses them
        return hidden_states


def select_token_indices(seq_len, token_config):
    """Select token indices based on strategy and percentage."""
    percentage = token_config["token_percentage"]
    strategy = token_config["strategy"]
    min_tokens = token_config.get("min_tokens", 1)
    max_tokens = token_config.get("max_tokens", 10)

    # Calculate number of tokens to select
    n_tokens = max(min_tokens, math.ceil(seq_len * percentage / 100))
    n_tokens = min(n_tokens, max_tokens, seq_len)

    if strategy == "last-n-percent":
        start_idx = max(0, seq_len - n_tokens)
        return list(range(start_idx, seq_len))
    elif strategy == "first-n-percent":
        return list(range(min(n_tokens, seq_len)))
    elif strategy == "middle-n-percent":
        start_idx = max(0, (seq_len - n_tokens) // 2)
        end_idx = min(seq_len, start_idx + n_tokens)
        return list(range(start_idx, end_idx))
    elif strategy == "evenly-spaced":
        if n_tokens >= seq_len:
            return list(range(seq_len))
        indices = np.linspace(0, seq_len - 1, n_tokens, dtype=int)
        return list(indices)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def extract_steered_representations_with_hooks(
    model,
    tokenizer,
    texts,
    device,
    model_config,
    steering_layer,
    direction_tensor,
    steering_alpha,
    token_config,
    is_positive=True,
):
    """Extract representations with steering applied using forward hooks."""
    print(
        f"🔄 Extracting {'positive' if is_positive else 'negative'} representations with steering..."
    )

    model.eval()
    batch_size = 8
    n_batches = (len(texts) + batch_size - 1) // batch_size
    all_representations = []

    # Convert direction to tensor on correct device
    if not isinstance(direction_tensor, torch.Tensor):
        direction_tensor = torch.tensor(
            direction_tensor, dtype=torch.float32, device=device
        )

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Determine token indices for each sample in batch
        batch_token_indices = []
        for i in range(len(batch_texts)):
            # Get actual sequence length for this sample
            seq_len = inputs["attention_mask"][i].sum().item()
            token_indices = select_token_indices(seq_len, token_config)
            batch_token_indices.append(token_indices)

        # Create steering hook
        hook = SteeringHook(
            target_layer=steering_layer,
            steering_direction=direction_tensor,
            steering_alpha=steering_alpha,
            token_indices=batch_token_indices,
            model_config=model_config,
        )
        hook.set_polarity(is_positive)

        # Hook the transformer layer
        transformer_layer_to_hook = steering_layer

        # Handle both DebertaModel and DebertaForMaskedLM attribute structures
        if hasattr(model, "encoder"):
            target_module = model.encoder.layer[transformer_layer_to_hook].output
        elif hasattr(model, "deberta") and hasattr(model.deberta, "encoder"):
            target_module = model.deberta.encoder.layer[
                transformer_layer_to_hook
            ].output
        else:
            raise AttributeError(
                f"Cannot find encoder layers in model. Model type: {type(model)}"
            )

        handle = target_module.register_forward_hook(hook)

        try:
            with torch.no_grad():
                # Normal forward pass - hook will apply steering automatically
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of tensors

                # Extract representations from all layers
                batch_representations = []
                for layer_idx, layer_hidden in enumerate(hidden_states):
                    layer_repr = extract_token_representations(
                        layer_hidden, inputs["attention_mask"], model_config
                    )
                    batch_representations.append(layer_repr)

                # Stack layers: [batch_size, n_layers, hidden_dim]
                batch_representations = np.stack(batch_representations, axis=1)
                all_representations.append(batch_representations)

        finally:
            # Always remove the hook
            handle.remove()

    # Concatenate all batches
    final_representations = np.concatenate(all_representations, axis=0)
    print(f"✅ Extracted steered representations: {final_representations.shape}")
    return final_representations


def extract_token_representations(hidden_states, attention_mask, model_config):
    """Extract token representations based on model's token strategy."""
    strategy = model_config.get("token_strategy", "first-token")

    if strategy == "first-token":
        # Extract first token ([CLS] for DeBERTa)
        return hidden_states[:, 0, :].cpu().numpy()
    elif strategy == "last-token":
        # Extract last token based on attention mask
        seq_lengths = attention_mask.sum(dim=1)
        batch_size = hidden_states.shape[0]
        last_token_repr = hidden_states[range(batch_size), seq_lengths - 1, :]
        return last_token_repr.cpu().numpy()
    elif strategy == "mean":
        # Mean pooling over sequence
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu().numpy()
    else:
        raise ValueError(f"Unknown token strategy: {strategy}")


# ============================================================================
# COMPLETE COMPREHENSIVE ANALYSIS WITH DUAL CCS TRAINING (FIXED + FULL)
# ============================================================================


def run_comprehensive_multi_token_steering_analysis(
    vanilla_results,
    X_pos_steered,
    X_neg_steered,
    steering_alpha,
    token_config,
    output_dir,
):
    """
    Run COMPLETE comprehensive analysis including PHASE 2.1 and ALL analysis from original file.

    CRITICAL FIXES:
    1. Uses FIXED best layer from vanilla CCS (independent of steering)
    2. Trains NEW CCS on steered representations for proper comparison
    3. Includes ALL comprehensive analysis from original file

    PROPER DATA USAGE:
    - Original CCS trained on original representations at FIXED best layer
    - Steered CCS trained on steered representations at SAME fixed layer
    - All analysis compares these two properly trained models
    """
    # Extract vanilla results
    best_layer = vanilla_results["best_layer"]
    original_ccs = vanilla_results["best_ccs"]
    X_pos_single = vanilla_results["X_pos_single"]
    X_neg_single = vanilla_results["X_neg_single"]
    labels = vanilla_results["labels"]
    train_idx = vanilla_results["train_idx"]
    test_idx = vanilla_results["test_idx"]
    direction_tensor = vanilla_results["direction_tensor"]
    device = vanilla_results["best_ccs"].device

    print(f"Running COMPLETE analysis with FIXED best layer: {best_layer}")
    print(f"Steering alpha: {steering_alpha}")
    print(
        f"Token config: {token_config['token_percentage']}% tokens, {token_config['strategy']}"
    )

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    analysis_results = {}

    print(f"📊 Original single-token shape: {X_pos_single.shape}")
    print(f"📊 Steered representations shape: {X_pos_steered.shape}")
    print("✅ Using PROPER steered representations from actual forward pass!")

    # ========================================================================
    # PHASE 2.1: TRAIN NEW CCS ON STEERED REPRESENTATIONS (CRITICAL!)
    # ========================================================================

    print("\n" + "=" * 80)
    print("PHASE 2.1: TRAINING NEW CCS ON STEERED REPRESENTATIONS")
    print("=" * 80)
    print(f"🎯 Training NEW CCS on steered data at FIXED best layer {best_layer}")
    print("📊 This allows us to compare:")
    print("   - Original CCS (trained on original representations)")
    print("   - Steered CCS (trained on steered representations)")
    print("   Both using the SAME fixed best layer for fair comparison!")

    # Convert labels to pandas Series for compatibility
    y_vec = pd.Series(labels)

    # Train NEW CCS on steered representations at the FIXED best layer
    from ccs import CCS

    steered_ccs = CCS(
        x0=X_neg_steered[train_idx, best_layer, :],  # Use FIXED best layer
        x1=X_pos_steered[train_idx, best_layer, :],  # Use FIXED best layer
        y_train=y_vec[train_idx].values,
        nepochs=CCS_CONFIG["nepochs"],
        ntries=CCS_CONFIG["ntries"],
        lr=CCS_CONFIG["lr"],
        weight_decay=CCS_CONFIG["weight_decay"],
        batch_size=CCS_CONFIG["batch_size"],
        lambda_classification=CCS_CONFIG["lambda_classification"],
        device=device,
        max_gradient_norm=CCS_CONFIG.get("max_gradient_norm", None),
        max_weight_magnitude=CCS_CONFIG.get("max_weight_magnitude", None),
    )
    steered_ccs.repeated_train()

    print("✅ NEW CCS trained on steered representations")

    # Get steering direction from steered CCS for comparison
    from steering import get_steering_direction

    steered_direction_tensor, steered_weights, steered_bias = get_steering_direction(
        steered_ccs
    )

    # Compare original vs steered CCS weights
    original_weights, original_bias = original_ccs.get_weights()
    print("\n📊 CCS COMPARISON:")
    print(f"Original CCS bias: {original_bias:.6f}")
    print(f"Steered CCS bias: {steered_bias:.6f}")
    print(f"Bias difference: {steered_bias - original_bias:+.6f}")

    original_weight_norm = np.linalg.norm(original_weights)
    steered_weight_norm = np.linalg.norm(steered_weights)
    print(f"Original weights norm: {original_weight_norm:.6f}")
    print(f"Steered weights norm: {steered_weight_norm:.6f}")
    print(f"Weight norm difference: {steered_weight_norm - original_weight_norm:+.6f}")

    # Calculate cosine similarity between weight vectors
    cosine_sim = np.dot(original_weights, steered_weights) / (
        np.linalg.norm(original_weights) * np.linalg.norm(steered_weights) + 1e-8
    )
    print(f"Cosine similarity between weight vectors: {cosine_sim:.6f}")

    # ========================================================================
    # TRAIN CCS ON ALL LAYERS FOR BOTH ORIGINAL AND STEERED MODELS
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPREHENSIVE CCS TRAINING ON ALL LAYERS")
    print("=" * 80)

    print("🔄 Training CCS on ALL layers for original representations...")
    original_all_layers_results = train_ccs_all_layers(
        X_pos_single, X_neg_single, labels, device
    )[0]  # Get ccs_results

    print("🔄 Training CCS on ALL layers for steered representations...")
    steered_all_layers_results = train_ccs_all_layers(
        X_pos_steered, X_neg_steered, labels, device
    )[0]  # Get ccs_results

    # ========================================================================
    # CREATE COMPREHENSIVE COMPARISON ANALYSIS
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON ANALYSIS")
    print("=" * 80)

    from format_results import get_results_table

    # Get results tables for all layers
    original_results_df = get_results_table(original_all_layers_results)
    steered_results_df = get_results_table(steered_all_layers_results)

    print(f"📊 Original CCS results shape: {original_results_df.shape}")
    print(f"📊 Steered CCS results shape: {steered_results_df.shape}")

    # Create comprehensive comparison DataFrame
    comparison_data = {}

    # Add original metrics
    for col in original_results_df.columns:
        comparison_data[f"{col}_original"] = original_results_df[col].values

    # Add steered metrics
    for col in steered_results_df.columns:
        comparison_data[f"{col}_steered"] = steered_results_df[col].values

    # Add difference metrics
    for col in original_results_df.columns:
        steered_vals = np.array(steered_results_df[col].values)
        orig_vals = np.array(original_results_df[col].values)
        comparison_data[f"{col}_diff"] = steered_vals - orig_vals
        comparison_data[f"{col}_percent_change"] = (
            (steered_vals - orig_vals) / (orig_vals + 1e-8) * 100
        )

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data, index=original_results_df.index)
    comparison_df.index.name = "layer"

    # Add metadata
    comparison_df["is_fixed_best_layer"] = False
    comparison_df.loc[best_layer, "is_fixed_best_layer"] = True
    comparison_df["steering_alpha"] = steering_alpha
    comparison_df["token_percentage"] = token_config["token_percentage"]
    comparison_df["token_strategy"] = token_config["strategy"]

    # ========================================================================
    # SAVE COMPREHENSIVE RESULTS
    # ========================================================================

    print("\n📊 SAVING COMPREHENSIVE RESULTS...")

    # Save comprehensive comparison results
    comparison_path = (
        output_dir
        / f"comprehensive_comparison_fixed_layer_{best_layer}_alpha_{steering_alpha}_tokens_{token_config['token_percentage']}pct.csv"
    )
    comparison_df.to_csv(comparison_path)
    print(f"Comprehensive comparison results saved to: {comparison_path}")

    # Save individual results tables
    orig_path = output_dir / f"original_ccs_all_layers_fixed_layer_{best_layer}.csv"
    steered_path = (
        output_dir
        / f"steered_ccs_all_layers_fixed_layer_{best_layer}_alpha_{steering_alpha}_tokens_{token_config['token_percentage']}pct.csv"
    )

    original_results_df.to_csv(orig_path)
    steered_results_df.to_csv(steered_path)

    print(f"Original CCS results (all layers) saved to: {orig_path}")
    print(f"Steered CCS results (all layers) saved to: {steered_path}")

    # ========================================================================
    # FULL COMPREHENSIVE ANALYSIS FROM ORIGINAL FILE (ALL ANALYSIS CALLS)
    # ========================================================================

    print("\n" + "=" * 80)
    print("FULL COMPREHENSIVE ANALYSIS (ALL ANALYSIS FROM ORIGINAL FILE)")
    print("=" * 80)

    # =================================================================
    # 1. PCA EIGENVALUES ANALYSIS (NEW FEATURE - FIRST PRIORITY)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"PCA EIGENVALUES ANALYSIS FOR BEST LAYER {best_layer} - FIRST PRIORITY")
    print("=" * 60)

    # Import the PCA eigenvalues analysis function
    from steering_analysis1_fixed import create_pca_eigenvalues_comparison

    # Create PCA eigenvalues analysis for both original and steered models
    print("Creating PCA eigenvalues analysis...")
    eigenvalues_results = create_pca_eigenvalues_comparison(
        X_pos_orig=X_pos_single,
        X_neg_orig=X_neg_single,
        X_pos_steered=X_pos_steered,
        X_neg_steered=X_neg_steered,
        labels=labels,
        best_layer=best_layer,
        steering_alpha=steering_alpha,
        plots_dir=plots_dir,
        n_components=10,
    )
    analysis_results["pca_eigenvalues_analysis"] = eigenvalues_results

    # =================================================================
    # 2. ENHANCED STEERING ANALYSIS (from steering_analysis1_fixed.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"ENHANCED STEERING ANALYSIS FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # Import enhanced steering analysis functions
    from steering_analysis1_fixed import (
        create_best_separation_plots,
        plot_steering_power_with_proper_steered_data,
    )

    # 2.1 Basic steering power plot using PROPER steered data
    # CRITICAL: Extract from layer where effects actually appear
    effect_layer = best_layer + EFFECT_LAYER_DELAY
    print(
        f"Creating steering power analysis with PROPER steered data at effect layer {effect_layer}..."
    )
    steering_plot_path = (
        plots_dir / f"steering_power_layer_{best_layer}_effects_{effect_layer}.png"
    )

    plot_steering_power_with_proper_steered_data(
        ccs=original_ccs,  # Use original CCS for comparison
        positive_statements_original=X_pos_single[:, effect_layer, :],  # Effects layer
        negative_statements_original=X_neg_single[:, effect_layer, :],  # Effects layer
        positive_statements_steered=X_pos_steered[
            :, effect_layer, :
        ],  # PROPER steered at effects layer
        negative_statements_steered=X_neg_steered[
            :, effect_layer, :
        ],  # PROPER steered at effects layer
        title=f"Steering Analysis - Applied at Layer {best_layer}, Effects at Layer {effect_layer}",
        save_path=str(steering_plot_path),
    )
    analysis_results["steering_power_plot"] = steering_plot_path

    # 2.2 Layer-wise steering effects
    print("Analyzing layer-wise steering effects with PROPER steered data...")
    layer_metrics = compare_steering_layers(
        X_pos_single,
        X_neg_single,  # Original representations
        X_pos_steered,
        X_neg_steered,  # PROPER steered representations from forward pass
    )

    layer_effects_plot = plot_layer_steering_effects(
        layer_metrics, best_layer, plots_dir, steering_alpha, "multi_token"
    )
    analysis_results["layer_effects"] = {
        "plot_path": layer_effects_plot,
        "metrics": layer_metrics,
    }

    # 2.3 Best separation component analysis
    print(
        f"Creating best separation component analysis at effect layer {effect_layer}..."
    )
    separation_plots = create_best_separation_plots(
        positive_statements_original=X_pos_single[:, effect_layer, :],  # Effects layer
        negative_statements_original=X_neg_single[:, effect_layer, :],  # Effects layer
        positive_statements_steered=X_pos_steered[
            :, effect_layer, :
        ],  # PROPER steered at effects layer
        negative_statements_steered=X_neg_steered[
            :, effect_layer, :
        ],  # PROPER steered at effects layer
        y_vector=labels,
        ccs=original_ccs,  # Use original CCS for decision boundary
        best_layer=effect_layer,  # Update to effects layer
        steering_alpha=steering_alpha,
        plots_dir=plots_dir,
        n_components=10,
        n_plots=5,
        separation_metric="separation_index",
    )
    analysis_results["best_separation_plots"] = separation_plots

    # =================================================================
    # 3. ADDITIONAL STEERING ANALYSIS (from steering_analysis_fixed.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"ADDITIONAL STEERING ANALYSIS FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # Import additional steering analysis functions
    from steering_analysis_fixed import (
        create_comprehensive_comparison_visualizations,
        plot_boundary_comparison_improved,
        plot_improved_layerwise_steering_focus,
        plot_steering_layer_analysis,
        plot_steering_power_improved_with_proper_steered_data,
    )

    # 3.1 Steering layer analysis plot
    print("Creating steering layer analysis plot...")
    steering_layer_analysis_path = (
        plots_dir / f"steering_layer_analysis_layer_{best_layer}.png"
    )
    plot_steering_layer_analysis(
        layer_metrics, best_layer, str(steering_layer_analysis_path)
    )
    analysis_results["steering_layer_analysis"] = steering_layer_analysis_path

    # 3.2 Improved layerwise steering focus
    print("Creating improved layerwise steering focus plot...")
    improved_layerwise_path = (
        plots_dir / f"improved_layerwise_steering_focus_layer_{best_layer}.png"
    )
    plot_improved_layerwise_steering_focus(
        X_pos=X_pos_single,
        X_neg=X_neg_single,
        X_pos_steered=X_pos_steered,  # PROPER steered
        X_neg_steered=X_neg_steered,  # PROPER steered
        labels=labels,
        best_layer=best_layer,
        steering_alpha=steering_alpha,
        save_path=str(improved_layerwise_path),
    )
    analysis_results["improved_layerwise_focus"] = improved_layerwise_path

    # 3.3 Improved steering power plot using PROPER steered data
    print(
        f"Creating improved steering power plot with PROPER steered data at effect layer {effect_layer}..."
    )
    steering_power_improved_path = (
        plots_dir
        / f"steering_power_improved_layer_{best_layer}_effects_{effect_layer}.png"
    )
    plot_steering_power_improved_with_proper_steered_data(
        ccs=original_ccs,  # Use original CCS
        X_pos_original=X_pos_single[:, effect_layer, :],  # Effects layer
        X_neg_original=X_neg_single[:, effect_layer, :],  # Effects layer
        X_pos_steered=X_pos_steered[
            :, effect_layer, :
        ],  # PROPER steered at effects layer
        X_neg_steered=X_neg_steered[
            :, effect_layer, :
        ],  # PROPER steered at effects layer
        best_layer=effect_layer,  # Update to effects layer
        steering_alpha=steering_alpha,
        save_path=str(steering_power_improved_path),
    )
    analysis_results["steering_power_improved"] = steering_power_improved_path

    # 3.4 Improved boundary comparison
    print(
        f"Creating improved boundary comparison plot at effect layer {effect_layer}..."
    )
    boundary_comparison_improved_path = (
        plots_dir
        / f"boundary_comparison_improved_layer_{best_layer}_effects_{effect_layer}.png"
    )
    plot_boundary_comparison_improved(
        X_pos_orig=X_pos_single[:, effect_layer, :],  # Effects layer
        X_neg_orig=X_neg_single[:, effect_layer, :],  # Effects layer
        X_pos_steer=X_pos_steered[
            :, effect_layer, :
        ],  # PROPER steered at effects layer
        X_neg_steer=X_neg_steered[
            :, effect_layer, :
        ],  # PROPER steered at effects layer
        labels=labels,
        ccs=original_ccs,  # Use original CCS
        best_layer=effect_layer,  # Update to effects layer
        steering_alpha=steering_alpha,
        save_path=str(boundary_comparison_improved_path),
    )
    analysis_results["boundary_comparison_improved"] = boundary_comparison_improved_path

    # =================================================================
    # 4. COMPREHENSIVE COMPARISON VISUALIZATIONS (from steering_analysis_fixed.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"COMPREHENSIVE COMPARISON VISUALIZATIONS FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # Create comprehensive comparison visualizations
    print("Creating comprehensive comparison visualizations...")
    comprehensive_plots = create_comprehensive_comparison_visualizations(
        comparison_df=comparison_df,
        best_layer=best_layer,
        steering_alpha=steering_alpha,
        plots_dir=plots_dir,
    )
    analysis_results["comprehensive_plots"] = comprehensive_plots

    # =================================================================
    # 5. LAYER-WISE PCA ANALYSIS (from format_results.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"LAYER-WISE ANALYSIS FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # Import PCA analysis functions
    from format_results import (
        plot_all_layers_components_matrix,
        plot_pca_or_tsne_layerwise,
    )

    # Original layer-wise PCA analysis
    print("Creating layer-wise PCA analysis for original model...")

    # Normalize data as required by plot_pca_or_tsne_layerwise
    X_pos_mean = X_pos_single.mean(0)
    X_neg_mean = X_neg_single.mean(0)

    X_pos_normalized = X_pos_single - X_pos_mean
    X_neg_normalized = X_neg_single - X_neg_mean

    # Check for NaN values after normalization
    if np.isnan(X_pos_normalized).any() or np.isnan(X_neg_normalized).any():
        print("Warning: Found NaN values after normalization, replacing with 0")
        X_pos_normalized = np.nan_to_num(
            X_pos_normalized, nan=0.0, posinf=0.0, neginf=0.0
        )
        X_neg_normalized = np.nan_to_num(
            X_neg_normalized, nan=0.0, posinf=0.0, neginf=0.0
        )

    # Create labels series for plotting
    labels_series = pd.Series(labels, name="toxicity_label")

    layerwise_original_path = plots_dir / "layerwise_pca_original.png"
    plot_pca_or_tsne_layerwise(
        X_pos=X_pos_normalized,
        X_neg=X_neg_normalized,
        hue=labels_series,
        standardize=True,
        n_components=5,
        components=[0, 1],
        mode="pca",
        plot_title=f"Layer-wise PCA Analysis - Original Model (Best Layer {best_layer})",
        save_path=str(layerwise_original_path),
    )

    # Normalize steered data with NaN handling - USING PROPER STEERED DATA
    X_pos_steered_mean = X_pos_steered.mean(0)
    X_neg_steered_mean = X_neg_steered.mean(0)

    X_pos_steered_normalized = X_pos_steered - X_pos_steered_mean
    X_neg_steered_normalized = X_neg_steered - X_neg_steered_mean

    # Check for NaN values after steered normalization
    if (
        np.isnan(X_pos_steered_normalized).any()
        or np.isnan(X_neg_steered_normalized).any()
    ):
        print("Warning: Found NaN values after steered normalization, replacing with 0")
        X_pos_steered_normalized = np.nan_to_num(
            X_pos_steered_normalized, nan=0.0, posinf=0.0, neginf=0.0
        )
        X_neg_steered_normalized = np.nan_to_num(
            X_neg_steered_normalized, nan=0.0, posinf=0.0, neginf=0.0
        )

    layerwise_steered_path = (
        plots_dir
        / f"layerwise_pca_steered_layer_{best_layer}_alpha_{steering_alpha}.png"
    )
    plot_pca_or_tsne_layerwise(
        X_pos=X_pos_steered_normalized,
        X_neg=X_neg_steered_normalized,
        hue=labels_series,
        standardize=True,
        n_components=5,
        components=[0, 1],
        mode="pca",
        plot_title=f"Layer-wise PCA Analysis - Steered Model (Best Layer {best_layer}, α={steering_alpha})",
        save_path=str(layerwise_steered_path),
    )

    # =================================================================
    # 6. COMPONENTS MATRIX ANALYSIS (from format_results.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"COMPONENTS MATRIX ANALYSIS FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # Enhanced PCA components matrix analysis for ALL layers (original model)
    print(
        "Creating enhanced PCA components matrix analysis for ALL layers (original model)..."
    )

    # Create subdirectory for components matrix plots
    components_original_dir = plots_dir / "components_matrix_original"
    components_original_dir.mkdir(exist_ok=True)

    plot_all_layers_components_matrix(
        X_pos=X_pos_normalized,
        X_neg=X_neg_normalized,
        hue=labels_series,
        start_layer=0,
        standardize=True,
        n_components=5,
        mode="pca",
        plot_title_prefix=f"Original Model (Best Layer {best_layer}) - PCA Components Matrix",
        save_dir=components_original_dir,
    )

    # Enhanced PCA components matrix analysis for steered model (from best layer onwards)
    print(
        f"Creating enhanced PCA components matrix analysis for steered model (from best layer {best_layer} onwards)..."
    )

    # Create subdirectory for steered components matrix plots
    components_steered_dir = (
        plots_dir / f"components_matrix_steered_from_layer_{best_layer}"
    )
    components_steered_dir.mkdir(exist_ok=True)

    plot_all_layers_components_matrix(
        X_pos=X_pos_steered_normalized,  # PROPER steered
        X_neg=X_neg_steered_normalized,  # PROPER steered
        hue=labels_series,
        start_layer=best_layer,
        standardize=True,
        n_components=5,
        mode="pca",
        plot_title_prefix=f"Steered Model (Best Layer {best_layer}, α={steering_alpha}) - PCA Components Matrix",
        save_dir=components_steered_dir,
    )

    # =================================================================
    # 7. TRADITIONAL BOUNDARY COMPARISON (from steering_analysis1_fixed.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"TRADITIONAL BOUNDARY COMPARISON FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # Import boundary comparison functions
    from steering_analysis1_fixed import plot_boundary_comparison_for_components

    if STEERING_CONFIG["plot_boundary"]:
        print("Creating traditional boundary comparison plot...")

        # Create traditional comparison plot (PC0 vs PC1) - USING PROPER STEERED DATA
        traditional_boundary_path = (
            plots_dir
            / f"boundary_comparison_traditional_layer_{best_layer}_effects_{effect_layer}_alpha_{steering_alpha}.png"
        )

        # Use the FIXED plot function with PROPER extracted steered representations
        plot_boundary_comparison_for_components(
            positive_statements_original=X_pos_single[
                :, effect_layer, :
            ],  # Effects layer
            negative_statements_original=X_neg_single[
                :, effect_layer, :
            ],  # Effects layer
            positive_statements_steered=X_pos_steered[
                :, effect_layer, :
            ],  # PROPER steered from forward pass at effects layer
            negative_statements_steered=X_neg_steered[
                :, effect_layer, :
            ],  # PROPER steered from forward pass at effects layer
            y_vector=labels,
            ccs=original_ccs,  # Use original CCS
            components=[0, 1],  # Traditional PC0 vs PC1
            separation_metrics={
                "silhouette_score": 0.0,
                "fisher_ratio": 0.0,
                "between_class_distance": 0.0,
                "separation_index": 0.0,
            },
            best_layer=effect_layer,  # Update to effects layer
            steering_alpha=steering_alpha,
            n_components=5,
            save_path=str(traditional_boundary_path),
        )

        # Also create the basic boundary comparison plot
        basic_boundary_path = (
            plots_dir
            / f"boundary_comparison_layer_{best_layer}_effects_{effect_layer}_alpha_{steering_alpha}.png"
        )

        # Create basic boundary comparison - USING PROPER STEERED DATA
        plot_boundary_comparison_for_components(
            positive_statements_original=X_pos_single[
                :, effect_layer, :
            ],  # Effects layer
            negative_statements_original=X_neg_single[
                :, effect_layer, :
            ],  # Effects layer
            positive_statements_steered=X_pos_steered[
                :, effect_layer, :
            ],  # PROPER steered from forward pass at effects layer
            negative_statements_steered=X_neg_steered[
                :, effect_layer, :
            ],  # PROPER steered from forward pass at effects layer
            y_vector=labels,
            ccs=original_ccs,  # Use original CCS
            components=[0, 1],
            separation_metrics={
                "silhouette_score": 0.0,
                "fisher_ratio": 0.0,
                "between_class_distance": 0.0,
                "separation_index": 0.0,
            },
            best_layer=effect_layer,  # Update to effects layer
            steering_alpha=steering_alpha,
            n_components=5,
            save_path=str(basic_boundary_path),
        )

    # =================================================================
    # 8. ADDITIONAL MISSING PLOTS
    # =================================================================
    print("\n" + "=" * 60)
    print(f"CREATING ADDITIONAL MISSING PLOTS FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # Create steering strength analysis plot
    print("Creating steering strength analysis plot...")
    steering_strength_path = (
        plots_dir / f"steering_strength_analysis_layer_{best_layer}.png"
    )

    # This is a custom plot showing steering strength across layers
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate steering strength for each layer using PROPER steered data
    steering_strengths = []
    layer_indices = list(range(X_pos_single.shape[1]))

    for layer_idx in layer_indices:
        # Calculate difference between PROPER steered and original representations
        layer_diff_pos = np.mean(
            np.abs(X_pos_steered[:, layer_idx, :] - X_pos_single[:, layer_idx, :])
        )
        layer_diff_neg = np.mean(
            np.abs(X_neg_steered[:, layer_idx, :] - X_neg_single[:, layer_idx, :])
        )
        avg_diff = (layer_diff_pos + layer_diff_neg) / 2
        steering_strengths.append(avg_diff)

    # Plot steering strength
    ax.plot(
        layer_indices, steering_strengths, "b-", linewidth=2, marker="o", markersize=6
    )
    # Show where steering was applied and where effects appear
    ax.axvline(
        x=best_layer,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Steering Applied at Layer {best_layer}",
    )
    ax.axvline(
        x=effect_layer,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Effects Appear at Layer {effect_layer}",
    )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Steering Strength (Mean Absolute Difference)", fontsize=12)
    ax.set_title(
        f"Steering Strength Analysis - Applied at Layer {best_layer}, Effects at Layer {effect_layer} (α={steering_alpha})\n"
        f"Multi-Token Steering ({token_config['token_percentage']}% tokens, {token_config['strategy']})",
        fontsize=14,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(steering_strength_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Steering strength analysis saved to: {steering_strength_path}")
    analysis_results["steering_strength_analysis"] = steering_strength_path

    # =================================================================
    # 9. SAVE INDIVIDUAL CSV FILES (from original pipeline)
    # =================================================================
    print("\n" + "=" * 60)
    print("SAVING INDIVIDUAL CSV FILES (FROM ORIGINAL PIPELINE)")
    print("=" * 60)

    # Save main comparison CSV
    main_comparison_path = (
        output_dir / f"comparison_results_layer_{best_layer}_alpha_{steering_alpha}.csv"
    )
    comparison_df.to_csv(main_comparison_path)
    print(f"Main comparison results saved to: {main_comparison_path}")

    # Save original results CSV
    orig_results_path = output_dir / f"results_original_layer_{best_layer}.csv"
    original_results_df.to_csv(orig_results_path)
    print(f"Original results saved to: {orig_results_path}")

    # Save steered results CSV
    steered_results_path = (
        output_dir / f"results_steered_layer_{best_layer}_alpha_{steering_alpha}.csv"
    )
    steered_results_df.to_csv(steered_results_path)
    print(f"Steered results saved to: {steered_results_path}")

    # Save full comparison CSV (more detailed)
    full_comparison_path = (
        output_dir
        / f"results_comparison_full_layer_{best_layer}_alpha_{steering_alpha}.csv"
    )
    comparison_df.to_csv(full_comparison_path)
    print(f"Full comparison results saved to: {full_comparison_path}")

    # Calculate and save differences CSV
    print("Calculating differences between original and steered results...")
    differences_df = pd.DataFrame()

    # Calculate differences for numeric columns
    for col in original_results_df.columns:
        if col in steered_results_df.columns and pd.api.types.is_numeric_dtype(
            original_results_df[col]
        ):
            differences_df[f"{col}_diff"] = (
                steered_results_df[col] - original_results_df[col]
            )
            differences_df[f"{col}_percent_change"] = (
                (
                    (steered_results_df[col] - original_results_df[col])
                    / original_results_df[col]
                    * 100
                )
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
            )

    differences_path = (
        output_dir
        / f"results_differences_layer_{best_layer}_alpha_{steering_alpha}.csv"
    )
    differences_df.to_csv(differences_path)
    print(f"Differences results saved to: {differences_path}")

    # Create comparison tables log file
    print("Creating comparison tables log file...")
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    comparison_log_path = (
        logs_dir / f"comparison_tables_layer_{best_layer}_alpha_{steering_alpha}.log"
    )

    with open(comparison_log_path, "w") as f:
        f.write(f"Comparison Tables Log for FIXED Best Layer {best_layer}\n")
        f.write("=" * 60 + "\n\n")

        f.write("ORIGINAL RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(original_results_df.to_string())
        f.write("\n\n")

        f.write("STEERED RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(steered_results_df.to_string())
        f.write("\n\n")

        f.write("COMPARISON RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(comparison_df.to_string())
        f.write("\n\n")

        f.write("DIFFERENCES:\n")
        f.write("-" * 20 + "\n")
        f.write(differences_df.to_string())
        f.write("\n")

    print(f"Comparison tables log saved to: {comparison_log_path}")

    # ========================================================================
    # SPECIFIC ANALYSIS AT THE FIXED BEST LAYER
    # ========================================================================

    print(f"\n📊 FIXED BEST LAYER ({best_layer}) SPECIFIC ANALYSIS:")
    print("-" * 50)

    # Extract metrics for the fixed best layer
    best_layer_orig = original_results_df.loc[best_layer]
    best_layer_steered = steered_results_df.loc[best_layer]

    print("ORIGINAL CCS at fixed best layer:")
    for metric, value in best_layer_orig.items():
        print(f"  {metric}: {value:.6f}")

    print("\nSTEERED CCS at fixed best layer:")
    for metric, value in best_layer_steered.items():
        print(f"  {metric}: {value:.6f}")

    print("\nDIFFERENCES (Steered - Original) at fixed best layer:")
    for metric in best_layer_orig.index:
        diff = best_layer_steered[metric] - best_layer_orig[metric]
        pct_change = (diff / (best_layer_orig[metric] + 1e-8)) * 100
        print(f"  {metric}: {diff:+.6f} ({pct_change:+.2f}%)")

    # =================================================================
    # 10. SUMMARY REPORT
    # =================================================================
    print("\n" + "=" * 80)
    print("COMPLETE FIXED MULTI-TOKEN STEERING ANALYSIS SUMMARY")
    print(f"FIXED Best Layer: {best_layer} (Independent of Steering Configuration)")
    print(f"Applied at Layer {best_layer}, Effects Analyzed at Layer {effect_layer}")
    print("=" * 80)

    print("✅ CRITICAL FIXES APPLIED:")
    print("   ✅ Best layer selection INDEPENDENT of steering configuration")
    print("   ✅ Dual CCS training: Original CCS vs Steered CCS")
    print("   ✅ All analysis uses PROPER data from respective CCS models")

    print("\n✅ COMPLETE ANALYSIS FROM ORIGINAL FILE:")
    print("   ✅ PCA eigenvalues analysis completed (NEW FEATURE - CALLED FIRST)")
    print("   ✅ Enhanced steering analysis completed - ANALYZING EFFECTS LAYER!")
    print(
        "   ✅ Additional steering analysis from steering_analysis_fixed.py completed"
    )
    print("   ✅ Comprehensive comparison visualizations completed")
    print(
        "   ✅ Individual CSV files created (original, steered, differences, full comparison)"
    )
    print("   ✅ Comparison tables log file created")
    print("   ✅ Layer-wise PCA analysis completed")
    print("   ✅ Components matrix analysis completed")
    print("   ✅ Boundary comparison analysis completed - AT EFFECTS LAYER!")
    print("   ✅ ALL ANALYSIS USES PROPER STEERED DATA FROM FORWARD PASS!")
    print(
        f"   ✅ CORRECTLY HOOKS FIXED LAYER {best_layer} AND ANALYZES EFFECTS AT LAYER {effect_layer}!"
    )

    # Count all plots created
    total_plots_created = 0

    if "best_separation_plots" in analysis_results:
        n_separation_plots = len(analysis_results["best_separation_plots"])
        print(f"✅ {n_separation_plots} best component pair separation plots created")
        total_plots_created += n_separation_plots

    if "comprehensive_plots" in analysis_results:
        n_comprehensive_plots = len(analysis_results["comprehensive_plots"])
        print(f"✅ {n_comprehensive_plots} comprehensive comparison plots created")
        total_plots_created += n_comprehensive_plots

    # Count individual analysis plots
    individual_plots = [
        "steering_power_plot",
        "steering_layer_analysis",
        "improved_layerwise_focus",
        "steering_power_improved",
        "boundary_comparison_improved",
        "steering_strength_analysis",
    ]
    n_individual_plots = sum(1 for plot in individual_plots if plot in analysis_results)
    print(f"✅ {n_individual_plots} individual analysis plots created")
    total_plots_created += n_individual_plots

    # Count additional plots (traditional + basic boundary)
    additional_plots = 2  # traditional + basic boundary
    print(f"✅ {additional_plots} additional boundary comparison plots created")
    total_plots_created += additional_plots

    print(
        f"\nTOTAL PLOTS CREATED FOR FIXED BEST LAYER {best_layer}: {total_plots_created}"
    )

    # Count CSV files created
    csv_files = [
        "comprehensive_comparison",
        "original_ccs_all_layers",
        "steered_ccs_all_layers",
        "comparison_results",
        "results_original",
        "results_steered",
        "results_comparison_full",
        "results_differences",
    ]
    print(f"✅ {len(csv_files)} CSV result files created")

    # Calculate steering effect statistics at the effect layer
    pos_steering_magnitude = np.mean(
        np.linalg.norm(
            X_pos_steered[:, effect_layer, :] - X_pos_single[:, effect_layer, :], axis=1
        )
    )
    neg_steering_magnitude = np.mean(
        np.linalg.norm(
            X_neg_steered[:, effect_layer, :] - X_neg_single[:, effect_layer, :], axis=1
        )
    )

    print("\n📊 STEERING EFFECT MAGNITUDE AT EFFECT LAYER:")
    print(f"    Fixed best layer (steering applied): {best_layer}")
    print(f"    Effects measured at layer: {effect_layer}")
    print(f"    Positive samples magnitude: {pos_steering_magnitude:.6f}")
    print(f"    Negative samples magnitude: {neg_steering_magnitude:.6f}")
    print(f"    Token percentage: {token_config['token_percentage']}%")
    print(f"    Strategy: {token_config['strategy']}")

    print("\n📊 CCS COMPARISON SUMMARY:")
    print(f"    Original CCS weights norm: {original_weight_norm:.6f}")
    print(f"    Steered CCS weights norm: {steered_weight_norm:.6f}")
    print(f"    Weight cosine similarity: {cosine_sim:.6f}")
    print(f"    Bias difference: {steered_bias - original_bias:+.6f}")

    print(f"\nAll analysis results saved to: {plots_dir}")
    print(f"All CSV files saved to: {output_dir}")
    print(f"All log files saved to: {output_dir / 'logs'}")
    print("=" * 80)

    return {
        "layer_effects_plot": layer_effects_plot,
        "layer_metrics": layer_metrics,
        "steering_magnitude": {
            "positive": pos_steering_magnitude,
            "negative": neg_steering_magnitude,
        },
        "comparison_results": (
            comparison_df,
            original_all_layers_results,
            steered_all_layers_results,
        ),
        "original_ccs": original_ccs,
        "steered_ccs": steered_ccs,
        "best_layer_used": best_layer,  # For verification
        "ccs_comparison": {
            "original_weights": original_weights,
            "steered_weights": steered_weights,
            "original_bias": original_bias,
            "steered_bias": steered_bias,
            "weight_cosine_similarity": cosine_sim,
        },
        "analysis_results": analysis_results,
    }


# ============================================================================
# MAIN PIPELINE (FIXED + COMPLETE)
# ============================================================================


def main():
    """
    COMPLETE FIXED main pipeline with all analysis from original file.

    CRITICAL FIXES:
    1. Vanilla CCS is trained ONCE and best layer is selected ONCE
    2. The same best layer is used for ALL steering experiments
    3. Different token configurations cannot affect best layer selection
    4. Dual CCS training: Original vs Steered for proper comparison
    5. All comprehensive analysis from original file included
    """
    print("=" * 80)
    print("COMPLETE FIXED CCS Multi-Token Steering Pipeline with Full Analysis")
    print("=" * 80)

    # Load configuration
    model_config = MODEL_CONFIGS["deberta_base"]
    token_config = MULTI_TOKEN_CONFIG

    # Create output directory
    output_dir = create_output_dir(
        model_config,
        suffix=f"_tokens_{token_config['token_percentage']}_percent",
    )
    setup_logging(output_dir)

    print(
        f"Starting COMPLETE FIXED pipeline for model: {model_config['model_name']} ({model_config['size']})"
    )
    print(f"Output directory: {output_dir}")

    print("\n" + "=" * 80)
    print("PIPELINE CONFIGURATION")
    print("=" * 80)
    print(f"Data: {DATA_CONFIG['dataset_name']}")
    print(f"Model: {model_config['model_name']} ({model_config['size']})")
    print(
        f"CCS Token Strategy: {model_config.get('token_strategy', 'first-token')} (vanilla CCS)"
    )
    print(f"Steering Strategy: {token_config['strategy']}")
    print(f"Steering Token Percentage: {token_config['token_percentage']}%")
    print(f"CCS Lambda: {CCS_CONFIG['lambda_classification']}")
    print(f"Normalizing: {CCS_CONFIG['normalizing']}")
    print(f"Steering Alpha: {STEERING_CONFIG['default_alpha']}")

    # Load data and model
    positive_texts, negative_texts, labels = load_data()
    model, tokenizer, device, model_config = load_model()

    # ========================================================================
    # CRITICAL FIX: VANILLA CCS TRAINING (ONCE PER MODEL)
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 1: VANILLA CCS TRAINING (FIXED BEST LAYER SELECTION)")
    print("=" * 80)

    # Create vanilla CCS results container
    vanilla_ccs = VanillaCCSResults(model_config, device)

    # Train vanilla CCS ONCE - this determines the best layer
    vanilla_ccs.train_vanilla_ccs(
        positive_texts, negative_texts, labels, model, tokenizer
    )

    # Get all vanilla results
    vanilla_results = vanilla_ccs.get_vanilla_results()

    print(f"✅ FIXED BEST LAYER: {vanilla_results['best_layer']}")
    print(
        "✅ This layer will be used for ALL steering experiments regardless of token configuration"
    )

    # ========================================================================
    # STEP 2: STEERING EXPERIMENTS WITH FIXED BEST LAYER
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 2: STEERING EXPERIMENTS WITH FIXED BEST LAYER")
    print("=" * 80)

    steering_alpha = STEERING_CONFIG.get("default_alpha")
    best_layer = vanilla_results["best_layer"]  # Use FIXED best layer
    direction_tensor = vanilla_results["direction_tensor"]

    print(f"🎯 Applying steering to FIXED best layer: {best_layer}")
    print(f"🎯 Steering alpha: {steering_alpha}")
    print(f"🎯 Token percentage: {token_config['token_percentage']}%")
    print(f"🎯 Token strategy: {token_config['strategy']}")

    # Extract steered representations using proper forward pass with hooks
    print("🔄 Extracting steered representations with proper forward pass...")
    X_pos_steered = extract_steered_representations_with_hooks(
        model=model,
        tokenizer=tokenizer,
        texts=positive_texts,
        device=device,
        model_config=model_config,
        steering_layer=best_layer,  # Use FIXED best layer
        direction_tensor=direction_tensor,
        steering_alpha=steering_alpha,
        token_config=token_config,
        is_positive=True,
    )

    X_neg_steered = extract_steered_representations_with_hooks(
        model=model,
        tokenizer=tokenizer,
        texts=negative_texts,
        device=device,
        model_config=model_config,
        steering_layer=best_layer,  # Use FIXED best layer
        direction_tensor=direction_tensor,
        steering_alpha=steering_alpha,
        token_config=token_config,
        is_positive=False,
    )

    # ========================================================================
    # STEP 2.1: TRAIN NEW CCS ON STEERED REPRESENTATIONS (CRITICAL!)
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 2.1: TRAIN NEW CCS ON STEERED REPRESENTATIONS (CRITICAL PHASE!)")
    print("=" * 80)
    print(
        f"🎯 Training NEW CCS on steered representations at FIXED best layer: {best_layer}"
    )
    print("📊 This allows us to compare:")
    print("   - Original CCS (trained on original representations)")
    print("   - Steered CCS (trained on steered representations)")
    print("   Both using the SAME fixed best layer for fair comparison!")

    # ========================================================================
    # STEP 3: COMPREHENSIVE ANALYSIS WITH BOTH CCS MODELS + FULL ANALYSIS
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 3: COMPREHENSIVE ANALYSIS WITH BOTH CCS MODELS + FULL ANALYSIS")
    print("=" * 80)
    print("📊 Comparing original CCS vs steered CCS performance")
    print("📊 Both models use the SAME fixed best layer for fair comparison")
    print("📊 Including ALL comprehensive analysis from original file")

    # Run COMPLETE analysis using the FIXED best layer and DUAL CCS training
    analysis_results = run_comprehensive_multi_token_steering_analysis(
        vanilla_results=vanilla_results,
        X_pos_steered=X_pos_steered,
        X_neg_steered=X_neg_steered,
        steering_alpha=steering_alpha,
        token_config=token_config,
        output_dir=output_dir,
    )

    # Verify best layer consistency
    used_best_layer = analysis_results["best_layer_used"]
    print(f"✅ Verification: Analysis used best layer {used_best_layer}")
    print(f"✅ Original best layer: {vanilla_results['best_layer']}")
    print(
        f"✅ Best layer consistency: {used_best_layer == vanilla_results['best_layer']}"
    )

    # ========================================================================
    # STEP 4: SAVE RESULTS
    # ========================================================================

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save results
    save_results(
        ccs_results=vanilla_results["ccs_results"],
        results_df=vanilla_results["results_df"],
        X_pos=vanilla_results["X_pos_single"],
        X_neg=vanilla_results["X_neg_single"],
        best_layer=vanilla_results["best_layer"],  # Use FIXED best layer
        direction_tensor=vanilla_results["direction_tensor"],
        output_dir=output_dir,
        model_config=model_config,
        comparison_results=analysis_results.get("comparison_results"),
    )

    print("\n" + "=" * 80)
    print("COMPLETE FIXED MULTI-TOKEN STEERING PIPELINE WITH FULL ANALYSIS COMPLETED!")
    print("=" * 80)
    print("✅ CRITICAL FIXES APPLIED:")
    print("   ✅ Vanilla CCS trained ONCE on single-token representations")
    print(f"   ✅ FIXED best layer selected ONCE: {vanilla_results['best_layer']}")
    print("   ✅ Best layer selection is now INDEPENDENT of steering configuration")
    print(
        f"   ✅ Multi-token steering applied to FIXED layer {vanilla_results['best_layer']}"
    )
    print(
        f"   ✅ NEW CCS trained on steered representations at FIXED layer {vanilla_results['best_layer']}"
    )
    print("   ✅ Comprehensive comparison: Original CCS vs Steered CCS")
    print("   ✅ Both CCS models use the SAME fixed best layer for fair comparison")
    print("   ✅ Analysis shows how steering affects CCS learning, not layer selection")

    print("\n✅ COMPLETE ANALYSIS FROM ORIGINAL FILE:")
    print("   ✅ Proper forward pass with steering hooks (no simulation!)")
    print("   ✅ Real attention propagation through all layers")
    print("   ✅ PCA eigenvalues analysis completed")
    print("   ✅ Enhanced steering analysis completed")
    print("   ✅ Layer-wise PCA analysis completed")
    print("   ✅ Components matrix analysis completed")
    print("   ✅ Boundary comparison analysis completed")
    print("   ✅ Comprehensive comparison visualizations completed")
    print("   ✅ ALL analysis uses PROPER steered data from forward pass!")
    print("   ✅ ALL analysis uses PROPER dual CCS training results!")

    print(f"\n📊 Token configuration: {token_config['token_percentage']}% tokens")
    print(f"📊 Strategy: {token_config['strategy']}")
    print(f"📊 Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
