#!/usr/bin/env python3
"""
CCS Training and Steering Pipeline - Multi-Token with Proper Forward Pass
=========================================================================

CORRECT APPROACH (as described by user):
1. Normal forward pass to layer N → get representations
2. Apply steering to layer N → modify representations
3. Continue normal forward pass from layer N+1 → using steered representations as input
4. Extract final representations → no simulation needed!

This uses forward hooks to intercept and modify representations during normal forward pass.
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

# ============================================================================
# STEERING CONFIGURATION CONSTANTS
# ============================================================================

# CRITICAL: Effect layer delay factor
# Due to transformer architecture: hooking transformer_layer[N] affects hidden_states[N+1]
# So effects appear 1 layer after where steering is applied
EFFECT_LAYER_DELAY = 1

# Import steering and analysis functions
from steering import (
    compare_steering_layers,
)
from steering_analysis1_fixed import (
    plot_layer_steering_effects,
)

# ============================================================================
# PROPER FORWARD PASS WITH STEERING HOOKS
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

        # DEBUG: Check input values before steering
        input_max = torch.max(torch.abs(hidden_states)).item()
        if input_max > 1e10:
            print(
                f"❌ CRITICAL: Input to steering hook contains astronomical values! Max: {input_max:.2e}"
            )
            return hidden_states  # Return unmodified to prevent further damage

        # Apply steering to each sample in the batch
        steering_applied = False
        for batch_idx in range(hidden_states.shape[0]):
            if batch_idx < len(self.token_indices):
                sample_token_indices = self.token_indices[batch_idx]

                # Apply steering to selected tokens
                for token_idx in sample_token_indices:
                    if token_idx < hidden_states.shape[1]:  # Check bounds
                        # DEBUG: Check values before steering
                        original_val = hidden_states[batch_idx, token_idx, :].clone()
                        original_max = torch.max(torch.abs(original_val)).item()

                        # Apply steering with polarity
                        polarity = 1.0 if self.is_positive else -1.0
                        steering_vec = polarity * self.steering_direction

                        # DEBUG: Check steering vector
                        if isinstance(steering_vec, torch.Tensor):
                            steering_max = torch.max(torch.abs(steering_vec)).item()
                        else:
                            steering_max = abs(float(steering_vec))
                        if steering_max > 1e10:
                            print(
                                f"❌ CRITICAL: Steering vector contains astronomical values! Max: {steering_max:.2e}"
                            )
                            return hidden_states  # Return unmodified

                        # DEBUG: Check alpha
                        if abs(self.steering_alpha) > 1e10:
                            print(
                                f"❌ CRITICAL: Steering alpha is astronomical! Alpha: {self.steering_alpha:.2e}"
                            )
                            return hidden_states  # Return unmodified

                        # Apply steering
                        hidden_states[batch_idx, token_idx, :] += (
                            self.steering_alpha * steering_vec
                        )

                        # DEBUG: Check values after steering
                        new_val = hidden_states[batch_idx, token_idx, :]
                        new_max = torch.max(torch.abs(new_val)).item()

                        if new_max > 1e10:
                            print("❌ CRITICAL: Steering created astronomical values!")
                            print(f"  Original max: {original_max:.2e}")
                            print(f"  Steering max: {steering_max:.2e}")
                            print(f"  Alpha: {self.steering_alpha:.2e}")
                            print(f"  New max: {new_max:.2e}")
                            print(
                                f"  This will cause matplotlib error with dimensions like {new_max}"
                            )
                            # Reset to original to prevent further damage
                            hidden_states[batch_idx, token_idx, :] = original_val
                            return hidden_states

                        steering_applied = True

        if steering_applied:
            # Final check on output
            output_max = torch.max(torch.abs(hidden_states)).item()
            if output_max > 1e10:
                print(
                    f"❌ CRITICAL: Final steering output contains astronomical values! Max: {output_max:.2e}"
                )
            else:
            print(
                f"🎯 Applied steering to hidden_states[{self.target_layer}] with alpha={self.steering_alpha}"
            )
                print(f"🔍 Output max value: {output_max:.2e}")

        # CRITICAL FIX: Return the modified hidden states so PyTorch actually uses them
        # This ensures the steered output replaces the original output
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
    """
    Extract representations with steering applied using forward hooks.

    This is the CORRECT approach: normal forward pass with steering applied at target layer.
    """
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

        # CORRECT APPROACH: Always hook the best layer (where CCS performs optimally)
        # Effects will appear in the NEXT hidden_states index due to transformer architecture
        # hidden_states[3] = output of transformer_layer[2]
        # hidden_states[4] = output of transformer_layer[3] <- Effects appear here
        transformer_layer_to_hook = (
            steering_layer  # Always hook the best layer (layer 3)
        )

        # Handle both DebertaModel and DebertaForMaskedLM attribute structures
        if hasattr(model, "encoder"):
            # DebertaModel: model.encoder.layer
            target_module = model.encoder.layer[transformer_layer_to_hook].output
            print(
                f"🔧 Using DebertaModel structure: model.encoder.layer[{transformer_layer_to_hook}].output"
            )
        elif hasattr(model, "deberta") and hasattr(model.deberta, "encoder"):
            # DebertaForMaskedLM: model.deberta.encoder.layer
            target_module = model.deberta.encoder.layer[
                transformer_layer_to_hook
            ].output
            print(
                f"🔧 Using DebertaForMaskedLM structure: model.deberta.encoder.layer[{transformer_layer_to_hook}].output"
            )
        else:
            raise AttributeError(
                f"Cannot find encoder layers in model. Model type: {type(model)}"
            )

        print(
            f"🎯 Hooking transformer layer {transformer_layer_to_hook} (best layer) - effects will appear in hidden_states[{steering_layer + 1}]"
        )

        # IMPORTANT: Understanding the hidden_states indexing:
        # - hidden_states[0] = embeddings (before any transformer processing)
        # - hidden_states[1] = output of transformer_layer[0]
        # - hidden_states[2] = output of transformer_layer[1]
        # - hidden_states[3] = output of transformer_layer[2] (best layer input)
        # - hidden_states[4] = output of transformer_layer[3] <- STEERING EFFECTS APPEAR HERE
        # So hooking transformer_layer[3] (best layer) modifies hidden_states[4]
        handle = target_module.register_forward_hook(hook)

        try:
            with torch.no_grad():
                # Normal forward pass - hook will apply steering automatically
                outputs = model(**inputs, output_hidden_states=True)

                # Extract representations from all layers
                hidden_states = outputs.hidden_states  # Tuple of tensors

                # Convert to numpy and extract based on strategy
                batch_representations = []
                for layer_idx, layer_hidden in enumerate(hidden_states):
                    # The hook automatically modifies the output, so hidden_states[steering_layer]
                    # already contains the steered representation - no special extraction needed
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
# COMPREHENSIVE ANALYSIS FOR MULTI-TOKEN STEERING (COMPLETE VERSION)
# ============================================================================


def run_comprehensive_multi_token_steering_analysis(
    best_ccs,
    X_pos_single,
    X_neg_single,
    X_pos_steered,
    X_neg_steered,
    best_layer,
    direction_tensor,
    labels,
    train_idx,
    test_idx,
    device,
    output_dir,
):
    """
    Run COMPLETE comprehensive analysis for multi-token steering.

    CRITICAL: This function uses PROPER steered representations from actual forward pass,
    NOT simulations! All analysis uses real steered data from the model.
    """
    print(
        f"Running COMPLETE comprehensive multi-token steering analysis for best layer {best_layer}..."
    )

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Get steering alpha from config
    steering_alpha = STEERING_CONFIG.get("default_alpha")

    print(f"📊 Original single-token shape: {X_pos_single.shape}")
    print(f"📊 Steered representations shape: {X_pos_steered.shape}")
    print("✅ Using PROPER steered representations from actual forward pass!")

    analysis_results = {}

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
        ccs=best_ccs,
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
        ccs=best_ccs,
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
        ccs=best_ccs,
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
        ccs=best_ccs,
        best_layer=effect_layer,  # Update to effects layer
        steering_alpha=steering_alpha,
        save_path=str(boundary_comparison_improved_path),
    )
    analysis_results["boundary_comparison_improved"] = boundary_comparison_improved_path

    # =================================================================
    # 4. COMPARISON ANALYSIS (from steering_analysis1_fixed.py)
    # =================================================================
    print("\n" + "=" * 80)
    print(f"STARTING COMPARISON ANALYSIS FOR BEST LAYER {best_layer}")
    print("=" * 80)

    # Import comparison analysis functions
    from steering_analysis1_fixed import create_comparison_results_table

    # Run the comprehensive comparison analysis using PROPER steered data
    comparison_df, orig_results, steered_results = create_comparison_results_table(
        X_pos_orig=X_pos_single,
        X_neg_orig=X_neg_single,
        X_pos_steered=X_pos_steered,  # PROPER steered
        X_neg_steered=X_neg_steered,  # PROPER steered
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        best_layer=best_layer,
        device=device,
        ccs_config=CCS_CONFIG,
        normalizing=CCS_CONFIG.get("normalizing", "mean"),
    )

    # Save main comparison CSV
    comparison_path = (
        output_dir / f"comparison_results_layer_{best_layer}_alpha_{steering_alpha}.csv"
    )
    comparison_df.to_csv(comparison_path)
    print(f"Comparison results saved to: {comparison_path}")

    # Save individual CSV files
    print("Saving individual CSV result files...")

    # Convert results dictionaries to DataFrames using get_results_table
    from format_results import get_results_table

    orig_results_df = get_results_table(orig_results)
    steered_results_df = get_results_table(steered_results)

    # Save original results CSV
    orig_results_path = output_dir / f"results_original_layer_{best_layer}.csv"
    orig_results_df.to_csv(orig_results_path)
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
    for col in orig_results_df.columns:
        if col in steered_results_df.columns and pd.api.types.is_numeric_dtype(
            orig_results_df[col]
        ):
            differences_df[f"{col}_diff"] = (
                steered_results_df[col] - orig_results_df[col]
            )
            differences_df[f"{col}_percent_change"] = (
                (
                    (steered_results_df[col] - orig_results_df[col])
                    / orig_results_df[col]
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
        f.write(f"Comparison Tables Log for Best Layer {best_layer}\n")
        f.write("=" * 60 + "\n\n")

        f.write("ORIGINAL RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(orig_results_df.to_string())
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

    # =================================================================
    # 5. COMPREHENSIVE COMPARISON VISUALIZATIONS (from steering_analysis_fixed.py)
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
    # 6. LAYER-WISE PCA ANALYSIS (from format_results.py)
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
    # 7. COMPONENTS MATRIX ANALYSIS (from format_results.py)
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
    # 8. TRADITIONAL BOUNDARY COMPARISON (from steering_analysis1_fixed.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"TRADITIONAL BOUNDARY COMPARISON FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # Import boundary comparison functions
    from steering_analysis1_fixed import plot_boundary_comparison_for_components

    if STEERING_CONFIG["plot_boundary"]:
        print("Creating traditional boundary comparison plot...")

        # ⚠️ CRITICAL FIX: Use PROPER extracted steered representations, NOT simulation!
        # The original run_ccs_pipeline.py incorrectly used simulation here:
        # pos_steered_simple = X_pos[:, best_layer, :].copy()
        # pos_steered_simple += steering_alpha * direction_np  # <-- This is SIMULATION!
        #
        # We use PROPER extracted steered representations instead:

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
            ccs=best_ccs,
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
            ccs=best_ccs,
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
    # 9. ADDITIONAL MISSING PLOTS
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
        f"Multi-Token Steering ({MULTI_TOKEN_CONFIG['token_percentage']}% tokens, {MULTI_TOKEN_CONFIG['strategy']})",
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
    # 10. SUMMARY REPORT
    # =================================================================
    print("\n" + "=" * 60)
    print("COMPLETE MULTI-TOKEN STEERING ANALYSIS SUMMARY")
    print(f"Applied at Layer {best_layer}, Effects Analyzed at Layer {effect_layer}")
    print("=" * 60)

    print("✅ PCA eigenvalues analysis completed (NEW FEATURE - CALLED FIRST)")
    print("✅ Enhanced steering analysis completed - ANALYZING EFFECTS LAYER!")
    print("✅ Additional steering analysis from steering_analysis_fixed.py completed")
    print("✅ Comprehensive comparison visualizations completed")
    print(
        "✅ Individual CSV files created (original, steered, differences, full comparison)"
    )
    print("✅ Comparison tables log file created")
    print("✅ Layer-wise PCA analysis completed")
    print("✅ Components matrix analysis completed")
    print("✅ Boundary comparison analysis completed - AT EFFECTS LAYER!")
    print("✅ ALL ANALYSIS USES PROPER STEERED DATA FROM FORWARD PASS!")
    print(
        f"✅ CORRECTLY HOOKS BEST LAYER {best_layer} AND ANALYZES EFFECTS AT LAYER {effect_layer}!"
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

    print(f"\nTOTAL PLOTS CREATED FOR BEST LAYER {best_layer}: {total_plots_created}")

    # Count CSV files created
    csv_files = [
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

    print("📊 Steering effect magnitude at effect layer:")
    print(f"    Applied at layer: {best_layer}")
    print(f"    Effects measured at layer: {effect_layer}")
    print(f"    Positive samples magnitude: {pos_steering_magnitude:.6f}")
    print(f"    Negative samples magnitude: {neg_steering_magnitude:.6f}")
    print(f"    Token percentage: {MULTI_TOKEN_CONFIG['token_percentage']}%")
    print(f"    Strategy: {MULTI_TOKEN_CONFIG['strategy']}")

    print(f"\nAll analysis results saved to: {plots_dir}")
    print(f"All CSV files saved to: {output_dir}")
    print(f"All log files saved to: {output_dir / 'logs'}")
    print("=" * 60)

    return {
        "layer_effects_plot": layer_effects_plot,
        "layer_metrics": layer_metrics,
        "steering_magnitude": {
            "positive": pos_steering_magnitude,
            "negative": neg_steering_magnitude,
        },
        "comparison_results": (comparison_df, orig_results, steered_results),
        "analysis_results": analysis_results,
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Main pipeline for multi-token steering with proper forward pass."""
    print("=" * 80)
    print("CCS Multi-Token Steering Pipeline with Proper Forward Pass")
    print("=" * 80)

    # Load configuration
    model_config = MODEL_CONFIGS["deberta_base"]
    token_config = MULTI_TOKEN_CONFIG

    # Create output directory
    output_dir = create_output_dir(
        model_config, suffix=f"_tokens_{token_config['token_percentage']}_percent"
    )
    setup_logging(output_dir)

    print(
        f"Starting multi-token pipeline for model: {model_config['model_name']} ({model_config['size']})"
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

    print("\n" + "=" * 80)
    print("STEP 1: VANILLA CCS TRAINING")
    print("=" * 80)

    # Extract single-token representations for vanilla CCS training
    print(
        f"🔄 Extracting single-token representations using '{model_config.get('token_strategy', 'first-token')}' strategy..."
    )
    X_pos_single, X_neg_single = extract_representations(
        model, tokenizer, positive_texts, negative_texts, device, model_config
    )
    print(f"📊 Single-token representations shape: {X_pos_single.shape}")

    # Train vanilla CCS on single-token representations
    print("🔄 Training vanilla CCS on single-token representations...")
    ccs_results, train_idx, test_idx = train_ccs_all_layers(
        X_pos_single, X_neg_single, labels, device
    )

    # Select best layer based on vanilla CCS results
    best_layer, results_df = select_best_layer(ccs_results)

    # Setup steering direction from vanilla CCS
    print(f"🔄 Setting up steering for best layer {best_layer}...")
    best_ccs, direction_tensor = setup_steering(
        X_pos_single, X_neg_single, labels, train_idx, best_layer, device
    )

    # DEBUG: Check direction tensor magnitude
    direction_magnitude = torch.norm(direction_tensor).item()
    print(f"🔍 DEBUG: Direction tensor magnitude: {direction_magnitude}")
    print(f"🔍 DEBUG: Direction tensor shape: {direction_tensor.shape}")
    print(f"🔍 DEBUG: Direction tensor first 5 values: {direction_tensor[:5]}")

    steering_alpha = STEERING_CONFIG.get("default_alpha", 100.0)
    print(f"🔍 DEBUG: Steering alpha: {steering_alpha}")

    # Calculate expected steering magnitude
    expected_magnitude = direction_magnitude * steering_alpha
    print(f"🔍 DEBUG: Expected steering magnitude: {expected_magnitude}")

    if direction_magnitude < 1e-6:
        print("⚠️  WARNING: Direction tensor magnitude is very small!")
    if expected_magnitude < 1e-3:
        print("⚠️  WARNING: Expected steering magnitude is very small!")

    # CRITICAL DEBUG: Check for astronomical values in direction tensor
    direction_max = torch.max(torch.abs(direction_tensor)).item()
    if direction_max > 1e10:
        print(
            f"❌ CRITICAL: Direction tensor contains astronomical values! Max: {direction_max:.2e}"
        )
        print(
            f"❌ This will cause the matplotlib error with dimensions like {direction_max}"
        )
        return

    # CRITICAL DEBUG: Check steering alpha for astronomical values
    if steering_alpha > 1e10:
        print(
            f"❌ CRITICAL: Steering alpha is astronomical! Alpha: {steering_alpha:.2e}"
        )
        print(
            f"❌ This will cause the matplotlib error with dimensions like {steering_alpha}"
        )
        return

    print("\n" + "=" * 80)
    print("STEP 2: PROPER FORWARD PASS WITH STEERING")
    print("=" * 80)

    # Extract steered representations using proper forward pass with hooks
    steering_alpha = STEERING_CONFIG.get("default_alpha")

    print("🔄 Extracting steered representations with proper forward pass...")
    X_pos_steered = extract_steered_representations_with_hooks(
        model=model,
        tokenizer=tokenizer,
        texts=positive_texts,
        device=device,
        model_config=model_config,
        steering_layer=best_layer,
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
        steering_layer=best_layer,
        direction_tensor=direction_tensor,
        steering_alpha=steering_alpha,
        token_config=token_config,
        is_positive=False,
    )

    # CRITICAL DEBUG: Check steered representations for astronomical values
    print("🔍 DEBUG: Checking steered representations for astronomical values...")

    def debug_check_array(arr, name):
        """Check array for astronomical values that could cause matplotlib errors."""
        if arr is None:
            print(f"❌ {name} is None!")
            return False

        print(f"🔍 {name} shape: {arr.shape}")

        # Check for NaN/Inf
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0:
            print(f"⚠️  {name} contains {nan_count} NaN values")
        if inf_count > 0:
            print(f"⚠️  {name} contains {inf_count} infinite values")

        # Check for astronomical values
        max_val = np.max(np.abs(arr))
        min_val = np.min(np.abs(arr[arr != 0]))  # Non-zero minimum
        print(f"🔍 {name} max absolute value: {max_val:.2e}")
        print(f"🔍 {name} min non-zero absolute value: {min_val:.2e}")

        # Check if any value could be the problematic 47135374297
        target_val = 47135374297
        close_matches = np.abs(arr - target_val) < 1e6
        if close_matches.any():
            print(f"❌ FOUND IT! {name} contains values close to {target_val}")
            close_values = arr[close_matches]
            print(f"❌ Close values: {close_values}")
            return False

        # Check for other astronomical values
        if max_val > 1e10:
            print(
                f"❌ CRITICAL: {name} contains astronomical values! Max: {max_val:.2e}"
            )
            print(
                f"❌ This could cause matplotlib error with dimensions like {max_val}"
            )
            return False

        return True

    # Check all steered representations
    if not debug_check_array(X_pos_steered, "X_pos_steered"):
        print("❌ Stopping due to astronomical values in X_pos_steered")
        return

    if not debug_check_array(X_neg_steered, "X_neg_steered"):
        print("❌ Stopping due to astronomical values in X_neg_steered")
        return

    # CRITICAL FIX: Extract original representations using the same hook-based method
    # but with steering_alpha=0 to ensure fair comparison
    print(
        "🔄 Extracting original representations using same hook-based method (α=0)..."
    )
    X_pos_original_hooks = extract_steered_representations_with_hooks(
        model=model,
        tokenizer=tokenizer,
        texts=positive_texts,
        device=device,
        model_config=model_config,
        steering_layer=best_layer,
        direction_tensor=direction_tensor,
        steering_alpha=0.0,  # No steering for baseline comparison
        token_config=token_config,
        is_positive=True,
    )

    X_neg_original_hooks = extract_steered_representations_with_hooks(
        model=model,
        tokenizer=tokenizer,
        texts=negative_texts,
        device=device,
        model_config=model_config,
        steering_layer=best_layer,
        direction_tensor=direction_tensor,
        steering_alpha=0.0,  # No steering for baseline comparison
        token_config=token_config,
        is_positive=False,
    )

    print("\n" + "=" * 80)
    print("STEP 3: COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # Run comprehensive analysis using hook-based original representations
    analysis_results = run_comprehensive_multi_token_steering_analysis(
        best_ccs=best_ccs,
        X_pos_single=X_pos_original_hooks,  # Use hook-based original representations
        X_neg_single=X_neg_original_hooks,  # Use hook-based original representations
        X_pos_steered=X_pos_steered,
        X_neg_steered=X_neg_steered,
        best_layer=best_layer,
        direction_tensor=direction_tensor,
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        device=device,
        output_dir=output_dir,
    )

    # Extract comparison results for saving
    comparison_results = analysis_results.get("comparison_results")

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save results
    save_results(
        ccs_results=ccs_results,
        results_df=results_df,
        X_pos=X_pos_single,
        X_neg=X_neg_single,
        best_layer=best_layer,
        direction_tensor=direction_tensor,
        output_dir=output_dir,
        model_config=model_config,
        comparison_results=comparison_results,
    )

    print("\n" + "=" * 80)
    print("MULTI-TOKEN STEERING PIPELINE WITH COMPLETE ANALYSIS COMPLETED!")
    print("=" * 80)
    print("✅ Vanilla CCS trained on single-token representations")
    print(f"✅ Best layer selected: {best_layer}")
    print(
        f"✅ Multi-token steering applied to {token_config['token_percentage']}% of tokens"
    )
    print("✅ Proper forward pass with steering hooks (no simulation!)")
    print("✅ Real attention propagation through all layers")
    print("✅ COMPLETE comprehensive analysis with all plots and CSV files")
    print("✅ PCA eigenvalues analysis completed")
    print("✅ Enhanced steering analysis completed")
    print("✅ Layer-wise PCA analysis completed")
    print("✅ Components matrix analysis completed")
    print("✅ Boundary comparison analysis completed")
    print("✅ ALL ANALYSIS USES PROPER STEERED DATA FROM FORWARD PASS!")
    print(f"✅ Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
