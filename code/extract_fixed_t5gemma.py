import numpy as np
import torch
from tqdm import tqdm


def get_llm_type(model_cfg) -> str:
    """
    Determine the model type based on configuration
    """
    if model_cfg.config.is_encoder_decoder:
        return "encoder-decoder"

    model_type = model_cfg.config.model_type.lower()
    if model_type in [
        "bert",
        "roberta",
        "distilbert",
        "albert",
        "deberta",
        "deberta-v2",
    ]:
        return "encoder"
    elif model_type in [
        "gpt2",
        "gpt",
        "gptj",
        "gpt_neo",
        "gpt_neox",
        "llama",
        "bloom",
        "opt",
        "falcon",
        "gemma",
    ]:
        return "decoder"

    return "unknown"


def extract_representation(
    model,
    tokenizer,
    text,
    layer_index=None,
    get_all_hs=False,
    strategy="last-token",
    model_type=None,
    use_decoder=False,
    device=None,
    token_number=None,
):
    """
    FIXED: Extracts the vector representation from the given model layer or all model layers.
    MAJOR CHANGE: Proper encoder-decoder handling with meaningful decoder input for different strategies.

    Args:
        - model: HuggingFace model
        - tokenizer: model tokenizer
        - text: input string
        - layer_index: layer number (as default middle layer)
        - strategy: 'first-token' (for encoders), 'last-token' (for decoders), 'mean', or 'custom'
        - model_type: "encoder", "encoder-decoder", "decoder"
        - use_decoder: if True use decoder hidden states (for encoder-decoder models)
        - device: computation device
        - token_number: specific token position for 'custom' strategy

    Return:
        - torch.Tensor (n_layers, dim) if get_all_hs=True, else (dim,) - kept on GPU
    """
    # Enhanced device handling - prefer GPU
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        elif hasattr(model, "device"):
            device = model.device
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Enhanced model type detection
    if model_type is None:
        model_type = get_llm_type(model)

    # MAJOR CHANGE: Proper encoder-decoder handling for T5GemmaForConditionalGeneration
    if model_type == "encoder-decoder":
        # Tokenize input for encoder
        enc_inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        
        # Check for empty inputs
        if enc_inputs["input_ids"].shape[1] == 0:
            raise ValueError("Empty input tensor detected - check tokenization")

        if use_decoder:
            # CHANGED: Create decoder input sequence for multiple token positions
            # The decoder receives encoder outputs internally, we just need decoder input tokens
            
            # Get appropriate decoder start token
            if hasattr(tokenizer, 'decoder_start_token_id') and tokenizer.decoder_start_token_id is not None:
                decoder_start_token_id = tokenizer.decoder_start_token_id
            elif hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                decoder_start_token_id = tokenizer.bos_token_id
            elif hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                decoder_start_token_id = tokenizer.pad_token_id
            else:
                decoder_start_token_id = 0  # fallback
            
            batch_size = enc_inputs["input_ids"].shape[0]
            encoder_length = enc_inputs["input_ids"].shape[1]
            
            # MAJOR FIX: Create decoder input with multiple tokens so different strategies work
            # Use the full encoder input as decoder input (this is common practice)
            # The encoder-decoder attention will handle the actual encoder output connection
            decoder_input_ids = torch.cat([
                torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device),
                enc_inputs["input_ids"][:, :-1]  # Shift right: remove last token, prepend start token
            ], dim=1)
            
            # Create proper attention mask for decoder input
            decoder_attention_mask = torch.cat([
                torch.ones((batch_size, 1), device=device),  # Start token is always attended
                enc_inputs.get("attention_mask", torch.ones_like(enc_inputs["input_ids"]))[:, :-1]
            ], dim=1)
            
            with torch.no_grad():
                # CHANGED: Normal encoder-decoder forward pass
                # Encoder processes input, decoder receives encoder outputs + decoder input
                outputs = model(
                    input_ids=enc_inputs["input_ids"],
                    attention_mask=enc_inputs.get("attention_mask", None),
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
                
                # Use decoder hidden states - now has multiple positions for different strategies
                hs_tuple = outputs.decoder_hidden_states
                attention_mask = decoder_attention_mask  # Use decoder attention mask for strategies
                
        else:
            # CHANGED: For encoder extraction, use encoder_hidden_states from full forward pass
            with torch.no_grad():
                # Use a minimal decoder input to get encoder hidden states
                dummy_decoder_input = torch.full(
                    (enc_inputs["input_ids"].shape[0], 1), 
                    tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                    dtype=torch.long, 
                    device=device
                )
                
                outputs = model(
                    input_ids=enc_inputs["input_ids"],
                    attention_mask=enc_inputs.get("attention_mask", None),
                    decoder_input_ids=dummy_decoder_input,
                    output_hidden_states=True
                )
                
                hs_tuple = outputs.encoder_hidden_states
                attention_mask = enc_inputs.get("attention_mask", None)

    # Encoder only OR Decoder only models
    else:
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        
        attention_mask = inputs.get("attention_mask", None)
        
        # Check for empty inputs
        if inputs["input_ids"].shape[1] == 0:
            raise ValueError("Empty input tensor detected - check tokenization")
            
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hs_tuple = outputs["hidden_states"]

    # Enhanced extraction logic with proper handling for all cases, keep on GPU
    if get_all_hs:
        reps = []
        for layer_idx, layer_h in enumerate(hs_tuple):
            # Keep on GPU and convert to float32
            layer_h = layer_h.float()

            # Check for empty tensors
            if layer_h.shape[1] == 0:
                print(f"Warning: Empty hidden states at layer {layer_idx}, skipping...")
                continue

            # Handle different extraction strategies properly
            if strategy == "first-token":
                # Use first token (good for encoder models)
                extracted = layer_h[:, 0, :].squeeze(0)

            elif strategy == "last-token":
                # Use attention mask to find actual last token, not padding
                if attention_mask is not None and attention_mask.shape[1] > 0:
                    # Find last non-padding token for each sequence
                    # FIXED: Ensure indices are long integers
                    last_token_indices = (attention_mask.sum(dim=1) - 1).long()
                    batch_size = layer_h.shape[0]

                    if batch_size == 1:
                        # Single sequence case
                        last_hidden = layer_h[0, last_token_indices[0], :]
                        extracted = last_hidden
                    else:
                        # Multiple sequences case
                        last_hidden = layer_h[
                            torch.arange(batch_size, device=device, dtype=torch.long), last_token_indices
                        ]
                        extracted = last_hidden.squeeze()
                else:
                    # Fallback to simple last token if no attention mask
                    extracted = layer_h[:, -1, :].squeeze(0)

            elif strategy == "mean":
                # Use attention mask for proper mean pooling (exclude padding), keep on GPU
                if attention_mask is not None and attention_mask.shape[1] > 0:
                    # Expand mask to match hidden states dimensions
                    expanded_mask = (
                        attention_mask.unsqueeze(-1).expand(layer_h.size()).float()
                    )
                    # Mask out padding tokens
                    masked_hidden = layer_h * expanded_mask
                    # Sum and divide by actual sequence length
                    summed = torch.sum(masked_hidden, dim=1)
                    lengths = torch.sum(expanded_mask, dim=1)
                    # Avoid division by zero
                    lengths = torch.clamp(lengths, min=1.0)
                    mean_hidden = summed / lengths
                    extracted = mean_hidden.squeeze(0)
                else:
                    # Fallback to simple mean if no attention mask
                    extracted = layer_h.mean(dim=1).squeeze(0)

            elif strategy == "custom":
                # Add bounds checking for custom token number
                if token_number is not None:
                    seq_len = layer_h.shape[1]
                    if token_number < seq_len:
                        extracted = layer_h[:, token_number, :].squeeze(0)
                    else:
                        # Fallback to last available token if token_number is out of bounds
                        extracted = layer_h[:, -1, :].squeeze(0)
                        print(
                            f"Warning: token_number {token_number} >= seq_len {seq_len}, using last token"
                        )
                else:
                    raise ValueError(
                        "token_number must be specified for 'custom' strategy"
                    )

            else:
                raise ValueError(
                    "strategy must be 'mean', 'last-token', 'first-token', or 'custom'"
                )

            # Keep as GPU tensor, ensure float32 and handle any remaining numerical issues
            extracted = extracted.float()

            # Replace any inf/nan values with zeros on GPU
            extracted = torch.where(torch.isfinite(extracted), extracted, torch.zeros_like(extracted))

            reps.append(extracted)

        # Return as GPU tensor stack
        if len(reps) == 0:
            raise ValueError("No valid hidden states extracted")
            
        result = torch.stack(reps).float()

        # Final check for extreme values and clip if necessary, on GPU
        max_abs_val = torch.max(torch.abs(result))
        if max_abs_val > 1000:
            # print(
            #     f"Warning: Extreme values detected (max: {max_abs_val:.2f}), applying clipping..."
            # )
            clip_value = torch.quantile(torch.abs(result.flatten()), 0.995)
            result = torch.clamp(result, -clip_value, clip_value)

        return result

    else:
        # Single layer extraction
        if layer_index is None:
            layer_index = len(hs_tuple) // 2  # middle layer as default

        # Add bounds checking for layer_index
        if layer_index >= len(hs_tuple):
            layer_index = len(hs_tuple) - 1
            print(f"Warning: layer_index out of bounds, using layer {layer_index}")

        hs = hs_tuple[layer_index][0].float()  # Convert to float32

        # Check for empty hidden states
        if hs.shape[0] == 0:
            raise ValueError("Empty hidden states detected")

        # Apply same enhanced logic for single layer extraction
        if strategy == "mean":
            if attention_mask is not None and attention_mask.shape[1] > 0:
                expanded_mask = attention_mask.unsqueeze(-1).expand(hs.size()).float()
                masked_hidden = hs * expanded_mask
                summed = torch.sum(masked_hidden, dim=0)
                length = torch.sum(expanded_mask, dim=0)
                length = torch.clamp(length, min=1.0)  # Avoid division by zero
                result = (summed / length).float()
            else:
                result = hs.mean(dim=0).float()

        elif strategy == "first-token":
            result = hs[0].float()

        elif strategy == "last-token":
            if attention_mask is not None and attention_mask.shape[1] > 0:
                # FIXED: Ensure indices are long integers
                last_token_idx = (attention_mask.sum(dim=1) - 1).long()
                result = hs[last_token_idx[0]].float()
            else:
                result = hs[-1].float()

        elif strategy == "custom":
            if token_number is not None:
                seq_len = hs.shape[0]
                if token_number < seq_len:
                    result = hs[token_number].float()
                else:
                    result = hs[-1].float()
                    print(
                        f"Warning: token_number {token_number} >= seq_len {seq_len}, using last token"
                    )
            else:
                raise ValueError("token_number must be specified for 'custom' strategy")
        else:
            raise ValueError(
                "strategy must be 'mean', 'last-token', 'first-token', or 'custom'"
            )

        # Clean up result on GPU
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
        return result


def vectorize_df(
    df_text,
    model,
    tokenizer,
    layer_index=None,
    strategy="last-token",
    model_type=None,
    use_decoder=False,
    get_all_hs=False,
    device=None,
    token_number=None,
):
    """
    Converts the df_text to an embedding matrix keeping everything on GPU
    Enhanced with better error handling and progress reporting.

    Args:
        df_text: list or pandas Series with text data
        model: HuggingFace model
        tokenizer: tokenizer
        layer_index: layer index (int or None - middle layer as default)
        strategy: 'first-token', 'last-token', 'mean', or 'custom'
        model_type: str ('encoder', 'encoder-decoder', 'decoder')
        use_decoder: use decoder hidden states for encoder-decoder models
        get_all_hs: extract all hidden states layers
        device: computation device
        token_number: specific token position for 'custom' strategy

    Return:
        torch.Tensor (n_samples, n_layers, hidden_dim) if get_all_hs=True - kept on GPU
        torch.Tensor (n_samples, hidden_dim) if get_all_hs=False - kept on GPU
    """
    # Prefer GPU device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        elif hasattr(model, "device"):
            device = model.device
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    embeddings = []
    failed_extractions = 0

    # Better progress bar with more information
    for idx in tqdm(
        range(len(df_text)), desc=f"Extracting embeddings (strategy: {strategy})"
    ):
        # Handle different input types (list, pandas Series, etc.)
        if hasattr(df_text, "iloc"):
            text = df_text.iloc[idx]
        else:
            text = df_text[idx]

        # Skip empty or None texts
        if text is None or (isinstance(text, str) and len(text.strip()) == 0):
            print(f"Warning: Empty text at index {idx}, skipping...")
            failed_extractions += 1
            continue

        # Wrapped in explicit error checking instead of try-except
        text_valid = isinstance(text, str) and len(text.strip()) > 0
        if not text_valid:
            print(f"Warning: Invalid text at index {idx}, skipping...")
            failed_extractions += 1
            continue

        vec = extract_representation(
            model=model,
            tokenizer=tokenizer,
            text=text,
            layer_index=layer_index,
            strategy=strategy,
            model_type=model_type,
            use_decoder=use_decoder,
            get_all_hs=get_all_hs,
            device=device,
            token_number=token_number,
        )
        embeddings.append(vec)

    if failed_extractions > 0:
        print(
            f"Warning: {failed_extractions} extractions failed and were skipped"
        )

    # Better error handling for empty results
    if len(embeddings) == 0:
        raise ValueError("No successful extractions - all texts failed to process")

    # Stack tensors on GPU instead of converting to numpy
    first_shape = embeddings[0].shape
    all_same_shape = all(emb.shape == first_shape for emb in embeddings)
    
    if not all_same_shape:
        print("Warning: Inconsistent embedding shapes detected, attempting to fix...")
        # Find max dimensions
        if len(first_shape) == 1:
            max_dim = max(emb.shape[0] for emb in embeddings)
            padded_embeddings = []
            for emb in embeddings:
                if emb.shape[0] < max_dim:
                    padding = torch.zeros(max_dim - emb.shape[0], device=device, dtype=emb.dtype)
                    padded = torch.cat([emb, padding], dim=0)
                    padded_embeddings.append(padded)
                else:
                    padded_embeddings.append(emb)
            embeddings = padded_embeddings
        elif len(first_shape) == 2:
            max_dim0 = max(emb.shape[0] for emb in embeddings)
            max_dim1 = max(emb.shape[1] for emb in embeddings)
            padded_embeddings = []
            for emb in embeddings:
                if emb.shape[0] < max_dim0 or emb.shape[1] < max_dim1:
                    padded = torch.zeros(max_dim0, max_dim1, device=device, dtype=emb.dtype)
                    padded[:emb.shape[0], :emb.shape[1]] = emb
                    padded_embeddings.append(padded)
                else:
                    padded_embeddings.append(emb)
            embeddings = padded_embeddings

    # Stack as GPU tensor
    result = torch.stack(embeddings).float()

    # Final validation and cleanup on GPU
    print(f"Extraction completed: shape={result.shape}, dtype={result.dtype}, device={result.device}")
    result_max = torch.max(result)
    result_min = torch.min(result)
    result_std = torch.std(result)
    print(f"Stats: max={result_max:.3f}, min={result_min:.3f}, std={result_std:.3f}")

    # Final check for extreme values on GPU
    max_abs_val = torch.max(torch.abs(result))
    if max_abs_val > 1000:
        # print("Warning: Extreme values detected in final result, applying global clipping...")
        clip_value = torch.quantile(torch.abs(result.flatten()), 0.999)
        result = torch.clamp(result, -clip_value, clip_value)
        print(f"Applied clipping at ±{clip_value:.2f}")

    # Final check for inf/nan on GPU
    inf_count = torch.sum(torch.isinf(result))
    nan_count = torch.sum(torch.isnan(result))
    if inf_count > 0 or nan_count > 0:
        print(f"Warning: Found {inf_count} inf and {nan_count} nan values, replacing with zeros...")
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))

    return result


# Additional utility functions for model-specific configurations
def get_recommended_strategy(model_type):
    """
    Get recommended extraction strategy based on model type
    """
    if model_type == "encoder":
        return "first-token"
    elif model_type == "decoder":
        return "last-token"
    elif model_type == "encoder-decoder":
        return "mean"  # or "last-token" if using decoder
    else:
        return "mean"  # safe default


def validate_extraction_params(model, tokenizer, model_type, strategy):
    """
    Validate that extraction parameters make sense for the given model
    """
    detected_type = get_llm_type(model)

    if model_type != detected_type:
        print(
            f"Warning: model_type '{model_type}' doesn't match detected type '{detected_type}'"
        )

    recommended_strategy = get_recommended_strategy(detected_type)
    if strategy != recommended_strategy:
        print(
            f"Info: Using strategy '{strategy}', but '{recommended_strategy}' is recommended for {detected_type} models"
        )


# Example usage configurations for different models
def get_deberta_config():
    """Configuration for DeBERTa model"""
    return {
        "strategy": "first-token",
        "model_type": "encoder",
        "use_decoder": False,
        "get_all_hs": True,
    }


def get_gemma_config():
    """Configuration for Gemma model"""
    return {
        "strategy": "last-token",
        "model_type": "decoder",
        "use_decoder": False,
        "get_all_hs": True,
    }


def get_t5_config(use_decoder=True):
    """Configuration for T5 model"""
    return {
        "strategy": "last-token" if use_decoder else "first-token",
        "model_type": "encoder-decoder",
        "use_decoder": use_decoder,
        "get_all_hs": True,
    }