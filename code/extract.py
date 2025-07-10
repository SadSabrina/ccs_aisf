import numpy as np
import torch
from tqdm import tqdm


def get_llm_type(model_cfg) -> str:
    # Changed: More robust model type detection with debug output

    # First check the model config's model_type attribute
    if hasattr(model_cfg.config, "model_type"):
        model_type = model_cfg.config.model_type.lower()
        print(f"   Model type from config: {model_type}")

        # BERT and similar encoder models
        if model_type in [
            "bert",
            "roberta",
            "distilbert",
            "albert",
            "deberta",
            "deberta-v2",
        ]:
            print("   → Classified as encoder model")
            return "encoder"
        # Decoder models
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
        ]:
            print("   → Classified as decoder model")
            return "decoder"

    # Only check is_encoder_decoder for actual encoder-decoder models
    if (
        hasattr(model_cfg.config, "is_encoder_decoder")
        and model_cfg.config.is_encoder_decoder
    ):
        print("   → Found is_encoder_decoder=True, classified as encoder-decoder")
        return "encoder-decoder"

    print("   → Unknown model type, defaulting to encoder")
    return "encoder"  # Default to encoder for safety


def extract_representation(
    model,
    tokenizer,
    text,
    layer_index=None,
    get_all_hs=False,
    strategy="first-token",
    model_type=None,
    use_decoder=False,
    device=None,
):
    """
    Extracts the vector representation from the given model layer or all model layers.

    Args:
        - model: HuggingFace model
        - tokenizer: model tokenizer
        - text: input string
        - layer_index: layer number (as default middle layer)
        - strategy: 'first-token' (as default) 'last-token' or 'mean'
        - model_type: "encoder", "encoder-decoder", "decoder"
        - use_decoder: if True use decoder hidden states (for encoder-decoder models)
        - device: computing device

    Return:
        - numpy-array (n_layers, dim)
    """
    # Changed: Explicit device checking instead of multiple fallbacks
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(model.device)

    # Tokenize and keep on GPU
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # model_type detection
    if model_type is None:
        model_type = get_llm_type(model)

    # Forward pass
    if model_type == "encoder-decoder":  # Encoder-Decoder models
        enc_inputs = tokenizer(text, return_tensors="pt").to(model.device)
        dec_inputs = tokenizer("", return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(
                enc_inputs.input_ids,
                decoder_input_ids=dec_inputs.input_ids,
                output_hidden_states=True,
            )

    # Encoder only OR Decoder only models
    else:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden states - keep on GPU as long as possible
    if use_decoder and "decoder_hidden_states" in outputs:
        hs_tuple = outputs["decoder_hidden_states"]
    elif "encoder_hidden_states" in outputs:
        hs_tuple = outputs["encoder_hidden_states"]
    # Encoder only OR Decoder only models
    else:
        hs_tuple = outputs["hidden_states"]

    # Select the layer
    if get_all_hs:
        reps = []
        for layer_h in hs_tuple:
            # shape: (1, seq_len, dim) - keep on GPU during processing
            if strategy == "first-token":
                rep = layer_h[:, 0, :].squeeze(0)
            elif strategy == "last-token":
                rep = layer_h[:, -1, :].squeeze(0)
            elif strategy == "mean":
                rep = layer_h.mean(dim=1).squeeze(0)
            else:
                raise ValueError(
                    "strategy must be 'mean', 'last-token' or 'first-token'"
                )

            # Changed: Only move to CPU at the very end
            reps.append(rep.cpu().numpy())
        return np.stack(reps)

    else:
        if layer_index is None:
            layer_index = len(hs_tuple) // 2  # middle layer as default

        hs = hs_tuple[layer_index][0]  # shape: (seq_len, dim)

        # Strategy for single layer
        if strategy == "mean":
            result = hs.mean(dim=0)
        elif (
            strategy == "first-token"
        ):  # for encoder models (or encoder part if encoder-decoder)
            result = hs[0]
        elif (
            strategy == "last-token"
        ):  # for decoder models (or decoder part if encoder-decoder)
            result = hs[-1]
        else:
            raise ValueError("strategy must be 'mean', 'last-token' or 'first-token'")

        # Changed: Only move to CPU at the very end
        return result.detach().cpu().numpy()


def vectorize_df(
    df_text,
    model,
    tokenizer,
    layer_index=None,
    strategy="last_token",
    model_type=None,
    use_decoder=False,
    get_all_hs=False,
    device=None,
):
    """
    Converts the df["text"] column to an embedding matrix (n_samples, n_layers, hidden_dim)

    Args:
        df_text: list of text strings (can be pandas Series or list)
        model: HuggingFace model
        tokenizer: tokenizer
        layer_index: layer index (int or None - mean as default)
        strategy: 'first-token', 'last-token' or 'mean'
        model_type: str ('encoder', 'encoder-decoder', 'decoder')
        use_decoder: use decoder hidden states for encoder-decoder models
        get_all_hs: whether to extract all hidden states
        device: computing device

    Return:
        numpy-array (n_samples, n_layers, hidden_dim)
    """
    # Changed: Pre-allocate device if not provided
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    embeddings = []

    # Changed: Process in batches to optimize GPU usage for large datasets
    batch_size = min(8, len(df_text))  # Small batch size to avoid memory issues

    if len(df_text) > batch_size:
        # Process in batches for better GPU utilization
        for batch_start in tqdm(
            range(0, len(df_text), batch_size), desc="Extracting embeddings (batched)"
        ):
            batch_end = min(batch_start + batch_size, len(df_text))
            batch_texts = df_text[batch_start:batch_end]

            batch_embeddings = []
            for text in batch_texts:
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
                )
                batch_embeddings.append(vec)

            embeddings.extend(batch_embeddings)
    else:
        # Process individually for small datasets
        for text in tqdm(range(0, len(df_text)), desc="Extracting embeddings"):
            vec = extract_representation(
                model=model,
                tokenizer=tokenizer,
                text=df_text[text],
                layer_index=layer_index,
                strategy=strategy,
                model_type=model_type,
                use_decoder=use_decoder,
                get_all_hs=get_all_hs,
                device=device,
            )
            embeddings.append(vec)

    return np.stack(embeddings)
