import torch
from tqdm import tqdm


def extract_representation(
    model, tokenizer, text,
    strategy="first-token",
    model_type=None, use_decoder=False, device=None, token_number=None
):
    """
    Extracts the vector representation from the given model layer or all model layers.

    Args:
        - model: HuggingFace model
        - tokenizer: model tokenizer
        - text: input string or list of strings
        - strategy: 'first-token' (as default) 'last-token' or 'mean'
        - model_type: "encoder", "encoder-decoder", "decoder"
        - use_decoder: if True use decoder hidden states (for encoder-decoder models)
        - device 

    Return:
        - torch.Tensor (batch_size, n_layers, dim) or (n_layers, dim) for single text
    """
    # CHANGED: Simplified device handling
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    is_single = isinstance(text, str)
    if is_single:
        text = [text]

    # Forward pass 
    if model_type == 'encoder-decoder':
        enc_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(model.device)
        dec_inputs = tokenizer([""] * len(text), return_tensors="pt", padding=True, truncation=True, max_length=64).to(model.device)

        with torch.no_grad():
            outputs = model(enc_inputs.input_ids, 
                            decoder_input_ids=dec_inputs.input_ids, 
                            output_hidden_states=True)
            # Extract hidden states
        if use_decoder and "decoder_hidden_states" in outputs:
            hs_tuple = outputs["decoder_hidden_states"]
            attention_mask = dec_inputs.attention_mask
        elif "encoder_hidden_states" in outputs:
            hs_tuple = outputs["encoder_hidden_states"]
            attention_mask = enc_inputs.attention_mask
    else: 
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        attention_mask = inputs.attention_mask
        hs_tuple = outputs["hidden_states"]

    # CHANGED: Removed unnecessary variables, direct processing
    reps = []
    for layer_h in hs_tuple:
        if strategy == "first-token":
            reps.append(layer_h[:, 0, :])
        elif strategy == "last-token":
            # CHANGED: Direct indexing without intermediate variables
            reps.append(layer_h[torch.arange(layer_h.size(0), device=layer_h.device), 
                                attention_mask.sum(dim=1) - 1, :])
        elif strategy == "mean":
            # CHANGED: Direct calculation without intermediate variables
            reps.append((layer_h * attention_mask.unsqueeze(-1).float()).sum(dim=1) / 
                       attention_mask.sum(dim=1, keepdim=True).float())
        elif strategy == "custom":
            reps.append(layer_h[:, token_number, :])
        else:
            raise ValueError("strategy must be 'mean', 'last-token' or 'first-token'")
    
    result = torch.stack(reps, dim=1)
    return result.squeeze(0) if is_single else result


def vectorize_df(df_text, model, tokenizer, batch_size=32, strategy="last_token", 
                 model_type=None, use_decoder=False, device=None, token_number=None):
    """
    Converts the df["text"] column to an embedding matrix using batching

    Args:
        df_text: list, array, or pandas Series of texts
        model: HuggingFace model
        tokenizer: tokenizer
        batch_size: batch size for processing
        strategy: 'first-token', 'last-token' or 'mean'
        model_type: str ('encoder', 'encoder-decoder', 'decoder')
        use_decoder: use decoder hidden states for encoder-decoder models

    Return:
        torch.Tensor (n_samples, n_layers, hidden_dim) on GPU
    """
    # CHANGED: Direct conversion without intermediate variable
    texts = df_text.tolist() if hasattr(df_text, 'tolist') else list(df_text)
    
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        # CHANGED: Direct function call without storing batch_vec
        embeddings.append(extract_representation(
            model=model,
            tokenizer=tokenizer,
            text=texts[i:i + batch_size],
            strategy=strategy,
            model_type=model_type,
            use_decoder=use_decoder,
            device=device,
            token_number=token_number
        ))
    
    return torch.cat(embeddings, dim=0)