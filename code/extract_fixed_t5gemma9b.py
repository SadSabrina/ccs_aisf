import numpy as np
import torch
from tqdm import tqdm


def extract_representation_batch(
    model,
    tokenizer,
    texts,
    layer_index=None,
    get_all_hs=False,
    strategy="last-token",
    model_type=None,
    use_decoder=False,
    device=None,
    token_number=None,
    max_length=256
):
    """
    CHANGED: Batch version of extract_representation - processes multiple texts at once
    Much faster than processing one text at a time.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type is None:
        model_type = get_llm_type(model)
    
    # CHANGED: Batch tokenization with consistent padding
    if model_type == "encoder-decoder":
        # CHANGED: Batch tokenize all texts at once
        enc_inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length  # Shorter sequences = faster
        ).to(device)
        
        if enc_inputs["input_ids"].shape[1] == 0:
            return [None] * len(texts)
        
        if use_decoder:
            # CHANGED: Batch decoder input creation
            batch_size = enc_inputs["input_ids"].shape[0]
            
            if hasattr(tokenizer, 'decoder_start_token_id') and tokenizer.decoder_start_token_id is not None:
                decoder_start_token_id = tokenizer.decoder_start_token_id
            elif hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                decoder_start_token_id = tokenizer.bos_token_id
            else:
                decoder_start_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            
            decoder_input_ids = torch.cat([
                torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device),
                enc_inputs["input_ids"][:, :-1]
            ], dim=1)
            
            decoder_attention_mask = torch.cat([
                torch.ones((batch_size, 1), device=device),
                enc_inputs.get("attention_mask", torch.ones_like(enc_inputs["input_ids"]))[:, :-1]
            ], dim=1)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=enc_inputs["input_ids"],
                    attention_mask=enc_inputs.get("attention_mask", None),
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
                hs_tuple = outputs.decoder_hidden_states
                attention_mask = decoder_attention_mask
        else:
            # CHANGED: Encoder extraction with minimal decoder input
            dummy_decoder_input = torch.full(
                (enc_inputs["input_ids"].shape[0], 1), 
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                dtype=torch.long, 
                device=device
            )
            
            with torch.no_grad():
                outputs = model(
                    input_ids=enc_inputs["input_ids"],
                    attention_mask=enc_inputs.get("attention_mask", None),
                    decoder_input_ids=dummy_decoder_input,
                    output_hidden_states=True
                )
                hs_tuple = outputs.encoder_hidden_states
                attention_mask = enc_inputs.get("attention_mask", None)
    else:
        # CHANGED: Batch processing for encoder/decoder only models
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(device)
        
        attention_mask = inputs.get("attention_mask", None)
        
        if inputs["input_ids"].shape[1] == 0:
            return [None] * len(texts)
            
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hs_tuple = outputs["hidden_states"]
    
    # CHANGED: Batch extraction logic
    batch_size = len(texts)
    
    if get_all_hs:
        # CHANGED: Process all layers for all samples at once
        all_layer_results = []
        
        for layer_idx, layer_h in enumerate(hs_tuple):
            layer_h = layer_h.float()
            
            if layer_h.shape[1] == 0:
                continue
            
            # CHANGED: Batch strategy application
            if strategy == "first-token":
                layer_extracted = layer_h[:, 0, :]
            elif strategy == "last-token":
                if attention_mask is not None and attention_mask.shape[1] > 0:
                    last_token_indices = (attention_mask.sum(dim=1) - 1).long()
                    layer_extracted = layer_h[torch.arange(batch_size, device=device), last_token_indices]
                else:
                    layer_extracted = layer_h[:, -1, :]
            elif strategy == "mean":
                if attention_mask is not None and attention_mask.shape[1] > 0:
                    expanded_mask = attention_mask.unsqueeze(-1).expand(layer_h.size()).float()
                    masked_hidden = layer_h * expanded_mask
                    summed = torch.sum(masked_hidden, dim=1)
                    lengths = torch.sum(expanded_mask, dim=1)
                    lengths = torch.clamp(lengths, min=1.0)
                    layer_extracted = summed / lengths
                else:
                    layer_extracted = layer_h.mean(dim=1)
            elif strategy == "custom":
                if token_number is not None:
                    seq_len = layer_h.shape[1]
                    if token_number < seq_len:
                        layer_extracted = layer_h[:, token_number, :]
                    else:
                        layer_extracted = layer_h[:, -1, :]
                else:
                    raise ValueError("token_number must be specified for 'custom' strategy")
            else:
                raise ValueError("strategy must be 'mean', 'last-token', 'first-token', or 'custom'")
            
            # CHANGED: Clean up on GPU
            layer_extracted = torch.where(torch.isfinite(layer_extracted), layer_extracted, torch.zeros_like(layer_extracted))
            all_layer_results.append(layer_extracted)
        
        if len(all_layer_results) == 0:
            return [None] * len(texts)
        
        # CHANGED: Stack layers and return list of per-sample tensors
        stacked_layers = torch.stack(all_layer_results, dim=1)  # (batch_size, n_layers, hidden_dim)
        return [stacked_layers[i] for i in range(batch_size)]
    
    else:
        # CHANGED: Single layer batch processing
        if layer_index is None:
            layer_index = len(hs_tuple) // 2
        
        if layer_index >= len(hs_tuple):
            layer_index = len(hs_tuple) - 1
        
        hs = hs_tuple[layer_index].float()
        
        if hs.shape[1] == 0:
            return [None] * len(texts)
        
        # CHANGED: Apply strategy to entire batch
        if strategy == "mean":
            if attention_mask is not None and attention_mask.shape[1] > 0:
                expanded_mask = attention_mask.unsqueeze(-1).expand(hs.size()).float()
                masked_hidden = hs * expanded_mask
                summed = torch.sum(masked_hidden, dim=1)
                lengths = torch.sum(expanded_mask, dim=1)
                lengths = torch.clamp(lengths, min=1.0)
                batch_result = summed / lengths
            else:
                batch_result = hs.mean(dim=1)
        elif strategy == "first-token":
            batch_result = hs[:, 0, :]
        elif strategy == "last-token":
            if attention_mask is not None and attention_mask.shape[1] > 0:
                last_token_indices = (attention_mask.sum(dim=1) - 1).long()
                batch_result = hs[torch.arange(batch_size, device=device), last_token_indices]
            else:
                batch_result = hs[:, -1, :]
        elif strategy == "custom":
            if token_number is not None:
                seq_len = hs.shape[1]
                if token_number < seq_len:
                    batch_result = hs[:, token_number, :]
                else:
                    batch_result = hs[:, -1, :]
            else:
                raise ValueError("token_number must be specified for 'custom' strategy")
        else:
            raise ValueError("strategy must be 'mean', 'last-token', 'first-token', or 'custom'")
        
        # CHANGED: Clean up and return list
        batch_result = torch.where(torch.isfinite(batch_result), batch_result, torch.zeros_like(batch_result))
        return [batch_result[i] for i in range(batch_size)]


def vectorize_df_batched(
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
    batch_size=16,  # CHANGED: Added batch processing
    max_length=256,  # CHANGED: Reduced from 512 for speed
):
    """
    CHANGED: Batch processing version of vectorize_df for much faster extraction
    All data stays on GPU throughout the process.
    
    New Args:
        batch_size: Number of texts to process simultaneously
        max_length: Maximum sequence length (shorter = faster)
    """
    # CHANGED: GPU device preference
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # CHANGED: Enhanced model type detection
    if model_type is None:
        model_type = get_llm_type(model)
    
    embeddings = []
    failed_extractions = 0
    
    # CHANGED: Process in batches instead of one-by-one
    num_batches = (len(df_text) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Processing batches (size={batch_size})"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df_text))
        
        # CHANGED: Prepare batch texts
        batch_texts = []
        for idx in range(start_idx, end_idx):
            if hasattr(df_text, "iloc"):
                text = df_text.iloc[idx]
            else:
                text = df_text[idx]
            
            # CHANGED: Skip invalid texts but continue batch processing
            if text is None or not isinstance(text, str) or len(text.strip()) == 0:
                batch_texts.append("")  # Use empty string as placeholder
                failed_extractions += 1
            else:
                batch_texts.append(text.strip())
        
        # CHANGED: Skip empty batches
        if not any(batch_texts):
            continue
            
        # CHANGED: Extract batch embeddings
        batch_embeddings = extract_representation_batch(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            layer_index=layer_index,
            strategy=strategy,
            model_type=model_type,
            use_decoder=use_decoder,
            get_all_hs=get_all_hs,
            device=device,
            token_number=token_number,
            max_length=max_length
        )
        
        # CHANGED: Add valid embeddings to results
        for i, embedding in enumerate(batch_embeddings):
            if embedding is not None:
                embeddings.append(embedding)
    
    if failed_extractions > 0:
        print(f"Warning: {failed_extractions} extractions failed and were skipped")
    
    if len(embeddings) == 0:
        raise ValueError("No successful extractions - all texts failed to process")
    
    # CHANGED: Stack on GPU
    result = torch.stack(embeddings).float()
    
    # CHANGED: Quick stats without moving to CPU
    print(f"Batch extraction completed: shape={result.shape}, device={result.device}")
    
    return result