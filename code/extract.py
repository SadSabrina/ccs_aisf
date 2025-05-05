import torch
from tqdm import tqdm
import numpy as np


def get_llm_type(model_cfg) -> str:

    if model_cfg.config.is_encoder_decoder:
        return "encoder-decoder"

    model_type = model_cfg.config.model_type.lower()
    if model_type in ["bert", "roberta", "distilbert", "albert", "deberta", "deberta-v2"]:
        return "encoder"
    elif model_type in ["gpt2", "gpt", "gptj", "gpt_neo", "gpt_neox", "llama", "bloom", "opt", "falcon"]:
        return "decoder"

    return "unknown"


def extract_representation(
    model, tokenizer, text,
    layer_index=None, get_all_hs=False, strategy="first-token",
    model_type=None, use_decoder=False, device=None
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
        - device 

    Return:
        - numpy-array (n_layers, dim)
    """
    if device is None:

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    model = model.to(device)

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")  

    # model_type 
    if model_type is None:
        model_type = get_llm_type(model)

    # Forward pass 
    if model_type == 'encoder-decoder': # Encoder-Decoder models

      enc_inputs = tokenizer(text, return_tensors="pt").to(device)
      dec_inputs = tokenizer("", return_tensors="pt").to(device)

      with torch.no_grad():
        outputs = model(enc_inputs.input_ids, 
                        decoder_input_ids=dec_inputs.input_ids, 
                        output_hidden_states=True)
    
    # Encoder only OR Decoder only models
    else: 
      with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden states
    if use_decoder and "decoder_hidden_states" in outputs:
        hs_tuple = outputs["decoder_hidden_states"]

    elif "encoder_hidden_states" in outputs:
        hs_tuple = outputs["encoder_hidden_states"]

    # Encoder only OR Decoder only models
    else:
        hs_tuple = outputs["hidden_states"]

    # Select the layer

    if get_all_hs:
      if (model_type == 'encoder') or (model_type == 'encoder-decoder' and use_decoder==False):
        hs = np.array(hs_tuple)
        hs = hs[:, :, 0].reshape(hs.shape[0], -1)  #  CLS token

        return hs

      if model_type == 'decoder' or (model_type == 'encoder-decoder' and use_decoder==True):
        hs = np.array(hs_tuple)
        hs = hs[:, :, -1].reshape(hs.shape[0], -1)  # last token

        return hs

    else:
      if layer_index is None: 
        layer_index = len(hs_tuple) // 2  # middle layer as default
          
      hs = hs_tuple[layer_index][0]  # shape: (seq_len, dim)
    # Strategy for single layer

    if strategy == "mean":
        return hs.mean(dim=0).detach().cpu().numpy()

    elif strategy == "first-token": # for encoder models (or encoder part if encoder-decoder)
      return hs[0].detach().cpu().numpy()
    
    elif strategy == "last-token": # for decoder models (or decoder part if encoder-decoder)
      return hs[0].detach().cpu().numpy()

    else:
        raise ValueError("strategy must be 'mean', 'last-token' or 'first-token'")
    

def vectorize_df(df_text, model, tokenizer, layer_index=None, strategy="last_token", model_type=None, use_decoder=False, get_all_hs=False):
    """
    Converts the df["text"] column to an embedding matrix (n_samples, n_layers, hidden_dim)

    Args:
        df: pandas DataFrame with column "text" (can be othet column name)
        model: HuggingFace model
        tokenizer: tokenizef
        layer_index: layer index (int or None - mean as default)
        strategy: 'first-token', 'last-token' or 'mean'
        model_type: str ('encoder', 'encoder-decoder', 'decoder')
        use_decoder: use decoder hidden states for encoder-decoder models

    Return:
        numpy-array (n_samples, n_layers, hidden_dim)
    """
    embeddings = []
    for text in tqdm(range(0, len(df_text)), desc="Extracting embeddings"):
        vec = extract_representation(
            model=model,
            tokenizer=tokenizer,
            text=df_text[text],
            layer_index=layer_index,
            strategy=strategy,
            model_type=model_type,
            use_decoder=use_decoder,
            get_all_hs=get_all_hs
        )
        embeddings.append(vec)
    return np.stack(embeddings)