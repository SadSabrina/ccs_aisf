# 🧪 List of Models for Experiments

---

## 1. Encoder-only: DeBERTa

**Token position:** `first`  
**Model:** `AutoTokenizer` + `AutoModelForMaskedLM`

| Name | Description | Hugging Face ID |
|------|-------------|------------------|
| DeBERTa Base | 13 layers | `microsoft/deberta-base` |
| DeBERTa V3 Small (pretrained) | 7 layers, hate speech fine-tuned | `Narrativaai/deberta-v3-small-finetuned-hate_speech18` |
| DeBERTa Large | 25 layers | `microsoft/deberta-large` |
| DeBERTa Large (pretrained) | 25 layers, hate speech fine-tuned | `Elron/deberta-v3-large-hate` |

---

## 2. Decoder-only: GPT

**Token position:** `last`  
**Model:** `AutoTokenizer` + `AutoModelForCausalLM`

| Name | Description | Hugging Face ID |
|------|-------------|------------------|
| GPT-2 | 13 layers | `gpt2` |
| GPT-2 Large | 37 layers | `gpt2-large` |
| GPT-Neo (pretrained) | 13 layers, detox objective | `ybelkada/gpt-neo-125m-detox` |

---

## 3. Encoder-Decoder: BERT (Hate Speech)

**Token positions:**  
- `encoder`: first token  
- `decoder`: last token  

**Model:** `BertTokenizer` + `EncoderDecoderModel`

| Name | Description | Hugging Face ID |
|------|-------------|------------------|
| BERT Base | 12 + 12 layers | `google-bert/bert-base-uncased` |
| BERT Base (pretrained) | fine-tuned on hate speech | `ayushdh96/HateSpeech_Bert_Base_Uncased_Fine_Tuned` |
| BERT Base (5 + 5) | fine-tuned on English + Turkish tweets | `ctoraman/hate-speech-bert` |

---

## 4. Decoder-only big models: OpenLLaMA (Causal), Gemma

**Token position:** `last`  
**Model:** `LlamaTokenizer` + `AutoModelForCausalLM` (Llama)
**Model:** `AutoTokenizer` + `AutoModelForCausalLM` (Gemma)


| Name | Version | Description | Hugging Face ID |
|------|---------|-------------|------------------|
| OpenLLaMA 3B | v1 | base | `openlm-research/open_llama_3b` |
| OpenLLaMA 13B | v1 | base | `openlm-research/open_llama_13b` |
| GEMMA 2B | - | base | `google/gemma-2b` |
| GEMMA 7B | - | base | `google/gemma-7b` |
