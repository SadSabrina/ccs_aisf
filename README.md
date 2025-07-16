#

--------

### upd 13.07 from lena with data

* _i hate notebooks and push to main with whole my soul!!!_

Clear data folder:

* [data_dep](data_dep) folder - all old dataset files for back compatibility
* [data](data) folder - with correct data and only deprecated _total_hate2.0.csv_ files
* [notebooks/SC_on_statements.ipynb](notebooks/SC_on_statements.ipynb) - sanity check notebook on data from [data](data) folder

Data to run models:

* (_deprecated NOT_) [data/raw/total_hate2.0.csv](data/raw/total_hate2.0.csv) - __NOT__ data from Sabrina ([commit link](https://github.com/SadSabrina/ccs_aisf/commit/bde79cb583f66bb65d19977bdf4ed1b1a23040c5))
  * reference yes [data/yes_no/hate_total_not_yes_data.csv](data/yes_no/hate_total_not_yes_data.csv)
  * reference no [data/yes_no/hate_total_not_no_data.csv](data/yes_no/hate_total_not_no_data.csv)

  ```
  Label 0 (harmful): 625 total, 168 contain 'not' (26.9%)
  Label 1 (safe): 625 total, 459 contain 'not' (73.4%)
  ```

* [data/raw/total_hate_data.csv](data/raw/total_hate_data.csv) - mixed data
  * reference yes [data/yes_no/hate_total_yes_data.csv](data/yes_no/hate_total_yes_data.csv)
  * reference no [data/yes_no/hate_total_no_data.csv](data/yes_no/hate_total_no_data.csv)

  ```
  Label 0 (harmful): 622 total, 37 contain 'not' (5.9%)
  Label 1 (safe): 622 total, 290 contain 'not' (46.6%)
  ```

* (__current balanced NOT__) [data/raw/total_hate3.0.csv](data/raw/total_hate3.0.csv) - __NOT__ data from Lena (this)
  * referenced yes [data/yes_no/hate_total3.0_yes.csv](data/yes_no/hate_total3.0_yes.csv)
  * referenced no [data/yes_no/hate_total3.0_no.csv](data/yes_no/hate_total3.0_no.csv)

  ```
  Label 0 (harmful): 625 total, 320 contain 'not' (51.2%)
  Label 1 (safe): 625 total, 297 contain 'not' (47.5%)
  ```

# KONI

[ccs_aisf/koni](https://drive.google.com/drive/folders/1w9OdbXfB5CRtGrMhA6lQ088ARh1EcRH9?usp=drive_link) - folder with npz and pkl files on google

## Main Models

| Model Name | Type | Size/Architecture | Hugging Face Link | Notebook Link | Data Folder |
|------------|------|-------------------|-------------------|---------------|-------------|
| __Gemma 2 Series__ |
| gemma-2-2b | Base | 2B parameters | | [notebooks/CCS_on_statements_gemma-2-2b.ipynb](notebooks/CCS_on_statements_gemma-2-2b.ipynb)| |
| gemma-2-2b-it | Instruction Tuned | 2B parameters | | [notebooks/CCS_on_statements_gemma-2-2b-it.ipynb](notebooks/CCS_on_statements_gemma-2-2b-it.ipynb)| |
| gemma-2-9b | Base | 9B parameters | | [notebooks/CCS_on_statements_gemma-2-9b.ipynb](notebooks/CCS_on_statements_gemma-2-9b.ipynb) | |
| gemma-2-9b-it | Instruction Tuned | 9B parameters | | [notebooks/CCS_on_statements_gemma-2-9b-it.ipynb](notebooks/CCS_on_statements_gemma-2-9b-it.ipynb) | |
| __Gemma 3 Series__ |
| gemma-3-1b-it | Instruction Tuned | 1B parameters | | | |
| gemma-3-4b-it | Instruction Tuned | 4B parameters | | | |
| gemma-3-12b-it | Instruction Tuned | 12B parameters | | | |
| gemma-3-27b-it | Instruction Tuned | 27B parameters (float16) | | | |
| __Meta Llama 3 Series__ |
| Meta-Llama-3-8B | Base | 8B parameters | | [notebooks/CCS_on_statement_Meta-Llama-3-8B.ipynb](notebooks/CCS_on_statement_Meta-Llama-3-8B.ipynb) | |
| Meta-Llama-3-8B-Instruct | Instruction Tuned | 8B parameters | | [notebooks/CCS_on_statement_Meta-Llama-3-8B-Instruct.ipynb](notebooks/CCS_on_statement_Meta-Llama-3-8B-Instruct.ipynb) | |
| Meta-Llama-3-70B | Base | 70B parameters | | | |
| Meta-Llama-3-70B-Instruct | Instruction Tuned | 70B parameters | | | |
| __Meta Llama 2 Guard__ |
| Llama-Guard-2-8B | Safety classifier based on Llama 3 8B | 8B parameters | | | |
| __Meta Llama 3.1 Series__ |
| Meta-Llama-3.1-8B | Base | 8B parameters | | | |
| Meta-Llama-3.1-8B-Instruct | Instruction Tuned | 8B parameters | | | |
| Meta-Llama-3.1-405B | Base | 405B parameters | | | |
| Meta-Llama-3.1-405B-Instruct | Instruction Tuned | 405B parameters | | | |
| __Llama 3.2 Series__ |
| Llama-3.2-1B | Base | 1B parameters | | | |
| Llama-3.2-3B | Base | 3B parameters | | | |
| Llama-3.2-3B-Instruct | Instruction Tuned | 3B parameters | | | |
| __Llama 4 Series__ |
| Llama-4-Scout | Base | MoE 16 experts (17B) | | | |
| Llama-4-Scout-Instruct | Instruction Tuned | MoE 16 experts (17B) | | | |
| Llama-4-Maverick | Base | MoE 107 experts (17B) | | | |
| Llama-4-Maverick-Instruct | Instruction Tuned | MoE 107 experts (17B) | | | |

## Additional Models (по остаточному принципу)

| Model Name | Type | Size | Hugging Face Link | Notebook Link | Data Folder |
|------------|------|------|-------------------|---------------|-------------|
| gemma-3n-e2b-it | Instruction Tuned | 2B parameters | | | |
| gemma-3n-e4b-it | Instruction Tuned | 4B parameters | | | |

## Notes

* __it__ = Instruction Tuning
* __Scout__ = MoE (Mixture of Experts) with 16 experts, ~17B parameters
* __Maverick__ = MoE (Mixture of Experts) with 107 experts, ~17B parameters
* __float16__ = Model uses 16-bit floating point precision

## Quick Reference

### Model Categories

* __Base Models__: Pre-trained but not instruction-tuned

* __Instruction Tuned__: Fine-tuned for following instructions and chat
* __MoE__: Mixture of Experts architecture for better efficiency

### Size Categories

* __Small__: 1B-3B parameters

* __Medium__: 8B-12B parameters  
* __Large__: 27B+ parameters
* __Extra Large__: 405B parameters
