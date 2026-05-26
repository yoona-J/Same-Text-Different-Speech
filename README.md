# When the Same Transcript Sounds Different

Official repository for:

**When the Same Transcript Sounds Different: Diagnosing and Mitigating Recognition Instability in Low-Information Conversational ASR**

This repository provides code, experiment configurations, and analysis resources for diagnosing hallucination and recognition instability in low-information Korean conversational ASR and for reproducing the proposed Acoustic Conditioning framework.

> Note  
> This work uses Korean regional conversational speech data.  
> While experiments are conducted in Korean, the proposed analysis framework targets a broader ASR failure mode in which limited lexical evidence increases sensitivity to acoustic/prosodic variation.

---

# Overview

Modern ASR systems achieve strong average performance, but spontaneous conversational speech still produces:

- hallucination
- unstable decoding
- prediction inconsistency
- robustness degradation under low-information speech

This work introduces:

- **Low-information speech**
- **Same-transcript instability**
- **Acoustic Conditioning**

The proposed framework analyzes recognition instability caused by acoustic/prosodic variation and mitigates it using utterance-level acoustic representations injected into the encoder.

---

# Main Contributions

- Definition of low-information conversational speech
- Introduction of same-transcript instability
- Hallucination and instability analysis in Korean conversational ASR
- Lightweight Acoustic Conditioning for ESPnet
- Improved CER and hallucination under matched settings

---

# Repository Structure

```text
Same-Text-Different-Speech/
│
├── README.md
│
├── espnet/
│   └── espnet2/
│       ├── asr/
│       │   ├── acoustic_conditioning.py
│       │   └── espnet_model.py
│       └── bin/
│           └── asr_inference.py
│
├── espnet_asr/
│   └── recipe/
│       └── asr1/
│           ├── conf/
│           ├── local/
│           ├── scripts/
│           ├── dump/
│           ├── exp/
│           ├── logs/
│           ├── run_decode1.sh
│           ├── run_decode2.sh
│           ├── run_training.sh
```

---

# Environment

| Component | Setting |
|---|---|
| GPU | NVIDIA RTX 4090 |
| Framework | ESPnet / PyTorch |
| PyTorch | 2.5+ |
| Tokenization | Character-level Korean |
| Acoustic Extraction | Parselmouth + librosa |

---

# Dataset

The dataset is not directly included because of size constraints.

Expected ESPnet dump format: 

```text
dump/raw/
├── train/
├── dev/
└── test/
```

Each split contains:

```text
wav.scp
text
utt2spk
spk2utt
```

Optional evaluation subset:

```text
dump/raw/test_1k/
```

---

# Low-Information Definition

Low-information speech is defined as:

```text
duration <= 2.0 sec
OR
token_count <= 2
```

This reflects Korean conversational utterances with limited lexical information.

---

# Acoustic Features

11 utterance-level acoustic/prosodic features are used:

```text
f0_std
f0_range
token_per_sec
duration_per_token
silence_ratio
num_pauses
mean_pause
rms_energy
hnr_mean
jitter_local
shimmer_local
```

These features are projected through an MLP and injected into encoder representations through gated residual conditioning.

---

# ESPnet Source Modifications

The proposed framework modifies original ESPnet source code.

Modified files:

```text
espnet/espnet2/asr/acoustic_conditioning.py
espnet/espnet2/asr/espnet_model.py
espnet/espnet2/bin/asr_inference.py
```

## acoustic_conditioning.py

Purpose:

- Implements Acoustic Conditioning module
- Projects acoustic features
- Computes conditioning vector
- Applies gated residual conditioning

Role:

```text
acoustic_feat
        ↓
MLP projection
        ↓
sigmoid gate
        ↓
encoder input conditioning
```

---

## espnet_model.py

Purpose:

- Adds acoustic feature support into ESPnet ASR model
- Receives `acoustic_feat`
- Passes conditioned representation into encoder

Main modification:

```text
speech + acoustic conditioning
            ↓
encoder
```

---

## asr_inference.py

Purpose:

- Extends Speech2Text inference
- Allows decoding with:

```text
acoustic_feat,npy
```

Without this modification, inference cannot receive Acoustic Conditioning inputs.

---

# Reproduction Pipeline

Move to recipe:

```bash
cd espnet_asr/recipe/asr1
. ./path.sh
```

---

# Step 1. Baseline Encoder Training

Train baseline encoders:

```bash
./run_training.sh
```

These scripts launch:

- Conformer
- Transformer
- E-Branchformer
- Bi-LSTM

training experiments.

---

# Step 2. Acoustic Feature Extraction

Extract acoustic features from raw ESPnet data.

Script:

```text
local/extract_hallu_acoustic_features.py
```

Purpose:

- Reads wav.scp
- Reads transcript text
- Computes utterance-level acoustic features
- Saves normalized numpy features

Command:

```bash
python local/extract_hallu_acoustic_features.py \
--train_wav_scp dump/raw/train/wav.scp \
--train_text dump/raw/train/text \
--dev_wav_scp dump/raw/dev/wav.scp \
--dev_text dump/raw/dev/text \
--test_wav_scp dump/raw/test/wav.scp \
--test_text dump/raw/test/text \
--out_root dump/hallu_acoustic_feats
```

Output:

```text
dump/hallu_acoustic_feats/
├── acoustic_stats.json
├── train/acoustic_feats.scp
├── dev/acoustic_feats.scp
└── test/acoustic_feats.scp
```

For 1k evaluation:

```text
local/extract_hallu_acoustic_test_only.py
```

```text
dump/hallu_acoustic_feats/test_1k/
```

---

# Step 3. Collect Statistics

Example:

Acoustic Conditioning + E‑Branchformer + Transformer Decoder

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m espnet2.bin.asr_train \
  --collect_stats true \
  --allow_variable_data_keys true \
  --use_preprocessor true \
  --bpemodel none \
  --token_type char \
  --token_list data/token_list/char/tokens.txt \
  --non_linguistic_symbols none \
  --cleaner none \
  --g2p none \
  --train_data_path_and_name_and_type dump/raw/train/wav.scp,speech,sound \
  --train_data_path_and_name_and_type dump/raw/train/text,text,text \
  --train_data_path_and_name_and_type dump/hallu_acoustic_feats/train/acoustic_feats.scp,acoustic_feat,npy \
  --valid_data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound \
  --valid_data_path_and_name_and_type dump/raw/dev/text,text,text \
  --valid_data_path_and_name_and_type dump/hallu_acoustic_feats/dev/acoustic_feats.scp,acoustic_feat,npy \
  --config conf/tuning/train_asr_ebranchformer_hybrid_hallu.yaml \
  --output_dir exp/asr_stats_ebranchformer_hybrid_hallu \
  --ngpu 1
```

---

# Step 4. Training

## Baseline Hybrid

E‑Branchformer + CTC + Transformer Decoder

Training:

```bash
CUDA_VISIBLE_DEVICES=1 ./asr.sh \
  --stage 11 \
  --stop_stage 11 \
  --ngpu 1 \
  --use_lm false \
  --use_word_lm false \
  --token_type char \
  --train_set train \
  --valid_set dev \
  --test_sets test \
  --asr_config conf/tuning/train_asr_ebranchformer_hybrid.yaml \
  --asr_tag ebranchformer_hybrid
```

---

## Acoustic Conditioning + CTC-only

Training:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m espnet2.bin.asr_train ...
```

Config:

```text
train_asr_ebranchformer_ctc_hallu.yaml
```

---

## Acoustic Conditioning + Hybrid

Training:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m espnet2.bin.asr_train ...
```

Config:

```text
train_asr_ebranchformer_hybrid_hallu.yaml
```

---

# Step 5. Decoding

Prepare keys:

```bash
cut -d' ' -f1 dump/raw/test/wav.scp > dump/raw/test/keys.txt
```

Run scripts:

```bash
./run_decode_gpu0.sh
./run_decode_gpu1.sh
```

---

## Baseline Hybrid Decode

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m espnet2.bin.asr_inference ...
```

---

## Acoustic Conditioning Hybrid Decode

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m espnet2.bin.asr_inference ...
```

---

## Acoustic Conditioning CTC Decode

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m espnet2.bin.asr_inference ...
```

---

# Script Descriptions

## run_conformer_models.sh

Purpose:

- launches conformer experiments
- manages training automation

---

## run_transformer_ebranchformer_models.sh

Purpose:

- launches transformer and E‑Branchformer experiments
- handles hybrid model training

---

## run_decode_gpu0.sh / run_decode_gpu1.sh

Purpose:

- GPU-parallel decoding
- automatic inference execution
- experiment log generation

---

## local/

Contains:

- preprocessing
- feature extraction
- analysis utilities

Main script:

```text
extract_hallu_acoustic_features.py
```

---

## conf/

Contains:

- training configs
- decoding configs
- acoustic-conditioning configs

Examples:

```text
train_asr_ebranchformer_hybrid.yaml
train_asr_ebranchformer_ctc_hallu.yaml
train_asr_ebranchformer_hybrid_hallu.yaml
```

---

## exp/

Contains:

- trained checkpoints
- logs
- averaged models
- decoding outputs

---

## logs/

Contains:

- training logs
- decoding logs
- collect-stats logs

---

# Proposed Model

```text
Speech Features
        │
        ▼
Acoustic Conditioning
        │
        ▼
E‑Branchformer Encoder
        ├── CTC branch
        └── Transformer Decoder
```

No external LM is used unless explicitly configured.

---
