#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export LC_ALL=C

COMMON_ARGS="\
  --ngpu 1 \
  --inference_nj 8 \
  --use_lm false \
  --use_word_lm false \
  --token_type char \
  --train_set train \
  --valid_set dev \
  --test_sets test"

mkdir -p logs

# BiLSTM
./asr.sh \
  --stage 12 \
  --stop_stage 12 \
  $COMMON_ARGS \
  --asr_config conf/tuning/train_asr_bilstm_ctc.yaml \
  --asr_tag bilstm_ctc_full \
  2>&1 | tee logs/decode_bilstm_$(date +%Y%m%d_%H%M%S).log

# Transformer
./asr.sh \
  --stage 12 \
  --stop_stage 12 \
  $COMMON_ARGS \
  --asr_config conf/tuning/train_asr_transformer_ctc.yaml \
  --asr_tag transformer_ctc_full \
  2>&1 | tee logs/decode_transformer_$(date +%Y%m%d_%H%M%S).log