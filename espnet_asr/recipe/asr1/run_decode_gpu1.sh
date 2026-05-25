#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1
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

# Conformer
./asr.sh \
  --stage 12 \
  --stop_stage 12 \
  $COMMON_ARGS \
  --asr_config conf/tuning/train_asr_conformer_ctc.yaml \
  --asr_tag conformer_ctc_full \
  2>&1 | tee logs/decode_conformer_$(date +%Y%m%d_%H%M%S).log

# E-Branchformer
./asr.sh \
  --stage 12 \
  --stop_stage 12 \
  $COMMON_ARGS \
  --asr_config conf/tuning/train_asr_ebranchformer_ctc.yaml \
  --asr_tag ebranchformer_ctc_full \
  2>&1 | tee logs/decode_ebranchformer_$(date +%Y%m%d_%H%M%S).log