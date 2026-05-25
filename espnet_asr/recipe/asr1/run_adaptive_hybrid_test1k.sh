#!/usr/bin/env bash
set -e

. ./path.sh

mkdir -p logs

MODEL_TAG="transformer_hybrid"
EXP_DIR="exp/asr_transformer_hybrid"
CONFIG="${EXP_DIR}/config.yaml"
MODEL="${EXP_DIR}/valid.loss.best.pth"
GPU=0

python make_adaptive_buckets.py

run_decode () {
  BUCKET=$1
  CTC_WEIGHT=$2
  LM_WEIGHT=$3

  TEST_SET="test_1k_${BUCKET}"
  OUT_DIR="${EXP_DIR}/adaptive_${BUCKET}"

  echo "========================================="
  echo "[Decode] ${BUCKET} | ctc_weight=${CTC_WEIGHT}, lm_weight=${LM_WEIGHT}"
  echo "========================================="

  CUDA_VISIBLE_DEVICES=${GPU} python3 -m espnet2.bin.asr_inference \
    --batch_size 1 \
    --ngpu 1 \
    --data_path_and_name_and_type dump/raw/${TEST_SET}/wav.scp,speech,sound \
    --key_file dump/raw/${TEST_SET}/keys.txt \
    --asr_train_config ${CONFIG} \
    --asr_model_file ${MODEL} \
    --lm_train_config exp/lm_train_char/config.yaml \
    --lm_file exp/lm_train_char/valid.loss.best.pth \
    --ctc_weight ${CTC_WEIGHT} \
    --lm_weight ${LM_WEIGHT} \
    --beam_size 5 \
    --maxlenratio 1.0 \
    --output_dir ${OUT_DIR} \
    2>&1 | tee logs/adaptive_${BUCKET}.log
}

# 안정적 발화: acoustic/CTC 비중 높임
run_decode stable 0.5 0.1

# 중간 발화: 균형
run_decode mid 0.3 0.3

# 불안정 발화: decoder/LM prior 비중 높임
run_decode unstable 0.1 0.5

MERGE_DIR="${EXP_DIR}/adaptive_merged_test1k"
mkdir -p "${MERGE_DIR}/output.1/1best_recog"

cat \
  ${EXP_DIR}/adaptive_stable/1best_recog/text \
  ${EXP_DIR}/adaptive_mid/1best_recog/text \
  ${EXP_DIR}/adaptive_unstable/1best_recog/text \
  | sort > "${MERGE_DIR}/output.1/1best_recog/text"

python calc_cer_wer.py \
  dump/raw/test_1k/text \
  ${MERGE_DIR} \
  | tee logs/adaptive_hybrid_test1k_cer_wer.txt
