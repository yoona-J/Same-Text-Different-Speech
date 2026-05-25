#!/usr/bin/env bash
set -e

# ============================================================
# Adaptive Gated Decoding for test_1k buckets
#
# stable   : CTC 비중 높임
# mid      : CTC / LM 균형
# unstable : LM prior 비중 높임
# ============================================================

. ./path.sh

mkdir -p logs

GPU=0

EXP_DIR="exp/asr_transformer_hybrid"
ASR_CONFIG="${EXP_DIR}/config.yaml"
ASR_MODEL="${EXP_DIR}/valid.loss.best.pth"

LM_CONFIG="exp/lm_train_char/config.yaml"
LM_MODEL="exp/lm_train_char/valid.loss.best.pth"

MERGE_DIR="${EXP_DIR}/adaptive_gated_test1k"

# 기존 adaptive 결과 제거
rm -rf "${EXP_DIR}/adaptive_stable"
rm -rf "${EXP_DIR}/adaptive_mid"
rm -rf "${EXP_DIR}/adaptive_unstable"
rm -rf "${MERGE_DIR}"

run_decode () {
  BUCKET=$1
  CTC_WEIGHT=$2
  LM_WEIGHT=$3

  TEST_SET="test_1k_${BUCKET}"
  OUT_DIR="${EXP_DIR}/adaptive_${BUCKET}"

  echo "========================================="
  echo "[Decode] bucket=${BUCKET}"
  echo "test_set=${TEST_SET}"
  echo "ctc_weight=${CTC_WEIGHT}"
  echo "lm_weight=${LM_WEIGHT}"
  echo "output=${OUT_DIR}"
  echo "========================================="

  CUDA_VISIBLE_DEVICES=${GPU} python3 -m espnet2.bin.asr_inference \
    --batch_size 1 \
    --ngpu 1 \
    --data_path_and_name_and_type dump/raw/${TEST_SET}/wav.scp,speech,sound \
    --key_file dump/raw/${TEST_SET}/keys.txt \
    --asr_train_config ${ASR_CONFIG} \
    --asr_model_file ${ASR_MODEL} \
    --lm_train_config ${LM_CONFIG} \
    --lm_file ${LM_MODEL} \
    --ctc_weight ${CTC_WEIGHT} \
    --lm_weight ${LM_WEIGHT} \
    --beam_size 5 \
    --maxlenratio 1.0 \
    --output_dir ${OUT_DIR} \
    2>&1 | tee logs/adaptive_${BUCKET}_decode.log

  echo "[Done] ${BUCKET}"
}

# ============================================================
# Bucket-wise adaptive weighting
# ============================================================

run_decode stable 0.5 0.1
run_decode mid 0.3 0.3
run_decode unstable 0.1 0.5

# ============================================================
# Merge bucket results
# ============================================================

mkdir -p "${MERGE_DIR}/output.1/1best_recog"

cat \
  "${EXP_DIR}/adaptive_stable/1best_recog/text" \
  "${EXP_DIR}/adaptive_mid/1best_recog/text" \
  "${EXP_DIR}/adaptive_unstable/1best_recog/text" \
  | sort > "${MERGE_DIR}/output.1/1best_recog/text"

# ============================================================
# CER/WER calculation
# ============================================================

echo "========================================="
echo "[Evaluate] Adaptive Gated Decoding"
echo "========================================="

python calc_cer_wer.py \
  dump/raw/test_1k/text \
  "${MERGE_DIR}" \
  | tee logs/adaptive_gated_test1k_cer_wer.txt

echo "========================================="
echo "[DONE]"
echo "Result saved to logs/adaptive_gated_test1k_cer_wer.txt"
echo "========================================="
