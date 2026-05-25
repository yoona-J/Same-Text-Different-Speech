#!/usr/bin/env bash
set -e

. ./path.sh
mkdir -p logs

cut -d' ' -f1 dump/raw/test/wav.scp > dump/raw/test/keys.txt

run_decode () {
  GPU=$
  TAG=$2
  CONFIG=$3
  MODEL=$4

  OUT_DIR="exp/asr_${TAG}/direct_gpu_test"
  LOG_FILE="logs/decode_${TAG}_full_test_$(date +%Y%m%d_%H%M%S).log"

  echo "========================================="
  echo "[Decode] ${TAG}"
  echo "GPU=${GPU}"
  echo "OUT=${OUT_DIR}"
  echo "========================================="

  rm -rf "${OUT_DIR}"

  CUDA_VISIBLE_DEVICES=${GPU} python3 -m espnet2.bin.asr_inference \
    --batch_size 1 \
    --ngpu 1 \
    --num_workers 0 \
    --data_path_and_name_and_type dump/raw/test/wav.scp,speech,sound \
    --key_file dump/raw/test/keys.txt \
    --asr_train_config "exp/asr_${TAG}/config.yaml" \
    --asr_model_file "exp/asr_${TAG}/${MODEL}" \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"
}

run_decode 1 transformer_ctc_full conf/tuning/train_asr_transformer_ctc.yaml valid.acc.ave.pth
run_decode 1 bilstm_ctc_full conf/tuning/train_asr_bilstm_ctc.yaml valid.acc.ave.pth

echo "DONE"
