#!/usr/bin/env bash
set -e

. ./path.sh
export LC_ALL=C

mkdir -p logs
# cut -d' ' -f1 dump/raw/test/wav.scp > dump/raw/test/keys.txt
cut -d' ' -f1 dump/raw/test_1k/wav.scp > dump/raw/test_1k/keys.txt

run_decode () {
  GPU=$1
  TAG=$2
  MODEL=$3

  OUT_DIR="exp/asr_${TAG}/direct_gpu_test_all_1K"

  rm -rf "${OUT_DIR}"

  CUDA_VISIBLE_DEVICES=${GPU} python3 -m espnet2.bin.asr_inference \
    --batch_size 1 \
    --ngpu 1 \
    --num_workers 0 \
    --data_path_and_name_and_type dump/raw/test_1k/wav.scp,speech,sound \
    --key_file dump/raw/test_1k/keys.txt \
    --asr_train_config exp/asr_${TAG}/config.yaml \
    --asr_model_file exp/asr_${TAG}/${MODEL} \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee logs/decode_${TAG}_direct_$(date +%Y%m%d_%H%M%S).log
}

# run_decode 0 bilstm_ctc_full valid.acc.ave.pth
## 예전 transformer, conformer임
run_decode 0 transformer_ctc_full valid.acc.ave.pth