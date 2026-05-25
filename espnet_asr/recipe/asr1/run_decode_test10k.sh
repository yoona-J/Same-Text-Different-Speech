#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1
export LC_ALL=C

TEST_SET="test_1k"
NUM_SAMPLES=1000

mkdir -p dump/raw/${TEST_SET}
mkdir -p logs

echo "========================================="
echo "[1] Random sampling ${NUM_SAMPLES} utterances"
echo "========================================="

# 랜덤 utt 추출
shuf dump/raw/test/wav.scp | head -${NUM_SAMPLES} | cut -d' ' -f1 > /tmp/${TEST_SET}_keys

# 필요한 파일 필터링
for f in wav.scp text utt2spk utt2num_samples; do
  awk 'NR==FNR {a[$1]=1; next} ($1 in a)' \
    /tmp/${TEST_SET}_keys \
    dump/raw/test/${f} \
    > dump/raw/${TEST_SET}/${f}
done

# 메타 파일 복사
cp dump/raw/test/feats_type dump/raw/${TEST_SET}/
cp dump/raw/test/audio_format dump/raw/${TEST_SET}/

# spk2utt 생성
utils/utt2spk_to_spk2utt.pl \
  dump/raw/${TEST_SET}/utt2spk \
  > dump/raw/${TEST_SET}/spk2utt

echo
echo "[INFO] Sample count:"
wc -l dump/raw/${TEST_SET}/wav.scp

echo
echo "========================================="
echo "[2] Decode : BiLSTM"
echo "========================================="

./asr.sh \
  --stage 12 \
  --stop_stage 12 \
  --ngpu 1 \
  --inference_nj 2 \
  --use_lm false \
  --use_word_lm false \
  --token_type char \
  --train_set train \
  --valid_set dev \
  --test_sets ${TEST_SET} \
  --asr_config conf/tuning/train_asr_bilstm_ctc.yaml \
  --asr_tag bilstm_ctc_full \
  2>&1 | tee logs/decode_bilstm_${TEST_SET}_$(date +%Y%m%d_%H%M%S).log

echo
echo "========================================="
echo "[3] Decode : Transformer"
echo "========================================="

./asr.sh \
  --stage 12 \
  --stop_stage 12 \
  --ngpu 1 \
  --inference_nj 2 \
  --use_lm false \
  --use_word_lm false \
  --token_type char \
  --train_set train \
  --valid_set dev \
  --test_sets ${TEST_SET} \
  --asr_config conf/tuning/train_asr_transformer_ctc.yaml \
  --asr_tag transformer_ctc_full \
  2>&1 | tee logs/decode_transformer_${TEST_SET}_$(date +%Y%m%d_%H%M%S).log

echo
echo "========================================="
echo "[4] Decode : Conformer"
echo "========================================="

./asr.sh \
  --stage 12 \
  --stop_stage 12 \
  --ngpu 1 \
  --inference_nj 2 \
  --use_lm false \
  --use_word_lm false \
  --token_type char \
  --train_set train \
  --valid_set dev \
  --test_sets ${TEST_SET} \
  --asr_config conf/tuning/train_asr_conformer_ctc.yaml \
  --asr_tag conformer_ctc_full \
  2>&1 | tee logs/decode_conformer_${TEST_SET}_$(date +%Y%m%d_%H%M%S).log

echo
echo "========================================="
echo "[5] Decode : E-Branchformer"
echo "========================================="

./asr.sh \
  --stage 12 \
  --stop_stage 12 \
  --ngpu 1 \
  --inference_nj 2 \
  --use_lm false \
  --use_word_lm false \
  --token_type char \
  --train_set train \
  --valid_set dev \
  --test_sets ${TEST_SET} \
  --asr_config conf/tuning/train_asr_ebranchformer_ctc.yaml \
  --asr_tag ebranchformer_ctc_full \
  2>&1 | tee logs/decode_ebranchformer_${TEST_SET}_$(date +%Y%m%d_%H%M%S).log

echo
echo "========================================="
echo "[DONE]"
echo "========================================="