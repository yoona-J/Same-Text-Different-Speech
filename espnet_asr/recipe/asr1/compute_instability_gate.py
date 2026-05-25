#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# Input / Output
# ============================================================

IN_CSV = Path("logs/test_1k_acoustic_features_decoder.csv")
OUT_CSV = Path("logs/test_1k_instability_gate.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# Utility
# ============================================================

def zscore(s):
    x = pd.to_numeric(s, errors="coerce").astype(float)
    return (x - x.mean()) / (x.std() + 1e-8)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ============================================================
# Load
# ============================================================

df = pd.read_csv(IN_CSV)

# error row 제거
df = df[df["status"] == "ok"].copy()

# ============================================================
# Acoustic instability risk score
#
# 위험도 증가 방향:
# - f0_std ↑              : pitch 변동성 증가
# - f0_range ↑            : 억양 범위 증가
# - duration_per_token ↑  : 토큰당 발화 길이 증가 / 늘임
# - mean_pause ↑          : pause 증가
# - max_pause ↑           : 긴 pause 존재
# - silence_ratio ↑       : 무음 비율 증가
# - jitter_local ↑        : voice perturbation 증가
# - shimmer_local ↑       : amplitude perturbation 증가
# - hnr ↓                 : harmonicity 낮음 / 음질 불안정
# - burst_score ↑         : 비정상적으로 빠른 token/sec
#
# token/sec는 단순히 빠를수록 안정/불안정이라고 보기 어렵기 때문에,
# 6 token/sec 초과분만 burst_score로 분리해서 반영.
# ============================================================

df["burst_score"] = np.maximum(df["token_per_sec"] - 6.0, 0.0)

risk = (
    1.4 * zscore(df["f0_std"]) +
    1.2 * zscore(df["f0_range"]) +
    1.5 * zscore(df["duration_per_token"]) +
    1.2 * zscore(df["mean_pause"]) +
    0.8 * zscore(df["max_pause"]) +
    1.0 * zscore(df["silence_ratio"]) +
    0.9 * zscore(df["jitter_local"]) +
    0.7 * zscore(df["jitter_rap"]) +
    0.9 * zscore(df["shimmer_local"]) +
    0.7 * zscore(df["shimmer_apq3"]) -
    1.0 * zscore(df["hnr"]) +
    1.3 * zscore(df["burst_score"])
)

# 안정적으로 보기 위해 평균 0, 표준편차 1로 다시 정규화
risk = (risk - risk.mean()) / (risk.std() + 1e-8)

df["risk_score"] = risk

# sigmoid gate
# gate가 높을수록 unstable speech로 간주
df["gate"] = sigmoid(df["risk_score"])

# ============================================================
# Bucket assignment
#
# gate 하위 33%: stable
# 중간 33%: mid
# 상위 33%: unstable
# ============================================================

q1, q2 = df["gate"].quantile([0.33, 0.66])

def assign_bucket(g):
    if g <= q1:
        return "stable"
    elif g <= q2:
        return "mid"
    else:
        return "unstable"

df["bucket"] = df["gate"].apply(assign_bucket)

# ============================================================
# Recommended adaptive weights
#
# stable:
#   acoustic evidence / CTC 비중 높임
#
# mid:
#   CTC와 decoder/LM 균형
#
# unstable:
#   linguistic prior / LM 비중 높임
# ============================================================

def assign_ctc_weight(bucket):
    if bucket == "stable":
        return 0.5
    elif bucket == "mid":
        return 0.3
    else:
        return 0.1

def assign_lm_weight(bucket):
    if bucket == "stable":
        return 0.1
    elif bucket == "mid":
        return 0.3
    else:
        return 0.5

df["adaptive_ctc_weight"] = df["bucket"].apply(assign_ctc_weight)
df["adaptive_lm_weight"] = df["bucket"].apply(assign_lm_weight)

# ============================================================
# Save
# ============================================================

cols = [
    "utt",
    "risk_score",
    "gate",
    "bucket",
    "adaptive_ctc_weight",
    "adaptive_lm_weight",
    "duration",
    "token_count",
    "token_per_sec",
    "duration_per_token",
    "silence_ratio",
    "mean_pause",
    "max_pause",
    "f0_mean",
    "f0_std",
    "f0_range",
    "hnr",
    "jitter_local",
    "jitter_rap",
    "shimmer_local",
    "shimmer_apq3",
    "burst_score",
]

df[cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"Saved: {OUT_CSV}")
print()
print(df["bucket"].value_counts())
print()
print(df.groupby("bucket")[["risk_score", "gate", "adaptive_ctc_weight", "adaptive_lm_weight"]].mean())
