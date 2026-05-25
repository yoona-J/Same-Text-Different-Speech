#!/usr/bin/env python3

import csv
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from tqdm import tqdm

# ============================================================
# Config
# ============================================================

WAV_SCP = Path("dump/raw/test_1k/wav.scp")
TEXT_PATH = Path("dump/raw/test_1k/text")

OUT_CSV = Path("logs/test_1k_acoustic_features_decoder.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000


# ============================================================
# Utils
# ============================================================

def read_kv(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            k, v = line.rstrip("\n").split(maxsplit=1)
            data[k] = v
    return data


def safe_stats(x):
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
        }

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.max(x) - np.min(x)),
    }


# ============================================================
# Praat voice quality
# ============================================================

def extract_praat_features(wav_path):

    snd = parselmouth.Sound(str(wav_path))

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    pitch = snd.to_pitch()

    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]

    pitch_stats = safe_stats(pitch_values)

    # --------------------------------------------------------
    # Harmonicity (HNR)
    # --------------------------------------------------------

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    hnr = call(harmonicity, "Get mean", 0, 0)

    # --------------------------------------------------------
    # Jitter
    # --------------------------------------------------------

    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

    jitter_local = call(
        point_process,
        "Get jitter (local)",
        0, 0,
        0.0001,
        0.02,
        1.3
    )

    jitter_rap = call(
        point_process,
        "Get jitter (rap)",
        0, 0,
        0.0001,
        0.02,
        1.3
    )

    # --------------------------------------------------------
    # Shimmer
    # --------------------------------------------------------

    shimmer_local = call(
        [snd, point_process],
        "Get shimmer (local)",
        0, 0,
        0.0001,
        0.02,
        1.3,
        1.6
    )

    shimmer_apq3 = call(
        [snd, point_process],
        "Get shimmer (apq3)",
        0, 0,
        0.0001,
        0.02,
        1.3,
        1.6
    )

    return {
        "f0_mean": pitch_stats["mean"],
        "f0_std": pitch_stats["std"],
        "f0_range": pitch_stats["range"],

        "hnr": float(hnr),

        "jitter_local": float(jitter_local),
        "jitter_rap": float(jitter_rap),

        "shimmer_local": float(shimmer_local),
        "shimmer_apq3": float(shimmer_apq3),
    }


# ============================================================
# librosa features
# ============================================================

def extract_librosa_features(wav_path, text):

    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    duration = len(y) / sr

    tokens = text.split()
    token_count = len(tokens)

    token_per_sec = token_count / (duration + 1e-8)
    duration_per_token = duration / max(token_count, 1)

    # --------------------------------------------------------
    # silence / pause
    # --------------------------------------------------------

    abs_y = np.abs(y)

    silence_ratio = float(np.mean(abs_y < 0.01))

    intervals = librosa.effects.split(y, top_db=30)

    pauses = []

    prev_end = 0

    for start, end in intervals:
        pause = (start - prev_end) / sr

        if pause > 0:
            pauses.append(pause)

        prev_end = end

    pauses = np.asarray(pauses, dtype=np.float32)

    mean_pause = float(np.mean(pauses)) if len(pauses) > 0 else 0.0
    max_pause = float(np.max(pauses)) if len(pauses) > 0 else 0.0

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = librosa.feature.rms(y=y)[0]
    rms_stats = safe_stats(rms)

    # --------------------------------------------------------
    # Spectral
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    centroid_stats = safe_stats(centroid)
    bandwidth_stats = safe_stats(bandwidth)
    rolloff_stats = safe_stats(rolloff)
    zcr_stats = safe_stats(zcr)

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    row = {

        "duration": duration,

        "token_count": token_count,

        "token_per_sec": token_per_sec,
        "duration_per_token": duration_per_token,

        "silence_ratio": silence_ratio,

        "mean_pause": mean_pause,
        "max_pause": max_pause,

        "rms_mean": rms_stats["mean"],
        "rms_std": rms_stats["std"],

        "spectral_centroid_mean": centroid_stats["mean"],
        "spectral_bandwidth_mean": bandwidth_stats["mean"],
        "spectral_rolloff_mean": rolloff_stats["mean"],

        "zcr_mean": zcr_stats["mean"],
    }

    for i in range(13):
        row[f"mfcc{i+1}_mean"] = float(mfcc_means[i])
        row[f"mfcc{i+1}_std"] = float(mfcc_stds[i])

    return row


# ============================================================
# Main
# ============================================================

def main():

    wavscp = read_kv(WAV_SCP)
    text_map = read_kv(TEXT_PATH)

    rows = []

    for utt, wav_path in tqdm(wavscp.items()):

        text = text_map.get(utt, "")

        try:

            librosa_feat = extract_librosa_features(wav_path, text)

            praat_feat = extract_praat_features(wav_path)

            feat = {
                "utt": utt,
                "status": "ok",
            }

            feat.update(librosa_feat)
            feat.update(praat_feat)

        except Exception as e:

            feat = {
                "utt": utt,
                "status": f"error:{type(e).__name__}"
            }

        rows.append(feat)

    fieldnames = []

    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUT_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()