#!/usr/bin/env python3
import argparse
import random
import numpy as np
from jiwer import wer as jiwer_wer

# ----------------------------
# Edit distance
# ----------------------------
def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]

    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[n][m]


# ----------------------------
# Load text
# ----------------------------
def load_text(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            uttid, text = parts
            data[uttid] = text.strip()
    return data


# ----------------------------
# Utterance metrics
# ----------------------------
def compute_utt_metrics(refs, hyps):

    cer_errors = []
    cer_lens = []

    wer_errors = []
    wer_lens = []

    for uttid in refs:
        if uttid not in hyps:
            continue

        ref = refs[uttid]
        hyp = hyps[uttid]

        # CER
        ref_char = list(ref.replace(" ", ""))
        hyp_char = list(hyp.replace(" ", ""))

        c_err = edit_distance(ref_char, hyp_char)
        c_len = max(1, len(ref_char))

        cer_errors.append(c_err)
        cer_lens.append(c_len)

        # WER
        ref_word = ref.split()
        hyp_word = hyp.split()

        w_err = edit_distance(ref_word, hyp_word)
        w_len = max(1, len(ref_word))

        wer_errors.append(w_err)
        wer_lens.append(w_len)

    return (
        np.array(cer_errors),
        np.array(cer_lens),
        np.array(wer_errors),
        np.array(wer_lens)
    )


# ----------------------------
# Bootstrap
# ----------------------------
def bootstrap_metric(errors, lens, n_boot=1000):

    N = len(errors)
    scores = []

    for _ in range(n_boot):
        idx = np.random.randint(0, N, N)

        err = errors[idx].sum()
        ln = lens[idx].sum()

        scores.append(100 * err / ln)

    scores = np.array(scores)

    mean = scores.mean()
    low = np.percentile(scores, 2.5)
    high = np.percentile(scores, 97.5)

    return mean, low, high, scores


# ----------------------------
# Main
# ----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ref", required=True)
    parser.add_argument("--base_hyp", required=True)
    parser.add_argument("--ours_hyp", required=True)
    parser.add_argument("--n_boot", type=int, default=1000)

    args = parser.parse_args()

    refs = load_text(args.ref)
    base = load_text(args.base_hyp)
    ours = load_text(args.ours_hyp)

    base_ce, base_cl, base_we, base_wl = compute_utt_metrics(refs, base)
    ours_ce, ours_cl, ours_we, ours_wl = compute_utt_metrics(refs, ours)

    # CER
    b_cer, b_l, b_h, b_scores = bootstrap_metric(
        base_ce, base_cl, args.n_boot
    )

    o_cer, o_l, o_h, o_scores = bootstrap_metric(
        ours_ce, ours_cl, args.n_boot
    )

    # WER
    b_wer, bw_l, bw_h, bw_scores = bootstrap_metric(
        base_we, base_wl, args.n_boot
    )

    o_wer, ow_l, ow_h, ow_scores = bootstrap_metric(
        ours_we, ours_wl, args.n_boot
    )

    # Difference CI
    cer_diff = o_scores - b_scores
    wer_diff = ow_scores - bw_scores

    cd_l = np.percentile(cer_diff, 2.5)
    cd_h = np.percentile(cer_diff, 97.5)

    wd_l = np.percentile(wer_diff, 2.5)
    wd_h = np.percentile(wer_diff, 97.5)

    print("\n===== CER =====")
    print(f"Baseline : {b_cer:.3f}% "
          f"[{b_l:.3f}, {b_h:.3f}]")

    print(f"Ours     : {o_cer:.3f}% "
          f"[{o_l:.3f}, {o_h:.3f}]")

    print(f"Diff(Ours-Baseline): "
          f"{o_cer-b_cer:.3f}%p "
          f"[{cd_l:.3f}, {cd_h:.3f}]")

    print("\n===== WER =====")
    print(f"Baseline : {b_wer:.3f}% "
          f"[{bw_l:.3f}, {bw_h:.3f}]")

    print(f"Ours     : {o_wer:.3f}% "
          f"[{ow_l:.3f}, {ow_h:.3f}]")

    print(f"Diff(Ours-Baseline): "
          f"{o_wer-b_wer:.3f}%p "
          f"[{wd_l:.3f}, {wd_h:.3f}]")

    print()


if __name__ == "__main__":
    main()
