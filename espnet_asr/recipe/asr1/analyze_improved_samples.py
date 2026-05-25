import pandas as pd
import sys

compare_csv = sys.argv[1]
meta_csv = sys.argv[2]

compare_df = pd.read_csv(compare_csv)
meta_df = pd.read_csv(meta_csv)

df = compare_df.merge(meta_df, on="utt_id", how="left")

print("=" * 60)
print(
    df.groupby("condition")["cer_diff"]
      .agg(["count", "mean", "median", "std", "min", "max"])
)
print("=" * 60)

print("\nTop Improved")
print(
    df.sort_values("cer_diff", ascending=False)[
        [
            "utt_id",
            "text",
            "condition",
            "cer_baseline",
            "cer_acoustic",
            "cer_diff",
        ]
    ].head(30)
)

print("\nTop Degraded")
print(
    df.sort_values("cer_diff", ascending=True)[
        [
            "utt_id",
            "text",
            "condition",
            "cer_baseline",
            "cer_acoustic",
            "cer_diff",
        ]
    ].head(30)
)

df.to_csv("compare_lm_vs_acoustic_with_metadata.csv", index=False)
print("\nSaved to compare_lm_vs_acoustic_with_metadata.csv")