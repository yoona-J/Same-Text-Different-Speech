import pandas as pd
import sys

base_csv = sys.argv[1]
new_csv = sys.argv[2]
out_csv = sys.argv[3]

base_name = sys.argv[4]
new_name = sys.argv[5]

base_df = pd.read_csv(base_csv)
new_df = pd.read_csv(new_csv)

df = base_df.merge(
    new_df,
    on="utt_id",
    suffixes=(f"_{base_name}", f"_{new_name}")
)

df["cer_diff"] = (
    df[f"cer_{base_name}"]
    - df[f"cer_{new_name}"]
)

print("=" * 60)
print(df["cer_diff"].describe())
print("=" * 60)

print("\nTop Improved Samples")
print(
    df.sort_values("cer_diff", ascending=False)[
        [
            "utt_id",
            f"cer_{base_name}",
            f"cer_{new_name}",
            "cer_diff",
        ]
    ].head(20)
)

df.to_csv(out_csv, index=False)

print(f"\nSaved to: {out_csv}")
