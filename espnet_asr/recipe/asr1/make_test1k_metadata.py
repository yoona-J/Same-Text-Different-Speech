import pandas as pd
import numpy as np
import sys

ref_path = "dump/raw/test_1k/text"

rows = []

with open(ref_path, "r", encoding="utf-8") as f:

    for line in f:

        parts = line.strip().split(maxsplit=1)

        if len(parts) < 2:
            continue

        utt_id = parts[0]
        text = parts[1]

        num_chars = len(text.replace(" ", ""))
        num_tokens = len(text.split())

        # low-info heuristic
        if num_chars <= 5 or num_tokens <= 2:
            condition = "low_info"
        else:
            condition = "normal"

        rows.append({
            "utt_id": utt_id,
            "text": text,
            "num_chars": num_chars,
            "num_tokens": num_tokens,
            "condition": condition,
        })

df = pd.DataFrame(rows)

df.to_csv("test_1k_metadata.csv", index=False)

print(df["condition"].value_counts())

print("\nSaved to test_1k_metadata.csv")
