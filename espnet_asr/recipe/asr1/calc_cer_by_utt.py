import pandas as pd
from jiwer import cer
import sys

ref_path = sys.argv[1]
hyp_path = sys.argv[2]
out_csv = sys.argv[3]


def load_text(path):
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)

            if len(parts) < 2:
                continue

            utt_id = parts[0]
            text = parts[1]

            data[utt_id] = text

    return data


ref_dict = load_text(ref_path)
hyp_dict = load_text(hyp_path)

rows = []

for utt_id in ref_dict:

    ref = ref_dict[utt_id]
    hyp = hyp_dict.get(utt_id, "")

    try:
        utt_cer = cer(ref, hyp)
    except:
        utt_cer = 1.0

    rows.append({
        "utt_id": utt_id,
        "ref": ref,
        "hyp": hyp,
        "cer": utt_cer,
    })

df = pd.DataFrame(rows)

print("=" * 50)
print(f"Num samples: {len(df)}")
print(f"Mean CER: {df['cer'].mean() * 100:.2f}%")
print("=" * 50)

df.to_csv(out_csv, index=False)

print(f"Saved to: {out_csv}")
