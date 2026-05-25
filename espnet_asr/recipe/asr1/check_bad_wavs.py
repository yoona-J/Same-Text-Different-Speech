#!/usr/bin/env python3
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

WAV_SCP = Path("dump/raw/test/wav.scp")
OUT = Path("logs/bad_wavs_test.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

bad = []

with open(WAV_SCP, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

for line in tqdm(lines):
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        bad.append((parts[0], "INVALID_LINE"))
        continue

    utt, path = parts

    try:
        info = sf.info(path)
        if info.frames <= 0:
            bad.append((utt, f"EMPTY frames={info.frames} path={path}"))
    except Exception as e:
        bad.append((utt, f"{type(e).__name__}: {e} path={path}"))

with open(OUT, "w", encoding="utf-8") as f:
    for utt, err in bad:
        f.write(f"{utt}\t{err}\n")

print(f"bad files: {len(bad)}")
print(f"saved: {OUT}")
