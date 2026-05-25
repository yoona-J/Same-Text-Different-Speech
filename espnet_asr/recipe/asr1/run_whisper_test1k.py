import whisper
from pathlib import Path
from tqdm import tqdm

wav_scp = Path("dump/raw/test/wav.scp")
out_dir = Path("exp/whisper_medium_test/1best_recog")
out_dir.mkdir(parents=True, exist_ok=True)

model = whisper.load_model("medium")

# 전체 라인 수 미리 계산
lines = wav_scp.read_text(encoding="utf-8").splitlines()

with open(out_dir / "text", "w", encoding="utf-8") as out:

    for line in tqdm(lines, desc="Whisper Decode", ncols=100):
        utt_id, wav_path = line.strip().split(maxsplit=1)

        result = model.transcribe(
            wav_path,
            language="ko",
            task="transcribe",
            fp16=True,
            verbose=None,   # 중요
        )

        hyp = result["text"].strip()
        out.write(f"{utt_id} {hyp}\n")