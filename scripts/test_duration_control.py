"""Valida a premissa central do pipeline: o TTS honra a duração pedida?

Toda a estratégia de isocronia depende de poder pedir "diga isto em 4,2 segundos"
e receber algo próximo disso. Se o erro for grande, o plano cai para
time-stretch pesado, que é audivelmente pior.

Uso: uv run python scripts/test_duration_control.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_ID = "mlx-community/OmniVoice-bf16"
VOICE = Path("dublador/voices/luciana")
OUT = Path("/tmp/duration_test")

TEXT = "A gente precisa entender que isso muda tudo daqui pra frente."
TARGETS = [2.5, 3.5, 4.5, 5.5]


def main() -> None:
    from mlx_audio.tts.utils import load_model

    OUT.mkdir(exist_ok=True)
    ref_text = (VOICE / "ref.txt").read_text(encoding="utf-8").strip()

    print(f"carregando {MODEL_ID} ...")
    t0 = time.time()
    model = load_model(MODEL_ID)
    print(f"carregado em {time.time() - t0:.1f}s\n")

    print(f"texto: {TEXT!r}")
    print(f"{'alvo':>6} {'obtido':>8} {'erro':>8} {'RTF':>7}")
    print("-" * 34)

    for target in TARGETS:
        t0 = time.time()
        chunks = []
        sample_rate = 24000
        for result in model.generate(
            text=TEXT,
            language="portuguese",
            ref_audio=str(VOICE / "ref.wav"),
            ref_text=ref_text,
            duration_s=target,
            num_steps=32,
        ):
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            chunks.append(audio)
            sample_rate = getattr(result, "sample_rate", sample_rate) or sample_rate
        elapsed = time.time() - t0

        audio = np.concatenate(chunks)
        actual = len(audio) / sample_rate
        path = OUT / f"target_{target:.1f}s.wav"
        sf.write(path, audio, sample_rate)

        error = (actual - target) / target
        rtf = elapsed / max(actual, 1e-6)
        print(f"{target:6.2f} {actual:8.2f} {error:+7.1%} {rtf:7.3f}")

    print(f"\narquivos em {OUT}")


if __name__ == "__main__":
    main()
