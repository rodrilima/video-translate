"""Isola as causas de uma dublagem com som ruim.

Três variáveis se confundem quando o áudio sai "embolado":
  1. compressão temporal — pedir menos tempo do que o texto precisa;
  2. qualidade do clipe de referência — o clonador herda os artefatos dele;
  3. o próprio motor.

Este script varia uma de cada vez, mantendo as outras fixas.

Uso: uv run python scripts/diagnose_quality.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from dublador.tts.base import Voice
from dublador.utils.syllables import count_syllables, estimated_duration

OUT = Path("samples/diagnostico")
TEXT = "A gente precisa entender que isso muda tudo daqui pra frente."


def save(audio: np.ndarray, sr: int, name: str) -> float:
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(OUT / f"{name}.wav", audio, sr)
    return len(audio) / sr


def main() -> None:
    from dublador.tts.omnivoice import OmniVoiceBackend

    natural = estimated_duration(TEXT)
    print(f"texto: {TEXT!r}")
    print(f"sílabas: {count_syllables(TEXT)}  ->  duração natural estimada: {natural:.2f}s")
    print()

    voice = Voice.load("luciana")
    backend = OmniVoiceBackend()

    print("carregando modelo...")
    backend.warmup()
    print()

    # Variável 1: compressão temporal, com a mesma referência.
    print("--- variando a duração alvo (referência: luciana) ---")
    print(f"{'condição':>16} {'obtido':>8} {'vs natural':>11}")
    cases: list[tuple[str, float | None]] = [
        ("livre", None),
        ("natural", round(natural, 2)),
        ("folgado", round(natural * 1.15, 2)),
        ("apertado", round(natural * 0.80, 2)),
    ]
    for label, target in cases:
        t0 = time.time()
        result = backend.synthesize(TEXT, voice, duration_s=target)
        actual = save(result.audio, result.sample_rate, f"dur_{label}")
        ratio = actual / natural
        print(f"{label:>16} {actual:8.2f}s {ratio:10.2f}x   ({time.time() - t0:.1f}s)")

    print(f"\narquivos em {OUT}")
    print("compare dur_livre.wav com dur_apertado.wav: se só o apertado soa")
    print("embolado, o problema é orçamento de tempo, não o motor.")


if __name__ == "__main__":
    main()
