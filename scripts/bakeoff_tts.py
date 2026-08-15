"""Compara motores e vozes de TTS na mesma frase.

Não existe benchmark público de qualidade TTS segmentado por pt-BR, então a
escolha do motor só se resolve ouvindo. Este script produz as amostras lado a
lado, sempre sem restrição de duração, para que a comparação seja de timbre e
naturalidade e não de compressão.

Uso: uv run python scripts/bakeoff_tts.py
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import soundfile as sf

from dublador.tts.base import Voice
from dublador.utils.syllables import count_syllables

OUT = Path("samples/bakeoff")

TEXT = ("Olha, no fim das contas o que importa é quanto tempo "
        "você aguenta insistir.")

# Vozes embutidas do Kokoro para português (não precisam de clipe de referência).
KOKORO_VOICES = ["pf_dora", "pm_alex", "pm_santa"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"texto: {TEXT!r}  ({count_syllables(TEXT)} sílabas)\n")
    print(f"{'amostra':>24} {'dur':>7} {'RTF':>7}  status")
    print("-" * 60)

    _run_omnivoice()
    _run_kokoro()

    print(f"\narquivos em {OUT}")


def _report(name: str, duration: float, elapsed: float) -> None:
    print(f"{name:>24} {duration:6.2f}s {elapsed / max(duration, 1e-6):7.3f}  ok")


def _fail(name: str, exc: Exception) -> None:
    print(f"{name:>24} {'-':>7} {'-':>7}  FALHOU: {type(exc).__name__}: {exc}")


def _run_omnivoice() -> None:
    from dublador.tts.omnivoice import OmniVoiceBackend

    try:
        backend = OmniVoiceBackend()
        voice = Voice.load("luciana")
        t0 = time.time()
        result = backend.synthesize(TEXT, voice, duration_s=None)
        elapsed = time.time() - t0
        sf.write(OUT / "omnivoice_luciana.wav", result.audio, result.sample_rate)
        _report("omnivoice/luciana", result.duration, elapsed)
    except Exception as exc:  # noqa: BLE001 - o ponto é registrar e seguir
        _fail("omnivoice/luciana", exc)
        traceback.print_exc()


def _run_kokoro() -> None:
    from dublador.tts.kokoro import KokoroBackend

    backend = KokoroBackend()
    for voice_id in KOKORO_VOICES:
        try:
            voice = Voice(name=voice_id, ref_audio=Path("."), ref_text="",
                          meta={"kokoro_voice": voice_id})
            t0 = time.time()
            result = backend.synthesize(TEXT, voice, duration_s=None)
            elapsed = time.time() - t0
            sf.write(OUT / f"kokoro_{voice_id}.wav", result.audio,
                     result.sample_rate)
            _report(f"kokoro/{voice_id}", result.duration, elapsed)
        except Exception as exc:  # noqa: BLE001
            _fail(f"kokoro/{voice_id}", exc)


if __name__ == "__main__":
    main()
