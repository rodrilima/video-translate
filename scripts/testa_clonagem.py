"""Compara voz fixa do catálogo com clonagem da voz real do locutor.

A pergunta é se vale trocar o timbre estável do Kokoro pela voz do próprio
locutor do vídeo. A tentativa anterior de clonagem usava uma voz sintética do
macOS como referência — o pior insumo possível para um clonador — e soou
robótica. Aqui a referência é fala humana real, extraída pelos turnos da
diarização.

O clonador recebe áudio em inglês e gera português: é clonagem entre idiomas, e
é justamente isso que precisa ser ouvido antes de decidir.

Uso: uv run python scripts/testa_clonagem.py
"""

from __future__ import annotations

import time
from pathlib import Path

import soundfile as sf

from dublador.model import load_segments
from dublador.config import JobPaths
from dublador.tts.base import Voice

JOB = "iq5VjE31Eig"
OUT = Path("samples/clonagem")
REF_AUDIO = Path("/tmp/ref_locutor.wav")
REF_TEXT = Path("/tmp/ref_locutor.txt")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # frases reais do vídeo, já adaptadas pelo pipeline
    segments = [s for s in load_segments(JobPaths(JOB).segments)
                if s.text_pt_final and 8 <= len(s.text_pt_final.split()) <= 16]
    frases = [s.text_pt_final for s in segments[:3]]

    print("frases de teste:")
    for f in frases:
        print(f"  {f}")
    print()

    _kokoro(frases)
    _omnivoice(frases)
    print(f"\narquivos em {OUT}/")


def _kokoro(frases: list[str]) -> None:
    from dublador.tts.kokoro import KokoroBackend

    backend = KokoroBackend()
    voice = Voice.load("alex")
    backend.warmup()

    for index, texto in enumerate(frases, start=1):
        inicio = time.time()
        r = backend.synthesize(texto, voice, duration_s=None)
        sf.write(OUT / f"{index}_fixa_alex.wav", r.audio, r.sample_rate)
        print(f"kokoro/alex   frase {index}: {r.duration:.2f}s "
              f"(RTF {(time.time() - inicio) / r.duration:.2f})")


def _omnivoice(frases: list[str]) -> None:
    from dublador.tts.omnivoice import OmniVoiceBackend

    voice = Voice(name="locutor", ref_audio=REF_AUDIO,
                  ref_text=REF_TEXT.read_text(encoding="utf-8").strip(),
                  meta={})
    backend = OmniVoiceBackend()
    backend.warmup()

    for index, texto in enumerate(frases, start=1):
        inicio = time.time()
        r = backend.synthesize(texto, voice, duration_s=None)
        sf.write(OUT / f"{index}_clonada.wav", r.audio, r.sample_rate)
        print(f"omnivoice     frase {index}: {r.duration:.2f}s "
              f"(RTF {(time.time() - inicio) / r.duration:.2f})")


if __name__ == "__main__":
    main()
