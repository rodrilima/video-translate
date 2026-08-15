"""Descobre quanta compressão temporal o TTS aguenta antes de degradar.

Isso define um parâmetro central do pipeline. Se o motor só soa bem na duração
que ele escolhe sozinho, a isocronia precisa ser resolvida inteiramente no
texto — encurtando a tradução — e o `duration_s` passa a ser um ajuste fino,
não uma alavanca de compressão.

Gera a mesma frase em frações da duração natural, para escuta cega.

Uso: uv run python scripts/find_compression_limit.py
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from dublador.tts.base import Voice
from dublador.tts.omnivoice import OmniVoiceBackend
from dublador.utils.syllables import count_syllables

OUT = Path("samples/limite_compressao")

SENTENCES = [
    "Olha, no fim das contas o que importa é quanto tempo você aguenta insistir.",
    "Eu não tô dizendo que é fácil, mas é o caminho mais curto que existe.",
]

RATIOS = [1.00, 0.95, 0.90, 0.85, 0.80]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    backend = OmniVoiceBackend()
    voice = Voice.load("luciana")
    backend.warmup()

    for index, sentence in enumerate(SENTENCES, start=1):
        print(f"\nfrase {index}: {sentence!r}")
        print(f"sílabas: {count_syllables(sentence)}")

        free = backend.synthesize(sentence, voice, duration_s=None)
        natural = free.duration
        sf.write(OUT / f"f{index}_ratio_1.00_livre.wav", free.audio,
                 free.sample_rate)
        print(f"duração livre: {natural:.2f}s  "
              f"({count_syllables(sentence) / natural:.2f} síl/s)")
        print(f"{'razão':>7} {'alvo':>7} {'obtido':>8}")

        for ratio in RATIOS:
            target = round(natural * ratio, 2)
            result = backend.synthesize(sentence, voice, duration_s=target)
            name = f"f{index}_ratio_{ratio:.2f}.wav"
            sf.write(OUT / name, result.audio, result.sample_rate)
            print(f"{ratio:7.2f} {target:6.2f}s {result.duration:7.2f}s")

    print(f"\narquivos em {OUT}/")
    print("Ouça em ordem decrescente de razão e diga onde começa a incomodar.")
    print("Esse ponto vira o piso de compressão do pipeline.")


if __name__ == "__main__":
    main()
