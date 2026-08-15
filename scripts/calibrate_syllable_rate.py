"""Mede a taxa silábica real do motor de TTS.

O orçamento de sílabas por segmento é o que impede a tradução de ficar longa
demais para o tempo disponível. Se a taxa estiver alta, o pipeline pede texto
mais curto do que precisa e a fala sai comprimida; se estiver baixa, sobra
silêncio. Um erro de 20% aqui degrada todos os segmentos do vídeo.

Sintetiza frases sem restrição de duração e divide sílabas por segundo medido.

Uso: uv run python scripts/calibrate_syllable_rate.py [--voice luciana]
"""

from __future__ import annotations

import argparse
import statistics

from dublador.tts.base import Voice, get_backend
from dublador.utils.syllables import count_syllables

# Frases de comprimento variado, em registro falado — que é o registro que a
# dublagem produz. Textos literários dariam uma taxa diferente.
SENTENCES = [
    "Beleza, vamos direto ao ponto.",
    "A gente precisa entender que isso muda tudo daqui pra frente.",
    "Eu não tô dizendo que é fácil, mas é o caminho mais curto que existe.",
    "Olha, no fim das contas o que importa é quanto tempo você aguenta insistir.",
    "Isso aqui não é teoria, é uma coisa que eu vi acontecer várias vezes.",
    "Se você parar pra pensar, a maior parte das pessoas desiste antes da metade.",
    "E aí, quando o resultado aparece, todo mundo acha que foi sorte.",
    "Não é sorte, é volume de tentativa ao longo de muito tempo.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="luciana")
    parser.add_argument("--backend", default="omnivoice")
    args = parser.parse_args()

    voice = Voice.load(args.voice)
    backend = get_backend(args.backend)
    print(f"backend={backend.name} voz={voice.name}")
    backend.warmup()

    print(f"\n{'síl':>4} {'seg':>7} {'síl/s':>7}  texto")
    print("-" * 72)

    rates: list[float] = []
    for sentence in SENTENCES:
        syllables = count_syllables(sentence)
        result = backend.synthesize(sentence, voice, duration_s=None)
        rate = syllables / result.duration
        rates.append(rate)
        print(f"{syllables:4d} {result.duration:7.2f} {rate:7.2f}  {sentence[:44]}")

    mean = statistics.mean(rates)
    median = statistics.median(rates)
    spread = statistics.pstdev(rates)

    print("-" * 72)
    print(f"média {mean:.2f} | mediana {median:.2f} | desvio {spread:.2f} síl/s")
    print()
    print(f"Ajuste SYLLABLE_RATE_PTBR em dublador/config.py para {median:.1f}")
    print("A mediana é preferível: uma frase atípica não deve mover o orçamento")
    print("de todas as outras.")


if __name__ == "__main__":
    main()
