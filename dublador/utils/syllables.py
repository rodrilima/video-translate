"""Contagem de sílabas em português brasileiro.

Serve para converter uma duração alvo em um orçamento de texto, que é a forma
mais eficaz de resolver isocronia: restringir a tradução na origem, em vez de
acelerar o áudio depois.

É uma heurística de núcleos vocálicos, não um silabador linguístico completo —
não precisa ser exata, precisa ser *consistente*, porque a taxa silábica é
calibrada empiricamente contra a saída real do TTS
(ver scripts/calibrate_syllable_rate.py).
"""

from __future__ import annotations

import re
import unicodedata

from ..config import SYLLABLE_RATE_PTBR

STRONG_VOWELS = set("aeoáéóâêôãõà")
WEAK_VOWELS = set("iuíúy")
VOWELS = STRONG_VOWELS | WEAK_VOWELS

# Vogais nasais: absorvem a vogal seguinte num ditongo ("não", "irmão", "põe").
NASAL_VOWELS = set("ãõ")
# Vogais fracas acentuadas: o acento marca justamente a quebra do ditongo
# ("saída", "país", "baú").
ACCENTED_WEAK = set("íú")
# Vogais fortes acentuadas: formam hiato quando precedidas de vogal fraca
# ("experiência", "influência").
ACCENTED_STRONG = set("áéóâêô")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_MARKUP_RE = re.compile(r"\[[^\]]*\]")  # marcadores tipo [pause]


def count_word_syllables(word: str) -> int:
    """Sílabas de uma palavra, contando núcleos vocálicos.

    Vogais adjacentes normalmente formam um núcleo só (ditongo). As exceções,
    em ordem de precedência:
      - vogal nasal antes de vogal é sempre ditongo ("não", "irmão");
      - vogal fraca acentuada quebra o ditongo ("sa-í-da", "pa-ís");
      - vogal fraca seguida de forte acentuada quebra ("ex-pe-ri-ên-cia");
      - duas vogais fortes formam hiato ("ca-os", "po-e-ta").
    """
    w = word.lower()
    nuclei = 0
    previous_char = ""

    for char in w:
        if char not in VOWELS:
            previous_char = ""
            continue

        if not previous_char:
            nuclei += 1
        elif previous_char in NASAL_VOWELS:
            pass  # ditongo nasal: a vogal seguinte não abre sílaba
        elif char in ACCENTED_WEAK or previous_char in ACCENTED_WEAK:
            nuclei += 1
        elif previous_char in WEAK_VOWELS and char in ACCENTED_STRONG:
            nuclei += 1
        elif previous_char in STRONG_VOWELS and char in STRONG_VOWELS:
            nuclei += 1

        previous_char = char

    if nuclei == 0 and _has_letters(w):
        return 1  # siglas e monossílabos atípicos ainda ocupam tempo de fala
    return nuclei


def _has_letters(word: str) -> bool:
    return any(unicodedata.category(c).startswith("L") for c in word)


def count_syllables(text: str) -> int:
    """Sílabas de um trecho, ignorando marcadores de controle como [pause]."""
    clean = _MARKUP_RE.sub(" ", text)
    return sum(count_word_syllables(w) for w in _WORD_RE.findall(clean))


def syllable_budget(target_seconds: float,
                    rate: float = SYLLABLE_RATE_PTBR) -> int:
    """Quantas sílabas cabem confortavelmente num intervalo de tempo."""
    return max(1, round(target_seconds * rate))


def estimated_duration(text: str, rate: float = SYLLABLE_RATE_PTBR) -> float:
    """Duração estimada da fala, antes de sintetizar. Usada para descartar
    candidatos ruins sem pagar o custo do TTS."""
    return count_syllables(text) / rate
