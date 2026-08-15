"""Operações de áudio no nível de amostra."""

from __future__ import annotations

import numpy as np

# Margem preservada nas bordas. Cortar rente come o ataque de plosivas ("p",
# "t", "k"), que começam com um transiente de baixa energia — a fala fica com
# som de corte.
GUARD_MS = 25.0

# Limiar relativo ao pico. Abaixo disso é ruído de fundo do sintetizador, não
# fala.
SILENCE_THRESHOLD = 0.02


def trim_silence(audio: np.ndarray, sample_rate: int, *,
                 guard_ms: float = GUARD_MS,
                 threshold: float = SILENCE_THRESHOLD) -> np.ndarray:
    """Remove silêncio das pontas.

    Importa mais do que parece numa dublagem: o sintetizador coloca silêncio
    fixo antes e depois de cada trecho — medido aqui, ~206 ms no início em
    todos os segmentos. Como cada trecho é posicionado pelo tempo em que a
    pessoa começa a falar, esse silêncio vira atraso constante entre a boca e a
    voz ao longo do vídeo inteiro.

    O silêncio do fim também engana o encaixe, que o contava como fala e pedia
    compressão desnecessária.
    """
    if audio.size == 0:
        return audio

    envelope = np.abs(audio)
    peak = float(envelope.max())
    if peak <= 0:
        return audio

    voiced = np.where(envelope > peak * threshold)[0]
    if voiced.size == 0:
        return audio

    guard = int(sample_rate * guard_ms / 1000.0)
    start = max(0, int(voiced[0]) - guard)
    end = min(len(audio), int(voiced[-1]) + guard)
    return audio[start:end]


def leading_silence(audio: np.ndarray, sample_rate: int, *,
                    threshold: float = SILENCE_THRESHOLD) -> float:
    """Silêncio inicial em segundos. Usado para verificar a sincronia."""
    if audio.size == 0:
        return 0.0
    envelope = np.abs(audio)
    peak = float(envelope.max())
    if peak <= 0:
        return len(audio) / sample_rate
    voiced = np.where(envelope > peak * threshold)[0]
    return float(voiced[0]) / sample_rate if voiced.size else 0.0
