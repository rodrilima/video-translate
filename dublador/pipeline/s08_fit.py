"""Etapa 8: encaixar os áudios sintetizados na linha do tempo do vídeo.

A ordem das táticas segue o custo perceptual de cada uma, da mais barata para a
mais cara:

  1. começar mais cedo, invadindo o silêncio anterior — inaudível;
  2. deixar avançar sobre o silêncio seguinte, quando não há fala lá — inaudível;
  3. esticar o tempo com Rubber Band, dentro de 0,90x a 1,15x — audível de leve;
  4. aceitar sobreposição e marcar o segmento — audível, mas honesto.

Nunca esticamos além dos limites: uma fala reconhecivelmente acelerada estraga
mais a experiência do que uma leve sobreposição.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import soundfile as sf

from ..config import SAMPLE_RATE, STRETCH_MAX, STRETCH_MIN, JobPaths
from ..model import Segment, load_segments, save_segments
from ..utils import ffmpeg

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


def assemble(paths: JobPaths, *, progress: ProgressFn = _noop) -> list[Segment]:
    """Escreve dub.wav com a dublagem posicionada na linha do tempo."""
    segments = load_segments(paths.segments)

    total = ffmpeg.duration(paths.audio)
    track = np.zeros(int(total * SAMPLE_RATE) + SAMPLE_RATE, dtype=np.float32)

    progress(0.05, "montando linha do tempo")
    for index, segment in enumerate(segments):
        _place(track, segment, segments, paths)
        progress(0.05 + 0.85 * (index + 1) / len(segments),
                 f"encaixando {index + 1}/{len(segments)}")

    peak = float(np.abs(track).max())
    if peak > 1.0:
        track /= peak  # evita clipping antes da normalização de loudness

    progress(0.95, "gravando faixa de dublagem")
    sf.write(paths.dub, track[: int(total * SAMPLE_RATE)], SAMPLE_RATE)
    save_segments(segments, paths.segments)

    stretched = sum(1 for s in segments if abs(s.stretch - 1.0) > 0.01)
    overlapping = sum(1 for s in segments if "sobreposicao" in s.qa_flags)
    progress(1.0, f"{stretched} esticados, {overlapping} sobrepostos")
    return segments


def _place(track: np.ndarray, segment: Segment, segments: list[Segment],
           paths: JobPaths) -> None:
    if not segment.tts_path:
        return

    audio, rate = sf.read(paths.root / segment.tts_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = _resample(audio, rate, SAMPLE_RATE)

    available = _available_span(segment, segments)
    duration = len(audio) / SAMPLE_RATE

    # 1) antecipar o início, ocupando o silêncio anterior
    start = segment.start
    if duration > available and segment.gap_before > 0:
        start = max(0.0, segment.start - segment.gap_before)
        available += segment.gap_before

    # 3) esticar, dentro do que não degrada
    if duration > available:
        ratio = duration / available
        if ratio <= 1 / STRETCH_MIN:
            audio = _time_stretch(audio, ratio)
            segment.stretch = round(1 / ratio, 3)
            duration = len(audio) / SAMPLE_RATE
        else:
            audio = _time_stretch(audio, 1 / STRETCH_MIN)
            segment.stretch = STRETCH_MIN
            duration = len(audio) / SAMPLE_RATE

    # 4) o que sobrar vira sobreposição declarada
    if duration > available + 0.05 and "sobreposicao" not in segment.qa_flags:
        segment.qa_flags.append("sobreposicao")

    offset = int(start * SAMPLE_RATE)
    end = min(offset + len(audio), len(track))
    if end > offset:
        track[offset:end] += audio[: end - offset]


def _available_span(segment: Segment, segments: list[Segment]) -> float:
    """Tempo até a próxima fala começar — não a duração nominal do segmento.

    Silêncio depois do segmento é espaço utilizável: a dublagem pode avançar
    sobre ele sem atropelar nada.
    """
    following = [s for s in segments if s.start > segment.start]
    if not following:
        return max(segment.duration, segment.target_duration)
    return max(0.1, min(s.start for s in following) - segment.start)


def _time_stretch(audio: np.ndarray, ratio: float) -> np.ndarray:
    """Comprime (ratio > 1) ou estica (ratio < 1) preservando o tom.

    Rubber Band preserva formantes muito melhor que um phase vocoder, que é o
    que torna fala esticada reconhecível como artificial.
    """
    import pyrubberband as pyrb

    if abs(ratio - 1.0) < 0.005:
        return audio
    ratio = min(max(ratio, STRETCH_MIN), 1 / STRETCH_MIN)
    return pyrb.time_stretch(audio, SAMPLE_RATE, ratio).astype(np.float32)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Reamostragem linear: suficiente para 24 kHz -> 48 kHz, que é um fator
    inteiro exato e não introduz aliasing perceptível em fala."""
    if source_rate == target_rate:
        return audio
    duration = len(audio) / source_rate
    target_length = int(duration * target_rate)
    source_positions = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(source_positions, np.arange(len(audio)), audio).astype(
        np.float32
    )
