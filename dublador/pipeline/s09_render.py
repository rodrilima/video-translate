"""Etapas 9 e 10: mixar e montar o arquivo final.

A dublagem é normalizada antes de entrar na mixagem — sem isso o compressor de
sidechain dispara de forma inconsistente entre segmentos, porque o TTS produz
volumes ligeiramente diferentes a cada geração.

O leito sonoro ideal é o instrumental separado da faixa original, que preserva
trilha e efeitos sem a voz. Enquanto a separação de fontes não estiver
disponível, cai para o áudio original em volume reduzido: a voz em inglês
continua audível ao fundo, o que é pior, mas mantém o pipeline inteiro
executável e testável.
"""

from __future__ import annotations

from typing import Callable

from ..config import JobPaths
from ..model import load_segments
from ..utils import ffmpeg
from ..utils.srt import write_srt

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


# Volume do leito quando ele ainda contém a voz original: baixo o bastante para
# não competir com a dublagem.
FALLBACK_BED_GAIN = 0.18
SEPARATED_BED_GAIN = 0.80


def render(paths: JobPaths, *, progress: ProgressFn = _noop) -> dict:
    """Produz out.mp4 com dublagem, áudio original e legendas."""
    segments = load_segments(paths.segments)

    progress(0.05, "gerando legendas")
    write_srt(segments, paths.srt_pt, language="pt")
    write_srt(segments, paths.srt_en, language="en")

    progress(0.15, "normalizando dublagem")
    normalized = paths.root / "dub_norm.wav"
    ffmpeg.normalize_loudness(paths.dub, normalized)

    separated = paths.instrumental.exists()
    bed = paths.instrumental if separated else paths.audio
    gain = SEPARATED_BED_GAIN if separated else FALLBACK_BED_GAIN

    progress(0.45, "mixando" + ("" if separated else " (sem separação de fontes)"))
    ffmpeg.mix_with_ducking(normalized, bed, paths.mixed, bed_gain=gain)

    progress(0.75, "montando arquivo final")
    # Português primeiro: é a faixa que o player abre por padrão.
    ffmpeg.mux(paths.video, paths.mixed, paths.output, subtitles=[
        (paths.srt_pt, "por", "Português"),
        (paths.srt_en, "eng", "English"),
    ])

    loudness = ffmpeg.measure_loudness(paths.mixed)
    result = {
        "output": str(paths.output),
        "separated_bed": separated,
        "integrated_lufs": loudness.get("input_i"),
        "true_peak": loudness.get("input_tp"),
        "duration": ffmpeg.duration(paths.output),
    }
    progress(1.0, f"{paths.output.name} pronto")
    return result
