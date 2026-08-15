"""Etapa 7: sintetizar a fala de cada segmento.

A regra central vem de escuta, não de teoria: comprimir a fala abaixo de 0,90x
da duração natural degrada perceptivelmente. Então esta etapa nunca pede mais
compressão que isso — se o texto ainda não couber, o segmento é sintetizado no
menor tamanho aceitável e marcado, e o excedente é tratado no encaixe (etapa 8)
ou volta para o revisor encurtar o texto.

Comprimir áudio é sempre a última escolha; encurtar texto é a primeira.
"""

from __future__ import annotations

from typing import Callable

import soundfile as sf

from ..config import DURATION_FLOOR, OVERFLOW_TOLERANCE, JobPaths
from ..model import Segment, load_segments, save_segments
from ..tts.base import TTSBackend, Voice, get_backend
from ..utils.audio import trim_silence
from ..utils.syllables import estimated_duration

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


def synthesize(paths: JobPaths, *, backend_name: str, voice_name: str | None,
               progress: ProgressFn = _noop) -> list[Segment]:
    """Gera tts/seg_NNN.wav para cada segmento e anota a duração obtida.

    A voz vem do próprio segmento quando a diarização identificou quem fala;
    `voice_name` só entra como imposição do usuário ou fallback.
    """
    segments = load_segments(paths.segments)
    paths.ensure()

    backend = get_backend(backend_name)
    voices = _voice_cache(segments, voice_name)

    progress(0.02, f"carregando {backend_name}")
    backend.warmup()

    for index, segment in enumerate(segments):
        voice = voices[segment.voice if not voice_name else voice_name]
        _synthesize_segment(backend, segment, voice, paths)
        progress(0.05 + 0.92 * (index + 1) / len(segments),
                 f"sintetizando {index + 1}/{len(segments)}")
        if index % 10 == 0:
            save_segments(segments, paths.segments)

    save_segments(segments, paths.segments)

    overflowing = sum(1 for s in segments
                      if s.overflow_pct > OVERFLOW_TOLERANCE)
    used = sorted({s.voice for s in segments if s.voice})
    detail = f", {len(used)} vozes" if len(used) > 1 else ""
    progress(1.0,
             f"{len(segments)} sintetizados, {overflowing} acima do tempo{detail}")
    return segments


def _voice_cache(segments: list[Segment], forced: str | None) -> dict:
    """Carrega uma vez cada voz usada, em vez de a cada segmento."""
    names = {forced} if forced else {s.voice for s in segments if s.voice}
    names.discard(None)
    if not names:
        names = {"alex"}
    cache = {name: Voice.load(name) for name in names}
    # segmentos sem voz atribuída caem na primeira disponível
    default = cache[sorted(cache)[0]]
    return _DefaultDict(cache, default)


class _DefaultDict(dict):
    def __init__(self, data: dict, default) -> None:
        super().__init__(data)
        self._default = default

    def __missing__(self, key):
        return self._default


def _synthesize_segment(backend: TTSBackend, segment: Segment, voice: Voice,
                        paths: JobPaths) -> None:
    text = segment.text_for_tts.strip()
    if not text:
        segment.tts_path = None
        segment.tts_duration = 0.0
        return

    target = segment.target_duration
    natural = estimated_duration(text)
    requested = _requested_duration(natural, target)

    result = backend.synthesize(text, voice, duration_s=requested)

    # O sintetizador embute silêncio nas pontas. Mantê-lo atrasaria a entrada da
    # fala em relação à boca do locutor de forma constante ao longo do vídeo.
    audio = trim_silence(result.audio, result.sample_rate)
    duration = len(audio) / result.sample_rate

    filename = f"seg_{segment.id:04d}.wav"
    sf.write(paths.tts_dir / filename, audio, result.sample_rate)

    segment.tts_path = f"tts/{filename}"
    segment.tts_duration = round(duration, 3)
    segment.overflow_pct = round(max(0.0, duration / target - 1), 4)

    if segment.overflow_pct > OVERFLOW_TOLERANCE:
        _flag(segment, "audio_maior_que_slot")


def _requested_duration(natural: float, target: float) -> float:
    """Quanto tempo pedir ao motor.

    Cabendo, pede o tempo natural — esticar fala para preencher silêncio soa tão
    artificial quanto comprimi-la, e o silêncio restante é útil na montagem.
    Não cabendo, comprime até o piso e não além.
    """
    if natural <= target:
        return natural
    return max(target, natural * DURATION_FLOOR)


def _flag(segment: Segment, flag: str) -> None:
    if flag not in segment.qa_flags:
        segment.qa_flags.append(flag)
