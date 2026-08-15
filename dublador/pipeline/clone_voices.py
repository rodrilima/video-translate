"""Extrai uma voz de referência para cada locutor, a partir do próprio vídeo.

Serve à dublagem que preserva o timbre de quem fala. Só é possível porque a
diarização já separa os turnos: dela sai, para cada pessoa, um trecho contínuo
de fala limpa, que é exatamente o insumo que um clonador zero-shot pede.

Por que não é o padrão:

  - os pesos do motor de clonagem são CC-BY-NC, sem uso comercial, enquanto o
    motor de vozes fixas é Apache-2.0;
  - custa ~5x mais em memória, disco e tempo;
  - clonar a voz de uma pessoa real é uma decisão de quem publica, não um
    detalhe técnico que o software deva tomar sozinho.

A alternativa que parecia mais barata — misturar as vozes fixas do catálogo
para chegar perto do locutor — foi medida e não resolve: dentro de um mesmo
gênero as vozes disponíveis cobrem apenas 134 a 140 Hz, e de todo modo casar
frequência não aproxima timbre, que é o que se percebe como voz parecida.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import soundfile as sf

from ..config import JobPaths
from . import diarize

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


# Faixa de duração do clipe de referência. Curto demais não caracteriza a voz;
# longo demais não acrescenta e pesa na inferência.
MIN_REF = 4.0
MAX_REF = 10.0


def extract(paths: JobPaths, *, progress: ProgressFn = _noop) -> dict:
    """Escreva um clipe de referência por locutor em vozes_do_video/."""
    from ..model import load_segments

    data = diarize.load(paths)
    turns = data.get("turns") or []
    if not turns:
        raise RuntimeError(
            "sem diarização: a clonagem precisa saber quem fala em cada trecho"
        )

    source = paths.vocals if paths.vocals.exists() else paths.audio
    audio, rate = sf.read(source, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    segments = load_segments(paths.segments) if paths.segments.exists() else []
    outdir = paths.root / "vozes_do_video"
    outdir.mkdir(exist_ok=True)

    speakers = sorted({turn["speaker"] for turn in turns})
    profiles: dict[str, dict] = {}

    for index, speaker in enumerate(speakers):
        progress(0.1 + 0.8 * index / max(len(speakers), 1),
                 f"extraindo referência de {speaker}")

        window = _best_window(turns, speaker)
        if window is None:
            continue

        start, end = window
        clip = audio[int(start * rate):int(end * rate)]
        if clip.size == 0:
            continue

        path = outdir / f"{speaker}.wav"
        sf.write(path, clip, rate)

        text = _text_in(segments, start, end)
        (outdir / f"{speaker}.txt").write_text(text, encoding="utf-8")

        profiles[speaker] = {
            "ref_audio": str(path.relative_to(paths.root)),
            "ref_text": text,
            "window": [round(start, 2), round(end, 2)],
            "duration": round(end - start, 2),
        }

    (paths.root / "clones.json").write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")

    progress(1.0, f"{len(profiles)} voz(es) extraída(s) do vídeo")
    return profiles


def _best_window(turns: list[dict], speaker: str) -> tuple[float, float] | None:
    """Escolhe o trecho de referência: o turno contínuo mais longo da pessoa.

    Turno contínuo, e não a soma de vários, porque emendar trechos distantes
    introduz cortes que o clonador reproduz como artefato.
    """
    candidates = [(t["start"], t["end"]) for t in turns
                  if t["speaker"] == speaker and t["end"] - t["start"] >= MIN_REF]
    if not candidates:
        candidates = [(t["start"], t["end"]) for t in turns
                      if t["speaker"] == speaker]
    if not candidates:
        return None

    start, end = max(candidates, key=lambda w: w[1] - w[0])
    # descarta o primeiro instante do turno, onde costuma haver resíduo de quem
    # falava antes
    start = min(start + 0.2, end)
    return start, min(start + MAX_REF, end)


def _text_in(segments: list, start: float, end: float) -> str:
    """Transcrição correspondente ao clipe, que o clonador exige junto do áudio.

    Usa timestamps de palavra: pegar apenas sentenças inteiramente contidas na
    janela devolve texto curto demais para o áudio, e a incompatibilidade entre
    os dois degrada a clonagem.
    """
    words = [w.w for segment in segments for w in segment.words
             if start <= w.s and w.e <= end]
    return " ".join(words).strip()


def load(paths: JobPaths) -> dict:
    path = paths.root / "clones.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def voice_for(paths: JobPaths, speaker: str | None):
    """Voz clonada de um locutor, pronta para o backend de TTS."""
    from ..tts.base import Voice

    profile = load(paths).get(speaker or "")
    if not profile:
        return None
    return Voice(
        name=f"clone:{speaker}",
        ref_audio=paths.root / profile["ref_audio"],
        ref_text=profile["ref_text"],
        meta={"clonada": True},
    )
