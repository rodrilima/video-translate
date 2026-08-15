"""Diarização: descobrir quem fala em cada trecho.

Sem isso o vídeo inteiro recebe uma voz só, e uma conversa entre duas pessoas
sai como alguém falando sozinho. Três coisas dependem deste resultado:

  - a voz de cada fala, escolhida por locutor em vez de uma para o vídeo todo;
  - as fronteiras dos segmentos, que passam a cortar também na troca de quem
    fala — sem isso uma pergunta e sua resposta caem no mesmo bloco e ficam
    impossíveis de separar depois;
  - a concordância de gênero em português, que é por pessoa e não por vídeo.

Roda sobre o stem de vocais, não sobre o áudio bruto: música e efeitos ao fundo
degradam bastante a separação de locutores.
"""

from __future__ import annotations

import json
from typing import Callable

from ..config import JobPaths
from ..utils import ffmpeg

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


# O senko aceita apenas 16 kHz mono 16 bits.
DIARIZATION_RATE = 16000

# Trechos muito curtos costumam ser ruído de fronteira, não fala de alguém.
MIN_TURN = 0.35


def diarize(paths: JobPaths, *, progress: ProgressFn = _noop) -> dict:
    """Escreve diarization.json com os turnos de fala."""
    from senko import Diarizer

    source = paths.vocals if paths.vocals.exists() else paths.audio
    if not source.exists():
        raise FileNotFoundError(f"sem áudio para diarizar em {paths.root}")

    prepared = paths.root / "diarize_input.wav"
    progress(0.05, "preparando áudio")
    ffmpeg.extract_audio(source, prepared, rate=DIARIZATION_RATE, mono=True)

    progress(0.15, "carregando o modelo")
    diarizer = Diarizer(device="auto", quiet=True)

    progress(0.40, "identificando locutores")
    raw = diarizer.diarize(str(prepared))
    prepared.unlink(missing_ok=True)

    turns = _normalize(raw)
    speakers = sorted({turn["speaker"] for turn in turns})

    payload = {
        "speakers": speakers,
        "turns": turns,
        "speaking_time": {
            speaker: round(sum(t["end"] - t["start"]
                               for t in turns if t["speaker"] == speaker), 2)
            for speaker in speakers
        },
    }
    (paths.root / "diarization.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    progress(1.0, f"{len(speakers)} locutor(es), {len(turns)} turnos")
    return payload


def load(paths: JobPaths) -> dict:
    path = paths.root / "diarization.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def speaker_at(turns: list[dict], start: float, end: float) -> str | None:
    """Quem fala num intervalo, pelo turno de maior sobreposição.

    Sobreposição, e não o instante inicial: uma fala pode começar durante a
    cauda do turno anterior, e o dono do trecho é quem ocupa mais tempo dele.
    """
    best, best_overlap = None, 0.0
    for turn in turns:
        overlap = min(end, turn["end"]) - max(start, turn["start"])
        if overlap > best_overlap:
            best, best_overlap = turn["speaker"], overlap
    return best


def _normalize(raw) -> list[dict]:
    """Uniformiza a saída da biblioteca.

    Usamos `raw_segments`, não `merged_segments`: o segundo agrupa turnos do
    mesmo locutor ao longo de todo o áudio e devolve poucos blocos enormes, sem
    a granularidade necessária para cortar segmentos na troca de fala. A fusão
    de turnos próximos é feita aqui, com critério de tempo controlado.
    """
    if isinstance(raw, dict):
        entries = raw.get("raw_segments") or raw.get("segments") or []
    else:
        entries = raw or []

    turns: list[dict] = []
    for entry in entries or []:
        start = float(entry.get("start", entry.get("begin", 0.0)))
        end = float(entry.get("end", entry.get("stop", 0.0)))
        speaker = str(entry.get("speaker", entry.get("label", "SPEAKER_00")))
        if end - start >= MIN_TURN:
            turns.append({"start": round(start, 3), "end": round(end, 3),
                          "speaker": speaker})

    turns.sort(key=lambda t: t["start"])
    return _merge_adjacent(turns)


def _merge_adjacent(turns: list[dict], gap: float = 0.25) -> list[dict]:
    """Junta turnos consecutivos do mesmo locutor separados por pausa curta."""
    merged: list[dict] = []
    for turn in turns:
        if merged and merged[-1]["speaker"] == turn["speaker"] \
                and turn["start"] - merged[-1]["end"] <= gap:
            merged[-1]["end"] = turn["end"]
        else:
            merged.append(dict(turn))
    return merged
