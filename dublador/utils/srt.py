"""Geração de legendas SRT a partir dos segmentos."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import Segment

# Marcadores de controle que o revisor insere para o TTS; não vão para a tela.
_MARKUP_RE = re.compile(r"\[[^\]]*\]")


def write_srt(segments: list[Segment], path: Path, *,
              language: str = "pt") -> Path:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = (segment.text_pt_final or segment.text_pt_raw or ""
                ) if language == "pt" else segment.text_en
        text = _MARKUP_RE.sub("", text).strip()
        if not text:
            continue
        blocks.append(
            f"{index}\n"
            f"{_timestamp(segment.start)} --> {_timestamp(segment.end)}\n"
            f"{text}\n"
        )

    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def _timestamp(seconds: float) -> str:
    """Formato SRT: HH:MM:SS,mmm — vírgula, não ponto."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
