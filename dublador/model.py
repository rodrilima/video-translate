"""Modelo de dados do pipeline.

Um único `segments.json` carrega o estado de todas as etapas. É inspecionável,
diffável e editável à mão — corrigir uma tradução e re-renderizar só o TTS é
uma operação de editor de texto.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Word:
    """Palavra com timestamps. É o insumo de toda a isocronia, por isso vem do
    decoder TDT do Parakeet e não de DTW sobre cross-attention."""

    w: str
    s: float
    e: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text_en: str
    words: list[Word] = field(default_factory=list)

    # quem fala, quando a diarização identifica; None em vídeo de locutor único
    speaker: str | None = None
    # voz do catálogo atribuída a este segmento
    voice: str | None = None

    # silêncio livre antes do segmento; absorve estouro sem tocar no áudio da fala
    gap_before: float = 0.0
    # pausas internas (offset relativo ao início, duração), para marcar [pause]
    pauses: list[tuple[float, float]] = field(default_factory=list)

    text_pt_raw: str | None = None      # etapa 5, tradutor
    text_pt_final: str | None = None    # etapa 6, revisor
    syllable_budget: int = 0

    tts_path: str | None = None
    tts_duration: float | None = None
    stretch: float = 1.0

    cut_note: str = ""          # o que o revisor sacrificou para caber
    qa_flags: list[str] = field(default_factory=list)
    attempts: int = 0
    overflow_pct: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def target_duration(self) -> float:
        """Espaço realmente disponível: o segmento mais o silêncio à frente."""
        return self.duration + self.gap_before

    @property
    def text_for_tts(self) -> str:
        """O texto que vai ao sintetizador, com o melhor estágio disponível."""
        return self.text_pt_final or self.text_pt_raw or self.text_en

    def to_dict(self) -> dict:
        d = asdict(self)
        d["words"] = [asdict(w) if not isinstance(w, dict) else w for w in self.words]
        d["pauses"] = [list(p) for p in self.pauses]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        data = dict(d)
        data["words"] = [Word(**w) if isinstance(w, dict) else w
                         for w in data.get("words", [])]
        data["pauses"] = [tuple(p) for p in data.get("pauses", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def save_segments(segments: list[Segment], path: Path) -> None:
    payload = {"version": 1, "segments": [s.to_dict() for s in segments]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_segments(path: Path) -> list[Segment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Segment.from_dict(s) for s in payload["segments"]]
