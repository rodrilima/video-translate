"""Interface comum dos motores de síntese.

Existe porque a escolha de motor é uma decisão em aberto: não há benchmark
público de qualidade TTS segmentado por pt-BR, e a licença do melhor motor
(OmniVoice, pesos CC-BY-NC) restringe uso comercial. Trocar de motor precisa
ser barato.

A distinção que importa entre motores é `supports_duration`: quem honra uma
duração alvo resolve isocronia na geração; quem não honra empurra o problema
para o time-stretch, que degrada o áudio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import VOICES_DIR


@dataclass(frozen=True)
class Voice:
    """Uma voz do catálogo.

    Existem dois tipos, e a diferença é o que o motor precisa receber:
      - voz *nativa* do modelo (Kokoro): identificada só por um nome em
        `meta.json`, sem áudio;
      - voz *clonada* (OmniVoice, Chatterbox): precisa de `ref.wav` e da
        transcrição correspondente em `ref.txt`.
    """

    name: str
    ref_audio: Path | None
    ref_text: str
    meta: dict

    @property
    def is_native(self) -> bool:
        """True quando a voz vive dentro do modelo e dispensa referência."""
        return self.ref_audio is None

    @classmethod
    def load(cls, name: str) -> Voice:
        directory = VOICES_DIR / name
        if not directory.is_dir():
            raise FileNotFoundError(
                f"voz '{name}' não encontrada em {VOICES_DIR}."
                f" Disponíveis: {', '.join(list_voices()) or 'nenhuma'}"
            )

        meta_path = directory / "meta.json"
        meta = (json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.exists() else {})

        ref_audio = directory / "ref.wav"
        ref_text_path = directory / "ref.txt"
        return cls(
            name=name,
            ref_audio=ref_audio if ref_audio.exists() else None,
            ref_text=ref_text_path.read_text(encoding="utf-8").strip()
            if ref_text_path.exists() else "",
            meta=meta,
        )


@dataclass
class Synthesis:
    audio: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate


class TTSBackend:
    """Contrato dos motores. Implementações carregam o modelo preguiçosamente,
    porque o pipeline instancia o backend antes de saber se vai usá-lo."""

    name: str = "base"
    supports_duration: bool = False
    sample_rate: int = 24000

    def synthesize(self, text: str, voice: Voice, *,
                   duration_s: float | None = None) -> Synthesis:
        raise NotImplementedError

    def warmup(self) -> None:
        """Carrega pesos e paga o custo da primeira inferência antecipadamente."""
        return None


def get_backend(name: str) -> TTSBackend:
    """Fábrica por nome, usada pelos presets."""
    if name == "omnivoice":
        from .omnivoice import OmniVoiceBackend

        return OmniVoiceBackend()
    if name == "kokoro":
        from .kokoro import KokoroBackend

        return KokoroBackend()
    raise ValueError(f"backend de TTS desconhecido: {name}")


def list_voices() -> list[str]:
    """Vozes do catálogo: ou têm meta.json (nativas) ou ref.wav (clonadas)."""
    if not VOICES_DIR.exists():
        return []
    return sorted(
        d.name for d in VOICES_DIR.iterdir()
        if d.is_dir() and ((d / "meta.json").exists() or (d / "ref.wav").exists())
    )
