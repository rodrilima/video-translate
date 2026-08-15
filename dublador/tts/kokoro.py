"""Backend Kokoro via MLX.

Kokoro é pequeno (82M) e traz vozes de português embutidas, o que dispensa
clipe de referência — útil quando não há uma gravação de boa qualidade para
clonar, e rápido o bastante para o preset de rascunho.

O custo é controle: o modelo aceita `speed`, não uma duração alvo, então a
isocronia depende de estimar o fator antes de sintetizar e corrigir depois.
Licença Apache-2.0, sem restrição comercial.
"""

from __future__ import annotations

import numpy as np

from ..config import STRETCH_MAX, STRETCH_MIN
from ..utils.syllables import estimated_duration
from .base import Synthesis, TTSBackend, Voice

MODEL_ID = "mlx-community/Kokoro-82M-bf16"
DEFAULT_VOICE = "pf_dora"


class KokoroBackend(TTSBackend):
    name = "kokoro"
    supports_duration = False  # só velocidade, não duração alvo
    sample_rate = 24000

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            self._model = load_model(self.model_id)
        return self._model

    def warmup(self) -> None:
        self._ensure_model()

    def synthesize(self, text: str, voice: Voice, *,
                   duration_s: float | None = None) -> Synthesis:
        model = self._ensure_model()

        # Sem controle de duração: aproxima o alvo escolhendo a velocidade a
        # partir da duração estimada, limitada à faixa em que a fala não degrada.
        speed = 1.0
        if duration_s:
            natural = estimated_duration(text)
            if natural > 0:
                speed = min(max(natural / duration_s, STRETCH_MIN), STRETCH_MAX)

        voice_id = voice.meta.get("kokoro_voice", DEFAULT_VOICE)
        chunks: list[np.ndarray] = []
        sample_rate = self.sample_rate
        for result in model.generate(text=text, voice=voice_id, speed=speed,
                                     lang_code="p"):
            chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
            sample_rate = getattr(result, "sample_rate", sample_rate) or sample_rate

        if not chunks:
            raise RuntimeError(f"Kokoro não produziu áudio para: {text[:60]!r}")

        return Synthesis(audio=np.concatenate(chunks), sample_rate=sample_rate)
