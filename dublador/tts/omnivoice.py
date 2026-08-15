"""Backend OmniVoice via MLX.

É o motor padrão por um motivo específico: aceita `duration_s` e o honra com
erro medido abaixo de 1% neste hardware, o que resolve isocronia na geração em
vez de no pós-processamento.

Licença: o código do OmniVoice é Apache-2.0, mas os *pesos* são CC-BY-NC —
restrição herdada do dataset de treino. Livre para uso pessoal; para uso
comercial troque o backend (ver dublador/tts/base.py).
"""

from __future__ import annotations

import numpy as np

from .base import Synthesis, TTSBackend, Voice

MODEL_ID = "mlx-community/OmniVoice-bf16"

# Passos de difusão. 32 é o padrão do modelo; abaixo disso a qualidade cai
# visivelmente, acima o ganho não compensa o tempo.
NUM_STEPS = 32
LANGUAGE = "portuguese"


class OmniVoiceBackend(TTSBackend):
    name = "omnivoice"
    supports_duration = True
    sample_rate = 24000

    def __init__(self, model_id: str = MODEL_ID, num_steps: int = NUM_STEPS) -> None:
        self.model_id = model_id
        self.num_steps = num_steps
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
        if voice.is_native:
            raise ValueError(
                f"a voz '{voice.name}' é nativa de outro motor e não tem clipe de"
                " referência; OmniVoice só sintetiza por clonagem."
            )
        model = self._ensure_model()

        chunks: list[np.ndarray] = []
        sample_rate = self.sample_rate
        for result in model.generate(
            text=text,
            language=LANGUAGE,
            ref_audio=str(voice.ref_audio),
            ref_text=voice.ref_text,
            duration_s=duration_s,
            num_steps=self.num_steps,
        ):
            chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
            sample_rate = getattr(result, "sample_rate", sample_rate) or sample_rate

        if not chunks:
            raise RuntimeError(f"OmniVoice não produziu áudio para: {text[:60]!r}")

        return Synthesis(audio=np.concatenate(chunks), sample_rate=sample_rate)
