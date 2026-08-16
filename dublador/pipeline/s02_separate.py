"""Etapa 2: separar voz e trilha instrumental.

O que o pipeline consome daqui são os dois stems, para fins diferentes:

  - `instrumental.wav` vira o leito sonoro da mixagem final, preservando música
    e efeitos do vídeo original sob a dublagem. Sem ele, a voz em inglês
    continua audível ao fundo, que é o defeito mais perceptível do resultado.
  - `vocals.wav` vira a entrada do ASR, que transcreve melhor sem música
    competindo, e no futuro a entrada da diarização.

Escolha do modelo: como o stem que de fato entra na mixagem é o instrumental,
o critério é o SDR *instrumental*, não o vocal. Pelos scores publicados no
catálogo da própria biblioteca, o BS-Roformer lidera nesse stem (16,45), à
frente dos MelBand Roformer de topo — que pontuam mais alto em vocais mas são
modelos de stem único.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

from ..config import MODELS_DIR, JobPaths

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


MODELS = {
    # melhor SDR instrumental (16.45) — é o stem que vai para a mixagem
    "bs_roformer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    # melhor SDR vocal (12.60), útil se o objetivo for a transcrição
    "melband_roformer": "vocals_mel_band_roformer.ckpt",
    # mais leve, para o preset de rascunho
    "htdemucs": "htdemucs_ft.yaml",
}

DEFAULT_MODEL = "bs_roformer"


def separate(paths: JobPaths, *, model: str = DEFAULT_MODEL,
             progress: ProgressFn = _noop) -> dict:
    """Escreve vocals.wav e instrumental.wav a partir de audio48.wav."""
    from mlx_audio_separator import Separator

    if not paths.audio.exists():
        raise FileNotFoundError(
            f"{paths.audio} não existe; rode a etapa de download antes"
        )

    filename = MODELS.get(model, MODELS[DEFAULT_MODEL])
    workdir = paths.root / "stems"
    workdir.mkdir(exist_ok=True)

    progress(0.05, f"carregando {model}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    separator = Separator(output_dir=str(workdir), output_format="WAV",
                          model_file_dir=str(MODELS_DIR),
                          log_level=logging.WARNING)
    separator.load_model(model_filename=filename)

    progress(0.25, "separando voz e trilha")
    outputs = separator.separate(str(paths.audio))

    produced = [workdir / name if not Path(name).is_absolute() else Path(name)
                for name in outputs]
    progress(0.85, "organizando stems")

    vocals = _pick(produced, ("vocal",))
    instrumental = _pick(produced, ("instrument", "no_vocal", "accompan"))

    if vocals:
        shutil.move(str(vocals), paths.vocals)
    if instrumental:
        shutil.move(str(instrumental), paths.instrumental)

    shutil.rmtree(workdir, ignore_errors=True)

    result = {
        "model": model,
        "model_file": filename,
        "vocals": paths.vocals.exists(),
        "instrumental": paths.instrumental.exists(),
    }
    if not result["instrumental"]:
        raise RuntimeError(
            "a separação não produziu stem instrumental; arquivos gerados: "
            + ", ".join(p.name for p in produced)
        )

    progress(1.0, "vocals.wav e instrumental.wav prontos")
    return result


def _pick(paths: list[Path], keywords: tuple[str, ...]) -> Path | None:
    """Os nomes de saída variam por modelo, daí a busca por palavra-chave."""
    for path in paths:
        lowered = path.name.lower()
        if any(keyword in lowered for keyword in keywords):
            return path
    return None
