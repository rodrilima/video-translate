"""Move os LLMs de texto para a biblioteca do LM Studio.

O cache do Hugging Face e o LM Studio guardam o mesmo formato — MLX
safetensors — em layouts diferentes: o cache mantém os arquivos em `blobs/` e
monta o snapshot com links simbólicos, enquanto o LM Studio quer arquivos reais
numa pasta plana. Passar de um para o outro é renomear os blobs.

Só os LLMs de texto são movidos. O LM Studio não executa reconhecimento de fala
nem síntese de voz, então Parakeet, Kokoro, Whisper e OmniVoice continuam no
cache — levá-los para lá só criaria pastas que ele não sabe abrir.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
LMSTUDIO_DIR = Path.home() / ".lmstudio" / "models"


@dataclass
class Modelo:
    repo: str            # "mlx-community/Hy-MT2-7B-4bit"
    origem: Path         # pasta no cache do Hugging Face
    destino: Path        # pasta correspondente no LM Studio
    tamanho_gb: float

    @property
    def ja_migrado(self) -> bool:
        return (self.destino / "config.json").exists()


def llms_de_texto() -> set[str]:
    """Os modelos que o LM Studio consegue executar: os de texto dos presets."""
    from ..config import PRESETS

    repos = set()
    for preset in PRESETS.values():
        if preset.translator_model:
            repos.add(preset.translator_model)
        repos.add(preset.reviewer_model)
    return repos


def listar() -> list[Modelo]:
    """Modelos de texto presentes no cache, com destino calculado."""
    if not HF_HUB.exists():
        return []

    encontrados = []
    for repo in sorted(llms_de_texto()):
        org, nome = repo.split("/", 1)
        origem = HF_HUB / f"models--{org}--{nome}"
        if not origem.exists():
            continue
        encontrados.append(Modelo(
            repo=repo,
            origem=origem,
            destino=LMSTUDIO_DIR / org / nome,
            tamanho_gb=_tamanho_gb(origem),
        ))
    return encontrados


def mover(modelo: Modelo) -> None:
    """Leva um modelo para o LM Studio, sem cópia intermediária.

    Os blobs são renomeados para o destino em vez de copiados: como as duas
    pastas ficam no mesmo sistema de arquivos, isso é instantâneo e não exige
    o dobro de espaço livre.
    """
    snapshot = _snapshot_atual(modelo.origem)
    if snapshot is None:
        raise RuntimeError(f"snapshot não encontrado em {modelo.origem}")

    modelo.destino.mkdir(parents=True, exist_ok=True)

    for item in snapshot.iterdir():
        alvo = modelo.destino / item.name
        if alvo.exists():
            continue
        if item.is_symlink():
            os.replace(item.resolve(), alvo)
        elif item.is_file():
            shutil.copy2(item, alvo)

    shutil.rmtree(modelo.origem, ignore_errors=True)


def _snapshot_atual(origem: Path) -> Path | None:
    """A revisão baixada. Havendo mais de uma, fica com a mais recente."""
    snapshots = origem / "snapshots"
    if not snapshots.exists():
        return None
    candidatos = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidatos:
        return None
    return max(candidatos, key=lambda p: p.stat().st_mtime)


def _tamanho_gb(caminho: Path) -> float:
    total = 0
    for item in caminho.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return round(total / 1073741824, 2)
