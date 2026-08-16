"""Configuração central: caminhos, presets e constantes do pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = Path(__file__).resolve().parent / "voices"

# Onde as pastas de job são criadas. O padrão é o diretório de onde o comando
# foi chamado, para o resultado aparecer onde a pessoa está trabalhando em vez
# de dentro da instalação do programa.
_JOBS_DIR: Path | None = None

# Local usado antes de a saída passar a seguir o diretório atual. Continua
# sendo consultado na busca por jobs antigos, mas nada novo é criado nele.
LEGACY_JOBS_DIR = ROOT / "data" / "jobs"


def jobs_dir() -> Path:
    """Diretório onde as pastas de job vivem.

    Precedência: o que a linha de comando definiu, depois a variável de
    ambiente DUBLADOR_DIR, depois o diretório atual.
    """
    if _JOBS_DIR is not None:
        return _JOBS_DIR
    from_env = os.environ.get("DUBLADOR_DIR")
    if from_env:
        return Path(from_env).expanduser().resolve()
    return Path.cwd()


def set_jobs_dir(path: Path | str | None) -> None:
    global _JOBS_DIR
    _JOBS_DIR = Path(path).expanduser().resolve() if path else None

# Onde ficam os modelos do separador de fontes. O padrão da biblioteca é /tmp,
# que o macOS limpa periodicamente — e são 1,2 GB rebaixados a cada limpeza.
MODELS_DIR = Path.home() / ".cache" / "dublador" / "models"

# Taxa silábica do português brasileiro falado, em sílabas por segundo.
# MEDIDO, não estimado: 8 frases sintetizadas sem restrição de duração com
# OmniVoice deram mediana 5.17 síl/s (desvio 0.29). O valor de literatura que eu
# usara antes, 6.0, pedia ~16% menos tempo do que a fala precisa e comprimia
# todos os segmentos — o áudio saía perceptivelmente embolado.
# Recalibre ao trocar de motor ou de voz: scripts/calibrate_syllable_rate.py
SYLLABLE_RATE_PTBR = 5.2

# Tolerância de estouro de duração antes de acionar reescrita (fração da duração alvo).
OVERFLOW_TOLERANCE = 0.10

# Limites de time-stretch. Acima de 1.15x a fala fica audivelmente artificial,
# mesmo com Rubber Band.
STRETCH_MIN = 0.90
STRETCH_MAX = 1.15

# Teto empírico de compressão de texto: abaixo disso a tradução perde informação
# em vez de ficar mais concisa.
COMPRESSION_FLOOR = 0.49

# Piso de compressão temporal do TTS, VALIDADO POR ESCUTA: até 0.90x da duração
# natural a fala continua aceitável; abaixo disso incomoda. Esse número é a
# razão de a isocronia ser resolvida encurtando texto e não espremendo áudio.
DURATION_FLOOR = 0.90

# Alvos de loudness (EBU R128).
LUFS_TARGET = -16.0
TRUE_PEAK_MAX = -1.5

SAMPLE_RATE = 48000  # taxa de trabalho para separação e mixagem
ASR_SAMPLE_RATE = 16000  # o Parakeet espera 16 kHz mono


@dataclass(frozen=True)
class Preset:
    """Conjunto de escolhas de modelo por etapa, trocando qualidade por tempo."""

    name: str
    asr_model: str
    separator_model: str
    translator_model: str | None
    reviewer_model: str
    tts_backend: str
    fit_attempts: int

    @property
    def uses_translator(self) -> bool:
        """False quando um único LLM faz tradução e revisão numa passada só."""
        return self.translator_model is not None


# Os três presets usam o mesmo revisor. Ele é a etapa que domina o tempo, e o
# maior modelo que cabe com folga nos 18 GB disponíveis para a GPU já é o de
# 14B — subir dele exigiria abrir mão da residência dos modelos, que sozinha
# vale 16s por execução. O que separa os presets é o esforço: quantas rodadas
# de reescrita, qual separador de fontes e qual sintetizador.
PRESETS: dict[str, Preset] = {
    "draft": Preset(
        name="draft",
        asr_model="mlx-community/parakeet-tdt-0.6b-v3",
        separator_model="htdemucs",
        translator_model=None,  # passada única: o revisor traduz e adapta junto
        reviewer_model="mlx-community/Qwen3-14B-4bit",
        tts_backend="kokoro",
        fit_attempts=0,
    ),
    "balanced": Preset(
        name="balanced",
        asr_model="mlx-community/parakeet-tdt-0.6b-v3",
        separator_model="bs_roformer",
        translator_model="mlx-community/Hy-MT2-7B-4bit",
        reviewer_model="mlx-community/Qwen3-14B-4bit",
        tts_backend="kokoro",
        fit_attempts=3,
    ),
    "max": Preset(
        name="max",
        asr_model="mlx-community/parakeet-tdt-0.6b-v3",
        separator_model="bs_roformer",
        translator_model="mlx-community/Hy-MT2-7B-4bit",
        reviewer_model="mlx-community/Qwen3-14B-4bit",
        tts_backend="kokoro",
        fit_attempts=5,
    ),
}

DEFAULT_PRESET = "balanced"


def resolve_job_root(name: str) -> Path:
    """Localiza a pasta de um job aceitando o nome completo ou só o ID.

    As pastas são nomeadas `titulo-do-video-ID`, mas os subcomandos continuam
    aceitando apenas o ID: quem digita `dublador info ZTSI3DDP_4A` não deveria
    precisar lembrar do título.
    """
    destino = jobs_dir()

    # Procura no diretório atual e, só depois, no local antigo — jobs criados
    # antes de a saída seguir o diretório atual continuam acessíveis pelo ID.
    for base in (destino, LEGACY_JOBS_DIR):
        exact = base / name
        if exact.exists():
            return exact
        if base.exists():
            matches = sorted(p for p in base.glob(f"*-{name}") if p.is_dir())
            if matches:
                return matches[0]

    return destino / name


@dataclass
class JobPaths:
    """Localiza todos os artefatos de um job. Cada etapa lê e escreve aqui."""

    job_id: str
    root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = resolve_job_root(self.job_id)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.tts_dir.mkdir(exist_ok=True)

    @property
    def meta(self) -> Path:
        return self.root / "job.json"

    @property
    def video(self) -> Path:
        return self.root / "video.mkv"

    @property
    def audio(self) -> Path:
        return self.root / "audio48.wav"

    @property
    def vocals(self) -> Path:
        return self.root / "vocals.wav"

    @property
    def instrumental(self) -> Path:
        return self.root / "instrumental.wav"

    @property
    def asr(self) -> Path:
        return self.root / "asr.json"

    @property
    def segments(self) -> Path:
        return self.root / "segments.json"

    @property
    def glossary(self) -> Path:
        return self.root / "glossary.json"

    @property
    def tts_dir(self) -> Path:
        return self.root / "tts"

    @property
    def dub(self) -> Path:
        return self.root / "dub.wav"

    @property
    def mixed(self) -> Path:
        return self.root / "mixed.wav"

    @property
    def srt_pt(self) -> Path:
        return self.root / "pt.srt"

    @property
    def srt_en(self) -> Path:
        return self.root / "en.srt"

    @property
    def output(self) -> Path:
        return self.root / "out.mp4"
