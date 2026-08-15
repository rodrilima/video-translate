"""Wrappers finos sobre ffmpeg/ffprobe.

Regras que o pipeline depende e que quebram silenciosamente se esquecidas:
  - o vídeo nunca é reencodado (`-c:v copy`);
  - `amix` precisa de `normalize=0`, senão o ganho é dividido pelo número de
    entradas e a dublagem sai baixa;
  - `loudnorm` em duas passadas, porque a passada única é adaptativa e achata a
    dinâmica;
  - códigos de idioma em ISO 639-2 de três letras (`por`, `eng`), porque muitos
    players ignoram os de duas.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..config import LUFS_TARGET, SAMPLE_RATE, TRUE_PEAK_MAX


class FFmpegError(RuntimeError):
    pass


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise FFmpegError(
            f"{binary} não encontrado no PATH. Instale com: brew install ffmpeg"
        )
    return path


def run(args: list[str], *, desc: str = "ffmpeg") -> str:
    """Executa um comando e devolve stderr (onde o ffmpeg escreve o log)."""
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise FFmpegError(f"{desc} falhou (código {proc.returncode}):\n{tail}")
    return proc.stderr


def probe(path: Path) -> dict:
    """Metadados do container via ffprobe."""
    out = subprocess.run(
        [
            _require("ffprobe"), "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise FFmpegError(f"ffprobe falhou em {path}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def duration(path: Path) -> float:
    """Duração em segundos."""
    return float(probe(path)["format"]["duration"])


def extract_audio(source: Path, dest: Path, *, rate: int = SAMPLE_RATE,
                  mono: bool = False) -> Path:
    """Extrai áudio PCM. Mantenha 48 kHz estéreo para a separação de fontes;
    só converta para 16 kHz mono na entrada do ASR."""
    run(
        [
            _require("ffmpeg"), "-y", "-i", str(source),
            "-vn",
            "-ac", "1" if mono else "2",
            "-ar", str(rate),
            "-c:a", "pcm_s16le",
            str(dest),
        ],
        desc=f"extração de áudio de {source.name}",
    )
    return dest


def measure_loudness(path: Path) -> dict[str, float]:
    """Primeira passada do loudnorm: mede para alimentar a segunda."""
    stderr = run(
        [
            _require("ffmpeg"), "-i", str(path),
            "-af",
            f"loudnorm=I={LUFS_TARGET}:TP={TRUE_PEAK_MAX}:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        desc=f"medição de loudness de {path.name}",
    )
    # o JSON é o último bloco entre chaves no stderr
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1:
        raise FFmpegError(f"loudnorm não devolveu JSON para {path}")
    raw = json.loads(stderr[start : end + 1])
    return {k: float(v) for k, v in raw.items() if _is_number(v)}


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def normalize_loudness(source: Path, dest: Path) -> Path:
    """Loudnorm em duas passadas, preservando a dinâmica."""
    m = measure_loudness(source)
    loudnorm = (
        f"loudnorm=I={LUFS_TARGET}:TP={TRUE_PEAK_MAX}:LRA=11"
        f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
        f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
        f":offset={m.get('target_offset', 0.0)}:linear=true:print_format=summary"
    )
    run(
        [
            _require("ffmpeg"), "-y", "-i", str(source),
            "-af", loudnorm,
            "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
            str(dest),
        ],
        desc=f"normalização de {source.name}",
    )
    return dest


def mix_with_ducking(dub: Path, bed: Path, dest: Path, *,
                     bed_gain: float = 0.8,
                     threshold: float = 0.03,
                     ratio: float = 8.0,
                     attack: float = 20.0,
                     release: float = 300.0) -> Path:
    """Mixa a dublagem sobre a trilha instrumental, abaixando o leito quando há
    fala (sidechain). A dublagem deve chegar aqui já normalizada, senão o
    compressor dispara de forma inconsistente entre segmentos."""
    filtergraph = (
        f"[0:a]aresample={SAMPLE_RATE}[voice_src];"
        f"[1:a]aresample={SAMPLE_RATE},volume={bed_gain}[bed];"
        "[voice_src]asplit=2[voice][key];"
        f"[bed][key]sidechaincompress=threshold={threshold}:ratio={ratio}"
        f":attack={attack}:release={release}[ducked];"
        "[voice][ducked]amix=inputs=2:duration=longest:normalize=0[out]"
    )
    run(
        [
            _require("ffmpeg"), "-y", "-i", str(dub), "-i", str(bed),
            "-filter_complex", filtergraph,
            "-map", "[out]",
            "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
            str(dest),
        ],
        desc="mixagem com ducking",
    )
    return dest


def mux(video: Path, dubbed_audio: Path, dest: Path, *,
        subtitles: list[tuple[Path, str, str]] | None = None,
        keep_original_audio: bool = True) -> Path:
    """Monta o arquivo final sem reencodar o vídeo.

    `subtitles` é uma lista de (arquivo, idioma ISO 639-2, título). A primeira
    entrada vira a legenda padrão do player.

    Ordem das faixas: dublagem primeiro (padrão), áudio original depois. Os
    códigos de idioma têm de ter três letras — muitos players ignoram os de
    duas e mostram a faixa como "desconhecido".
    """
    tracks = list(subtitles or [])

    args = [_require("ffmpeg"), "-y", "-i", str(video), "-i", str(dubbed_audio)]
    for path, _language, _title in tracks:
        args += ["-i", str(path)]

    args += ["-map", "0:v:0", "-map", "1:a:0"]
    if keep_original_audio:
        args += ["-map", "0:a:0"]
    for index in range(len(tracks)):
        args += ["-map", str(2 + index)]

    args += [
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE),
        "-metadata:s:a:0", "language=por",
        "-metadata:s:a:0", "title=Dublagem PT-BR",
        "-disposition:a:0", "default",
    ]
    if keep_original_audio:
        args += [
            "-metadata:s:a:1", "language=eng",
            "-metadata:s:a:1", "title=Audio original",
            "-disposition:a:1", "0",
        ]

    if tracks:
        # mov_text é o único codec de legenda que o MP4 aceita de forma
        # confiável; ele não carrega estilo, o que aqui não faz falta.
        args += ["-c:s", "mov_text"]
        for index, (_path, language, title) in enumerate(tracks):
            args += [
                f"-metadata:s:s:{index}", f"language={language}",
                f"-metadata:s:s:{index}", f"title={title}",
                f"-disposition:s:{index}", "default" if index == 0 else "0",
            ]

    args += ["-movflags", "+faststart", str(dest)]
    run(args, desc="mux final")
    return dest
