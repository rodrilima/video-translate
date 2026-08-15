"""Mede o efeito do batch_size na separação de fontes.

O padrão da biblioteca é batch_size=1, que subutiliza a GPU. Batch maior é
puro paralelismo: a saída deve ser numericamente equivalente, e é isso que o
script confere — velocidade sem verificação de equivalência não vale nada,
porque um ganho que degrada o stem instrumental estraga a mixagem.

Uso: uv run python scripts/bench_separacao.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from dublador.config import JobPaths

JOB = "iq5VjE31Eig"
MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
BATCH_SIZES = [1, 4, 8]


def run(batch_size: int, outdir: Path) -> tuple[float, Path | None]:
    from mlx_audio_separator import Separator

    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    separator = Separator(
        output_dir=str(outdir),
        output_format="WAV",
        mdxc_params={
            "segment_size": 256,
            "override_model_segment_size": False,
            "batch_size": batch_size,
            "overlap": 8,
            "pitch_shift": 0,
        },
    )
    separator.load_model(model_filename=MODEL)

    start = time.time()
    outputs = separator.separate(str(JobPaths(JOB).audio))
    elapsed = time.time() - start

    produced = [outdir / Path(name).name for name in outputs]
    instrumental = next(
        (p for p in produced if "instrument" in p.name.lower()), None
    )
    return elapsed, instrumental


def load_mono(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32")
    return audio.mean(axis=1) if audio.ndim > 1 else audio


def main() -> None:
    base = Path("/tmp/bench_sep")
    reference: np.ndarray | None = None

    print(f"{'batch':>6} {'tempo':>8} {'ganho':>7}  equivalência ao batch=1")
    print("-" * 58)

    baseline = None
    for batch_size in BATCH_SIZES:
        elapsed, instrumental = run(batch_size, base / str(batch_size))
        if instrumental is None:
            print(f"{batch_size:6d} {'-':>8}  não produziu stem instrumental")
            continue

        audio = load_mono(instrumental)
        if reference is None:
            reference, baseline = audio, elapsed
            note = "(referência)"
        else:
            n = min(len(reference), len(audio))
            diff = float(np.abs(reference[:n] - audio[:n]).max())
            rms = float(np.sqrt(((reference[:n] - audio[:n]) ** 2).mean()))
            note = f"dif max {diff:.2e}  rms {rms:.2e}"

        speedup = baseline / elapsed if baseline else 1.0
        print(f"{batch_size:6d} {elapsed:7.1f}s {speedup:6.2f}x  {note}")

    shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
