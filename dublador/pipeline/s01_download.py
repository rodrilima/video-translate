"""Etapa 1: baixar o vídeo e extrair o áudio de trabalho.

Guarda o áudio em 48 kHz estéreo porque a separação de fontes precisa da banda
completa; a redução para 16 kHz mono acontece só na entrada do ASR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..config import SAMPLE_RATE, JobPaths
from ..utils import ffmpeg

ProgressFn = Callable[[float, str], None]

# O YouTube exige PO token no servidor de vídeo, o que devolve HTTP 403 na mídia
# mesmo com os metadados perfeitamente acessíveis. Na prática, medido neste
# projeto: os streams de áudio passam sem token em alguns clientes, mas todos os
# streams de vídeo (progressivos e adaptativos) são barrados. Cookies do
# navegador resolvem, então tentamos primeiro sem eles e caímos para cookies.
PLAYER_CLIENTS = ["android_vr", "tv_simply", "default"]
COOKIE_BROWSERS = ["chrome", "safari", "firefox", "brave", "edge"]


def _attempts(cookies_from_browser: str | None
              ) -> list[tuple[str, str | None]]:
    """Combinações de (cliente, navegador de cookies) em ordem de preferência.

    Sem cookies primeiro, porque ler cookies pode disparar prompt de keychain no
    macOS e não é necessário para todo vídeo.
    """
    if cookies_from_browser:
        return [(client, cookies_from_browser) for client in PLAYER_CLIENTS]
    plan: list[tuple[str, str | None]] = [(c, None) for c in PLAYER_CLIENTS]
    plan += [(c, b) for b in COOKIE_BROWSERS for c in ("android_vr", "default")]
    return plan


def _noop(pct: float, msg: str) -> None:
    return None


def download(url: str, paths: JobPaths, *, progress: ProgressFn = _noop,
             cookies_from_browser: str | None = None) -> dict:
    """Baixa o melhor vídeo+áudio disponível e extrai a trilha para WAV.

    Prefere áudio Opus (o YouTube serve ~160 kbps) e deixa o vídeo em MKV, que
    lida melhor com múltiplas faixas do que MP4.
    """
    import yt_dlp

    paths.ensure()
    progress(0.0, "consultando metadados")

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes", 0)
            if total:
                progress(0.05 + 0.75 * done / total, "baixando vídeo")
        elif d.get("status") == "finished":
            progress(0.80, "download concluído, remuxando")

    base_options: dict = {
        "format": "bv*+ba/b",
        "format_sort": ["res:1080", "fps", "hdr:sdr", "acodec:opus", "br"],
        "merge_output_format": "mkv",
        "outtmpl": str(paths.root / "video.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en.*"],
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 5,
        "concurrent_fragment_downloads": 4,
    }
    info = None
    strategy: tuple[str, str | None] | None = None
    failures: list[str] = []

    for client, browser in _attempts(cookies_from_browser):
        options = dict(base_options)
        if client != "default":
            options["extractor_args"] = {"youtube": {"player_client": [client]}}
        if browser:
            options["cookiesfrombrowser"] = (browser,)

        label = client + (f" + cookies:{browser}" if browser else "")
        try:
            progress(0.03, f"tentando {label}")
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
            strategy = (client, browser)
            break
        except Exception as exc:  # noqa: BLE001 - yt-dlp levanta vários tipos
            failures.append(f"{label}: {str(exc).splitlines()[-1][:110]}")
            for leftover in paths.root.glob("video.*"):
                leftover.unlink(missing_ok=True)

    if info is None:
        raise RuntimeError(
            "nenhuma combinação de cliente e cookies conseguiu baixar a mídia:\n  "
            + "\n  ".join(failures)
            + "\n\nÚltimo recurso: subir o provider de PO token com"
            " `docker run -d -p 4416:4416 brainicism/bgutil-ytdlp-pot-provider`."
        )

    if not paths.video.exists():
        produced = sorted(paths.root.glob("video.*"))
        candidates = [p for p in produced if p.suffix in {".mkv", ".mp4", ".webm"}]
        if not candidates:
            raise RuntimeError(
                f"yt-dlp não produziu arquivo de vídeo em {paths.root}"
            )
        candidates[0].rename(paths.video)

    progress(0.85, "extraindo áudio")
    ffmpeg.extract_audio(paths.video, paths.audio, rate=SAMPLE_RATE, mono=False)

    meta = {
        "url": url,
        "title": info.get("title"),
        "id": info.get("id"),
        "uploader": info.get("uploader"),
        "duration": info.get("duration"),
        "audio_duration": ffmpeg.duration(paths.audio),
        "download_strategy": {"player_client": strategy[0],
                              "cookies_from": strategy[1]},
    }
    _merge_meta(paths.meta, {"source": meta})
    progress(1.0, f"pronto: {meta['title']}")
    return meta


def _merge_meta(path: Path, update: dict) -> None:
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(update)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2),
                    encoding="utf-8")
