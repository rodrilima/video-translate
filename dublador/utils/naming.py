"""Nomes de pasta legíveis para os jobs.

Uma pasta chamada `ZTSI3DDP_4A` não diz nada; achar um trabalho antigo exige
abrir o job.json de cada uma. O nome passa a ser o título do vídeo seguido do
identificador — legível de relance, e ainda assim único quando dois vídeos têm
o mesmo título.
"""

from __future__ import annotations

import re
import unicodedata

# Nomes muito longos atrapalham no terminal e esbarram no limite de caminho de
# alguns sistemas de arquivos quando somados aos artefatos internos.
MAX_SLUG = 60


def slugify(text: str) -> str:
    """Reduz um título a letras, números e hifens.

    Remove acentos em vez de mantê-los: nomes de pasta acentuados complicam
    completar no terminal e variam de normalização entre sistemas.
    """
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_only).strip("-").lower()

    if len(slug) <= MAX_SLUG:
        return slug

    # corta na fronteira de palavra, para não terminar no meio de uma
    cut = slug[:MAX_SLUG].rsplit("-", 1)[0]
    return cut or slug[:MAX_SLUG]


def job_folder(title: str | None, video_id: str) -> str:
    """Nome da pasta de um job novo."""
    slug = slugify(title or "")
    return f"{slug}-{video_id}" if slug else video_id
