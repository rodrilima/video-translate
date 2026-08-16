"""Etapa final: resumo do conteúdo do vídeo, em português.

Serve para quem quer saber do que o vídeo trata sem assistir, e para arquivar
junto com o arquivo dublado. É um resumo de verdade — as ideias reescritas em
português — e não um recorte da transcrição.
"""

from __future__ import annotations

import json
from typing import Callable

from ..config import JobPaths
from ..llm.client import LocalLLM
from ..model import load_segments
from . import brief as brief_stage

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


SYSTEM = """Você resume vídeos em português brasileiro.

Escreva com suas próprias palavras, explicando as ideias — não recorte nem
traduza frases da transcrição. O leitor quer entender do que se trata e o que
foi defendido, sem assistir.

Seja concreto: se o vídeo dá números, exemplos ou recomendações, eles são o que
importa. Evite abertura genérica ("neste vídeo, o autor discute...") e vá
direto ao conteúdo.

Em "pontos", liste de três a seis afirmações distintas — cada uma um argumento,
dado ou recomendação diferente do vídeo. Um único ponto genérico não serve; se
o vídeo é curto, prefira pontos mais específicos a pontos mais amplos.

Responda APENAS com JSON."""


USER = """Transcrição de um vídeo em inglês, para você resumir em português.

{transcript}

Responda:
{{"titulo": "<título curto e específico, em português>",
  "resumo": "<dois a quatro parágrafos explicando o conteúdo>",
  "pontos": ["<afirmação, dado ou recomendação>", "<outra, distinta da anterior>", "..."],
  "para_quem": "<a quem este vídeo interessa, em uma frase>"}}"""

MAX_CHARS = 14000


def summarize(paths: JobPaths, *, model_id: str,
              progress: ProgressFn = _noop) -> dict:
    """Escreve resumo.txt."""
    segments = load_segments(paths.segments)
    transcript = _transcript(segments)

    llm = LocalLLM(model_id)
    progress(0.05, f"carregando {model_id}")
    llm.load()

    try:
        progress(0.30, "resumindo o conteúdo")
        payload = llm.chat_json(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER.format(transcript=transcript)}],
            max_tokens=1600, temperature=0.3,
        )
    finally:
        llm.unload()

    summary = {
        "titulo": str(payload.get("titulo", "")).strip(),
        "resumo": str(payload.get("resumo", "")).strip(),
        "pontos": [str(p).strip() for p in (payload.get("pontos") or [])],
        "para_quem": str(payload.get("para_quem", "")).strip(),
    }

    _write_text(paths, summary)

    progress(1.0, "resumo.txt pronto")
    return summary


def _write_text(paths: JobPaths, summary: dict) -> None:
    """Grava o resumo em texto puro, legível em qualquer lugar."""
    source = _source_meta(paths)
    brief = brief_stage.load(paths)

    linhas: list[str] = []
    titulo = summary["titulo"] or source.get("title", "Resumo")
    linhas.append(titulo)
    linhas.append("=" * len(titulo))
    linhas.append("")

    if summary.get("para_quem"):
        linhas.append(summary["para_quem"])
        linhas.append("")

    if source:
        duracao = source.get("audio_duration") or source.get("duration") or 0
        minutos, segundos = divmod(int(duracao), 60)
        if source.get("title"):
            linhas.append(f"Titulo original: {source['title']}")
        if source.get("uploader"):
            linhas.append(f"Canal: {source['uploader']}")
        linhas.append(f"Duracao: {minutos}:{segundos:02d}")
        if source.get("url"):
            linhas.append(f"Origem: {source['url']}")
        linhas.append("")

    linhas.append("RESUMO")
    linhas.append("")
    linhas += _wrap(summary["resumo"])
    linhas.append("")

    if summary["pontos"]:
        linhas.append("PONTOS PRINCIPAIS")
        linhas.append("")
        for ponto in summary["pontos"]:
            envolvido = _wrap(ponto, largura=74)
            linhas.append(f"- {envolvido[0]}")
            linhas += [f"  {linha}" for linha in envolvido[1:]]
        linhas.append("")

    glossario = brief.get("glossario") or {}
    if glossario:
        linhas.append("TERMOS")
        linhas.append("")
        largura = max(len(k) for k in glossario)
        linhas += [f"  {k.ljust(largura)}  {v}" for k, v in glossario.items()]
        linhas.append("")

    linhas.append("-" * 76)
    linhas.append("Resumo gerado automaticamente a partir da transcricao do video,")
    linhas.append("junto com a dublagem em portugues.")
    linhas.append("")

    (paths.root / "resumo.txt").write_text("\n".join(linhas), encoding="utf-8")


def _wrap(texto: str, largura: int = 76) -> list[str]:
    """Quebra em linhas de largura fixa, preservando os paragrafos."""
    import textwrap

    saida: list[str] = []
    for paragrafo in texto.split("\n"):
        if not paragrafo.strip():
            saida.append("")
            continue
        saida += textwrap.wrap(paragrafo.strip(), width=largura)
    return saida or [""]


def _source_meta(paths: JobPaths) -> dict:
    if not paths.meta.exists():
        return {}
    return json.loads(paths.meta.read_text(encoding="utf-8")).get("source", {})


def _transcript(segments: list) -> str:
    text = " ".join(s.text_en.strip() for s in segments if s.text_en.strip())
    if len(text) <= MAX_CHARS:
        return text
    half = MAX_CHARS // 2
    return f"{text[:half]}\n[...]\n{text[-half:]}"
