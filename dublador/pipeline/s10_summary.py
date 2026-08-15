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
    """Escreve resumo.md e resumo.json."""
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

    (paths.root / "resumo.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(paths, summary)

    progress(1.0, "resumo.md pronto")
    return summary


def _write_markdown(paths: JobPaths, summary: dict) -> None:
    source = _source_meta(paths)
    brief = brief_stage.load(paths)

    lines: list[str] = []
    lines.append(f"# {summary['titulo'] or source.get('title', 'Resumo')}")
    lines.append("")

    if summary.get("para_quem"):
        lines.append(f"*{summary['para_quem']}*")
        lines.append("")

    if source:
        duration = source.get("audio_duration") or source.get("duration") or 0
        minutes, seconds = divmod(int(duration), 60)
        lines.append("| | |")
        lines.append("|---|---|")
        if source.get("title"):
            lines.append(f"| Título original | {source['title']} |")
        if source.get("uploader"):
            lines.append(f"| Canal | {source['uploader']} |")
        lines.append(f"| Duração | {minutes}:{seconds:02d} |")
        if source.get("url"):
            lines.append(f"| Origem | {source['url']} |")
        lines.append("")

    lines.append("## Resumo")
    lines.append("")
    lines.append(summary["resumo"])
    lines.append("")

    if summary["pontos"]:
        lines.append("## Pontos principais")
        lines.append("")
        lines += [f"- {p}" for p in summary["pontos"]]
        lines.append("")

    glossary = brief.get("glossario") or {}
    if glossary:
        lines.append("## Termos")
        lines.append("")
        lines.append("| Original | Em português |")
        lines.append("|---|---|")
        lines += [f"| {k} | {v} |" for k, v in glossary.items()]
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Resumo gerado automaticamente a partir da transcrição do "
                 "vídeo, junto com a dublagem em português.")
    lines.append("")

    (paths.root / "resumo.md").write_text("\n".join(lines), encoding="utf-8")


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
