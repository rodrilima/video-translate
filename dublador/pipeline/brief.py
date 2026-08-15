"""Briefing do vídeo: contexto global para etapas que trabalham por segmento.

Tradutor e revisor enxergam poucas falas de cada lado. Isso resolve coesão
local, mas não resolve referência: um termo estabelecido no início do vídeo
volta trinta segundos depois já sem contexto à vista, e vira literalidade.
Foi assim que "to handle pun intended" — trocadilho com os "handles" de rede
social citados no começo — virou uma frase sobre piadas.

Uma única leitura da transcrição inteira produz assunto, registro e glossário,
que passam a acompanhar todos os segmentos. Custo: uma chamada de LLM por
vídeo.
"""

from __future__ import annotations

import json
from typing import Callable

from ..config import JobPaths
from ..llm.client import LocalLLM
from ..model import load_segments

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


SYSTEM = """Você prepara material de apoio para dublagem em português brasileiro.

Leia a transcrição e devolva o que um dublador precisaria saber ANTES de
começar: sobre o que é, em que tom se fala, e como traduzir os termos que se
repetem.

Identifique também quem fala. O gênero do locutor decide a voz da dublagem e a
concordância dos adjetivos em português ("sou empático" ou "sou empática"),
então vale reparar em nomes próprios, em como as pessoas se dirigem a quem
fala, e em como ele se descreve. Se a transcrição não deixar claro, diga
"indefinido" — chutar é pior que admitir.

No glossário, inclua apenas o que realmente reaparece ou o que erraria se
traduzido ao pé da letra: jargão da área, gírias, nomes de produtos e marcas,
trocadilhos. Nomes próprios de pessoas geralmente ficam como estão. Para cada
termo, dê a forma que soaria natural na fala brasileira — que às vezes é
manter o termo em inglês, quando é assim que se fala no Brasil.

Responda APENAS com JSON."""


USER = """Transcrição:
{transcript}

Responda:
{{"assunto": "<duas frases sobre o que é o vídeo>",
  "locutor": {{"genero": "masculina | feminina | indefinido",
              "evidencia": "<o que na transcrição indica isso>"}},
  "registro": "<como se fala: formal, coloquial, técnico, agressivo, etc.>",
  "glossario": {{"<termo em inglês>": "<como dizer em português>"}},
  "cuidados": ["<armadilha de tradução específica deste vídeo>"]}}"""

# Transcrições longas são truncadas: o começo e o fim carregam quase toda a
# informação de assunto, e o meio raramente muda o briefing.
MAX_CHARS = 12000


def build(paths: JobPaths, *, model_id: str,
          progress: ProgressFn = _noop) -> dict:
    """Escreve brief.json e o glossário inicial."""
    segments = load_segments(paths.segments)
    transcript = _transcript(segments)

    llm = LocalLLM(model_id)
    progress(0.05, f"carregando {model_id}")
    llm.load()

    try:
        progress(0.30, "lendo a transcrição inteira")
        payload = llm.chat_json(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER.format(transcript=transcript)}],
            max_tokens=1024, temperature=0.2,
        )
    finally:
        llm.unload()

    locutor = payload.get("locutor") or {}
    if not isinstance(locutor, dict):
        locutor = {}

    brief = {
        "assunto": str(payload.get("assunto", "")).strip(),
        "locutor": {
            "genero": str(locutor.get("genero", "indefinido")).strip().lower(),
            "evidencia": str(locutor.get("evidencia", "")).strip(),
        },
        "registro": str(payload.get("registro", "")).strip(),
        "glossario": {str(k): str(v)
                      for k, v in (payload.get("glossario") or {}).items()},
        "cuidados": [str(c) for c in (payload.get("cuidados") or [])],
    }

    (paths.root / "brief.json").write_text(
        json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.glossary.write_text(
        json.dumps(brief["glossario"], ensure_ascii=False, indent=2),
        encoding="utf-8")

    progress(1.0, f"{len(brief['glossario'])} termos no glossário")
    return brief


def load(paths: JobPaths) -> dict:
    path = paths.root / "brief.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_prompt_block(brief: dict) -> str:
    """Formata o briefing para entrar no prompt das etapas por segmento."""
    if not brief:
        return ""

    lines = []
    if brief.get("assunto"):
        lines.append(f"Sobre o vídeo: {brief['assunto']}")
    if brief.get("registro"):
        lines.append(f"Registro da fala: {brief['registro']}")
    if brief.get("cuidados"):
        lines.append("Armadilhas de tradução neste vídeo:")
        lines += [f"  - {c}" for c in brief["cuidados"][:5]]
    return "\n".join(lines)


def _transcript(segments: list) -> str:
    text = " ".join(s.text_en.strip() for s in segments if s.text_en.strip())
    if len(text) <= MAX_CHARS:
        return text
    half = MAX_CHARS // 2
    return f"{text[:half]}\n[...]\n{text[-half:]}"
