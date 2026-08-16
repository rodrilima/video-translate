"""Etapa 5: tradução base do inglês para o português.

Usa um modelo especialista em tradução (linhagem Hunyuan-MT), não um LLM
generalista. A diferença aparece em fidelidade terminológica e em não inventar
conteúdo — que é o erro mais caro aqui, porque uma alucinação na tradução vira
uma frase falada com convicção no vídeo final.

Esta etapa não tenta resolver isocronia nem registro coloquial; isso é trabalho
do revisor (etapa 6), que tem um modelo generalista melhor para reescrever sob
restrição.
"""

from __future__ import annotations

from typing import Callable

from ..config import JobPaths
from ..llm.client import LocalLLM
from ..model import Segment, load_segments, save_segments
from . import brief

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


# Formato de prompt da família Hunyuan-MT para pares que não envolvem chinês.
PROMPT = ("Translate the following segment into Portuguese (Brazil), "
          "without additional explanation.\n\n{text}")

# Mesma coisa, mas com as falas vizinhas à vista. Traduzir frase isolada produz
# erro de referência: neste material, "to handle pun intended" virou "lidar com
# as piadas" porque o modelo não sabia que o assunto era "handles" de rede
# social, dito três segmentos antes. O contexto não é traduzido, só orienta.
CONTEXT_PROMPT = (
    "{brief}"
    "Context (do not translate, for reference only):\n{context}\n\n"
    "Translate ONLY the segment below into Portuguese (Brazil), "
    "without additional explanation.\n\n{text}"
)

# Quantas falas vizinhas entram como contexto de cada lado.
CONTEXT_WINDOW = 3


def translate(paths: JobPaths, *, model_id: str,
              progress: ProgressFn = _noop) -> list[Segment]:
    """Preenche `text_pt_raw` em cada segmento."""
    segments = load_segments(paths.segments)
    brief_block = brief.as_prompt_block(brief.load(paths))
    llm = LocalLLM(model_id)

    progress(0.02, f"carregando {model_id}")
    llm.load()

    try:
        # Os segmentos são independentes, então vão todos juntos: uma chamada
        # por segmento deixa a GPU ociosa entre tokens.
        pending = [s for s in segments if s.text_en.strip()]
        for segment in segments:
            if not segment.text_en.strip():
                segment.text_pt_raw = ""

        progress(0.08, f"traduzindo {len(pending)} segmentos")

        def relatar(feitos: int, total: int) -> None:
            progress(0.08 + 0.90 * feitos / max(total, 1),
                     f"traduzindo {feitos}/{total}")

        conversations = [
            [{"role": "user",
              "content": _prompt_for(segment, segments, brief_block)}]
            for segment in pending
        ]
        max_tokens = max((_token_budget(s.text_en) for s in pending), default=96)
        responses = llm.chat_batch(conversations, max_tokens=max_tokens,
                                   temperature=0.0, on_progress=relatar)

        for segment, text in zip(pending, responses):
            segment.text_pt_raw = text.strip()
    finally:
        llm.unload()

    save_segments(segments, paths.segments)
    progress(1.0, f"{len(segments)} segmentos traduzidos")
    return segments


def _prompt_for(segment: Segment, segments: list[Segment],
                brief_block: str = "") -> str:
    """Monta o prompt do segmento com briefing e falas vizinhas."""
    index = segments.index(segment)
    before = segments[max(0, index - CONTEXT_WINDOW):index]
    after = segments[index + 1 : index + 1 + CONTEXT_WINDOW]
    context = " ".join(s.text_en.strip() for s in before + after).strip()

    if not context and not brief_block:
        return PROMPT.format(text=segment.text_en.strip())
    header = f"{brief_block}\n\n" if brief_block else ""
    return CONTEXT_PROMPT.format(brief=header, context=context,
                                 text=segment.text_en.strip())


def _token_budget(source: str) -> int:
    """Teto generoso: o português costuma render mais tokens que o inglês, e
    cortar a geração no meio produz frase truncada, que é pior que frase longa."""
    return max(96, int(len(source.split()) * 6) + 48)
