"""Mede se pedir um alvo mais apertado na primeira passada reduz reescritas.

Medido num vídeo real: 233 segmentos consumiram 404 rodadas de LLM. As 171
rodadas extras são 42% do trabalho da revisão, que por sua vez é 68% do
pipeline — ou seja, quase 30% do tempo total é retrabalho.

A hipótese é que o modelo tende a entregar perto do limite pedido. Pedindo
menos, a sobra natural cairia dentro da tolerância e a segunda rodada deixaria
de ser necessária.

O risco é comprimir demais e perder informação, então o teste mede as duas
coisas: quantos cabem de primeira e quão curto o texto fica.

Uso: uv run python scripts/teste_alvo_apertado.py
"""

from __future__ import annotations

import time

from dublador.config import OVERFLOW_TOLERANCE, JobPaths
from dublador.llm.client import LocalLLM, _extract_json
from dublador.model import load_segments
from dublador.pipeline import brief as B
from dublador.pipeline.s06_review import _build_messages, _clean
from dublador.utils.syllables import count_syllables

JOB = "ZTSI3DDP_4A"
MODEL = "mlx-community/Qwen3-14B-4bit"
AMOSTRA = 60

# Frações do orçamento real pedidas no prompt da primeira passada.
FATORES = [1.00, 0.90, 0.80]


def main() -> None:
    paths = JobPaths(JOB)
    todos = load_segments(paths.segments)
    segmentos = [s for s in todos if s.text_pt_raw][:AMOSTRA]
    bloco = B.as_prompt_block(B.load(paths))

    llm = LocalLLM(MODEL)
    llm.load()

    print(f"{len(segmentos)} segmentos, tolerância de {OVERFLOW_TOLERANCE:.0%}")
    print(f"{'alvo pedido':>12} {'tempo':>7} {'cabem':>10} {'sílabas':>9} "
          f"{'vs orçamento':>13}")
    print("-" * 60)

    for fator in FATORES:
        originais = [s.syllable_budget for s in segmentos]
        for s in segmentos:
            s.syllable_budget = max(1, round(s.syllable_budget * fator))

        conversas = [_build_messages(s, todos, {}, "masculina", retry=False,
                                     brief_block=bloco)
                     for s in segmentos]

        inicio = time.time()
        respostas = llm.chat_batch(conversas, max_tokens=384)
        decorrido = time.time() - inicio

        # restaura o orçamento real: é contra ele que o resultado é julgado
        for s, orig in zip(segmentos, originais):
            s.syllable_budget = orig

        cabem = 0
        razoes = []
        for segmento, bruto in zip(segmentos, respostas):
            try:
                texto = _clean(str(_extract_json(bruto).get("text", "")))
            except ValueError:
                continue
            if not texto:
                continue
            silabas = count_syllables(texto)
            razao = silabas / max(segmento.syllable_budget, 1)
            razoes.append(razao)
            if razao <= 1 + OVERFLOW_TOLERANCE:
                cabem += 1

        media = sum(razoes) / len(razoes) if razoes else 0
        print(f"{fator:11.0%} {decorrido:6.1f}s {cabem:4d}/{len(segmentos)} "
              f"{100 * cabem / len(segmentos):5.0f}% {media:12.2f}x")

    print()
    print("'vs orçamento' abaixo de 1.0 significa texto mais curto que o slot:")
    print("cabe com folga, mas pode ter perdido conteúdo.")


if __name__ == "__main__":
    main()
