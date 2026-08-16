"""Descobre se a CPU ociosa pode fazer trabalho útil sem atrasar a GPU.

Em Apple Silicon a memória é unificada: CPU e GPU disputam a mesma banda, que é
justamente o recurso saturado do pipeline — a geração de tokens roda a 92% do
teto de banda. Já medimos que trabalho de CPU pesado em memória deixa a GPU
1,81x mais lenta.

Falta o outro extremo: trabalho cujo conjunto de dados cabe no cache do
processador e quase não toca a memória principal. Se ele não atrapalhar, existe
espaço para rodar verificações de qualidade em paralelo, de graça.

Usa processos e não threads: trabalho de CPU em Python é limitado pelo GIL, e
threads não exercitariam os núcleos.

Uso: uv run python scripts/teste_cpu_livre.py
"""

from __future__ import annotations

import multiprocessing as mp
import time

JOB = "iq5VjE31Eig"
MODEL = "mlx-community/Qwen3-14B-4bit"


def carga_cache_residente(parar) -> None:
    """Aritmética sobre ~32 KB: cabe no cache L1/L2, quase não vai à memória."""
    import numpy as np

    dados = np.random.rand(4096).astype(np.float32)
    while not parar.is_set():
        for _ in range(500):
            dados = dados * np.float32(1.000001) + np.float32(1e-7)


def carga_texto(parar) -> None:
    """Processamento de texto, o tipo de verificação que faria sentido rodar
    junto: regex e contagem sobre strings curtas."""
    import re

    padrao = re.compile(r"[A-Za-zÀ-ÿ]+")
    frase = ("Uma frase de teste com pontuação, números 400 e nomes próprios. "
             * 6)
    while not parar.is_set():
        for _ in range(2000):
            padrao.findall(frase)
            frase.split()


def carga_banda(parar) -> None:
    """Referência do outro extremo: varre 32 MB, que só existe na memória."""
    import numpy as np

    dados = np.random.rand(4_000_000)
    while not parar.is_set():
        dados.sum()
        dados * 1.0001


def medir(llm, convs, alvo, n_processos: int) -> float:
    parar = mp.Event()
    processos = [mp.Process(target=alvo, args=(parar,), daemon=True)
                 for _ in range(n_processos)] if alvo else []
    for p in processos:
        p.start()
    time.sleep(0.5)  # deixa a carga estabilizar antes de cronometrar

    inicio = time.time()
    llm.chat_batch(convs, max_tokens=384)
    decorrido = time.time() - inicio

    parar.set()
    for p in processos:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()
    return decorrido


def main() -> None:
    from dublador.config import JobPaths
    from dublador.llm.client import LocalLLM
    from dublador.model import load_segments
    from dublador.pipeline import brief as B
    from dublador.pipeline.s06_review import _build_messages

    paths = JobPaths(JOB)
    segmentos = load_segments(paths.segments)
    bloco = B.as_prompt_block(B.load(paths))
    convs = [_build_messages(s, segmentos, {}, "masculina", retry=False,
                             brief_block=bloco)
             for s in segmentos]

    llm = LocalLLM(MODEL)
    llm.load()
    llm.chat_batch(convs[:4], max_tokens=384)  # aquece

    base = medir(llm, convs, None, 0)
    print(f"{'carga concorrente':32s} {'tempo':>8} {'impacto':>9}")
    print("-" * 52)
    print(f"{'nenhuma (referência)':32s} {base:7.1f}s {'—':>9}")

    for rotulo, alvo, n in (
        ("cache-residente, 4 processos", carga_cache_residente, 4),
        ("cache-residente, 8 processos", carga_cache_residente, 8),
        ("texto e regex, 4 processos", carga_texto, 4),
        ("varredura de memória, 4 proc.", carga_banda, 4),
    ):
        tempo = medir(llm, convs, alvo, n)
        print(f"{rotulo:32s} {tempo:7.1f}s {tempo / base:8.2f}x")

    print()
    print(f"núcleos disponíveis: {mp.cpu_count()}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
