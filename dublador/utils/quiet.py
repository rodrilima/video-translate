"""Mantém o painel do terminal limpo durante a execução.

O painel de progresso assume o controle da saída. Bibliotecas que escrevem por
fora dele — logging, barras de tqdm, downloads do Hugging Face — quebram o
layout e, pior, produzem tracebacks alarmantes de "I/O operation on closed
file": o handler de logging tenta escrever num fluxo que o painel substituiu, e
o erro ao emitir um aviso trivial parece uma falha grave.

Nada é descartado: tudo vai para um arquivo de log no diretório do job, que
continua disponível para diagnóstico.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path

# Bibliotecas que falam demais em nível INFO durante o pipeline.
NOISY = (
    "mdxc_separator", "common_separator", "core", "separator",
    "huggingface_hub", "transformers", "urllib3", "httpx", "filelock",
    "numba", "matplotlib", "speechbrain", "senko",
)


@contextlib.contextmanager
def quiet_output(log_path: Path):
    """Desvia a saída de erro e o logging das bibliotecas para um arquivo.

    Só a saída de erro é desviada. O painel escreve na saída padrão, que
    permanece intacta, e as exceções continuam sendo tratadas e exibidas pelo
    orquestrador — o desvio não engole falha nenhuma.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)

    original_stderr = sys.stderr
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_levels = {name: logging.getLogger(name).level for name in NOISY}

    file_handler = logging.StreamHandler(handle)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    original_root_level = root.level

    try:
        sys.stderr = handle
        root.handlers = [file_handler]
        # O root nasce em WARNING, o que descartaria a telemetria do pipeline
        # antes de ela chegar ao arquivo. As bibliotecas barulhentas continuam
        # contidas pelos níveis individuais logo abaixo.
        root.setLevel(logging.INFO)
        for name in NOISY:
            logging.getLogger(name).setLevel(logging.WARNING)
        _disable_progress_bars()
        yield log_path
    finally:
        sys.stderr = original_stderr
        root.handlers = original_handlers
        root.setLevel(original_root_level)
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)
        handle.close()


def _disable_progress_bars() -> None:
    """Desliga as barras que escrevem direto no terminal.

    As variáveis de ambiente cobrem bibliotecas que só as consultam na
    importação; a chamada ao huggingface_hub cobre o que já foi importado.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    with contextlib.suppress(Exception):
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
