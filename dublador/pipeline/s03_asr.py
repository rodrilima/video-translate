"""Etapa 3: transcrição com timestamps de palavra.

Usa Parakeet TDT, cujo decoder prevê durações diretamente — os timestamps saem
do próprio modelo, não de DTW sobre cross-attention como no Whisper. Como toda
a isocronia é construída sobre esses tempos, a precisão aqui se propaga para o
resultado final.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..config import ASR_SAMPLE_RATE, JobPaths
from ..model import Word
from ..utils import ffmpeg

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


# Segmentação de sentenças no nível do decoder: pausa longa fecha a sentença, e
# um teto de palavras evita legendas quilométricas em fala corrida.
SILENCE_GAP = 0.40
MAX_WORDS = 24
MAX_SENTENCE_DURATION = 12.0

# Trechos longos são processados em janelas para limitar o pico de memória.
CHUNK_DURATION = 300.0
OVERLAP_DURATION = 15.0


def transcribe(paths: JobPaths, *, model_id: str,
               progress: ProgressFn = _noop) -> dict:
    """Transcreve o áudio e devolve sentenças com palavras cronometradas."""
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.parakeet import DecodingConfig, SentenceConfig

    source = paths.vocals if paths.vocals.exists() else paths.audio
    if not source.exists():
        raise FileNotFoundError(
            f"nenhum áudio para transcrever em {paths.root}; rode a etapa de download antes"
        )

    # O Parakeet espera 16 kHz mono; o áudio de trabalho é 48 kHz estéreo.
    asr_input = paths.root / "asr_input.wav"
    progress(0.02, "preparando áudio")
    ffmpeg.extract_audio(source, asr_input, rate=ASR_SAMPLE_RATE, mono=True)

    progress(0.08, f"carregando {model_id}")
    model = from_pretrained(model_id)

    total = ffmpeg.duration(asr_input)

    def on_chunk(_info, current) -> None:
        if total:
            progress(0.10 + 0.85 * min(current / total, 1.0), "transcrevendo")

    progress(0.10, "transcrevendo")
    result = model.transcribe(
        asr_input,
        decoding_config=DecodingConfig(
            sentence=SentenceConfig(
                max_words=MAX_WORDS,
                silence_gap=SILENCE_GAP,
                max_duration=MAX_SENTENCE_DURATION,
            )
        ),
        chunk_duration=CHUNK_DURATION if total > CHUNK_DURATION else None,
        overlap_duration=OVERLAP_DURATION,
        chunk_callback=on_chunk,
    )

    sentences = [
        {
            "text": s.text.strip(),
            "start": float(s.start),
            "end": float(s.end),
            "words": [w.__dict__ for w in _tokens_to_words(s.tokens)],
        }
        for s in result.sentences
        if s.text.strip()
    ]

    payload = {
        "model": model_id,
        "language": "en",
        "duration": total,
        "text": result.text.strip(),
        "sentences": sentences,
    }
    paths.asr.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    asr_input.unlink(missing_ok=True)

    n_words = sum(len(s["words"]) for s in sentences)
    progress(1.0, f"{len(sentences)} sentenças, {n_words} palavras")
    return payload


def _tokens_to_words(tokens: list) -> list[Word]:
    """Agrupa tokens subword em palavras.

    O tokenizer do Parakeet é SentencePiece: o marcador `▁` (U+2581) indica
    início de palavra. Sem esse agrupamento os timestamps seriam de fragmentos
    como "▁transcri" + "ption", inúteis para medir pausas.

    Tokens numéricos são um caso à parte: costumam vir sem o marcador e, se
    apenas concatenados, produzem "produce400" e "first50" — texto corrompido
    que segue para o tradutor. Uma troca entre letra e dígito também abre
    palavra.
    """
    words: list[Word] = []
    for token in tokens:
        raw = token.text
        starts_word = raw.startswith("▁") or raw.startswith(" ")
        piece = raw.replace("▁", " ").strip()
        if not piece:
            continue

        if not starts_word and words:
            starts_word = _switches_alnum_class(words[-1].w, piece)

        if starts_word or not words:
            words.append(Word(w=piece, s=float(token.start), e=float(token.end)))
        else:
            words[-1].w += piece
            words[-1].e = float(token.end)
    return words


def _switches_alnum_class(previous: str, piece: str) -> bool:
    """True quando a fronteira troca de letra para dígito ou vice-versa."""
    if not previous or not piece:
        return False
    before, after = previous[-1], piece[0]
    return (before.isalpha() and after.isdigit()) or (
        before.isdigit() and after.isalpha()
    )


def load_asr(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
