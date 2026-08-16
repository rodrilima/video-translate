"""Etapa 4: transformar a transcrição em unidades de dublagem.

A saída do ASR não serve direto: o decoder fecha sentenças por pausa e por teto
de palavras, o que produz tanto fragmentos de duas palavras quanto blocos
cortados no meio de uma frase. Aqui isso vira segmentos com tamanho utilizável,
cada um com o orçamento de tempo que a tradução terá de respeitar.

Duas medidas importam para o resultado final:
  - `gap_before`, o silêncio livre antes do segmento. É espaço que a dublagem
    pode ocupar sem atropelar nada, e é a folga mais barata de usar.
  - `pauses`, os silêncios internos. Preservá-los é o que faz a dublagem
    respirar junto com o vídeo em vez de soar como narração corrida.
"""

from __future__ import annotations

from typing import Callable

from ..config import JobPaths
from ..model import Segment, Word, save_segments
from ..utils.syllables import syllable_budget
from . import diarize

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


# Um segmento curto demais não dá contexto ao tradutor e produz fala picotada.
MIN_DURATION = 1.2
# Longo demais acumula erro de sincronia e estoura o contexto do revisor.
MAX_DURATION = 12.0
# Pausa a partir da qual vale a pena cortar ou marcar respiração.
PAUSE_THRESHOLD = 0.30
# Silêncio máximo que um segmento pode "tomar emprestado" do que vem antes.
MAX_BORROWED_GAP = 0.50


def build_segments(paths: JobPaths, *, progress: ProgressFn = _noop) -> list[Segment]:
    """Lê asr.json e escreve segments.json."""
    from .s03_asr import load_asr

    progress(0.05, "lendo transcrição")
    asr = load_asr(paths.asr)

    progress(0.2, "lendo locutores")
    turns = diarize.load(paths).get("turns", [])

    progress(0.3, "segmentando")
    units = _pipeline(_from_asr(asr), turns)

    progress(0.9, "gravando segmentos")
    save_segments(units, paths.segments)
    progress(1.0, f"{len(units)} segmentos")
    return units


def _from_asr(asr: dict) -> list[list[Word]]:
    """Cada sentença do ASR vira uma lista de palavras."""
    groups: list[list[Word]] = []
    for sentence in asr["sentences"]:
        words = [Word(**w) if isinstance(w, dict) else w for w in sentence["words"]]
        if words:
            groups.append(words)
    return groups


def _split_long(groups: list[list[Word]]) -> list[list[Word]]:
    """Quebra grupos longos na maior pausa interna, recursivamente."""
    out: list[list[Word]] = []
    for words in groups:
        out.extend(_split_recursive(words))
    return out


def _split_recursive(words: list[Word], depth: int = 0) -> list[list[Word]]:
    span = words[-1].e - words[0].s
    if span <= MAX_DURATION or len(words) < 6 or depth > 6:
        return [words]

    # corta na maior pausa dentro da faixa central, para não gerar cacos
    best_index, best_gap = None, 0.0
    low, high = len(words) // 5, len(words) * 4 // 5
    for i in range(max(low, 1), max(high, 2)):
        gap = words[i].s - words[i - 1].e
        if gap > best_gap:
            best_index, best_gap = i, gap

    if best_index is None or best_gap < PAUSE_THRESHOLD / 2:
        best_index = len(words) // 2  # sem pausa boa: corta no meio mesmo

    return (_split_recursive(words[:best_index], depth + 1)
            + _split_recursive(words[best_index:], depth + 1))


def _merge_short(groups: list[list[Word]],
                 turns: list[dict] | None = None) -> list[list[Word]]:
    """Junta segmentos curtos ao vizinho, desde que o resultado não estoure e
    que não haja uma pausa longa separando os dois (pausa longa é fronteira de
    fala real, provavelmente troca de locutor)."""
    if not groups:
        return []

    merged: list[list[Word]] = [groups[0]]
    for words in groups[1:]:
        previous = merged[-1]
        duration = words[-1].e - words[0].s
        combined = words[-1].e - previous[0].s
        gap = words[0].s - previous[-1].e

        too_short = duration < MIN_DURATION or len(words) <= 2
        fits = combined <= MAX_DURATION
        continuous = gap < PAUSE_THRESHOLD * 2
        same_speaker = _same_speaker(previous, words, turns)

        if too_short and fits and continuous and same_speaker:
            merged[-1] = previous + words
        else:
            merged.append(words)
    return merged


def _same_speaker(previous: list[Word], words: list[Word],
                  turns: list[dict] | None) -> bool:
    """Impede que a fusão de segmentos curtos misture duas pessoas."""
    if not turns:
        return True
    a = diarize.speaker_at(turns, previous[0].s, previous[-1].e)
    b = diarize.speaker_at(turns, words[0].s, words[-1].e)
    return a == b


def _annotate_timing(groups: list[list[Word]]) -> list[Segment]:
    """Converte grupos de palavras em segmentos com folga e pausas medidas."""
    segments: list[Segment] = []
    previous_end = 0.0

    for index, words in enumerate(groups):
        start, end = words[0].s, words[-1].e
        gap = max(0.0, start - previous_end)

        pauses = [
            (round(words[i - 1].e - start, 3), round(words[i].s - words[i - 1].e, 3))
            for i in range(1, len(words))
            if words[i].s - words[i - 1].e >= PAUSE_THRESHOLD
        ]

        segment = Segment(
            id=index,
            start=round(start, 3),
            end=round(end, 3),
            text_en=" ".join(w.w for w in words),
            words=words,
            gap_before=round(min(gap, MAX_BORROWED_GAP), 3),
            pauses=pauses,
        )
        segment.syllable_budget = syllable_budget(segment.target_duration)
        segments.append(segment)
        previous_end = end

    return segments


# _annotate_timing devolve segmentos, mas build_segments precisa deles na ordem
# certa; mantidos separados para poder testar cada regra isoladamente.
def _pipeline(groups: list[list[Word]],
              turns: list[dict] | None = None) -> list[Segment]:
    """Ordem: separa locutores, depois quebra o que ficou longo, depois junta o
    que ficou curto. A troca de locutor vem primeiro porque é a única fronteira
    que nenhuma etapa posterior pode desfazer sem estragar a dublagem."""
    turns = turns or []
    groups = _split_by_speaker(groups, turns)
    groups = _merge_fragments(groups, turns)
    units = _annotate_timing(_merge_short(_split_long(groups), turns))
    _assign_speakers(units, turns)
    return units


# Pontuação que fecha uma ideia. Sem ela no fim de um trecho, a frase continua.
SENTENCE_END = ".!?…"
# Pausa longa demais para ser respiração no meio de uma frase.
FRAGMENT_GAP = 0.80


def _merge_fragments(groups: list[list[Word]],
                     turns: list[dict]) -> list[list[Word]]:
    """Junta trechos que são a mesma frase partida ao meio.

    O ASR fecha sentença por pausa, não por gramática: quem respira no meio de
    uma oração gera dois pedaços. Medido no material de teste, 61% dos
    segmentos eram fragmentos, incluindo um com a palavra "the" sozinha — que
    foi traduzida isolada, como "o".

    Traduzir meia oração é o que faz a dublagem soar solta: sem saber onde o
    trecho se encaixa, o modelo não tem como escolher a regência nem a ordem
    das palavras, que em português diferem do inglês.
    """
    if not groups:
        return []

    merged: list[list[Word]] = [groups[0]]
    for words in groups[1:]:
        previous = merged[-1]
        gap = words[0].s - previous[-1].e
        combined = words[-1].e - previous[0].s

        if (_continues_sentence(previous, words)
                and combined <= MAX_DURATION
                and gap <= FRAGMENT_GAP
                and _same_speaker(previous, words, turns)):
            merged[-1] = previous + words
        else:
            merged.append(words)
    return merged


def _continues_sentence(previous: list[Word], words: list[Word]) -> bool:
    """True quando o corte entre os dois trechos cai no meio de uma frase."""
    tail = previous[-1].w.rstrip()
    head = words[0].w.lstrip()
    if not tail or not head:
        return True

    unfinished = tail[-1] not in SENTENCE_END
    lowercase_start = head[0].islower()
    return unfinished or lowercase_start


def _split_by_speaker(groups: list[list[Word]],
                      turns: list[dict]) -> list[list[Word]]:
    """Corta onde muda quem fala.

    Sem isso uma pergunta e a resposta dela caem no mesmo segmento e recebem a
    mesma voz — foi o caso medido no material de teste, onde um segmento
    atravessava a troca de locutor aos 9,7s.
    """
    if not turns:
        return groups

    out: list[list[Word]] = []
    for words in groups:
        current: list[Word] = []
        current_speaker = None

        for word in words:
            speaker = diarize.speaker_at(turns, word.s, word.e)
            if current and speaker != current_speaker:
                out.append(current)
                current = []
            current.append(word)
            current_speaker = speaker

        if current:
            out.append(current)
    return out


def _assign_speakers(segments: list[Segment], turns: list[dict]) -> None:
    """Marca cada segmento com quem fala nele."""
    for segment in segments:
        segment.speaker = diarize.speaker_at(turns, segment.start, segment.end)
