"""Etapa 6: adaptar a tradução para fala natural que caiba no tempo.

É a etapa que substitui o editor humano. Medido no material de teste, um terço
dos segmentos precisa encolher mais de 25% para caber, e o motor de TTS degrada
audivelmente abaixo de ~0,9x da duração natural — logo, o encurtamento tem de
acontecer aqui, no texto, e não no áudio.

Trabalha em laço: reescreve, mede, e se ainda estourar reescreve de novo com a
folga real informada. Desiste depois de N tentativas e marca o segmento, em vez
de entregar algo que vai atropelar a cena seguinte sem aviso.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from ..config import COMPRESSION_FLOOR, OVERFLOW_TOLERANCE, JobPaths
from ..llm.client import LocalLLM, _extract_json
from ..llm.prompts import review as prompts
from ..model import Segment, load_segments, save_segments
from ..utils.syllables import count_syllables
from . import brief

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


# Quantos segmentos vizinhos entram no prompt como contexto.
CONTEXT_WINDOW = 2

# Quantas falas são adaptadas por vez. Casa com o tamanho de lote seguro do
# gerador, então não custa chamadas extras: é o mesmo número de requisições,
# só que montadas mais tarde, quando as anteriores já responderam.
CONTEXT_CHUNK = 4


def review(paths: JobPaths, *, model_id: str, attempts: int = 3,
           speaker_gender: str = "masculina",
           progress: ProgressFn = _noop) -> list[Segment]:
    """Preenche `text_pt_final`, `qa_flags` e `overflow_pct`."""
    segments = load_segments(paths.segments)
    glossary = _load_glossary(paths)
    brief_block = brief.as_prompt_block(brief.load(paths))
    genders = _speaker_genders(paths, speaker_gender)
    llm = LocalLLM(model_id)

    progress(0.02, f"carregando {model_id}")
    llm.load()

    try:
        # Rodadas em lote: a primeira processa tudo, as seguintes só quem ainda
        # não coube. Submeter os segmentos juntos mantém a GPU ocupada — um
        # prompt por vez a deixava em ~21% de uso.
        pending = list(segments)
        for round_index in range(max(1, attempts)):
            if not pending:
                break

            label = "revisando" if round_index == 0 else "encurtando"
            # A rodada 1 cobre quase tudo; as seguintes reprocessam poucos
            # segmentos. Dividir a barra pelo número de tentativas fazia a
            # rodada 1 completa marcar 35% e depois saltar para 100%.
            base = 0.05 if round_index == 0 else 0.95
            span = 0.90 if round_index == 0 else 0.04

            # Em blocos, e não tudo de uma vez: os prompts de um bloco são
            # montados depois de o anterior ter respondido, então cada fala vê
            # os vizinhos já adaptados em vez da tradução literal. Montar tudo
            # de antemão era mais rápido, mas nenhuma fala enxergava a versão
            # final da anterior — e é disso que vinha a sensação de frases
            # soltas, sem conectivo e sem continuidade de assunto.
            for start in range(0, len(pending), CONTEXT_CHUNK):
                chunk = pending[start : start + CONTEXT_CHUNK]
                conversations = [
                    _build_messages(segment, segments, glossary,
                                    genders.get(segment.speaker or "",
                                                speaker_gender),
                                    retry=round_index > 0,
                                    brief_block=brief_block)
                    for segment in chunk
                ]
                responses = llm.chat_batch(conversations, max_tokens=384)
                for segment, raw in zip(chunk, responses):
                    _apply_response(segment, raw)

                feitos = start + len(chunk)
                progress(base + span * feitos / max(len(pending), 1),
                         f"{label} {feitos}/{len(pending)}")

            pending = [s for s in pending
                       if s.overflow_pct > OVERFLOW_TOLERANCE
                       and "json_invalido" not in s.qa_flags]
            save_segments(segments, paths.segments)

        for segment in segments:
            _finalize(segment, glossary)
    finally:
        llm.unload()

    save_segments(segments, paths.segments)
    _save_glossary(paths, glossary)

    flagged = sum(1 for s in segments if s.qa_flags)
    progress(1.0, f"{len(segments)} revisados, {flagged} com ressalva")
    return segments


def _speaker_genders(paths: JobPaths, fallback: str) -> dict[str, str]:
    """Gênero de cada locutor, para a concordância sair certa por pessoa.

    Num vídeo com duas pessoas de gêneros diferentes, um único valor para todo
    o vídeo erraria os adjetivos da metade das falas.
    """
    import json

    path = paths.root / "voice.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {speaker: profile.get("gender", fallback)
            for speaker, profile in (data.get("speakers") or {}).items()}


def _build_messages(segment: Segment, all_segments: list[Segment],
                    glossary: dict[str, str], speaker_gender: str,
                    *, retry: bool, brief_block: str = "") -> list[dict[str, str]]:
    """Prompt de uma rodada. Na reescrita, informa a folga real que faltou."""
    source = (segment.text_pt_raw or "").strip()
    budget = segment.syllable_budget

    base = [
        {"role": "system", "content": prompts.SYSTEM},
        {"role": "user", "content": prompts.USER_TEMPLATE.format(
            brief=brief_block,
            id=segment.id,
            speaker=f"voz {speaker_gender} (concorde adjetivos com esse gênero)",
            context_en=_context_en(all_segments, segment.id) or "(sem contexto)",
            before=_context(all_segments, segment.id, -CONTEXT_WINDOW) or "(início)",
            after=_context(all_segments, segment.id, CONTEXT_WINDOW) or "(fim)",
            glossary=prompts.build_glossary_block(glossary),
            text_en=segment.text_en,
            text_pt=source,
            budget=budget,
            current=count_syllables(source),
            pauses=_describe_pauses(segment) or "nenhuma",
        )},
    ]
    if not retry:
        return base

    previous = segment.text_pt_final or source
    syllables = count_syllables(previous)
    return base + [
        {"role": "assistant", "content": json.dumps(
            {"id": segment.id, "text": previous}, ensure_ascii=False)},
        {"role": "user", "content": prompts.RETRY_TEMPLATE.format(
            id=segment.id, current=syllables, budget=budget,
            excess=round((syllables / max(budget, 1) - 1) * 100),
            previous=previous)},
    ]


def _apply_response(segment: Segment, raw: str) -> None:
    """Aceita a nova versão apenas se ela for melhor que a que já existe.

    Uma rodada de encurtamento pode devolver algo mais longo; nesse caso o
    resultado anterior é preservado.
    """
    source = (segment.text_pt_raw or "").strip()
    if not source:
        segment.text_pt_final = ""
        _flag(segment, "sem_traducao")
        return

    segment.attempts += 1
    budget = max(segment.syllable_budget, 1)

    try:
        payload = _extract_json(raw)
    except ValueError:
        _flag(segment, "json_invalido")
        _settle(segment, segment.text_pt_final or source, budget)
        return

    candidate = _clean(str(payload.get("text", "")).strip())
    if not candidate:
        _flag(segment, "resposta_vazia")
        _settle(segment, segment.text_pt_final or source, budget)
        return

    current = segment.text_pt_final or source
    best = candidate if count_syllables(candidate) < count_syllables(current) \
        else current

    note = str(payload.get("cut", "")).strip()
    if note and best == candidate:
        segment.cut_note = note[:80]

    _settle(segment, best, budget)


def _settle(segment: Segment, text: str, budget: int) -> None:
    segment.text_pt_final = text
    segment.overflow_pct = round(max(0.0, count_syllables(text) / budget - 1), 4)


def _finalize(segment: Segment, glossary: dict[str, str]) -> None:
    """Aplica as verificações de qualidade depois da última rodada."""
    if not segment.text_pt_final:
        return
    _flag_quality(segment, count_syllables(segment.text_pt_final),
                  max(segment.syllable_budget, 1),
                  segment.cut_note)
    _update_glossary(glossary, segment)


def _flag(segment: Segment, flag: str) -> None:
    if flag not in segment.qa_flags:
        segment.qa_flags.append(flag)


def _flag_quality(segment: Segment, syllables: int, budget: int,
                  cut_note: str) -> None:
    """Marca o que o operador precisa saber sem ouvir o vídeo inteiro."""
    if syllables > budget * (1 + OVERFLOW_TOLERANCE):
        _flag(segment, "estouro_de_tempo")

    ratio = syllables / max(count_syllables(segment.text_pt_raw or ""), 1)
    if ratio < COMPRESSION_FLOOR:
        # Abaixo desse piso a compressão deixa de ser concisão e vira perda.
        _flag(segment, "compressao_extrema")

    if _has_english_leftover(segment.text_pt_final or ""):
        _flag(segment, "ingles_residual")

    if _lost_numbers(segment.text_en, segment.text_pt_final or ""):
        _flag(segment, "numero_perdido")

    if cut_note.strip():
        _flag(segment, f"cortado:{cut_note.strip()[:60]}")


_NUMBER_RE = re.compile(r"\d[\d.,]*")
# Palavras funcionais inglesas que não existem em português: sinal barato de
# tradução incompleta, sem falso positivo com estrangeirismos já incorporados.
_ENGLISH_MARKERS = {
    "the", "and", "but", "with", "that", "this", "your", "you", "have",
    "what", "because", "about", "would", "should", "there", "their",
}


def _has_english_leftover(text: str) -> bool:
    words = {w.lower() for w in re.findall(r"[A-Za-z']+", text)}
    return bool(words & _ENGLISH_MARKERS)


def _lost_numbers(source_en: str, final_pt: str) -> bool:
    """Números somem com facilidade quando o modelo encurta agressivamente."""
    def normalize(text: str) -> set[str]:
        return {n.replace(".", "").replace(",", "")
                for n in _NUMBER_RE.findall(text)}

    return bool(normalize(source_en) - normalize(final_pt))


def _clean(text: str) -> str:
    """Remove aspas e rótulos que o modelo às vezes anexa."""
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(fala adaptada|texto|adaptação)\s*:\s*", "", text,
                  flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _context(segments: list[Segment], center: int, offset: int) -> str:
    """Texto dos vizinhos, para o modelo não repetir nem contradizer."""
    if offset < 0:
        window = segments[max(0, center + offset):center]
    else:
        window = segments[center + 1 : center + 1 + offset]
    return " ".join((s.text_pt_final or s.text_pt_raw or "") for s in window).strip()


def _context_en(segments: list[Segment], center: int) -> str:
    """Falas vizinhas no original.

    O revisor precisa do inglês ao redor, não só do português: o erro que ele
    tem de pegar nasceu justamente de o tradutor não ter visto esse contexto,
    então ler apenas a saída já traduzida herdaria o mesmo engano.
    """
    window = segments[max(0, center - CONTEXT_WINDOW):
                      center + CONTEXT_WINDOW + 1]
    return " ".join(s.text_en.strip() for s in window).strip()


def _describe_pauses(segment: Segment) -> str:
    if not segment.pauses:
        return ""
    return ", ".join(f"{offset:.1f}s (dura {length:.1f}s)"
                     for offset, length in segment.pauses)


def _update_glossary(glossary: dict[str, str], segment: Segment) -> None:
    """Fixa nomes próprios do inglês na forma que o revisor escolheu.

    Mantém consistência ao longo do vídeo: o mesmo nome não pode aparecer
    grafado de duas formas em falas diferentes.
    """
    final = segment.text_pt_final or ""
    for name in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", segment.text_en):
        if name.lower() in _ENGLISH_MARKERS or name in glossary:
            continue
        if name in final:
            glossary[name] = name


def _load_glossary(paths: JobPaths) -> dict[str, str]:
    if paths.glossary.exists():
        return json.loads(paths.glossary.read_text(encoding="utf-8"))
    return {}


def _save_glossary(paths: JobPaths, glossary: dict[str, str]) -> None:
    paths.glossary.write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
