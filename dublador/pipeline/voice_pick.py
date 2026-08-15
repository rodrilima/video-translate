"""Escolha automática da voz de dublagem.

A voz certa depende de quem fala no vídeo, então o software decide em vez de
perguntar. Dois sinais entram, nesta ordem:

  1. a transcrição, que costuma trazer evidência direta — o nome de quem fala,
     como as outras pessoas se dirigem a ele;
  2. a frequência fundamental do áudio, apenas como desempate.

A ordem é essa por medição, não por preferência. A tentativa inicial usava só
F0, com as faixas de referência para adultos (masculina ~85-155 Hz, feminina
~165-255 Hz), e errou no material real: um locutor masculino falando de forma
enfática deu 198 Hz — dentro da faixa tida como feminina. Duas implementações
independentes, autocorrelação própria e pYIN, concordaram nesse valor, o que
descarta erro de algoritmo. Fala expressiva simplesmente desloca a frequência o
bastante para invalidar a classificação, e F0 sozinho não resolve.

Nenhum dos dois sinais é confiável o suficiente para nunca errar. Por isso a
escolha vem sempre acompanhada da evidência que a motivou, avisa quando os dois
sinais discordam, e pode ser sobreposta com --voice.

O mesmo resultado alimenta a concordância gramatical do revisor: em português
"sou empático" e "sou empática" dependem de quem fala.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import soundfile as sf

from ..config import JobPaths

ProgressFn = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:
    return None


# Limite entre as faixas típicas. Fica na fronteira das duas distribuições.
GENDER_SPLIT_HZ = 160.0

# Faixa de busca. Fora disso não é voz humana falada, é ruído ou harmônico.
F0_MIN_HZ = 70.0
F0_MAX_HZ = 320.0

FRAME_MS = 40.0
HOP_MS = 20.0
ANALYSIS_RATE = 16000

# Quadros com energia abaixo desta fração do pico são silêncio entre palavras.
VOICED_THRESHOLD = 0.15


def choose(paths: JobPaths, *, progress: ProgressFn = _noop) -> dict:
    """Escolhe a voz de dublagem combinando transcrição e áudio.

    A transcrição tem prioridade porque costuma trazer evidência direta — o
    nome de quem fala, como as outras pessoas se dirigem a ele. O F0 entra só
    como desempate, porque isoladamente ele não decide: medido neste projeto,
    um locutor masculino falando de forma enfática deu 198 Hz, bem dentro da
    faixa tida como feminina. Fala expressiva desloca a frequência o
    suficiente para invalidar a classificação.

    Nenhum dos dois sinais é confiável o bastante para não errar, então a
    escolha vem acompanhada da evidência e pode ser sobreposta com --voice.
    """
    from ..tts.base import Voice, list_voices
    from . import brief as brief_stage

    source = paths.vocals if paths.vocals.exists() else paths.audio
    if not source.exists():
        raise FileNotFoundError(f"sem áudio para analisar em {paths.root}")

    progress(0.2, "medindo a voz do locutor")
    median_f0, frames = estimate_f0(source)
    f0_gender = ("indefinido" if median_f0 <= 0
                 else "masculina" if median_f0 < GENDER_SPLIT_HZ
                 else "feminina")

    progress(0.6, "consultando a transcrição")
    speaker = (brief_stage.load(paths).get("locutor") or {})
    text_gender = str(speaker.get("genero", "indefinido")).lower()

    if text_gender in {"masculina", "feminina"}:
        gender = text_gender
        basis = "transcrição"
        evidence = speaker.get("evidencia", "")
    elif f0_gender != "indefinido":
        gender = f0_gender
        basis = "frequência da voz"
        evidence = f"F0 mediana {median_f0:.0f} Hz"
    else:
        gender = "masculina"
        basis = "padrão"
        evidence = "sem evidência utilizável"

    progress(0.85, "escolhendo a voz")
    voice = _voice_for(gender, list_voices(), Voice)

    result = {
        "voice": voice,
        "gender": gender,
        "basis": basis,
        "evidence": evidence,
        "median_f0": round(median_f0, 1),
        "f0_gender": f0_gender,
        "agrees": f0_gender == gender,
        "voiced_frames": frames,
    }
    result["speakers"] = _assign_per_speaker(paths, gender, median_f0)
    _stamp_segments(paths, result)
    (paths.root / "voice.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    progress(1.0, f"{voice} ({gender}, por {basis})")
    return result


# Diferença de F0 a partir da qual dois locutores da mesma gravação são tratados
# como gêneros diferentes.
SPEAKER_SPLIT_HZ = 35.0


def _assign_per_speaker(paths: JobPaths, main_gender: str,
                        main_f0: float) -> dict:
    """Atribui uma voz a cada locutor detectado.

    O locutor principal herda o gênero apurado pela transcrição, que é a
    evidência mais forte disponível. Os demais são decididos por COMPARAÇÃO com
    ele, não por limiar absoluto — dois locutores da mesma gravação partilham
    condições de captação e estilo, e a diferença entre eles carrega mais
    informação do que o valor de cada um isolado. Foi o limiar absoluto que
    falhou antes, ao classificar como feminino um homem falando de forma
    enfática a 198 Hz.
    """
    from ..tts.base import Voice, list_voices
    from . import diarize as diarize_stage

    data = diarize_stage.load(paths)
    turns = data.get("turns") or []
    if not turns:
        return {}

    speaking = data.get("speaking_time") or {}
    order = sorted(speaking, key=lambda s: -speaking.get(s, 0.0))
    if not order:
        return {}

    available = list_voices()
    main = order[0]
    profiles: dict[str, dict] = {}
    used: list[str] = []

    for speaker in order:
        f0 = _speaker_f0(paths, turns, speaker)

        if speaker == main:
            gender, basis = main_gender, "transcrição"
        elif f0 <= 0 or main_f0 <= 0:
            gender, basis = main_gender, "sem medição"
        elif abs(f0 - main_f0) < SPEAKER_SPLIT_HZ:
            gender, basis = main_gender, "voz parecida com a principal"
        else:
            gender = "feminina" if f0 > main_f0 else "masculina"
            basis = "voz diferente da principal"

        # Prefere uma voz ainda não usada, para os locutores não se
        # confundirem. Mas o gênero vem antes da distinção: dar voz feminina a
        # um homem é erro audível, dois homens com a mesma voz é só perda de
        # nuance. Quando o catálogo não tem voz livre do gênero certo, repete.
        free = [v for v in available if v not in used]
        voice = (_match_gender(gender, free, Voice)
                 or _match_gender(gender, available, Voice)
                 or (free or available)[0])
        used.append(voice)

        profiles[speaker] = {
            "voice": voice,
            "gender": gender,
            "basis": basis,
            "median_f0": round(f0, 1),
            "speaking_time": round(speaking.get(speaker, 0.0), 1),
        }

    return profiles


def _stamp_segments(paths: JobPaths, result: dict) -> None:
    """Grava em cada segmento a voz que vai dublá-lo.

    Fica no segmento, e não numa tabela à parte, porque as etapas seguintes já
    leem segments.json — e porque assim dá para corrigir uma atribuição errada
    editando o arquivo e re-renderizando só o TTS.
    """
    from ..model import load_segments, save_segments

    if not paths.segments.exists():
        return

    profiles = result.get("speakers") or {}
    segments = load_segments(paths.segments)
    for segment in segments:
        profile = profiles.get(segment.speaker or "")
        segment.voice = (profile or {}).get("voice") or result["voice"]
    save_segments(segments, paths.segments)


def _speaker_gender(result: dict, speaker: str | None) -> str:
    """Gênero de quem fala num segmento, para a concordância em português."""
    profile = (result.get("speakers") or {}).get(speaker or "")
    return (profile or {}).get("gender") or result.get("gender", "masculina")


def _speaker_f0(paths: JobPaths, turns: list[dict], speaker: str) -> float:
    """F0 mediana considerando apenas os trechos em que este locutor fala."""
    source = paths.vocals if paths.vocals.exists() else paths.audio
    windows = [(t["start"], t["end"]) for t in turns if t["speaker"] == speaker]
    if not windows:
        return 0.0
    return estimate_f0(source, windows=windows)[0]


def _match_gender(gender: str, available: list[str], voice_cls) -> str:
    """Primeira voz do gênero pedido, ou string vazia se não houver.

    Devolve vazio em vez de levantar, porque não achar é um resultado esperado
    quando o catálogo tem menos vozes de um gênero do que locutores.
    """
    for name in available:
        try:
            if voice_cls.load(name).meta.get("genero") == gender:
                return name
        except FileNotFoundError:
            continue
    return ""


def _voice_for(gender: str, available: list[str], voice_cls) -> str:
    """Primeira voz do catálogo com o gênero pedido."""
    for name in available:
        try:
            if voice_cls.load(name).meta.get("genero") == gender:
                return name
        except FileNotFoundError:
            continue
    if not available:
        raise RuntimeError("nenhuma voz no catálogo")
    return available[0]


def estimate_f0(path, windows: list[tuple[float, float]] | None = None
                ) -> tuple[float, int]:
    """F0 mediana da fala, por autocorrelação.

    A mediana, e não a média, porque estimativas de F0 erram para o dobro ou
    para a metade em alguns quadros; um punhado de outliers não pode mover a
    decisão.

    `windows` restringe a análise a intervalos de tempo, o que permite medir
    cada locutor separadamente a partir dos turnos da diarização.
    """
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != ANALYSIS_RATE:
        audio = _resample(audio, rate, ANALYSIS_RATE)
        rate = ANALYSIS_RATE

    frame = int(rate * FRAME_MS / 1000)
    hop = int(rate * HOP_MS / 1000)
    if len(audio) < frame:
        return 0.0, 0

    peak = float(np.abs(audio).max())
    if peak <= 0:
        return 0.0, 0

    min_lag = int(rate / F0_MAX_HZ)
    max_lag = int(rate / F0_MIN_HZ)

    allowed = None
    if windows:
        allowed = [(int(a * rate), int(b * rate)) for a, b in windows]

    estimates: list[float] = []
    for start in range(0, len(audio) - frame, hop):
        if allowed is not None and not any(
                low <= start < high for low, high in allowed):
            continue
        window = audio[start : start + frame]
        if float(np.abs(window).max()) < peak * VOICED_THRESHOLD:
            continue  # silêncio entre palavras

        window = window - window.mean()
        correlation = np.correlate(window, window, mode="full")[frame - 1:]
        if correlation[0] <= 0:
            continue

        segment = correlation[min_lag:max_lag]
        if segment.size == 0:
            continue

        lag = int(np.argmax(segment)) + min_lag
        # Pico fraco em relação à energia total indica quadro não periódico
        # (consoante fricativa, ruído): não é fala com altura definida.
        if correlation[lag] < correlation[0] * 0.3:
            continue

        lag = _correct_octave(correlation, lag, max_lag)
        estimates.append(rate / lag)

    if not estimates:
        return 0.0, 0
    return float(np.median(estimates)), len(estimates)


def _correct_octave(correlation: np.ndarray, lag: int, max_lag: int,
                    tolerance: float = 0.80) -> int:
    """Desfaz o erro de oitava da autocorrelação.

    A autocorrelação de uma voz tem picos no período e em todos os seus
    múltiplos, mas também casa forte com os harmônicos — cujo lag é uma fração
    do verdadeiro. Quando isso vence, a estimativa sai no dobro ou no triplo da
    frequência real: medido aqui, uma voz masculina de ~97 Hz era lida como
    195 Hz e classificada como feminina.

    A fundamental é o MAIOR lag entre os picos fortes, então procuramos picos
    em múltiplos do lag encontrado e ficamos com o mais longo que ainda tenha
    correlação comparável.
    """
    best = lag
    peak = correlation[lag]

    for multiple in (2, 3, 4):
        candidate = lag * multiple
        if candidate >= max_lag or candidate >= len(correlation):
            break
        # o pico verdadeiro pode estar levemente deslocado do múltiplo exato
        low = max(0, candidate - multiple)
        high = min(len(correlation), candidate + multiple + 1)
        local = correlation[low:high]
        if local.size == 0:
            continue
        if float(local.max()) >= peak * tolerance:
            best = low + int(np.argmax(local))

    return best


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    duration = len(audio) / source_rate
    target_length = int(duration * target_rate)
    positions = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
