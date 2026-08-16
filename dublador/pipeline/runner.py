"""Orquestrador: encadeia as etapas e informa o progresso de cada uma.

Cada etapa declara o artefato que produz, o que dá retomada de graça: rodar de
novo pula o que já existe em disco. Numa falha no TTS, o vídeo não é rebaixado
nem a transcrição refeita.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import Preset, JobPaths

ProgressFn = Callable[[float, str], None]
StageFn = Callable[..., Any]


@dataclass
class Stage:
    key: str
    label: str
    run: StageFn
    artifact: Callable[[JobPaths], Path]
    # Etapas opcionais só entram quando o contexto pede — a clonagem de voz é
    # uma escolha por vídeo, não parte do caminho padrão.
    enabled: Callable[[dict], bool] = lambda ctx: True

    def is_done(self, paths: JobPaths) -> bool:
        path = self.artifact(paths)
        return path.exists() and path.stat().st_size > 0


@dataclass
class StageState:
    stage: Stage
    status: str = "pendente"     # pendente, rodando, pronto, pulado, erro
    percent: float = 0.0
    message: str = ""
    seconds: float = 0.0
    error: str = ""


@dataclass
class RunState:
    """Estado observável do pipeline, para a interface desenhar."""

    job_id: str
    url: str
    voice: str
    preset: str
    stages: list[StageState] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    chosen: dict = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def finished(self) -> bool:
        return all(s.status in {"pronto", "pulado", "erro"} for s in self.stages)


def build_stages() -> list[Stage]:
    """A ordem importa: cada etapa consome o artefato da anterior."""
    from . import (brief, s01_download, s02_separate, s03_asr, s04_segment,
                   clone_voices, diarize, s05_translate, s06_review, s07_tts,
                   s08_fit, s09_render, s10_summary, voice_pick)

    return [
        Stage("download", "baixando vídeo",
              lambda ctx: s01_download.download(
                  ctx["url"], ctx["paths"], progress=ctx["progress"],
                  cookies_from_browser=ctx.get("cookies")),
              lambda p: p.audio),
        Stage("separate", "separando voz e trilha",
              lambda ctx: s02_separate.separate(
                  ctx["paths"], model=ctx["preset"].separator_model,
                  progress=ctx["progress"]),
              lambda p: p.instrumental),
        Stage("diarize", "identificando locutores",
              lambda ctx: diarize.diarize(
                  ctx["paths"], progress=ctx["progress"]),
              lambda p: p.root / "diarization.json"),
        Stage("asr", "transcrevendo",
              lambda ctx: s03_asr.transcribe(
                  ctx["paths"], model_id=ctx["preset"].asr_model,
                  progress=ctx["progress"]),
              lambda p: p.asr),
        Stage("segment", "segmentando",
              lambda ctx: s04_segment.build_segments(
                  ctx["paths"], progress=ctx["progress"]),
              lambda p: p.segments),
        Stage("brief", "lendo o vídeo inteiro",
              lambda ctx: brief.build(
                  ctx["paths"], model_id=ctx["preset"].reviewer_model,
                  progress=ctx["progress"]),
              lambda p: p.root / "brief.json"),
        Stage("voice", "escolhendo a voz",
              lambda ctx: ctx["shared"].update(
                  voice_pick.choose(ctx["paths"], progress=ctx["progress"])),
              lambda p: p.root / ".done_voice"),
        Stage("clones", "extraindo a voz de cada locutor",
              lambda ctx: clone_voices.extract(
                  ctx["paths"], progress=ctx["progress"]),
              lambda p: p.root / "clones.json",
              enabled=lambda ctx: bool(ctx.get("clone"))),
        Stage("translate", "traduzindo",
              lambda ctx: s05_translate.translate(
                  ctx["paths"], model_id=ctx["preset"].translator_model,
                  progress=ctx["progress"]),
              lambda p: p.root / ".done_translate"),
        Stage("review", "adaptando para fala natural",
              lambda ctx: s06_review.review(
                  ctx["paths"], model_id=ctx["preset"].reviewer_model,
                  attempts=max(1, ctx["preset"].fit_attempts),
                  speaker_gender=ctx["shared"].get("gender", ctx["gender"]),
                  progress=ctx["progress"]),
              lambda p: p.root / ".done_review"),
        Stage("tts", "sintetizando a voz",
              lambda ctx: s07_tts.synthesize(
                  ctx["paths"], backend_name=ctx["preset"].tts_backend,
                  voice_name=ctx["voice"] or ctx["shared"].get("voice", "alex"),
                  clone=bool(ctx.get("clone")), progress=ctx["progress"]),
              lambda p: p.root / ".done_tts"),
        Stage("fit", "encaixando na linha do tempo",
              lambda ctx: s08_fit.assemble(
                  ctx["paths"], progress=ctx["progress"]),
              lambda p: p.dub),
        Stage("render", "mixando e montando",
              lambda ctx: s09_render.render(
                  ctx["paths"], progress=ctx["progress"]),
              lambda p: p.output),
        Stage("summary", "resumindo o conteúdo",
              lambda ctx: s10_summary.summarize(
                  ctx["paths"], model_id=ctx["preset"].reviewer_model,
                  progress=ctx["progress"]),
              lambda p: p.root / "resumo.txt"),
    ]


def run(state: RunState, paths: JobPaths, preset: Preset, *,
        url: str, voice: str | None = None, gender: str = "masculina",
        cookies: str | None = None, force_from: str | None = None,
        clone: bool = False,
        on_change: Callable[[], None] = lambda: None) -> RunState:
    """Executa as etapas, atualizando `state` para a interface acompanhar."""
    paths.ensure()
    # Compartilhado entre etapas: a escolha de voz é feita no meio do pipeline
    # e precisa alcançar a revisão (concordância) e o TTS (timbre).
    shared: dict[str, Any] = {}
    # Vira True ao alcançar a etapa pedida em --refazer; daí em diante nada é
    # pulado. Sem --refazer permanece False, e tudo que já existe é aproveitado.
    forcing = False

    context = {"clone": clone}

    for entry in state.stages:
        stage = entry.stage
        if not stage.enabled(context):
            entry.status = "pulado"
            entry.message = "não solicitado"
            on_change()
            continue
        if force_from is not None and stage.key == force_from:
            forcing = True

        if not forcing and stage.is_done(paths):
            if stage.key == "voice":
                shared.update(_load_voice_choice(paths))
            entry.status = "pulado"
            entry.percent = 1.0
            entry.message = "já existia"
            on_change()
            continue

        def progress(percent: float, message: str, _entry=entry) -> None:
            _entry.percent = max(0.0, min(1.0, percent))
            _entry.message = message
            on_change()

        entry.status = "rodando"
        entry.message = ""
        on_change()
        started = time.time()

        try:
            stage.run({
                "paths": paths, "preset": preset, "url": url, "voice": voice,
                "gender": gender, "cookies": cookies, "progress": progress,
                "shared": shared, "clone": clone,
            })
        except Exception as exc:  # noqa: BLE001 - a falha é reportada, não engolida
            entry.status = "erro"
            entry.error = f"{type(exc).__name__}: {exc}"
            entry.seconds = time.time() - started
            on_change()
            raise

        # marcador para as etapas que alteram segments.json em vez de criar
        # arquivo próprio, permitindo retomada
        marker = stage.artifact(paths)
        if marker.name.startswith(".done_"):
            marker.write_text(str(time.time()), encoding="utf-8")

        entry.status = "pronto"
        entry.percent = 1.0
        entry.seconds = time.time() - started
        on_change()

    state.chosen = dict(shared)
    return state


def _load_voice_choice(paths: JobPaths) -> dict:
    """Relê a escolha de voz gravada, para retomadas não perderem o timbre."""
    import json

    path = paths.root / "voice.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
