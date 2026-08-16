"""Interface de linha de comando.

Cada etapa roda isolada sobre os artefatos já em disco, o que torna possível
depurar a tradução sem rebaixar o vídeo ou reajustar a mixagem sem ressintetizar
tudo. A UI web usa exatamente as mesmas funções.
"""

from __future__ import annotations

import hashlib
import re
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from typer.core import TyperGroup

from .config import DEFAULT_PRESET, PRESETS, JobPaths, set_jobs_dir
from .utils.quiet import quiet_output


class _UrlFirstGroup(TyperGroup):
    """Permite `dublador <url>` sem escrever `run`.

    Um argumento posicional no callback não serve: ele consome o nome do
    subcomando, e `dublador resumo <job>` passa a ler "resumo" como URL. Aqui a
    decisão acontece no roteamento — se o primeiro token não é um comando
    conhecido nem uma opção, ele só pode ser uma URL, e `run` é injetado antes.
    Assim os subcomandos continuam intactos e as opções podem vir depois do link.
    """

    # Só decidem pelo grupo; qualquer outra coisa pertence ao `run`.
    _GROUP_FLAGS = {"--help", "-h", "--install-completion",
                    "--show-completion"}

    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands \
                and args[0] not in self._GROUP_FLAGS:
            args = ["run", *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=_UrlFirstGroup,
    add_completion=False,
    help="Dublagem automática local de vídeos, inglês para português brasileiro."
         " Passe só o link do vídeo; os subcomandos servem para depurar etapas.",
)
console = Console()


# Um ID de vídeo do YouTube: 11 caracteres, sem nada que o shell interprete.
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def normalize_url(entrada: str) -> str:
    """Aceita URL completa, URL curta ou apenas o ID do vídeo.

    O ID puro existe para dispensar as aspas: uma URL do YouTube tem `?` e `&`,
    que o zsh trata como glob e como operador de background — o shell falha
    antes de o programa receber o argumento, então isso não tem como ser
    resolvido aqui dentro. Passar só o ID contorna o problema na origem.
    """
    entrada = entrada.strip().strip('"').strip("'")
    if _BARE_ID_RE.match(entrada):
        return f"https://www.youtube.com/watch?v={entrada}"
    return entrada


def job_id_for(url: str) -> str:
    """Identificador estável por URL: rodar de novo retoma o mesmo job."""
    video_id = _youtube_id(url)
    if video_id:
        return video_id
    return hashlib.sha256(url.encode()).hexdigest()[:11]


def _youtube_id(url: str) -> str | None:
    if _BARE_ID_RE.match(url.strip()):
        return url.strip()
    patterns = [
        r"(?:v=|/shorts/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _progress_bar(label: str):
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.fields[msg]}"),
        TimeElapsedColumn(),
        console=console,
    )


@app.command()
def run(
    url: Annotated[str, typer.Argument(
        help="URL do vídeo, ou só o ID (dispensa aspas)")],
    voice: Annotated[str | None, typer.Option(
        help="Força uma voz do catálogo; por padrão o dublador escolhe")] = None,
    preset: Annotated[str, typer.Option(help="draft, balanced ou max")] = DEFAULT_PRESET,
    cookies: Annotated[str | None, typer.Option(
        help="Navegador de onde ler cookies, se o YouTube bloquear")] = None,
    refazer: Annotated[str | None, typer.Option(
        help="Refaz a partir desta etapa (ex: review)")] = None,
    clonar: Annotated[bool, typer.Option(
        "--clonar",
        help="Usa a voz de cada locutor do próprio vídeo, em vez do catálogo")] = False,
    em: Annotated[str | None, typer.Option(
        help="Onde criar a pasta do vídeo. Padrão: o diretório atual")] = None,
) -> None:
    """Dubla um vídeo do começo ao fim, acompanhando pelo terminal."""
    from rich.live import Live

    from .pipeline import runner
    from .tts.base import Voice

    set_jobs_dir(em)
    url = normalize_url(url)
    # A pasta nasce com o identificador e é renomeada para o título assim que o
    # download o revela, o que evita uma consulta de metadados só para isso.
    paths = JobPaths(job_id_for(url))
    cfg = PRESETS[preset]

    gender = "masculina"
    if voice:
        try:
            gender = Voice.load(voice).meta.get("genero", gender)
        except FileNotFoundError:
            pass

    state = runner.RunState(job_id=paths.job_id, url=url, voice=voice or "auto",
                            preset=preset)
    state.stages = [runner.StageState(stage) for stage in runner.build_stages()]

    escolha = ("vozes clonadas do vídeo" if clonar
               else "voz automática" if not voice else f"voz {voice} (forçada)")
    console.print(f"[bold]{paths.job_id}[/bold]  {escolha}  preset {preset}")
    _avisar_pasta_de_fora(paths)

    failed = None
    paths.ensure()
    log_path = paths.root / "run.log"

    # As bibliotecas do pipeline escrevem no terminal por fora do painel, o que
    # quebra o layout e gera tracebacks de "closed file" ao tentarem logar num
    # fluxo que o painel substituiu. Tudo isso vai para o arquivo de log.
    with quiet_output(log_path):
        with Live(_render_dashboard(state), console=console,
                  refresh_per_second=8, transient=False) as live:
            def redraw() -> None:
                live.update(_render_dashboard(state))

            try:
                runner.run(state, paths, cfg, url=url, voice=voice,
                           gender=gender, cookies=cookies, force_from=refazer,
                           clone=clonar, on_change=redraw)
            except Exception as exc:  # noqa: BLE001 - mostrado abaixo, sem traceback
                failed = exc
            redraw()

    if failed is not None:
        console.print(f"\n[red]falhou:[/red] {failed}")
        console.print(f"[dim]detalhes em:[/dim] {log_path}")
        console.print(
            f"[dim]retome de onde parou com:[/dim] uv run dublador {paths.job_id}"
        )
        raise typer.Exit(1)

    console.print(f"\n[green]pronto:[/green] {paths.output}")
    resumo_path = paths.root / "resumo.txt"
    if resumo_path.exists():
        console.print(f"[green]resumo:[/green] {resumo_path}")
    _print_voice(state.chosen, forced=voice)
    _print_quality(paths)


def _avisar_pasta_de_fora(paths: JobPaths) -> None:
    """Avisa quando o job reaproveitado não está no diretório atual.

    Sem isso a execução parece não ter feito nada: a pessoa roda o comando numa
    pasta, o programa reaproveita um trabalho anterior guardado em outro lugar,
    e um `ls` no diretório atual não mostra nada.
    """
    from .config import jobs_dir

    destino = jobs_dir()
    if paths.root.exists() and paths.root.parent != destino:
        console.print(
            f"[yellow]nota:[/yellow] este vídeo já foi dublado em "
            f"[bold]{paths.root}[/bold], e será reaproveitado de lá."
        )
        console.print(
            f"[dim]para refazer do zero aqui:[/dim] mova ou apague aquela pasta"
        )


_STATUS_STYLE = {
    "pendente": ("[dim]·[/dim]", "dim"),
    "rodando": ("[yellow]▸[/yellow]", "yellow"),
    "pronto": ("[green]✓[/green]", "green"),
    "pulado": ("[blue]·[/blue]", "blue"),
    "erro": ("[red]✗[/red]", "red"),
}


def _render_dashboard(state) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(width=2)
    table.add_column(width=28)
    table.add_column(width=22)
    table.add_column(width=8, justify="right")
    table.add_column(overflow="ellipsis")

    for entry in state.stages:
        marker, style = _STATUS_STYLE.get(entry.status, ("·", "dim"))
        bar = _bar(entry.percent) if entry.status == "rodando" else ""
        tempo = f"{entry.seconds:.1f}s" if entry.seconds else ""
        detail = entry.error or entry.message
        if entry.status == "pulado":
            detail = "já existia"
        table.add_row(marker, f"[{style}]{entry.stage.label}[/{style}]",
                      bar, tempo, f"[dim]{detail}[/dim]")

    table.add_row("", f"[bold]total[/bold]", "", f"[bold]{state.elapsed:.0f}s[/bold]", "")
    return table


def _bar(percent: float, width: int = 20) -> str:
    filled = int(round(percent * width))
    return f"[yellow]{'━' * filled}[/yellow][dim]{'━' * (width - filled)}[/dim]"


def _print_voice(chosen: dict, *, forced: str | None) -> None:
    """Mostra qual voz foi usada e por quê, para a escolha ser contestável."""
    if forced:
        console.print(f"[dim]voz {forced}, escolhida por você[/dim]")
        return
    if not chosen:
        return

    linha = (f"[dim]voz {chosen.get('voice')} "
             f"({chosen.get('gender')}), por {chosen.get('basis')}[/dim]")
    console.print(linha)
    if chosen.get("basis") == "transcrição" and not chosen.get("agrees", True):
        console.print(
            f"[yellow]atenção:[/yellow] a frequência da voz sugeria "
            f"{chosen.get('f0_gender')} (F0 {chosen.get('median_f0')} Hz). "
            f"Se a voz soar errada, force com --voice."
        )


def _print_quality(paths: JobPaths) -> None:
    """Fecha com o que o operador precisa saber sem assistir ao vídeo."""
    from collections import Counter

    from .model import load_segments

    if not paths.segments.exists():
        return
    segments = load_segments(paths.segments)
    fitting = sum(1 for s in segments if s.overflow_pct <= 0.10)
    stretched = sum(1 for s in segments if abs(s.stretch - 1.0) > 0.01)
    overlap = sum(1 for s in segments if "sobreposicao" in s.qa_flags)

    console.print(
        f"[dim]{len(segments)} falas | {fitting} no tempo | "
        f"{stretched} ajustadas | {overlap} sobrepostas[/dim]"
    )
    flags = Counter(f.split(":")[0] for s in segments for f in s.qa_flags
                    if not f.startswith("cortado"))
    problems = {k: v for k, v in flags.items() if k != "audio_maior_que_slot"}
    if problems:
        resumo = ", ".join(f"{k} ({v})" for k, v in problems.items())
        console.print(f"[yellow]revisar:[/yellow] {resumo}")


@app.command()
def download(
    url: Annotated[str, typer.Argument(help="URL do vídeo")],
    cookies: Annotated[str | None, typer.Option(
        help="Navegador de onde ler cookies (chrome, safari, firefox)")] = None,
) -> None:
    """Baixa o vídeo e extrai o áudio de trabalho em 48 kHz."""
    from .pipeline import s01_download

    paths = JobPaths(job_id_for(url))
    with _progress_bar("download") as bar:
        task = bar.add_task("download", total=1.0, msg="")
        meta = s01_download.download(
            url, paths,
            progress=lambda p, m: bar.update(task, completed=p, msg=m),
            cookies_from_browser=cookies,
        )
    console.print(f"[green]OK[/green] {meta['title']}")
    console.print(f"     job: [bold]{paths.job_id}[/bold]  ->  {paths.root}")
    console.print(f"     duração: {meta['audio_duration']:.1f}s")


@app.command()
def separate(
    job: Annotated[str, typer.Argument(help="ID do job")],
    preset: Annotated[str, typer.Option(help="draft, balanced ou max")] = DEFAULT_PRESET,
) -> None:
    """Separa voz e trilha instrumental do áudio original."""
    from .pipeline import s02_separate

    paths = JobPaths(job)
    cfg = PRESETS[preset]
    with _progress_bar("separate") as bar:
        task = bar.add_task("separate", total=1.0, msg="")
        result = s02_separate.separate(
            paths, model=cfg.separator_model,
            progress=lambda p, m: bar.update(task, completed=p, msg=m),
        )
    console.print(f"[green]OK[/green] {result['model']} -> "
                  f"vocals={result['vocals']} instrumental={result['instrumental']}")


@app.command()
def asr(
    job: Annotated[str, typer.Argument(help="ID do job (ou a URL)")],
    preset: Annotated[str, typer.Option(help="draft, balanced ou max")] = DEFAULT_PRESET,
) -> None:
    """Transcreve o áudio com timestamps de palavra."""
    from .pipeline import s03_asr

    paths = JobPaths(job if "/" not in job else job_id_for(job))
    cfg = PRESETS[preset]
    with _progress_bar("asr") as bar:
        task = bar.add_task("asr", total=1.0, msg="")
        result = s03_asr.transcribe(
            paths, model_id=cfg.asr_model,
            progress=lambda p, m: bar.update(task, completed=p, msg=m),
        )

    table = Table("início", "fim", "texto", title="Primeiras sentenças")
    for s in result["sentences"][:8]:
        table.add_row(f"{s['start']:.2f}", f"{s['end']:.2f}", s["text"][:70])
    console.print(table)
    console.print(f"[green]OK[/green] {paths.asr}")


@app.command()
def segment(job: Annotated[str, typer.Argument(help="ID do job")]) -> None:
    """Converte a transcrição em segmentos de dublagem com orçamento de tempo."""
    from .pipeline import s04_segment

    paths = JobPaths(job)
    with _progress_bar("segment") as bar:
        task = bar.add_task("segment", total=1.0, msg="")
        segments = s04_segment.build_segments(
            paths, progress=lambda p, m: bar.update(task, completed=p, msg=m)
        )

    durations = sorted(s.duration for s in segments)
    total = sum(s.duration for s in segments)
    console.print(
        f"[green]OK[/green] {len(segments)} segmentos | "
        f"fala {total:.1f}s | mediana {durations[len(durations) // 2]:.2f}s | "
        f"min {durations[0]:.2f}s | max {durations[-1]:.2f}s"
    )

    table = Table("id", "início", "dur", "folga", "sílabas", "texto")
    for s in segments[:10]:
        table.add_row(str(s.id), f"{s.start:.2f}", f"{s.duration:.2f}",
                      f"{s.gap_before:.2f}", str(s.syllable_budget),
                      s.text_en[:52])
    console.print(table)


@app.command()
def brief(
    job: Annotated[str, typer.Argument(help="ID do job")],
    preset: Annotated[str, typer.Option(help="draft, balanced ou max")] = DEFAULT_PRESET,
) -> None:
    """Lê a transcrição inteira e prepara contexto global para a tradução."""
    from .pipeline import brief as brief_stage

    paths = JobPaths(job)
    cfg = PRESETS[preset]
    with _progress_bar("brief") as bar:
        task = bar.add_task("brief", total=1.0, msg="")
        result = brief_stage.build(
            paths, model_id=cfg.reviewer_model,
            progress=lambda p, m: bar.update(task, completed=p, msg=m),
        )

    console.print(f"[bold]assunto:[/bold] {result['assunto'][:150]}")
    console.print(f"[bold]registro:[/bold] {result['registro'][:80]}")
    if result["cuidados"]:
        console.print("[bold]cuidados:[/bold]")
        for c in result["cuidados"][:5]:
            console.print(f"  - {c[:110]}")
    if result["glossario"]:
        table = Table("termo", "em português")
        for k, v in list(result["glossario"].items())[:14]:
            table.add_row(k[:32], v[:44])
        console.print(table)


@app.command()
def translate(
    job: Annotated[str, typer.Argument(help="ID do job")],
    preset: Annotated[str, typer.Option(help="draft, balanced ou max")] = DEFAULT_PRESET,
) -> None:
    """Traduz os segmentos para português."""
    from .pipeline import s05_translate

    paths = JobPaths(job)
    cfg = PRESETS[preset]
    if not cfg.uses_translator:
        console.print(f"[yellow]preset {preset} não usa tradutor separado[/yellow]")
        raise typer.Exit(1)

    with _progress_bar("translate") as bar:
        task = bar.add_task("translate", total=1.0, msg="")
        segments = s05_translate.translate(
            paths, model_id=cfg.translator_model,
            progress=lambda p, m: bar.update(task, completed=p, msg=m),
        )

    table = Table("id", "dur", "orç", "inglês", "português")
    for s in segments[:10]:
        table.add_row(str(s.id), f"{s.duration:.1f}", str(s.syllable_budget),
                      s.text_en[:34], (s.text_pt_raw or "")[:38])
    console.print(table)


@app.command()
def review(
    job: Annotated[str, typer.Argument(help="ID do job")],
    preset: Annotated[str, typer.Option(help="draft, balanced ou max")] = DEFAULT_PRESET,
) -> None:
    """Adapta a tradução para fala natural que caiba no tempo do vídeo."""
    from .pipeline import s06_review

    paths = JobPaths(job)
    cfg = PRESETS[preset]
    with _progress_bar("review") as bar:
        task = bar.add_task("review", total=1.0, msg="")
        segments = s06_review.review(
            paths, model_id=cfg.reviewer_model, attempts=max(1, cfg.fit_attempts),
            progress=lambda p, m: bar.update(task, completed=p, msg=m),
        )

    from .utils.syllables import count_syllables

    fitting = sum(1 for s in segments if s.overflow_pct <= 0.10)
    console.print(
        f"[green]OK[/green] {fitting}/{len(segments)} segmentos dentro do orçamento"
    )

    table = Table("id", "orç", "síl", "estouro", "adaptado")
    for s in segments[:12]:
        text = s.text_pt_final or ""
        table.add_row(str(s.id), str(s.syllable_budget), str(count_syllables(text)),
                      f"{s.overflow_pct:.0%}", text[:46])
    console.print(table)


@app.command()
def tts(
    job: Annotated[str, typer.Argument(help="ID do job")],
    voice: Annotated[str, typer.Option(help="Voz do catálogo")] = "alex",
    preset: Annotated[str, typer.Option(help="draft, balanced ou max")] = DEFAULT_PRESET,
) -> None:
    """Sintetiza a fala de cada segmento."""
    from .pipeline import s07_tts

    paths = JobPaths(job)
    cfg = PRESETS[preset]
    with _progress_bar("tts") as bar:
        task = bar.add_task("tts", total=1.0, msg="")
        segments = s07_tts.synthesize(
            paths, backend_name=cfg.tts_backend, voice_name=voice,
            progress=lambda p, m: bar.update(task, completed=p, msg=m),
        )

    over = [s for s in segments if s.overflow_pct > 0.10]
    console.print(
        f"[green]OK[/green] {len(segments)} segmentos | "
        f"{len(over)} maiores que o espaço disponível"
    )


@app.command()
def fit(job: Annotated[str, typer.Argument(help="ID do job")]) -> None:
    """Encaixa os áudios na linha do tempo do vídeo."""
    from .pipeline import s08_fit

    paths = JobPaths(job)
    with _progress_bar("fit") as bar:
        task = bar.add_task("fit", total=1.0, msg="")
        segments = s08_fit.assemble(
            paths, progress=lambda p, m: bar.update(task, completed=p, msg=m)
        )

    stretched = [s for s in segments if abs(s.stretch - 1.0) > 0.01]
    overlap = [s for s in segments if "sobreposicao" in s.qa_flags]
    console.print(
        f"[green]OK[/green] {paths.dub.name} | esticados {len(stretched)} | "
        f"sobrepostos {len(overlap)}"
    )


@app.command()
def render(job: Annotated[str, typer.Argument(help="ID do job")]) -> None:
    """Mixa, adiciona legendas e monta o vídeo final."""
    from .pipeline import s09_render

    paths = JobPaths(job)
    with _progress_bar("render") as bar:
        task = bar.add_task("render", total=1.0, msg="")
        result = s09_render.render(
            paths, progress=lambda p, m: bar.update(task, completed=p, msg=m)
        )

    if not result["separated_bed"]:
        console.print(
            "[yellow]aviso:[/yellow] sem separação de fontes — a voz original "
            "continua audível ao fundo"
        )
    console.print(f"[green]OK[/green] {result['output']}")
    console.print(
        f"     {result['duration']:.1f}s | "
        f"{result['integrated_lufs']:.1f} LUFS | pico {result['true_peak']:.1f} dBTP"
    )


@app.command()
def report(job: Annotated[str, typer.Argument(help="ID do job")]) -> None:
    """Mostra as métricas de qualidade e os segmentos com ressalva."""
    from collections import Counter

    from .model import load_segments

    paths = JobPaths(job)
    segments = load_segments(paths.segments)

    overflow = sorted(s.overflow_pct for s in segments)
    p95 = overflow[int(len(overflow) * 0.95)] if overflow else 0.0
    stretched = [s for s in segments if abs(s.stretch - 1.0) > 0.01]

    console.print(f"[bold]{len(segments)} segmentos[/bold]")
    console.print(f"  estouro p50 {overflow[len(overflow) // 2]:.0%} | p95 {p95:.0%}")
    console.print(f"  esticados: {len(stretched)}")
    console.print(f"  tentativas de reescrita: {sum(s.attempts for s in segments)}")

    counts = Counter(f.split(":")[0] for s in segments for f in s.qa_flags)
    if counts:
        console.print("\n[bold]ressalvas[/bold]")
        for flag, count in counts.most_common():
            console.print(f"  {flag}: {count}")

    worst = sorted(segments, key=lambda s: -s.overflow_pct)[:5]
    table = Table("id", "estouro", "texto")
    for s in worst:
        table.add_row(str(s.id), f"{s.overflow_pct:.0%}",
                      (s.text_pt_final or "")[:56])
    console.print(table)


@app.command()
def resumo(
    job: Annotated[str, typer.Argument(help="ID do job")],
    preset: Annotated[str, typer.Option(help="draft, balanced ou max")] = DEFAULT_PRESET,
) -> None:
    """Gera o resumo do conteúdo do vídeo em português."""
    from .pipeline import s10_summary

    paths = JobPaths(job)
    cfg = PRESETS[preset]
    with _progress_bar("resumo") as bar:
        task = bar.add_task("resumo", total=1.0, msg="")
        result = s10_summary.summarize(
            paths, model_id=cfg.reviewer_model,
            progress=lambda p, m: bar.update(task, completed=p, msg=m),
        )
    console.print(f"[green]OK[/green] {paths.root / 'resumo.txt'}")
    console.print(f"[bold]{result['titulo']}[/bold]")


@app.command()
def modelos(
    mover: Annotated[bool, typer.Option(
        "--mover",
        help="Executa de fato. Sem isto, apenas mostra o que seria feito")] = False,
) -> None:
    """Compartilha os LLMs de texto com a biblioteca do LM Studio.

    Move os modelos de linguagem para ~/.lmstudio/models, onde servem tanto a
    este projeto quanto ao LM Studio, em vez de ocupar espaço só aqui. Os
    modelos de fala ficam no cache: o LM Studio não os executa.
    """
    from .utils import modelos as mod

    encontrados = mod.listar()
    if not encontrados:
        console.print("[yellow]nenhum LLM de texto no cache[/yellow]")
        return

    table = Table("modelo", "tamanho", "situação")
    for m in encontrados:
        estado = ("[green]já no LM Studio[/green]" if m.ja_migrado
                  else "[yellow]no cache do HF[/yellow]")
        table.add_row(m.repo, f"{m.tamanho_gb:.1f} GB", estado)
    console.print(table)

    pendentes = [m for m in encontrados if not m.ja_migrado]
    if not pendentes:
        console.print("[green]tudo já compartilhado[/green]")
        return

    total = sum(m.tamanho_gb for m in pendentes)
    if not mover:
        console.print(f"\n[bold]{total:.1f} GB[/bold] seriam movidos para "
                      f"{mod.LMSTUDIO_DIR}")
        for m in pendentes:
            console.print(f"  {m.origem.name}")
            console.print(f"    -> {m.destino}")
        console.print("\n[dim]nada foi alterado. para executar:[/dim] "
                      "uv run dublador modelos --mover")
        return

    for m in pendentes:
        console.print(f"movendo {m.repo} ({m.tamanho_gb:.1f} GB)...")
        try:
            mod.mover(m)
        except Exception as exc:  # noqa: BLE001 - reportado, o resto continua
            console.print(f"  [red]falhou:[/red] {exc}")
            continue
        console.print(f"  [green]ok[/green] {m.destino}")

    console.print(f"\n[green]pronto.[/green] Os modelos agora servem aos dois "
                  f"usos e são encontrados automaticamente.")


@app.command()
def voices() -> None:
    """Lista as vozes disponíveis no catálogo."""
    from .tts.base import list_voices

    names = list_voices()
    if not names:
        console.print("[yellow]nenhuma voz no catálogo[/yellow]")
        return
    for name in names:
        console.print(f"  {name}")


@app.command()
def info(job: Annotated[str, typer.Argument(help="ID do job")]) -> None:
    """Mostra os artefatos já produzidos para um job."""
    paths = JobPaths(job)
    if not paths.root.exists():
        console.print(f"[red]job {job} não encontrado[/red]")
        raise typer.Exit(1)

    table = Table("artefato", "existe", "tamanho")
    for name in ("video", "audio", "vocals", "instrumental", "asr", "segments",
                 "dub", "mixed", "output"):
        p = getattr(paths, name)
        exists = p.exists()
        size = f"{p.stat().st_size / 1e6:.1f} MB" if exists else "-"
        table.add_row(name, "sim" if exists else "não", size)
    console.print(table)


if __name__ == "__main__":
    app()
