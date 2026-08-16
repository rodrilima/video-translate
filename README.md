# dublador

Dublagem automática de vídeos do inglês para o português brasileiro, rodando
inteiramente na sua máquina. Você passa um link do YouTube e recebe o vídeo
dublado, com legendas nos dois idiomas, o áudio original preservado como
segunda faixa e um resumo do conteúdo.

Nada sai do computador: transcrição, tradução e síntese de voz usam modelos
locais sobre MLX (Apple Silicon).

## Requisitos

- macOS com Apple Silicon. O pipeline usa MLX e CoreML; não há caminho para CPU
  ou CUDA.
- 16 GB de memória unificada é o mínimo confortável. O modelo de revisão sozinho
  ocupa ~8 GB.
- ~15 GB de disco para os modelos, baixados sob demanda na primeira execução.
  Ficam em `~/.cache/huggingface/hub` e `~/.cache/dublador/models`. Com
  `--clonar`, some mais 1,5 GB.

Se você usa o **LM Studio**, os modelos de linguagem podem ser compartilhados
com ele — é o mesmo formato, MLX safetensors:

```bash
uv run dublador modelos            # mostra o que seria movido
uv run dublador modelos --mover    # move para ~/.lmstudio/models
```

Os LLMs passam a servir aos dois usos em vez de ocupar espaço só aqui, e são
encontrados automaticamente. Modelos de fala não são movidos: o LM Studio não
executa reconhecimento de voz nem síntese.

## Instalação

```bash
brew install ffmpeg rubberband espeak-ng uv
uv sync --extra ml
```

`rubberband` é o binário usado no ajuste fino de tempo (o filtro embutido do
ffmpeg não serve). `espeak-ng` faz a fonetização do português no sintetizador.

Se a memória apertar durante a revisão, libere mais RAM para a GPU:

```bash
sudo sysctl iogpu.wired_limit_mb=20480
```

## Uso

```bash
uv run dublador iq5VjE31Eig
```

Só isso. O programa escolhe as vozes, mostra o progresso de cada etapa no
terminal e grava tudo numa pasta nomeada pelo título do vídeo, criada no
diretório atual: `./minha-estrategia-de-redes-sociais-ZTSI3DDP_4A/`. O
identificador no fim mantém a pasta única quando dois vídeos têm o mesmo
título, e os subcomandos aceitam só ele: `uv run dublador info ZTSI3DDP_4A`.

### Rodando de qualquer pasta

Defina um alias no seu `~/.zshrc`:

```bash
alias dublar='noglob uv run --project ~/projetos/translate dublador'
```

O `--project` faz o comando usar o ambiente do projeto de onde quer que você
esteja, sem duplicar a instalação. O `noglob` resolve o `?` da URL, que o zsh
expandiria como glob antes de o programa receber o argumento.

Com isso, a pasta do vídeo é criada **no diretório onde você está**:

```bash
cd ~/videos/dublados
dublar https://www.youtube.com/watch?v=iq5VjE31Eig
# cria ~/videos/dublados/titulo-do-video-iq5VjE31Eig/
```

Para mandar a saída para outro lugar, use `--em <pasta>` ou defina
`DUBLADOR_DIR` no ambiente.

Daí em diante:

```bash
dublar https://www.youtube.com/watch?v=iq5VjE31Eig       # URL completa, sem aspas
```

Sem o alias, estas formas de passar o vídeo funcionam do mesmo jeito:

```bash
uv run dublador iq5VjE31Eig                              # só o ID
uv run dublador https://youtu.be/iq5VjE31Eig             # URL curta
uv run dublador "https://www.youtube.com/watch?v=iq5..."  # URL completa, com aspas
```

O `noglob` não cobre URLs com `&` (playlists, por exemplo), porque o `&` manda
o comando para background. Nesses casos use aspas ou passe só o ID.

Rodar o mesmo link de novo aproveita tudo que já está em disco — uma execução
interrompida continua de onde parou, em vez de recomeçar.

### Opções

| opção | para quê |
|---|---|
| `--voice alex\|dora\|rocko` | Impõe uma voz, quando a escolha automática não agradar |
| `--clonar` | Usa a voz de cada locutor do próprio vídeo, em vez do catálogo |
| `--preset draft\|balanced\|max` | Troca qualidade por tempo. Padrão: `balanced` |
| `--refazer <etapa>` | Refaz a partir de um ponto, reaproveitando o resto |
| `--cookies chrome` | Quando o YouTube bloquear o download |
| `--em <pasta>` | Onde criar a pasta do vídeo. Padrão: o diretório atual |

`--refazer` é o que torna ajustes baratos. Trocar de voz leva ~11 s em vez de
reprocessar o vídeo inteiro:

```bash
uv run dublador --voice dora --refazer tts iq5VjE31Eig
```

### O que é gerado

Na pasta do job:

| arquivo | conteúdo |
|---|---|
| `out.mp4` | O vídeo final: dublagem, áudio original, legendas pt e en |
| `resumo.txt` | Resumo do conteúdo, com pontos principais e glossário |
| `segments.json` | Todas as falas, com tempos, tradução e voz atribuída |
| `pt.srt` / `en.srt` | Legendas soltas |
| `brief.json` | Assunto, registro e termos, apurados do vídeo inteiro |
| `diarization.json` | Quem fala em cada trecho |
| `run.log` | Log das bibliotecas, desviado para não sujar o painel |

O `segments.json` é editável: corrija uma tradução à mão e rode
`--refazer tts` para regerar só o áudio.

### Depurar etapas isoladas

Cada etapa roda sozinha sobre os artefatos já em disco:

```bash
uv run dublador asr <id>        # só transcreve
uv run dublador review <id>     # só reescreve o texto
uv run dublador resumo <id>     # só gera o resumo
uv run dublador info <id>       # o que já existe
uv run dublador report <id>     # métricas de qualidade
```

## Como funciona

Quatorze etapas, cada uma gravando seu resultado em disco:

```
download → separate → diarize → asr → segment → brief → voice
        → [clones] → translate → review → tts → fit → render → summary
```

| etapa | o que faz | ferramenta |
|---|---|---|
| download | baixa vídeo e áudio | yt-dlp |
| separate | separa voz de trilha e efeitos | BS-Roformer (MLX) |
| diarize | identifica quem fala quando | senko (CoreML) |
| asr | transcreve com tempo por palavra | Parakeet TDT 0.6B |
| segment | monta as falas e o orçamento de tempo | — |
| brief | lê o vídeo inteiro: assunto, termos, locutor | Qwen3-14B |
| voice | escolhe a voz de cada locutor | — |
| translate | tradução base, com contexto | Hy-MT2-7B |
| review | adapta para fala natural que caiba no tempo | Qwen3-14B |
| tts | sintetiza a voz | Kokoro-82M |
| fit | encaixa na linha do tempo do vídeo | rubberband |
| render | mixa, normaliza e monta o arquivo | ffmpeg |
| summary | escreve o resumo | Qwen3-14B |

### O problema central: isocronia

O português falado ocupa 20 a 30% mais tempo que o inglês para dizer a mesma
coisa. É onde a maioria dos projetos de dublagem falha, acelerando o áudio até
ficar artificial.

Aqui o problema é resolvido no texto, não no som. Cada fala recebe um orçamento
de sílabas calculado a partir do tempo disponível, e o revisor reescreve até
caber — encurtando de verdade, cortando ornamento e mantendo o fato. O áudio só
é comprimido como último recurso, e nunca abaixo de 0,90× da duração natural,
que é onde a fala começa a soar degradada.

Dois números que vieram de medição, não de literatura:

- **5,2 sílabas por segundo** é a taxa real do sintetizador em português. O
  valor de referência que se encontra publicado (6,0) pedia 16% menos tempo do
  que a fala precisa, e comprimia todos os segmentos.
- **0,90×** é o piso de compressão temporal, definido por escuta. Abaixo disso
  incomoda.

### Vozes

O catálogo tem três vozes fixas: `alex` e `rocko` (masculinas) e `dora`
(feminina). A escolha é automática — o gênero de quem fala é apurado da
transcrição, com a frequência da voz como desempate, e cada locutor recebe uma
voz diferente.

A frequência sozinha não decide, e isso foi medido: um locutor masculino
falando de forma enfática registrou 198 Hz, dentro da faixa tida como feminina,
com duas implementações independentes concordando. Fala expressiva desloca a
frequência o bastante para invalidar a classificação. Por isso o programa mostra
em que se baseou e avisa quando os dois sinais discordam — se errar, use
`--voice`.

Com `--clonar`, a voz de cada locutor é extraída do próprio vídeo e reproduzida
em português. Preserva o timbre, mas custa: a síntese fica ~11× mais lenta
(num vídeo de 25 min, 9,5 min contra 1 min) e os pesos do modelo são
**CC-BY-NC**, sem uso comercial. É opção, nunca padrão.

## Desempenho

Medido num MacBook Pro M5 Pro (24 GB), vídeo de 2 min 10 s:

| etapa | tempo |
|---|---|
| separação | 15,7 s |
| diarização | 6,5 s |
| transcrição | 2,1 s |
| briefing | 20 s |
| tradução | 15,7 s |
| **revisão** | **115 s** |
| síntese | 4,8 s |
| montagem e render | 7 s |
| **total** | **~184 s** (1,4× a duração) |

A revisão domina, e a proporção piora em vídeos longos: num vídeo de 25 minutos
ela deve responder por cerca de 22 dos ~32 minutos totais. É o primeiro lugar a
olhar se o tempo incomodar.

## Limitações conhecidas

- **A separação de trilha nunca foi testada com música de fundo.** O material de
  teste é fala pura, e o stem instrumental sai praticamente vazio. A etapa está
  objetivamente limpa (41 dB de separação), mas não foi exercitada de verdade.
- **Trocadilhos se perdem.** O pipeline evita traduzir errado, mas não recria o
  jogo de palavras — descarta e entrega o sentido.
- **O catálogo tem três vozes.** Um vídeo com quatro pessoas repete timbre.
- **Projeções para vídeos longos são extrapolação linear** a partir de um vídeo
  de dois minutos com densidade de fala atípica (98%).
- Baixar vídeos do YouTube contraria os Termos de Serviço da plataforma.
  Redistribuir conteúdo dublado de terceiros é outra conversa, com implicações
  próprias.

## Licenças dos modelos

| modelo | licença |
|---|---|
| Kokoro-82M (vozes fixas) | Apache-2.0 |
| Parakeet TDT | CC-BY-4.0 |
| senko | MIT |
| OmniVoice (clonagem) | pesos **CC-BY-NC**, sem uso comercial |

O caminho padrão do pipeline é livre para uso comercial. Só a clonagem não é.
