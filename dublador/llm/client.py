"""Cliente de LLM local sobre MLX.

Todo o pipeline roda em MLX — ASR, TTS e LLM — o que evita ter um servidor de
inferência separado no sistema e mantém uma única fila para a GPU.

O ponto delicado aqui é memória: tradutor e revisor não cabem juntos com o TTS
em 24 GB. `unload()` existe para ser chamado entre etapas, e o pipeline executa
uma etapa inteira antes de passar para a próxima em vez de intercalar.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any, Callable


# Onde o LM Studio guarda modelos. Ele usa o mesmo formato que usamos aqui —
# MLX safetensors em pasta plana — então um modelo que esteja lá serve aos dois
# sem duplicar gigabytes.
LMSTUDIO_DIR = Path.home() / ".lmstudio" / "models"


def resolve_model(model_id: str) -> str:
    """Prefere o modelo já presente no LM Studio, se houver.

    Recebe um identificador do Hugging Face (`org/nome`) e devolve um caminho
    local quando esse modelo existe na biblioteca do LM Studio. Não achando,
    devolve o identificador original e o download acontece como antes.
    """
    if "/" not in model_id or Path(model_id).exists():
        return model_id

    org, nome = model_id.split("/", 1)
    for candidato in (LMSTUDIO_DIR / org / nome, LMSTUDIO_DIR / nome):
        if (candidato / "config.json").exists():
            return str(candidato)
    return model_id


# Lote pequeno por segurança, não por falta de memória: ver chat_batch.
BATCH_SIZE = 4

# Tamanho mínimo do prefixo comum para valer a pena prefixar o cache. Abaixo
# disso, montar e copiar o cache custa mais do que reprocessar os tokens.
MIN_SHARED_PREFIX = 200


def _common_prefix_length(prompts: list[list[int]]) -> int:
    """Quantos tokens iniciais são iguais em todos os prompts."""
    shortest = min(len(p) for p in prompts)
    first = prompts[0]
    for index in range(shortest):
        token = first[index]
        if any(p[index] != token for p in prompts):
            return index
    return shortest

# Escritas que não têm o que fazer numa tradução para o português. A presença
# delas é sinal forte de contaminação entre sequências do lote, não de um
# estrangeirismo legítimo — que viria em alfabeto latino.
_FOREIGN_SCRIPT_RE = re.compile(
    r"[Ѐ-ӿ"      # cirílico
    r"Ͱ-Ͽ"       # grego
    r"一-鿿"       # han
    r"぀-ヿ"       # kana
    r"؀-ۿ"       # árabe
    r"֐-׿]"      # hebraico
)

# Pontuação colada na palavra seguinte ("tudo,o"), com letra e não dígito
# depois — dígito seria separador de milhar. Texto bem formado não faz isso.
_GLUED_PUNCT_RE = re.compile(r"[,;:][^\W\d_]")

# Sequência de maiúsculas grudada em minúsculas ("QUEocê"): não ocorre em
# português correto nem em siglas, que vêm isoladas.
_SHOUT_GLUE_RE = re.compile(r"[A-ZÁÉÍÓÚ]{2,}[a-záéíóúâêôãõç]")


def looks_corrupted(text: str) -> bool:
    """Heurística barata para saídas visivelmente quebradas.

    Serve como gatilho de nova tentativa, não como avaliação de qualidade.

    É calibrada para PRECISÃO, não para cobertura: um falso positivo custa uma
    regeneração, mas dispara em texto legítimo — a versão anterior desta função
    marcava "TikTok" e "iPhone" como corrompidos, 2 falsos positivos em 6
    frases boas, num material cheio de nomes de marca. Deixar passar uma
    corrupção rara é aceitável porque o lote pequeno já a torna rara; marcar
    texto bom constantemente não é.
    """
    if not text.strip():
        return True
    return (
        bool(_FOREIGN_SCRIPT_RE.search(text))
        or bool(_GLUED_PUNCT_RE.search(text))
        or bool(_SHOUT_GLUE_RE.search(text))
    )


# Quanto de memória os modelos residentes podem ocupar somados. O sistema
# reserva ~75% da RAM para a GPU (18 GB em 24 GB); este teto deixa folga para
# o cache de ativações, o sintetizador e o restante do processo.
RESIDENT_BUDGET_GB = 13.0

# Modelos já carregados, compartilhados entre as etapas do mesmo processo.
_LOADED: dict[str, tuple] = {}
_ORDER: list[str] = []


def _active_gb() -> float:
    import mlx.core as mx

    return mx.get_active_memory() / 1073741824


def _evict_until_fits() -> None:
    """Descarrega o modelo usado há mais tempo até caber no orçamento."""
    import mlx.core as mx

    while len(_ORDER) > 1 and _active_gb() > RESIDENT_BUDGET_GB:
        oldest = _ORDER.pop(0)
        _LOADED.pop(oldest, None)
        gc.collect()
        mx.clear_cache()


def release_all() -> None:
    """Libera todos os modelos. Para quando o processo vai fazer outra coisa."""
    import mlx.core as mx

    _LOADED.clear()
    _ORDER.clear()
    gc.collect()
    mx.clear_cache()


class LocalLLM:
    """Um modelo MLX, carregado sob demanda e mantido residente.

    O mesmo modelo serve três etapas do pipeline — briefing, revisão e resumo —
    e antes era carregado do zero em cada uma, a 3,2 s por vez. Como os dois
    modelos usados somam ~11,7 GB e há 18 GB disponíveis, mantê-los residentes
    é de graça em memória e poupa o recarregamento.

    O orçamento existe para o caso de um preset usar modelos maiores: passando
    do teto, o menos usado sai.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @property
    def _model(self):
        return _LOADED.get(self.model_id, (None, None))[0]

    @property
    def _tokenizer(self):
        return _LOADED.get(self.model_id, (None, None))[1]

    def load(self) -> None:
        if self.model_id in _LOADED:
            _ORDER.remove(self.model_id)
            _ORDER.append(self.model_id)
            return

        from mlx_lm import load

        _LOADED[self.model_id] = load(resolve_model(self.model_id))
        _ORDER.append(self.model_id)
        _evict_until_fits()

    def unload(self) -> None:
        """Mantém o modelo residente para a próxima etapa que o usar.

        As etapas chamam isto ao terminar, mas descarregar de fato só faz
        sentido quando a memória aperta — e aí o despejo por orçamento resolve.
        """
        import mlx.core as mx

        mx.clear_cache()

    def chat(self, messages: list[dict[str, str]], *,
             max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Uma rodada de conversa, devolvendo apenas o texto gerado."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self.load()
        prompt = self._apply_template(messages)
        text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )
        return _strip_reasoning(text).strip()

    def chat_batch(self, conversations: list[list[dict[str, str]]], *,
                   max_tokens: int = 512, temperature: float = 0.2,
                   batch_size: int = BATCH_SIZE,
                   on_progress: Callable[[int, int], None] | None = None
                   ) -> list[str]:
        """Gera várias respostas de uma vez.

        Chamadas de um prompt por vez deixam a GPU ociosa entre tokens: medido
        aqui, ~21% de uso. Como os segmentos são independentes dentro de uma
        rodada, dá para submetê-los juntos.

        Há um limite prático: com lotes grandes e prompts de comprimentos muito
        diferentes, sequências contaminam umas às outras — medido neste projeto,
        11 de 27 traduções saíram corrompidas com lote 16, algumas com
        caracteres cirílicos no meio do português, contra 0 de 27 com lote 4. O
        ganho de tempo entre os dois era irrisório (5,8s contra 7,4s), então o
        lote fica pequeno e ainda assim as saídas passam por verificação.
        """
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_sampler

        self.load()
        sampler = make_sampler(temp=temperature)
        outputs: list[str] = []

        for start in range(0, len(conversations), batch_size):
            chunk = conversations[start : start + batch_size]
            prompts = [
                self._tokenizer.encode(self._apply_template(messages))
                for messages in chunk
            ]
            prompts, caches = self._share_prefix(prompts)
            response = batch_generate(
                self._model, self._tokenizer, prompts=prompts,
                max_tokens=max_tokens, sampler=sampler, verbose=False,
                prompt_caches=caches,
            )
            texts = [_strip_reasoning(text).strip() for text in response.texts]

            # Rede de segurança: o que sair corrompido é refeito sozinho, onde
            # não existe vizinho para contaminar.
            for offset, text in enumerate(texts):
                if looks_corrupted(text):
                    texts[offset] = self.chat(
                        chunk[offset], max_tokens=max_tokens,
                        temperature=temperature,
                    )
            outputs.extend(texts)
            if on_progress is not None:
                on_progress(len(outputs), len(conversations))

        return outputs

    def _share_prefix(self, prompts: list[list[int]]):
        """Calcula uma vez o trecho que todos os prompts do lote têm em comum.

        Nas etapas de reescrita, instrução e briefing são idênticos entre os
        segmentos e respondem por cerca de três quartos de cada prompt: medido
        aqui, 963 de ~1.250 tokens, reprocessados a cada segmento. Prefixar o
        cache uma vez e reaproveitá-lo acelera a revisão em ~1,47x.

        A saída NÃO é bit a bit idêntica, ao contrário do que a ideia sugere.
        O prefixo é processado em lote de 1 e a geração em lote de 4, e essa
        diferença de forma muda ligeiramente os valores de ponto flutuante;
        onde dois tokens estão quase empatados, a escolha vira. Medido em
        material real: 16 de 23 respostas idênticas, e as 7 restantes com
        redação equivalente, não pior ("acho que deveria" contra "acho que
        você deveria"). Processar o prefixo já na largura do lote resolveria a
        numérica, mas a API de cache exige uma sequência por vez.

        Determinismo estrito nunca foi propriedade desta etapa: mudar o
        tamanho do lote já alterava os mesmos empates. Para desligar, basta
        elevar MIN_SHARED_PREFIX.

        Devolve os prompts já sem o prefixo e os caches correspondentes, ou os
        prompts originais e None quando não compensa.
        """
        import copy

        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        if len(prompts) < 2:
            return prompts, None

        shared = _common_prefix_length(prompts)
        # Abaixo deste tamanho o custo de montar e copiar o cache supera o
        # ganho. Também é preciso sobrar ao menos um token em cada prompt.
        if shared < MIN_SHARED_PREFIX or any(len(p) <= shared for p in prompts):
            return prompts, None

        cache = make_prompt_cache(self._model)
        self._model(mx.array([prompts[0][:shared]]), cache=cache)
        mx.eval([layer.state for layer in cache])

        return ([p[shared:] for p in prompts],
                [copy.deepcopy(cache) for _ in prompts])

    def _apply_template(self, messages: list[dict[str, str]]) -> str:
        """Monta o prompt desligando o modo de raciocínio quando existir.

        Modelos da família Qwen3 raciocinam por padrão: escrevem um bloco
        <think> inteiro antes de responder. Aqui isso é puro desperdício — a
        tarefa é reescrever uma frase curta sob restrição, não resolver um
        problema — e custava ~55s por segmento, dominando o tempo do pipeline.
        Nem todo tokenizer aceita a flag, daí o fallback.
        """
        try:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )

    def chat_json(self, messages: list[dict[str, str]], *,
                  max_tokens: int = 2048, temperature: float = 0.2,
                  attempts: int = 3) -> Any:
        """Igual a `chat`, mas exige JSON de volta.

        Modelos locais menores erram o formato com frequência, então tentamos de
        novo com a mensagem de erro anexada em vez de deixar a etapa quebrar.
        """
        conversation = list(messages)
        last_error = ""

        for attempt in range(attempts):
            raw = self.chat(conversation, max_tokens=max_tokens,
                            temperature=temperature if attempt == 0 else 0.0)
            try:
                return _extract_json(raw)
            except ValueError as exc:
                last_error = str(exc)
                conversation = list(messages) + [
                    {"role": "assistant", "content": raw[:800]},
                    {"role": "user", "content":
                     f"A resposta anterior não era JSON válido ({last_error})."
                     " Responda apenas com o JSON pedido, sem texto ao redor."},
                ]

        raise ValueError(
            f"{self.model_id} não devolveu JSON válido em {attempts} tentativas:"
            f" {last_error}"
        )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Remove blocos de raciocínio dos modelos que pensam antes de responder."""
    cleaned = _THINK_RE.sub("", text)
    # modelo que abriu <think> e não fechou: descarta tudo até o fim do bloco
    if "<think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1] if "</think>" in cleaned else ""
    return cleaned


def _extract_json(text: str) -> Any:
    """Encontra o JSON dentro da resposta, tolerando cercas e texto ao redor."""
    candidates: list[str] = []

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()

        # A estrutura externa é a que abre primeiro. Testar o colchete antes da
        # chave devolvia arrays internos: em {"assunto": ..., "cuidados": [...]}
        # o resultado era a lista de "cuidados", não o objeto.
        pairs = [("{", "}"), ("[", "]")]
        first_brace = candidate.find("{")
        first_bracket = candidate.find("[")
        if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
            pairs.reverse()

        for opener, closer in pairs:
            start = candidate.find(opener)
            end = candidate.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    continue

    raise ValueError("nenhum JSON encontrado na resposta")
