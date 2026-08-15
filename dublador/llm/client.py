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
from typing import Any


# Lote pequeno por segurança, não por falta de memória: ver chat_batch.
BATCH_SIZE = 4

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


class LocalLLM:
    """Um modelo MLX carregado sob demanda."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is None:
            from mlx_lm import load

            self._model, self._tokenizer = load(self.model_id)

    def unload(self) -> None:
        """Libera os pesos. Necessário antes de carregar o próximo modelo."""
        import mlx.core as mx

        self._model = None
        self._tokenizer = None
        gc.collect()
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
                   batch_size: int = BATCH_SIZE) -> list[str]:
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
            response = batch_generate(
                self._model, self._tokenizer, prompts=prompts,
                max_tokens=max_tokens, sampler=sampler, verbose=False,
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

        return outputs

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
