"""Escolha forçada entre as frases-alvo com o Whisper do Genérico."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from src.config import RAIZ_PROJETO
from src.transcritor import (
    TranscritorWhisperX, caminho_do_python, instrucao_de_instalacao,
)

NORMALIZACOES = ("soma", "media_por_token")
GRAFIAS = {
    "como_escrita": lambda frase: frase,
    "maiuscula_e_ponto": lambda frase: frase[:1].upper() + frase[1:] + ".",
    "maiuscula": lambda frase: frase[:1].upper() + frase[1:],
    "com_ponto": lambda frase: frase + ".",
}


class FalhaDaEscolhaForcada(RuntimeError):
    """Nenhuma gravação pôde ser pontuada: o comparador fica indisponível."""


def conferir_opcoes(normalizacao, grafias):
    """Recusa normalização ou grafia desconhecida, e lista de grafias vazia."""
    if normalizacao not in NORMALIZACOES:
        raise ValueError(
            "normalizacao da escolha forçada precisa ser uma de {}; veio {!r}".format(
                NORMALIZACOES, normalizacao))
    grafias = list(grafias or [])
    if not grafias:
        raise ValueError("a escolha forçada precisa de pelo menos uma grafia")
    desconhecidas = [g for g in grafias if g not in GRAFIAS]
    if desconhecidas:
        raise ValueError("grafia desconhecida {}; disponíveis: {}".format(
            desconhecidas, sorted(GRAFIAS)))
    if len(set(grafias)) != len(grafias):
        raise ValueError("grafia repetida em {}".format(grafias))
    return normalizacao, grafias


def textos_a_pontuar(frases_alvo, grafias):
    """Lista (rótulo, grafia, texto) na ordem dos rótulos e das grafias."""
    rotulos = sorted(str(r) for r in frases_alvo)
    if len(rotulos) < 2:
        raise ValueError("a escolha forçada precisa de pelo menos duas frases-alvo")
    itens = []
    for grafia in grafias:
        vistos = {}
        for rotulo in rotulos:
            frase = str(frases_alvo[rotulo]).strip()
            if not frase:
                raise ValueError("a frase-alvo de {!r} está vazia".format(rotulo))
            texto = GRAFIAS[grafia](frase)
            if texto in vistos:
                raise ValueError("{!r} e {!r} viram o mesmo texto {!r}".format(
                    vistos[texto], rotulo, texto))
            vistos[texto] = rotulo
            itens.append((rotulo, grafia, texto))
    return itens


def texto_das_sequencias(frases_alvo, grafias):
    """O texto canônico das sequências literais que a escolha forçada pontua."""
    _normalizacao, grafias = conferir_opcoes("soma", grafias)
    itens = textos_a_pontuar(frases_alvo, grafias)
    return "".join("{}\t{}\t{}\n".format(r, g, t) for r, g, t in itens), len(itens)


def sha256_das_sequencias(frases_alvo, grafias):
    """(sha256 em hexadecimal, número de sequências) do texto canônico."""
    texto, n = texto_das_sequencias(frases_alvo, grafias)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest(), n


def notas_por_frase(itens, pontuacoes):
    """Notas de cada rótulo nas duas normalizações."""
    if len(itens) != len(pontuacoes):
        raise RuntimeError("a ponte devolveu {} pontuações para {} textos".format(
            len(pontuacoes), len(itens)))
    notas = {n: {} for n in NORMALIZACOES}
    grafia_vencedora = {n: {} for n in NORMALIZACOES}
    for (rotulo, grafia, _), p in zip(itens, pontuacoes):
        soma = float(p["log_verossimilhanca"])
        tokens = int(p["tokens_pontuados"])
        if not math.isfinite(soma) or tokens < 2:
            raise RuntimeError("pontuação inválida para {!r}: {!r}".format(rotulo, p))
        for normalizacao, valor in (("soma", soma), ("media_por_token", soma / tokens)):
            if rotulo not in notas[normalizacao] or valor > notas[normalizacao][rotulo]:
                notas[normalizacao][rotulo] = valor
                grafia_vencedora[normalizacao][rotulo] = grafia
    return notas, grafia_vencedora


def escolher(notas):
    """Rótulo de maior nota e a lista das empatadas na maior nota."""
    rotulos = sorted(notas)
    maior = max(notas[r] for r in rotulos)
    empatados = [r for r in rotulos if notas[r] == maior]
    return empatados[0], empatados


def softmax_alinhado(classes, notas):
    """Softmax das notas na ordem de `classes`; classe sem nota recebe 0."""
    presentes = [c for c in classes if c in notas]
    valores = np.array([notas[c] for c in presentes], dtype=float)
    pesos = np.exp(valores - valores.max())
    pesos = pesos / pesos.sum()
    alinhado = np.zeros(len(classes), dtype=float)
    posicao = {c: i for i, c in enumerate(classes)}
    for c, p in zip(presentes, pesos):
        alinhado[posicao[c]] = float(p)
    return alinhado


def avaliar_escolha_forcada(sinais, rotulos, frases_alvo, pontuador, taxa,
                            normalizacao="soma", grafias=("como_escrita",)):
    """Avalia a escolha forçada nas gravações dadas, com a régua das três condições."""
    from src.evaluate import _resultado

    normalizacao, grafias = conferir_opcoes(normalizacao, grafias)
    rotulos = [str(r) for r in rotulos]
    if len(sinais) != len(rotulos):
        raise ValueError("Quantidade de sinais ({}) difere da de rótulos ({}).".format(
            len(sinais), len(rotulos)))
    if len(set(rotulos)) < 2:
        raise ValueError("A avaliação precisa de pelo menos 2 intenções diferentes.")
    frases_alvo = {str(r): f for r, f in frases_alvo.items()}
    faltando = sorted(set(rotulos) - set(frases_alvo))
    if faltando:
        raise ValueError("Rótulos sem frase-alvo: {}".format(faltando))
    itens = textos_a_pontuar(frases_alvo, grafias)
    classes = sorted(set(rotulos) | set(frases_alvo))
    outra = [n for n in NORMALIZACOES if n != normalizacao][0]

    por_normalizacao = {n: {"predicoes": [], "probabilidades": []} for n in NORMALIZACOES}
    por_gravacao, falhas = [], []
    for indice, sinal in enumerate(sinais):
        inicio = time.perf_counter()
        try:
            pontuacoes = pontuador.pontuar(sinal, taxa, [t for _, _, t in itens])
            notas, grafia = notas_por_frase(itens, pontuacoes)
        except ValueError as erro:
            falhas.append({"indice": indice, "motivo": str(erro)})
            for n in NORMALIZACOES:
                por_normalizacao[n]["predicoes"].append("FALHA_DA_ESCOLHA_FORCADA")
                por_normalizacao[n]["probabilidades"].append(np.zeros(len(classes)))
            por_gravacao.append({"indice": indice, "falhou": True})
            continue
        registro = {"indice": indice, "segundos": time.perf_counter() - inicio}
        for n in NORMALIZACOES:
            predicao, empatados = escolher(notas[n])
            por_normalizacao[n]["predicoes"].append(predicao)
            por_normalizacao[n]["probabilidades"].append(softmax_alinhado(classes, notas[n]))
            registro[n] = {"notas": notas[n], "grafia_vencedora": grafia[n],
                           "predicao": predicao, "empatados": empatados}
        por_gravacao.append(registro)

    if falhas and len(falhas) == len(rotulos):
        raise FalhaDaEscolhaForcada(
            "A escolha forçada falhou em todas as {} gravações; o comparador fica "
            "indisponível para esta pessoa. Primeiro motivo: {}".format(
                len(rotulos), falhas[0]["motivo"]))

    def _empacotar(n):
        return _resultado("escolha_forcada", rotulos, por_normalizacao[n]["predicoes"],
                          por_normalizacao[n]["probabilidades"], classes)

    resultado = _empacotar(normalizacao)
    resultado["normalizacao"] = normalizacao
    resultado["grafias"] = list(grafias)
    resultado["sensibilidade_da_normalizacao"] = dict(_empacotar(outra), normalizacao=outra)
    resultado["por_gravacao"] = por_gravacao
    resultado["falhas"] = falhas
    tempos = [r["segundos"] for r in por_gravacao if "segundos" in r]
    resultado["segundos_por_gravacao"] = {
        "n": len(tempos),
        "mediana": float(np.median(tempos)) if tempos else None,
        "maximo": float(max(tempos)) if tempos else None,
        "textos_por_gravacao": len(itens),
    }
    return resultado


class PontuadorWhisperForcado(TranscritorWhisperX):
    """Lado pai da ponte da escolha forçada."""

    def __init__(self, *args, caminho_ponte=None, **kwargs):
        super().__init__(*args, caminho_ponte=caminho_ponte or Path(__file__).with_name(
            "ponte_escolha_forcada.py"), **kwargs)
        self.nome = "whisper_forcado_{}".format(self.modelo)

    def transcrever(self, sinal, taxa):
        raise NotImplementedError("a escolha forçada não transcreve; use pontuar()")

    def pontuar(self, sinal, taxa, textos):
        self._garantir_processo()
        s = np.ascontiguousarray(np.asarray(sinal, dtype=np.float32))
        self._contador += 1
        pedido = {"id": self._contador, "taxa": int(taxa), "textos": list(textos),
                  "sinal_b64": base64.b64encode(s.tobytes()).decode("ascii")}
        self._processo.stdin.write(json.dumps(pedido, ensure_ascii=False) + "\n")
        self._processo.stdin.flush()
        linha = self._ler_linha(self.tempo_maximo_resposta, "pontuando as frases")
        try:
            resposta = json.loads(linha)
        except json.JSONDecodeError:
            self.fechar()
            raise RuntimeError("A ponte da escolha forçada respondeu fora do protocolo: "
                               "{!r}".format(linha.strip()))
        if resposta.get("id") != pedido["id"]:
            self.fechar()
            raise RuntimeError("A ponte da escolha forçada respondeu fora de ordem.")
        if "erro" in resposta:
            raise ValueError("A ponte da escolha forçada recusou o áudio: {}".format(
                resposta["erro"]))
        pontuacoes = resposta.get("pontuacoes")
        if not isinstance(pontuacoes, list) or len(pontuacoes) != len(textos):
            self.fechar()
            raise RuntimeError("A ponte da escolha forçada devolveu {!r} pontuações para "
                               "{} textos.".format(pontuacoes, len(textos)))
        return pontuacoes


def opcoes_da_escolha_forcada(config):
    """As opções de `condicoes.whisper_forcado`, conferidas."""
    opcoes = (config.get("condicoes") or {}).get("whisper_forcado") or {}
    normalizacao, grafias = conferir_opcoes(opcoes.get("normalizacao"), opcoes.get("grafias"))
    return {"ligado": bool(opcoes.get("ligado", False)),
            "normalizacao": normalizacao, "grafias": grafias}


def criar_pontuador(config):
    """Fábrica de produção: o mesmo modelo, idioma, revisão e venv do Genérico."""
    generico = (config.get("condicoes") or {}).get("generico") or {}
    modelo = str(generico.get("modelo", "medium"))
    venv = Path(generico.get("venv", ".venv-whisperx"))
    if not venv.is_absolute():
        venv = RAIZ_PROJETO / venv
    caminho_python = caminho_do_python(venv)
    if not caminho_python.exists():
        raise RuntimeError(
            "O venv dedicado do WhisperX não existe ({}). Instale com:\n{}".format(
                caminho_python, instrucao_de_instalacao(generico.get("venv", ".venv-whisperx"),
                                                        modelo)))
    from src.identidade_dos_pesos import identidade_do_whisper
    esperada = identidade_do_whisper(generico, com_vad=False)
    return PontuadorWhisperForcado(modelo=modelo, idioma=str(generico.get("idioma", "pt")),
                                   caminho_python=caminho_python,
                                   revisao=esperada["revisao"],
                                   identidade_esperada=esperada)

