"""Transcrição e casamento com frases-alvo: o caminho técnico do genérico."""
from __future__ import annotations

import atexit
import base64
import difflib
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import unicodedata
from pathlib import Path

import numpy as np

from src.config import RAIZ_PROJETO

_FIM_DO_FLUXO = object()


def caminho_do_python(venv):
    """Caminho do interpretador dentro de um venv, portável entre sistemas."""
    venv = Path(venv)
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def instrucao_de_instalacao(venv, modelo):
    """Instruções de instalação do venv da ponte, no dialeto do sistema atual."""
    if os.name == "nt":
        return (
            "    py -3.12 -m venv {venv}\n"
            "    {venv}\\Scripts\\pip install whisperx==3.8.6\n"
            "    {venv}\\Scripts\\python src\\ponte_whisperx.py --modelo {modelo} "
            "--aquecer".format(venv=venv, modelo=modelo)
        )
    return (
        "    python3.12 -m venv {venv}\n"
        "    {venv}/bin/pip install -r requirements-whisperx.lock.txt\n"
        "    {venv}/bin/python src/ponte_whisperx.py --modelo {modelo} "
        "--aquecer".format(venv=venv, modelo=modelo)
    )


def _bombear_saida(fluxo, fila):
    """Lê o stdout do filho linha a linha e empilha na fila, até o eof."""
    try:
        for linha in iter(fluxo.readline, ""):
            fila.put(linha)
    except (ValueError, OSError):
        pass
    finally:
        fila.put(_FIM_DO_FLUXO)

AVISO_FALA_TIPICA = (
    "Validação com fala típica/sintética confere o ENCANAMENTO do caminho de "
    "transcrição — não o desempenho em fala atípica (AFI), que é o cenário "
    "real do estudo."
)


def normalizar_texto(texto):
    """Normaliza texto para o casamento: minúsculas, sem acentos, sem pontuação."""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


SEM_CORRESPONDENCIA = "__sem_correspondencia__"

FALHA_DO_TRANSCRITOR = "__falha_do_transcritor__"


def taxa_de_erro_de_palavra(referencia, hipotese):
    """WER clássica: (substituições + inserções + remoções) / palavras da referência."""
    ref = normalizar_texto(referencia).split()
    hip = normalizar_texto(hipotese).split()
    if not ref:
        return None
    anterior = list(range(len(hip) + 1))
    for i, palavra_ref in enumerate(ref, start=1):
        atual = [i]
        for j, palavra_hip in enumerate(hip, start=1):
            custo = 0 if palavra_ref == palavra_hip else 1
            atual.append(min(anterior[j] + 1, atual[j - 1] + 1, anterior[j - 1] + custo))
        anterior = atual
    return float(anterior[-1]) / len(ref)


def _similaridade_por_caracteres(transcrito, frase):
    """A régua padrão: `SequenceMatcher` sobre caracteres sem acento."""
    return difflib.SequenceMatcher(None, transcrito, frase).ratio()


def _similaridade_por_palavras(transcrito, frase):
    """Régua alternativa: 1 − WER, limitada a [0, 1]."""
    wer = taxa_de_erro_de_palavra(frase, transcrito)
    if wer is None:
        return 0.0
    return max(0.0, 1.0 - wer)


REGUAS_DE_CASAMENTO = {
    "caracteres": _similaridade_por_caracteres,
    "palavras": _similaridade_por_palavras,
}
REGUA_PADRAO = "caracteres"


def obter_regua(nome):
    """A régua pedida, com erro nomeado se não existir."""
    if nome not in REGUAS_DE_CASAMENTO:
        raise ValueError(
            "Régua de casamento desconhecida: {!r}. Disponíveis: {}.".format(
                nome, sorted(REGUAS_DE_CASAMENTO)
            )
        )
    return REGUAS_DE_CASAMENTO[nome]


def casar_com_frase_alvo(texto, frases_alvo, regua=REGUA_PADRAO):
    """Casa um texto transcrito com a frase-alvo mais próxima do conjunto fechado."""
    if not frases_alvo:
        raise ValueError(
            "O conjunto de frases-alvo está vazio — defina a seção frases_alvo "
            "no config.yaml (rótulo → frase falada)."
        )
    normalizadas = {}
    for rotulo, frase in frases_alvo.items():
        frase_normalizada = normalizar_texto(frase)
        if not frase_normalizada:
            raise ValueError(
                "A frase-alvo de {!r} ({!r}) normaliza para vazio — use uma "
                "frase com letras/dígitos no config.yaml.".format(rotulo, frase)
            )
        normalizadas[str(rotulo)] = frase_normalizada

    transcrito = normalizar_texto(texto)
    if not transcrito:
        ranking_vazio = [(SEM_CORRESPONDENCIA, 1.0)] + [
            (rotulo, 0.0) for rotulo in sorted(normalizadas)
        ]
        return SEM_CORRESPONDENCIA, 0.0, ranking_vazio
    medir = obter_regua(regua)
    ranking = sorted(
        (
            (rotulo, float(medir(transcrito, frase)))
            for rotulo, frase in normalizadas.items()
        ),
        key=lambda par: (-par[1], par[0]),
    )
    rotulo, similaridade = ranking[0]
    return rotulo, similaridade, ranking


def frases_alvo_para(config, rotulos, grupo=None):
    """Monta {rotulo: frase} para os rótulos dados + todo o conjunto do config."""
    definidas = config.get("frases_alvo") or {}
    frases = {}
    for rotulo, frase in definidas.items():
        if not isinstance(rotulo, str):
            raise ValueError(
                "frases_alvo tem rótulo não-textual {!r} — no YAML, escreva o "
                "rótulo entre aspas.".format(rotulo)
            )
        if not isinstance(frase, str) or not frase.strip():
            raise ValueError(
                "frases_alvo.{} está vazia no config.yaml ({!r}) — preencha a "
                "frase falada ou remova a chave.".format(rotulo, frase)
            )
        frases[rotulo] = frase
    sem_frase = sorted({str(r) for r in rotulos} - set(frases))
    if sem_frase and grupo is not None and str(grupo) != "engenharia":
        raise ValueError(
            "Rótulo(s) sem frase em frases_alvo do config.yaml: {}. O participante "
            "é do grupo {!r}, e fora do grupo 'engenharia' o rótulo NÃO vira "
            "frase por conta própria: o Genérico seria pontuado contra texto "
            "errado. Escreva a frase falada de cada rótulo em frases_alvo."
            .format(", ".join(sem_frase), str(grupo))
        )
    for rotulo in sem_frase:
        frases[rotulo] = rotulo.replace("_", " ")
    return frases


class TranscritorWhisperX:
    """O lado pai da ponte: fala com src/ponte_whisperx.py por linhas JSON."""

    def __init__(self, modelo="medium", idioma="pt", caminho_python=None,
                 caminho_ponte=None, tempo_maximo_carga=3600, tempo_maximo_resposta=600,
                 revisao=None, identidade_esperada=None):
        self.modelo = str(modelo)
        self.revisao = None if revisao is None else str(revisao)
        self.identidade_esperada = dict(identidade_esperada or {})
        self.idioma = str(idioma)
        self.nome = "whisperx_{}".format(self.modelo)
        self.caminho_python = Path(caminho_python) if caminho_python is not None else (
            caminho_do_python(RAIZ_PROJETO / ".venv-whisperx")
        )
        self.caminho_ponte = Path(caminho_ponte) if caminho_ponte is not None else (
            Path(__file__).with_name("ponte_whisperx.py")
        )
        self.tempo_maximo_carga = tempo_maximo_carga
        self.tempo_maximo_resposta = tempo_maximo_resposta
        self._processo = None
        self._log = None
        self._contador = 0
        self._fila = None
        self._leitor = None
        self.procedencia = {
            "modelo": self.modelo,
            "idioma": self.idioma,
            "revisao_pedida": self.revisao,
            "estado": "NAO_LIDA: a ponte ainda não foi iniciada (partida preguiçosa)",
        }

    def _cauda_do_log(self):
        if self._log is None:
            return ""
        try:
            self._log.flush()
            self._log.seek(max(0, self._log.tell() - 2000))
            return self._log.read().strip()
        except Exception:
            return ""

    def _ler_linha(self, tempo_maximo, contexto):
        """Espera uma linha do filho, com timeout — portável entre sistemas."""
        try:
            linha = self._fila.get(timeout=tempo_maximo)
        except queue.Empty:
            cauda = self._cauda_do_log()
            self.fechar()
            raise RuntimeError(
                "A ponte do WhisperX não respondeu em {}s ({}). Log: {}".format(
                    tempo_maximo, contexto, cauda or "(vazio)"
                )
            )
        if linha is _FIM_DO_FLUXO:
            cauda = self._cauda_do_log()
            self.fechar()
            raise RuntimeError(
                "A ponte do WhisperX encerrou inesperadamente ({}). Log: {}".format(
                    contexto, cauda or "(vazio)"
                )
            )
        return linha

    def _garantir_processo(self):
        if self._processo is not None and self._processo.poll() is None:
            return
        if not self.caminho_python.exists():
            raise RuntimeError(
                "O venv dedicado do WhisperX não existe ({}). Instale com:\n"
                "{}\n(ver docs/REPRODUZIR.md, seção sobre a ponte de transcrição).".format(
                    self.caminho_python,
                    instrucao_de_instalacao(".venv-whisperx", self.modelo),
                )
            )
        self._log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        comando = [str(self.caminho_python), str(self.caminho_ponte),
                   "--modelo", self.modelo, "--idioma", self.idioma]
        if self.revisao is not None:
            comando += ["--revisao", self.revisao]
        self._processo = subprocess.Popen(
            comando,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, bufsize=1,
        )
        self._fila = queue.Queue()
        self._leitor = threading.Thread(
            target=_bombear_saida, args=(self._processo.stdout, self._fila), daemon=True
        )
        self._leitor.start()
        atexit.register(self.fechar)

        linha = self._ler_linha(self.tempo_maximo_carga, "carregando o modelo")
        try:
            saudacao = json.loads(linha)
        except json.JSONDecodeError:
            saudacao = {}
        if not saudacao.get("pronto"):
            cauda = self._cauda_do_log()
            self.fechar()
            raise RuntimeError(
                "A ponte do WhisperX não confirmou a carga do modelo "
                "(resposta: {!r}). Log: {}".format(linha.strip(), cauda or "(vazio)")
            )
        self.procedencia = self._procedencia_da_saudacao(saudacao)
        from src.identidade_dos_pesos import IdentidadeDivergente, conferir_pesos_do_whisper
        try:
            self.procedencia["identidade_conferida"] = conferir_pesos_do_whisper(
                saudacao.get("pesos"), self.identidade_esperada)
        except IdentidadeDivergente:
            self.fechar()
            raise

    def _procedencia_da_saudacao(self, saudacao):
        """Versões e parâmetros de decodificação que a ponte declarou ao subir."""
        procedencia = {"modelo": self.modelo, "idioma": self.idioma,
                       "revisao_pedida": self.revisao}
        for campo in ("versoes", "pesos", "decodificacao"):
            if campo in saudacao:
                procedencia[campo] = saudacao[campo]
            else:
                procedencia[campo] = (
                    "NAO_INFORMADO: a ponte não declarou '{}' na saudação".format(campo)
                )
        procedencia["estado"] = "LIDA da saudação da ponte"
        return procedencia

    def fechar(self):
        """Encerra o processo da ponte (idempotente); a próxima transcrição sobe um processo novo."""
        processo, self._processo = self._processo, None
        if processo is not None and processo.poll() is None:
            processo.terminate()
            try:
                processo.wait(timeout=5)
            except subprocess.TimeoutExpired:
                processo.kill()
        self._leitor = None
        self._fila = None
        log, self._log = self._log, None
        if log is not None:
            log.close()

    def transcrever(self, sinal, taxa):
        self._garantir_processo()
        s = np.ascontiguousarray(np.asarray(sinal, dtype=np.float32))
        self._contador += 1
        pedido = {
            "id": self._contador,
            "taxa": int(taxa),
            "sinal_b64": base64.b64encode(s.tobytes()).decode("ascii"),
        }
        self._processo.stdin.write(json.dumps(pedido) + "\n")
        self._processo.stdin.flush()

        linha = self._ler_linha(self.tempo_maximo_resposta, "transcrevendo")
        try:
            resposta = json.loads(linha)
        except json.JSONDecodeError:
            self.fechar()
            raise RuntimeError(
                "A ponte do WhisperX respondeu fora do protocolo: {!r}".format(linha.strip())
            )
        if resposta.get("id") != pedido["id"]:
            self.fechar()
            raise RuntimeError(
                "A ponte do WhisperX respondeu fora de ordem (pedido {}, "
                "resposta {}).".format(pedido["id"], resposta.get("id"))
            )
        if "erro" in resposta:
            raise ValueError(
                "O transcritor rejeitou o áudio: {}".format(resposta["erro"])
            )
        texto = resposta.get("texto")
        if not isinstance(texto, str):
            self.fechar()
            raise RuntimeError(
                "A ponte do WhisperX devolveu resposta sem texto: {!r}".format(resposta)
            )
        return texto


def criar_transcritor(config):
    """Fábrica de produção: WhisperX (padrão medium) via ponte por subprocesso."""
    opcoes = (config.get("condicoes") or {}).get("generico") or {}
    modelo = str(opcoes.get("modelo", "medium"))
    idioma = str(opcoes.get("idioma", "pt"))
    venv = Path(opcoes.get("venv", ".venv-whisperx"))
    if not venv.is_absolute():
        venv = RAIZ_PROJETO / venv
    caminho_python = caminho_do_python(venv)
    if not caminho_python.exists():
        raise RuntimeError(
            "O venv dedicado do WhisperX não existe ({}). Instale com:\n"
            "{}\n(ver docs/REPRODUZIR.md, seção sobre a ponte de transcrição). Para "
            "engenharia/testes, injete um mock com transcrever(sinal, taxa) "
            "-> str.".format(
                caminho_python,
                instrucao_de_instalacao(opcoes.get("venv", ".venv-whisperx"), modelo),
            )
        )
    from src.identidade_dos_pesos import identidade_do_whisper
    esperada = identidade_do_whisper(opcoes, com_vad=True)
    return TranscritorWhisperX(modelo=modelo, idioma=idioma, caminho_python=caminho_python,
                               revisao=esperada["revisao"], identidade_esperada=esperada)

