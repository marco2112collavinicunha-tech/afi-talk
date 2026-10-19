"""Identidade imutável dos pesos: revisão do Hub e sha256, conferidas na carga."""
import hashlib
import re
from collections.abc import Mapping

MARCA = "A PREENCHER"
REVISAO = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

COMANDO_PARA_MEDIR = (
    "na pasta do repositório, com o ambiente virtual ativo e o cache do Hub que as "
    "simulações usaram: medir a revisão e os sha256 do XLSR-53, do Whisper e do VAD "
    "e colar o trecho no config.yaml e em analise/chaves.json")


class IdentidadeAPreencher(RuntimeError):
    """Trava: a identidade dos pesos ainda não foi preenchida."""


class IdentidadeDivergente(RuntimeError):
    """Os pesos carregados não são os registrados."""


def estado(valor, padrao):
    """'fixado', 'a_preencher' ou 'ausente'; ValueError se mal escrito."""
    if valor is None:
        return "ausente"
    if isinstance(valor, str) and MARCA in valor:
        return "a_preencher"
    if isinstance(valor, str) and padrao.match(valor):
        return "fixado"
    raise ValueError("valor de identidade de pesos mal escrito: {!r}".format(valor))


def _recusar_a_preencher(nome):
    raise IdentidadeAPreencher(
        "A identidade dos pesos de {} ainda está marcada a preencher no config.yaml; "
        "o programa não carrega pesos reais sem ela (item técnico de 07/10/2026). "
        "Revisão a preencher pelo pesquisador responsável: {}.".format(nome, COMANDO_PARA_MEDIR))


def identidade_do_xlsr(config, checkpoint):
    """(revisão ou None, sha256 esperado ou None) do checkpoint, conferidos."""
    emb = config.get("embeddings") or {}
    revisoes = emb.get("revisoes") or {}
    hashes = emb.get("sha256_dos_tensores") or {}
    revisao, sha = revisoes.get(checkpoint), hashes.get(checkpoint)
    estados = (estado(revisao, REVISAO), estado(sha, SHA256))
    if "a_preencher" in estados:
        _recusar_a_preencher(checkpoint)
    return (revisao if estados[0] == "fixado" else None,
            sha if estados[1] == "fixado" else None)


def conferir_tensores(extrator, esperado):
    """Compara o sha256 dos tensores carregados com o registrado (se houver)."""
    if esperado is None:
        return "NAO_CONFERIDA: sem sha256 registrado no config"
    medido = extrator.hash_dos_pesos()
    if medido != esperado:
        raise IdentidadeDivergente(
            "Os tensores carregados de {} têm sha256 {} e o registrado é {}. Os pesos "
            "não são os da análise registrada; nada foi calculado.".format(
                extrator.checkpoint, medido, esperado))
    return "CONFERIDA"


def _tem_tensores(objeto):
    """True se `objeto` expõe `state_dict()` com pelo menos um tensor."""
    state_dict = getattr(objeto, "state_dict", None)
    if not callable(state_dict):
        return False
    try:
        return len(state_dict()) > 0
    except Exception:
        return False


def _filhos(objeto):
    """Os objetos dentro de `objeto`, em ordem fixa: atributos e valores de mapeamentos."""
    if isinstance(objeto, (dict, Mapping)):
        try:
            chaves = sorted(objeto.keys(), key=str)
        except Exception:
            return []
        return [objeto[c] for c in chaves]
    if isinstance(objeto, (list, tuple)):
        return list(objeto)
    if hasattr(objeto, "__dict__"):
        return [getattr(objeto, nome, None) for nome in sorted(vars(objeto))]
    return []


def modulo_com_tensores(objeto, profundidade=4, _vistos=None):
    """O primeiro objeto com `state_dict()` não vazio em `objeto` ou dentro dele."""
    if objeto is None or isinstance(objeto, (str, bytes, int, float, bool)):
        return None
    if _vistos is None:
        _vistos = set()
    if id(objeto) in _vistos:
        return None
    _vistos.add(id(objeto))
    if _tem_tensores(objeto):
        return objeto
    if profundidade <= 0:
        return None
    for filho in _filhos(objeto):
        achado = modulo_com_tensores(filho, profundidade - 1, _vistos)
        if achado is not None:
            return achado
    return None


def sha256_do_conjunto(arquivos):
    """Um sha256 para o conjunto {nome: sha256} dos arquivos do modelo."""
    if not isinstance(arquivos, dict) or not arquivos:
        raise ValueError("conjunto de arquivos vazio ou ilegível: {!r}".format(arquivos))
    texto = "\n".join("{}\t{}".format(n, arquivos[n]) for n in sorted(arquivos))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def identidade_do_whisper(opcoes_do_generico, com_vad=True):
    """{revisao, sha256_dos_arquivos, sha256_do_vad} esperados (None = ausente)."""
    opcoes = opcoes_do_generico or {}
    campos = {"revisao": (opcoes.get("revisao"), REVISAO),
              "sha256_dos_arquivos": (opcoes.get("sha256_dos_arquivos"), SHA256)}
    if com_vad:
        campos["sha256_do_vad"] = (opcoes.get("sha256_do_vad"), SHA256)
    saida = {}
    for nome, (valor, padrao) in campos.items():
        e = estado(valor, padrao)
        if e == "a_preencher":
            _recusar_a_preencher("Whisper/VAD ({})".format(nome))
        saida[nome] = valor if e == "fixado" else None
    return saida


def conferir_pesos_do_whisper(pesos, esperado):
    """Confere o bloco `pesos` da saudação da ponte com o registrado."""
    if not any(v is not None for v in esperado.values()):
        return "NAO_CONFERIDA: sem identidade registrada no config"
    if not isinstance(pesos, dict):
        raise IdentidadeDivergente("A ponte não declarou os pesos carregados ({!r}); "
                                   "não há como conferir a identidade.".format(pesos))
    divergencias = []
    if esperado.get("revisao") is not None and pesos.get("commit_carregado") != esperado["revisao"]:
        divergencias.append("commit {!r} (registrado {})".format(
            pesos.get("commit_carregado"), esperado["revisao"]))
    if esperado.get("sha256_dos_arquivos") is not None:
        try:
            medido = sha256_do_conjunto(pesos.get("sha256_dos_arquivos"))
        except ValueError as erro:
            medido = "ilegível ({})".format(erro)
        if medido != esperado["sha256_dos_arquivos"]:
            divergencias.append("arquivos {} (registrado {})".format(
                medido, esperado["sha256_dos_arquivos"]))
    if esperado.get("sha256_do_vad") is not None:
        vad = (pesos.get("vad") or {}) if isinstance(pesos.get("vad"), dict) else {}
        if vad.get("sha256_dos_tensores") != esperado["sha256_do_vad"]:
            divergencias.append("VAD {!r} (registrado {})".format(
                vad.get("sha256_dos_tensores"), esperado["sha256_do_vad"]))
    if divergencias:
        raise IdentidadeDivergente(
            "Os pesos que a ponte do Whisper carregou não são os registrados: {}. "
            "Nada foi transcrito.".format("; ".join(divergencias)))
    return "CONFERIDA"

