"""Ponte do WhisperX — o lado filho do subprocesso (roda no .venv-whisperx)."""
import argparse
import base64
import json
import os
import sys

DISPOSITIVO = "cpu"
TIPO_DE_COMPUTO = "int8"
TAMANHO_DO_LOTE = 8
PACOTES_DA_PROCEDENCIA = ("whisperx", "faster-whisper", "ctranslate2", "torch")


def versoes_dos_pacotes(pacotes=PACOTES_DA_PROCEDENCIA):
    """Versão instalada de cada pacote; ausente vira texto com o motivo."""
    from importlib import metadata

    versoes = {}
    for pacote in pacotes:
        try:
            versoes[pacote] = metadata.version(pacote)
        except metadata.PackageNotFoundError:
            versoes[pacote] = "AUSENTE: pacote não instalado neste venv"
    return versoes


def opcoes_efetivas(modelo):
    """As opções de decodificação que o modelo carregado vai usar, como dict."""
    import dataclasses

    opcoes = getattr(modelo, "options", None)
    if opcoes is None:
        return "NAO_LIDO: o modelo carregado não expõe `options`"
    if dataclasses.is_dataclass(opcoes):
        return dataclasses.asdict(opcoes)
    if hasattr(opcoes, "_asdict"):
        return dict(opcoes._asdict())
    return "NAO_LIDO: `options` de tipo {} não é conversível".format(type(opcoes).__name__)


def parametros_do_vad(modelo):
    """Classe do detector de voz e os limiares que o whisperx guarda no modelo."""
    detector = getattr(modelo, "vad_model", None)
    limiares = getattr(modelo, "_vad_params", None)
    return {
        "detector": type(detector).__name__ if detector is not None
        else "NAO_LIDO: o modelo carregado não expõe `vad_model`",
        "limiares": dict(limiares) if isinstance(limiares, dict)
        else "NAO_LIDO: o modelo carregado não expõe `_vad_params`",
    }


def _busca_de_modulo():
    """`modulo_com_tensores` de src/identidade_dos_pesos.py, carregado pelo caminho."""
    try:
        from src.identidade_dos_pesos import modulo_com_tensores
        return modulo_com_tensores
    except ImportError:
        import importlib.util
        caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "identidade_dos_pesos.py")
        spec = importlib.util.spec_from_file_location("identidade_dos_pesos_da_ponte", caminho)
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        return modulo.modulo_com_tensores


def _modulo_com_tensores(objeto, profundidade=4):
    """O primeiro objeto com `state_dict()` não vazio em `objeto` ou dentro dele."""
    return _busca_de_modulo()(objeto, profundidade)


def identidade_do_vad(modelo):
    """Classe do detector de voz e sha256 dos tensores que ele carregou."""
    import hashlib

    import torch

    detector = getattr(modelo, "vad_model", None)
    modulo = _modulo_com_tensores(detector)
    if modulo is None:
        return {"classe": type(detector).__name__ if detector is not None else None,
                "sha256_dos_tensores": "NAO_LIDO: nenhum módulo com tensores no VAD"}
    resumo = hashlib.sha256()
    for nome, tensor in sorted(modulo.state_dict().items()):
        dado = tensor.detach().to("cpu").contiguous()
        resumo.update(nome.encode("utf-8"))
        resumo.update(str(dado.dtype).encode("utf-8"))
        resumo.update(str(tuple(dado.shape)).encode("utf-8"))
        if dado.numel():
            resumo.update(dado.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"classe": type(detector).__name__, "modulo": type(modulo).__name__,
            "sha256_dos_tensores": resumo.hexdigest()}


def identidade_dos_pesos(caminho_do_modelo, revisao_pedida=None):
    """Quais pesos a ponte carregou: commit do snapshot e sha256 dos arquivos."""
    import hashlib

    pasta = os.path.realpath(str(caminho_do_modelo))
    pai = os.path.basename(os.path.dirname(pasta))
    commit = os.path.basename(pasta) if pai == "snapshots" else (
        "NAO_LIDO: o modelo não veio de um snapshot do cache do Hub"
    )
    arquivos = {}
    for nome in sorted(os.listdir(pasta)):
        alvo = os.path.join(pasta, nome)
        if not os.path.isfile(alvo):
            continue
        resumo = hashlib.sha256()
        with open(alvo, "rb") as fluxo:
            for bloco in iter(lambda: fluxo.read(1 << 20), b""):
                resumo.update(bloco)
        arquivos[nome] = resumo.hexdigest()
    return {
        "revisao_pedida": revisao_pedida if revisao_pedida is not None else (
            "NAO_FIXADA: sem --revisao, vale a versão corrente do Hub/cache"
        ),
        "commit_carregado": commit,
        "sha256_dos_arquivos": arquivos or "NAO_LIDO: a pasta do modelo não tem arquivos",
    }


def resolver_modelo(modelo, revisao=None, baixar=None):
    """Pasta local do modelo, na revisão pedida (None = a corrente)."""
    if os.path.isdir(str(modelo)):
        return str(modelo)
    if baixar is None:
        from faster_whisper.utils import download_model as baixar
    return baixar(str(modelo), revision=revisao)


def principal():
    parser = argparse.ArgumentParser(description="Ponte WhisperX (lado filho)")
    parser.add_argument("--modelo", default="medium")
    parser.add_argument("--idioma", default="pt")
    parser.add_argument(
        "--revisao", default=None,
        help="commit do Hub a carregar; ausente = versão corrente, declarada",
    )
    parser.add_argument(
        "--aquecer", action="store_true",
        help="só carrega/baixa o modelo, confirma e sai (sem servir transcrições)",
    )
    argumentos = parser.parse_args()

    protocolo = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    import numpy as np
    import whisperx

    pasta_do_modelo = resolver_modelo(argumentos.modelo, argumentos.revisao)
    modelo = whisperx.load_model(
        pasta_do_modelo, device=DISPOSITIVO, compute_type=TIPO_DE_COMPUTO,
        language=argumentos.idioma,
    )
    saudacao = {
        "pronto": True,
        "modelo": argumentos.modelo,
        "versoes": versoes_dos_pacotes(),
        "pesos": dict(identidade_dos_pesos(pasta_do_modelo, argumentos.revisao),
                      vad=identidade_do_vad(modelo)),
        "decodificacao": {
            "dispositivo": DISPOSITIVO,
            "tipo_de_computo": TIPO_DE_COMPUTO,
            "tamanho_do_lote": TAMANHO_DO_LOTE,
            "idioma": argumentos.idioma,
            "opcoes_do_modelo": opcoes_efetivas(modelo),
            "vad": parametros_do_vad(modelo),
        },
    }
    protocolo.write(json.dumps(saudacao, ensure_ascii=False, default=str) + "\n")
    if argumentos.aquecer:
        return 0

    for linha in sys.stdin:
        linha = linha.strip()
        if not linha:
            continue
        identificador = None
        try:
            pedido = json.loads(linha)
            identificador = pedido.get("id")
            taxa = int(pedido["taxa"])
            if taxa != 16000:
                raise ValueError(
                    "a ponte espera áudio a 16000 Hz (saída da padronização de áudio); veio {} Hz".format(taxa)
                )
            sinal = np.frombuffer(
                base64.b64decode(pedido["sinal_b64"]), dtype=np.float32
            ).copy()
            if sinal.size == 0:
                raise ValueError("o sinal chegou vazio à ponte")
            resultado = modelo.transcribe(sinal, batch_size=TAMANHO_DO_LOTE)
            texto = " ".join(
                segmento.get("text", "").strip()
                for segmento in resultado.get("segments", [])
            ).strip()
            resposta = {"id": identificador, "texto": texto}
        except Exception as erro:
            resposta = {
                "id": identificador,
                "erro": "{}: {}".format(type(erro).__name__, erro),
            }
        protocolo.write(json.dumps(resposta, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(principal())

