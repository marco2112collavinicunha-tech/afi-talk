"""Medida de inconsistência: variação de duração e de intensidade entre repetições."""
from __future__ import annotations

from pathlib import Path

import numpy as np

RMS_MINIMO = 1e-12

AVISO_DESCRITIVO = (
    "DESCRITIVO: publicado ao lado do STI, sem hipótese nem "
    "veredito. Intensidade em dBFS (não dB SPL), antes da normalização de "
    "volume; CV da duração e desvio em dB são escolha PROVISÓRIA."
)


def _carregar_sem_normalizar(item, config):
    if isinstance(item, (str, Path)):
        from src.preprocess import preprocessar_arquivo

        return np.asarray(preprocessar_arquivo(item, config, normalizar=False),
                          dtype=np.float64)
    sinal = np.asarray(item, dtype=np.float64)
    if sinal.ndim != 1 or sinal.size == 0 or not np.isfinite(sinal).all():
        raise ValueError(
            "Uma repetição não é sinal mono 1-D finito e não vazio — corrija a "
            "gravação antes de medir duração e intensidade."
        )
    return sinal


def medidas_da_repeticao(item, config):
    """`{"duracao_s", "intensidade_dbfs"}` de uma repetição."""
    taxa = int(config["inconsistencia"]["taxa_amostragem"])
    sinal = _carregar_sem_normalizar(item, config)
    rms = float(np.sqrt(np.mean(sinal ** 2)))
    return {
        "duracao_s": float(sinal.size / taxa),
        "intensidade_dbfs": float(20.0 * np.log10(max(rms, RMS_MINIMO))),
    }


def variacao_da_frase(lista_de_audios, config):
    """Variação de duração e de intensidade entre as N repetições de uma frase."""
    n = len(lista_de_audios)
    ddof = int(config["inconsistencia"]["ddof"])
    if n < 2 or ddof >= n:
        raise ValueError(
            "A variação entre repetições exige 2+ gravações da MESMA frase (e "
            "mais do que inconsistencia.ddof={}); veio {}.".format(ddof, n)
        )
    medidas = [medidas_da_repeticao(item, config) for item in lista_de_audios]
    duracoes = np.array([m["duracao_s"] for m in medidas])
    niveis = np.array([m["intensidade_dbfs"] for m in medidas])
    dp_duracao = float(duracoes.std(ddof=ddof))
    return {
        "n_repeticoes": n,
        "duracoes_s": duracoes.tolist(),
        "intensidades_dbfs": niveis.tolist(),
        "dp_duracao_s": dp_duracao,
        "cv_duracao": float(dp_duracao / duracoes.mean()),
        "dp_intensidade_db": float(niveis.std(ddof=ddof)),
    }


def variacao_por_frase(gravacoes_por_frase, config):
    """`{frase: variacao_da_frase(...)}` — mesma entrada de `stis_por_frase`."""
    if not gravacoes_por_frase:
        raise ValueError("Não há frases para medir a variação (dict vazio).")
    return {frase: variacao_da_frase(gravacoes_por_frase[frase], config)
            for frase in sorted(gravacoes_por_frase)}


def agregar_variacao(por_frase, config):
    """Índice do participante das duas medidas, pelo método do STI (`inconsistencia.agregacao`), extensão provisória."""
    if not por_frase:
        raise ValueError("Não há frases para agregar a variação (dict vazio).")
    metodo = config["inconsistencia"]["agregacao"]
    funcoes = {"media": np.mean, "mediana": np.median}
    if metodo not in funcoes:
        raise ValueError(
            "Método de agregação desconhecido: {!r} — use 'media' ou 'mediana' "
            "(inconsistencia.agregacao).".format(metodo))
    f = funcoes[metodo]
    return {
        "agregacao": metodo,
        "cv_duracao": float(f([v["cv_duracao"] for v in por_frase.values()])),
        "dp_intensidade_db": float(
            f([v["dp_intensidade_db"] for v in por_frase.values()])),
        "n_frases": len(por_frase),
        "aviso": AVISO_DESCRITIVO,
    }

