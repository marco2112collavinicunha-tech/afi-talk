"""Medida de inconsistência — STI acústico: índice espaço-temporal de (in)consistência da fala."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from inconsistencia.envoltoria import (
    extrair_envoltoria,
    normalizar_amplitude,
    normalizar_tempo,
)


def _carregar_repeticao(item, config):
    """Uma repetição: np.ndarray (já na taxa alvo) ou caminho de arquivo."""
    if isinstance(item, (str, Path)):
        taxa_da_fase1 = int(config["audio"]["taxa_amostragem"])
        taxa_alvo = int(config["inconsistencia"]["taxa_amostragem"])
        if taxa_da_fase1 != taxa_alvo:
            raise ValueError(
                "Config incoerente: audio.taxa_amostragem={} difere de "
                "inconsistencia.taxa_amostragem={} — iguale as duas (a padronização de áudio "
                "entrega os arquivos na primeira; o STI assume a segunda).".format(
                    taxa_da_fase1, taxa_alvo
                )
            )
        from src.preprocess import preprocessar_arquivo

        return preprocessar_arquivo(item, config)

    sinal = np.asarray(item, dtype=np.float64)
    if not np.isfinite(sinal).all():
        raise ValueError(
            "Uma repetição contém valores inválidos (NaN/infinito) — corrija a "
            "gravação antes de calcular o STI."
        )
    return sinal


def calcular_sti(lista_de_audios, config):
    """STI de uma frase de um participante: N repetições → um número."""
    opcoes = config["inconsistencia"]
    n_repeticoes = len(lista_de_audios)
    if n_repeticoes < 2:
        raise ValueError(
            "O STI compara repetições: são necessárias 2+ gravações da MESMA "
            "frase; veio {}.".format(n_repeticoes)
        )

    taxa = int(opcoes["taxa_amostragem"])
    pontos_no_tempo = int(opcoes["pontos_no_tempo"])
    pontos_do_desvio = int(opcoes["pontos_do_desvio"])
    ddof = int(opcoes["ddof"])
    if not 2 <= pontos_do_desvio <= pontos_no_tempo:
        raise ValueError(
            "inconsistencia.pontos_do_desvio deve estar entre 2 e "
            "pontos_no_tempo ({}); veio {}.".format(pontos_no_tempo, pontos_do_desvio)
        )
    if ddof >= n_repeticoes:
        raise ValueError(
            "inconsistencia.ddof={} exige mais repetições do que as {} "
            "recebidas (o desvio-padrão ficaria indefinido).".format(ddof, n_repeticoes)
        )

    envoltorias = []
    for item in lista_de_audios:
        sinal = _carregar_repeticao(item, config)
        envoltoria = extrair_envoltoria(sinal, taxa, opcoes["suavizacao"])
        envoltoria = normalizar_amplitude(envoltoria)
        envoltorias.append(normalizar_tempo(envoltoria, pontos_no_tempo))

    matriz = np.vstack(envoltorias)

    desvio_ponto_a_ponto = matriz.std(axis=0, ddof=ddof)
    blocos = np.array_split(desvio_ponto_a_ponto, pontos_do_desvio)
    desvios_por_bloco = np.array([bloco.mean() for bloco in blocos])

    return float(desvios_por_bloco.sum())

