"""Medida de inconsistência — envoltória de amplitude: extração, suavização e normalizações."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, hilbert, sosfiltfilt

TOLERANCIA_DE_ENVOLTORIA_CONSTANTE = 1e-9


def extrair_envoltoria(sinal, taxa, opcoes):
    """Extrai a envoltória de amplitude suavizada de uma repetição."""
    sinal = np.asarray(sinal, dtype=np.float64)
    if sinal.ndim != 1:
        raise ValueError(
            "A envoltória espera sinal mono 1-D (saída da padronização de áudio); veio "
            "forma {}.".format(sinal.shape)
        )
    if sinal.size < 8:
        raise ValueError(
            "Sinal curto demais para extrair a envoltória ({} amostras).".format(sinal.size)
        )

    metodo = opcoes["metodo"]
    if metodo in ("hilbert_butterworth", "meia_onda"):
        if metodo == "meia_onda":
            bruta = np.maximum(sinal, 0.0)
        else:
            bruta = np.abs(hilbert(sinal))
        corte = float(opcoes["frequencia_de_corte_hz"])
        if not 0.0 < corte < taxa / 2.0:
            raise ValueError(
                "inconsistencia.suavizacao.frequencia_de_corte_hz deve estar "
                "entre 0 e a metade da taxa ({} Hz); veio {}.".format(taxa / 2.0, corte)
            )
        ordem = int(opcoes["ordem_do_filtro"])
        sos = butter(ordem, corte, btype="low", fs=taxa, output="sos")
        padlen = 3 * (2 * len(sos) + 1)
        if sinal.size <= padlen:
            raise ValueError(
                "Sinal curto demais ({} amostras) para o filtro de ordem {} "
                "(precisa de mais de {} amostras). Use um sinal mais longo, "
                "reduza inconsistencia.suavizacao.ordem_do_filtro ou o método "
                "'rms'.".format(sinal.size, ordem, padlen)
            )
        return sosfiltfilt(sos, bruta)

    if metodo == "rms":
        janela = max(1, int(round(float(opcoes["janela_rms_ms"]) * taxa / 1000.0)))
        nucleo = np.ones(janela) / janela
        energia = np.convolve(sinal**2, nucleo, mode="same")
        return np.sqrt(energia)

    raise ValueError(
        "Método de suavização desconhecido: {!r} — use 'meia_onda', "
        "'hilbert_butterworth' ou 'rms' (inconsistencia.suavizacao.metodo).".format(metodo)
    )


def normalizar_amplitude(envoltoria):
    """z-score da envoltória: remove diferenças de volume entre repetições."""
    envoltoria = np.asarray(envoltoria, dtype=np.float64)
    desvio = float(envoltoria.std())
    escala = abs(float(envoltoria.mean()))
    variacao_relativa = desvio / escala if escala > 0.0 else desvio
    if variacao_relativa <= TOLERANCIA_DE_ENVOLTORIA_CONSTANTE:
        warnings.warn(
            "Envoltória constante (variação relativa {:.1e}, abaixo de {:.0e} — "
            "áudio silencioso, tom sustentado ou ruído estacionário?): z-score "
            "indefinido; usando envoltória de zeros no lugar.".format(
                variacao_relativa, TOLERANCIA_DE_ENVOLTORIA_CONSTANTE
            ),
            stacklevel=2,
        )
        return np.zeros_like(envoltoria)
    return (envoltoria - envoltoria.mean()) / desvio


def normalizar_tempo(envoltoria, pontos):
    """Reamostra a envoltória para `pontos` amostras por spline cúbico."""
    envoltoria = np.asarray(envoltoria, dtype=np.float64)
    if envoltoria.size < 4:
        raise ValueError(
            "Envoltória curta demais para o spline cúbico ({} pontos; "
            "mínimo 4).".format(envoltoria.size)
        )
    tempo_original = np.linspace(0.0, 1.0, envoltoria.size)
    tempo_normalizado = np.linspace(0.0, 1.0, int(pontos))
    return CubicSpline(tempo_original, envoltoria)(tempo_normalizado)

