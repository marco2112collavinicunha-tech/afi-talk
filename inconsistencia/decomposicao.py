"""Decomposição da variabilidade da envoltória em tempo e forma, atrás de chave."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from inconsistencia.envoltoria import (
    extrair_envoltoria,
    normalizar_amplitude,
    normalizar_tempo,
)
from inconsistencia.sti import _carregar_repeticao

N_BASES_PADRAO = 10
ORDEM_DAS_BASES = 4
PASSOS_DO_ALVO = 2
LIMIAR_RELATIVO_DE_PAUSA_PADRAO = 0.1
DURACAO_MINIMA_DE_PAUSA_MS_PADRAO = 150.0
ESCALAS_DO_GROSSO_AO_FINO = (0.08, 0.04, 0.02, 0.01, 0.0)
_PESO_DA_ANCORA = 1e-6


class DecomposicaoDesligada(RuntimeError):
    """A chave da decomposição tempo × forma está desligada (padrão)."""


def ligada_no_config(config):
    """A chave `inconsistencia.decomposicao_tempo_forma_ligada`."""
    return ((config.get("inconsistencia") or {})
            .get("decomposicao_tempo_forma_ligada", False)) is True


def _exigir_chave(ligada):
    if ligada is not True:
        raise DecomposicaoDesligada(
            "A decomposição tempo × forma está desligada. É análise secundária "
            "proposta; a flexibilidade da deformação depende da escolha do "
            "pesquisador responsável. "
            "Para uso de engenharia, passe ligada=True explicitamente."
        )


def _base_bspline(pontos, n_bases, ordem=ORDEM_DAS_BASES):
    """Matrizes (pontos × n_bases) da base B-spline e da sua derivada em [0, 1]."""
    grau = ordem - 1
    if n_bases < ordem:
        raise ValueError(
            "n_bases={} é menor que a ordem das B-splines ({}).".format(n_bases, ordem)
        )
    internos = np.linspace(0.0, 1.0, n_bases - grau + 1)[1:-1]
    nos = np.concatenate([np.zeros(ordem), internos, np.ones(ordem)])
    t = np.linspace(0.0, 1.0, pontos)
    base = np.empty((pontos, n_bases))
    derivada = np.empty((pontos, n_bases))
    for k in range(n_bases):
        coef = np.zeros(n_bases)
        coef[k] = 1.0
        spline = BSpline(nos, coef, grau, extrapolate=False)
        base[:, k] = np.nan_to_num(spline(t))
        derivada[:, k] = np.nan_to_num(spline.derivative()(t))
    base[-1, :] = 0.0
    base[-1, -1] = 1.0
    derivada[-1, :] = derivada[-2, :]
    return t, base, derivada


def _deformacao(coeficientes, base, t):
    """h(t) monótona com h(0)=0 e h(1)=1, a partir dos coeficientes de W."""
    w = base @ coeficientes
    integrando = np.exp(w - w.max())
    passo = np.diff(t)
    acumulado = np.concatenate([[0.0], np.cumsum(0.5 * (integrando[1:] + integrando[:-1]) * passo)])
    return acumulado / acumulado[-1]


def _registrar_uma(curva, alvo, t, base, derivada, lambda_suavidade, inicio,
                   escalas=ESCALAS_DO_GROSSO_AO_FINO):
    c = np.asarray(inicio, dtype=np.float64)
    for sigma in escalas:
        pontos_sigma = sigma * (t.size - 1)
        if pontos_sigma > 0:
            c = _registrar_numa_escala(
                gaussian_filter1d(curva, pontos_sigma, mode="nearest"),
                gaussian_filter1d(alvo, pontos_sigma, mode="nearest"),
                t, base, derivada, lambda_suavidade, c,
            )
        else:
            c = _registrar_numa_escala(curva, alvo, t, base, derivada, lambda_suavidade, c)
    return c


def _registrar_numa_escala(curva, alvo, t, base, derivada, lambda_suavidade, inicio):
    def custo(c):
        h = _deformacao(c, base, t)
        deformada = np.interp(h, t, curva)
        ajuste = np.trapezoid((deformada - alvo) ** 2, t)
        curvatura = derivada @ c
        penalidade = lambda_suavidade * np.trapezoid(curvatura ** 2, t)
        return ajuste + penalidade + _PESO_DA_ANCORA * float(c.mean()) ** 2

    resultado = minimize(custo, inicio, method="L-BFGS-B", options={"maxiter": 500})
    return resultado.x


def _desvio_em_blocos(matriz, pontos_do_desvio, ddof):
    """Os passos 5–6 do STI sobre uma matriz (N × pontos): soma das médias por bloco."""
    desvio = matriz.std(axis=0, ddof=ddof)
    blocos = np.array_split(desvio, pontos_do_desvio)
    return float(np.sum([b.mean() for b in blocos]))


def pausas_internas(envoltoria_bruta, taxa, limiar_relativo, duracao_minima_ms):
    """Pausas internas de uma repetição, sobre a envoltória antes do z-score."""
    env = np.asarray(envoltoria_bruta, dtype=np.float64)
    referencia = float(np.percentile(env, 95))
    if referencia <= 0.0:
        return []
    baixo = env < limiar_relativo * referencia
    duracoes = []
    n = env.size
    i = 0
    while i < n:
        if not baixo[i]:
            i += 1
            continue
        j = i
        while j < n and baixo[j]:
            j += 1
        toca_borda = i == 0 or j == n
        duracao_ms = 1000.0 * (j - i) / taxa
        if not toca_borda and duracao_ms >= duracao_minima_ms:
            duracoes.append(duracao_ms)
        i = j
    return duracoes


def decompor_tempo_forma(
    lista_de_audios,
    config,
    *,
    lambda_suavidade,
    ligada=None,
    n_bases=N_BASES_PADRAO,
    passos=PASSOS_DO_ALVO,
    limiar_relativo_de_pausa=LIMIAR_RELATIVO_DE_PAUSA_PADRAO,
    duracao_minima_de_pausa_ms=DURACAO_MINIMA_DE_PAUSA_MS_PADRAO,
):
    """N repetições da mesma frase → índices temporal e de forma, STI sem registro, pausas."""
    _exigir_chave(ligada_no_config(config) if ligada is None else ligada)
    lambda_suavidade = float(lambda_suavidade)
    if not np.isfinite(lambda_suavidade) or lambda_suavidade < 0.0:
        raise ValueError(
            "lambda_suavidade deve ser um número finito ≥ 0; veio {}.".format(lambda_suavidade)
        )
    if int(passos) < 1:
        raise ValueError("passos do alvo deve ser ≥ 1; veio {}.".format(passos))
    opcoes = config["inconsistencia"]
    n_repeticoes = len(lista_de_audios)
    if n_repeticoes < 2:
        raise ValueError(
            "A decomposição compara repetições: são necessárias 2+ gravações da MESMA "
            "frase; veio {}.".format(n_repeticoes)
        )
    taxa = int(opcoes["taxa_amostragem"])
    pontos_no_tempo = int(opcoes["pontos_no_tempo"])
    pontos_do_desvio = int(opcoes["pontos_do_desvio"])
    ddof = int(opcoes["ddof"])
    if ddof >= n_repeticoes:
        raise ValueError(
            "inconsistencia.ddof={} exige mais repetições do que as {} recebidas.".format(
                ddof, n_repeticoes
            )
        )

    curvas = []
    pausas = []
    for item in lista_de_audios:
        sinal = _carregar_repeticao(item, config)
        bruta = extrair_envoltoria(sinal, taxa, opcoes["suavizacao"])
        pausas.append(
            pausas_internas(bruta, taxa, limiar_relativo_de_pausa, duracao_minima_de_pausa_ms)
        )
        curvas.append(normalizar_tempo(normalizar_amplitude(bruta), pontos_no_tempo))
    matriz = np.vstack(curvas)

    t, base, derivada = _base_bspline(pontos_no_tempo, int(n_bases))
    coeficientes = np.zeros((n_repeticoes, int(n_bases)))
    alvo = matriz.mean(axis=0)
    alinhadas = matriz.copy()
    for _ in range(int(passos)):
        for i in range(n_repeticoes):
            coeficientes[i] = _registrar_uma(
                matriz[i], alvo, t, base, derivada, lambda_suavidade, coeficientes[i]
            )
        deformacoes = np.vstack([_deformacao(c, base, t) for c in coeficientes])
        alinhadas = np.vstack([np.interp(deformacoes[i], t, matriz[i]) for i in range(n_repeticoes)])
        alvo = alinhadas.mean(axis=0)

    deformacoes = np.vstack([_deformacao(c, base, t) for c in coeficientes])
    indice_temporal = float(np.mean((deformacoes - t).std(axis=0, ddof=ddof)))
    indice_de_forma = _desvio_em_blocos(alinhadas, pontos_do_desvio, ddof)
    sti_sem_registro = _desvio_em_blocos(matriz, pontos_do_desvio, ddof)

    contagens = [len(p) for p in pausas]
    totais = [float(sum(p)) for p in pausas]
    return {
        "indice_temporal": indice_temporal,
        "indice_de_forma": indice_de_forma,
        "sti_sem_registro": sti_sem_registro,
        "pausas": {
            "contagem_por_repeticao": contagens,
            "duracoes_ms_por_repeticao": [list(p) for p in pausas],
            "duracao_total_ms_por_repeticao": totais,
            "limiar_relativo": float(limiar_relativo_de_pausa),
            "duracao_minima_ms": float(duracao_minima_de_pausa_ms),
            "fonte_do_limiar": "SEM FONTE (descritor; depende das fonoaudiólogas)",
        },
        "parametros": {
            "lambda_suavidade": lambda_suavidade,
            "n_bases": int(n_bases),
            "ordem_das_bases": ORDEM_DAS_BASES,
            "passos_do_alvo": int(passos),
            "escalas_do_grosso_ao_fino": list(ESCALAS_DO_GROSSO_AO_FINO),
            "criterio": "minimos_quadrados (Koenig 2008 usa MINEIG; escala de λ não transferível)",
            "n_repeticoes": n_repeticoes,
            "estado_do_lambda": "provisório: depende da escolha do pesquisador responsável",
        },
    }

