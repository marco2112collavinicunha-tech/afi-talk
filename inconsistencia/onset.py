"""Onset robusto a groping: consenso entre repetições, com regra de acordo para um alinhamento externo."""
from __future__ import annotations

import numpy as np

class SemNucleoComum(ValueError):
    """As repetições não compartilham região comum acima do limiar."""


LIMIAR_DE_NUCLEO = 0.55

FRACAO_MINIMA_DO_NUCLEO = 0.35

CANDIDATOS_A_NUCLEO = 3

PONTOS_PARA_DTW = 500

TOLERANCIA_DA_GUARDA = 0.18
CANDIDATOS_SUSPEITOS_PARA_DISPARAR = 2


def _dtw_subsequencia(nucleo, inteiro):
    """Onde `nucleo` melhor encaixa dentro de `inteiro`."""
    nucleo = np.asarray(nucleo, dtype=float)
    inteiro = np.asarray(inteiro, dtype=float)
    n, m = len(nucleo), len(inteiro)
    custo = np.abs(nucleo[:, None] - inteiro[None, :])
    acumulado = np.full((n, m), np.inf)
    acumulado[0, :] = custo[0, :]
    for i in range(1, n):
        anterior = acumulado[i - 1]
        melhor = np.empty(m)
        melhor[0] = anterior[0]
        melhor[1:] = np.minimum(anterior[1:], anterior[:-1])
        base = custo[i] + melhor
        acumulada = np.cumsum(custo[i])
        acumulado[i] = acumulada + np.minimum.accumulate(base - acumulada)
    fim = int(np.argmin(acumulado[n - 1]))
    i, j = n - 1, fim
    while i > 0:
        opcoes = [(acumulado[i - 1, j], i - 1, j)]
        if j > 0:
            opcoes.append((acumulado[i - 1, j - 1], i - 1, j - 1))
            opcoes.append((acumulado[i, j - 1], i, j - 1))
        _, i, j = min(opcoes, key=lambda x: x[0])
    return j, fim, float(acumulado[n - 1, fim] / n)


def _similaridade(custo_medio, escala):
    """Custo médio → similaridade em [0, 1]."""
    if escala <= 0:
        return 0.0
    return float(np.exp(-custo_medio / escala))


def _decimar(envoltoria, pontos):
    """Reamostra para `pontos` por interpolação linear."""
    envoltoria = np.asarray(envoltoria, dtype=float)
    if len(envoltoria) <= pontos:
        return envoltoria, 1.0
    origem = np.linspace(0.0, 1.0, len(envoltoria))
    destino = np.linspace(0.0, 1.0, pontos)
    return np.interp(destino, origem, envoltoria), len(envoltoria) / pontos


def _prefixo_do_candidato(candidato, outros):
    """Quanto de prefixo não compartilhado o candidato tem, pelos olhos dos outros."""
    if not outros:
        return 0.0
    return float(np.median([_dtw_subsequencia(o, candidato)[0] for o in outros]))


def _consenso_entre_todos_os_pares(envoltorias):
    """Plano B da guarda: cada repetição é núcleo dentro de cada outra."""
    n = len(envoltorias)
    saida = []
    for k in range(n):
        vistos = [_dtw_subsequencia(envoltorias[j], envoltorias[k])[0]
                  for j in range(n) if j != k]
        saida.append(int(np.median(vistos)) if vistos else 0)
    return saida


def nucleo_comum(envoltorias, limiar=LIMIAR_DE_NUCLEO,
                 fracao_minima=FRACAO_MINIMA_DO_NUCLEO, com_guarda=True):
    """Onde começa o núcleo comum em cada repetição."""
    if len(envoltorias) < 2:
        raise ValueError(
            "O consenso compara repetições: são necessárias 2+; veio {}.".format(
                len(envoltorias)))

    n = len(envoltorias)
    escala = float(np.median([np.std(e) for e in envoltorias])) or 1.0
    ordem = np.argsort([len(e) for e in envoltorias])
    candidatos = [int(k) for k in ordem[:min(CANDIDATOS_A_NUCLEO, n)]]

    inicios_por_rep = [[] for _ in range(n)]
    similaridades = []
    for c in candidatos:
        for k, inteiro in enumerate(envoltorias):
            if k == c:
                inicios_por_rep[k].append(0)
                continue
            ini, _fim, custo_medio = _dtw_subsequencia(envoltorias[c], inteiro)
            inicios_por_rep[k].append(ini)
            similaridades.append(_similaridade(custo_medio, escala))

    similaridade_mediana = float(np.median(similaridades)) if similaridades else 0.0
    if similaridade_mediana < limiar:
        raise SemNucleoComum(
            "As repetições não compartilham região comum: similaridade mediana "
            "{:.3f}, abaixo do limiar {:.2f}. Isto NÃO é erro de cálculo — é a "
            "categoria de resultado 'inconsistência acima do mensurável'.".format(
                similaridade_mediana, limiar))

    inicios = [int(np.median(v)) if v else 0 for v in inicios_por_rep]

    guarda_disparou = False
    if com_guarda:
        tolerancia = TOLERANCIA_DA_GUARDA * len(envoltorias[0])
        fora = [e for k, e in enumerate(envoltorias) if k not in candidatos]
        suspeitos = sum(1 for c in candidatos
                        if _prefixo_do_candidato(envoltorias[c], fora) > tolerancia)
        guarda_disparou = suspeitos >= CANDIDATOS_SUSPEITOS_PARA_DISPARAR
    if guarda_disparou:
        inicios = _consenso_entre_todos_os_pares(envoltorias)

    restos = [len(e) - ini for e, ini in zip(envoltorias, inicios)]
    mediana_total = float(np.median([len(e) for e in envoltorias]))
    if min(restos) < fracao_minima * mediana_total:
        raise SemNucleoComum(
            "Depois de descontar o início não compartilhado, a repetição mais "
            "curta ficou com {} de {:.0f} pontos ({:.0%}), abaixo da fração "
            "mínima {:.0%}.".format(
                min(restos), mediana_total, min(restos) / max(mediana_total, 1),
                fracao_minima))

    return inicios, similaridade_mediana, guarda_disparou


def onset_por_consenso(envoltorias, quadros_por_amostra=1,
                       limiar=LIMIAR_DE_NUCLEO, pontos=PONTOS_PARA_DTW,
                       com_guarda=True):
    decimadas, fatores = [], []
    for e in envoltorias:
        d, f = _decimar(e, pontos)
        decimadas.append(d)
        fatores.append(f)
    inicios, similaridade, guarda = nucleo_comum(decimadas, limiar=limiar,
                                                 com_guarda=com_guarda)
    return ([int(round(i * f * quadros_por_amostra))
             for i, f in zip(inicios, fatores)], similaridade, guarda)


def onset_com_acordo(onsets_consenso, onsets_alinhamento, taxa,
                     tolerancia_ms=50.0):
    if len(onsets_consenso) != len(onsets_alinhamento):
        raise ValueError(
            "As duas camadas têm de cobrir as mesmas repetições: consenso tem "
            "{}, alinhamento tem {}.".format(
                len(onsets_consenso), len(onsets_alinhamento)))

    tolerancia = tolerancia_ms * taxa / 1000.0
    aceitos, revisar = [], []
    for i, (b, a) in enumerate(zip(onsets_consenso, onsets_alinhamento)):
        aceitos.append(b)
        if a is None or abs(b - a) > tolerancia:
            revisar.append(i)
    return aceitos, revisar

