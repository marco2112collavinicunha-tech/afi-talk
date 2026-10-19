#!/usr/bin/env python3
"""Teste t comum na comparação com o Entre-pessoas igualado (dados sintéticos).

Uso: python t_comum_igualado.py <mundo> <N> <replicas> <saida.json> [piloto]

Refaz os mesmos estudos simulados de h1b_igualado_dois_metodos.py (mesmas sementes, mesmo
piloto e K = 20 sorteios) e aplica aos ganhos o t de uma amostra do programa
(ganho_frases_fixas), que trata os ganhos dos participantes como independentes. É o teste que a
comparação usaria se ninguém cedesse gravações aos outros. Os sorteios do jackknife não entram
no t, mas são feitos e descartados para que cada estudo seja o mesmo da Figura 7.
"""
import json
import os
import platform
import sys
import time

import numpy as np
import scipy

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import h1b_igualado_dois_metodos as H

A, G, SJI = H.A, H.G, H.SJI


def descartar_sorteios_do_jackknife(N, rng):
    """Os N refazimentos sem um participante sorteiam (N - 1) vezes cada, de N - 2 doadores."""
    tam = (N - 2) * G.REP_TR
    for _ in range(N * (N - 1)):
        rng.random((SJI.K, 2, G.C, tam))


def p_do_t(g):
    return A.ganho_frases_fixas(np.asarray(g, dtype=float)[:, None], alfa=A.ALFA_DA_H1B)["p"]


def main(mundo, N, reps, saida, piloto=1000):
    idm = list(G.MUNDOS).index(mundo)
    par = G.MUNDOS[mundo]
    mu = G.centros(np.random.SeedSequence([H.SEMENTE, 99, idm]))
    rng = np.random.default_rng(np.random.SeedSequence([H.SEMENTE, 1, idm, N, 0]))
    th = 0.0
    for _ in range(piloto):
        Xtr, Xte = G.falantes(par, mu, N, rng)
        th += SJI.ganhos(G.distancias(Xtr, Xte).astype(np.float32), N, rng).mean()
    th /= piloto
    rng = np.random.default_rng(np.random.SeedSequence([H.SEMENTE, 1, idm, N, 1]))
    cont = {(f, a): 0 for f in ("centrado", "natural") for a in H.ALFAS}
    soma = 0.0
    t0 = time.time()
    for _ in range(reps):
        Xtr, Xte = G.falantes(par, mu, N, rng)
        g = SJI.ganhos(G.distancias(Xtr, Xte).astype(np.float32), N, rng)
        descartar_sorteios_do_jackknife(N, rng)
        soma += g.mean()
        for forma, desloc in (("centrado", th), ("natural", 0.0)):
            p = p_do_t(g - desloc)
            for a in H.ALFAS:
                cont[(forma, a)] += int(p is not None and p <= a)
    taxas = {}
    for (forma, a), k in cont.items():
        taxas.setdefault(forma, {})[str(a)] = {"taxa": k / reps, "wilson95": H.wilson(k, reps)}
    out = {"SINTETICO": True, "semente_raiz": H.SEMENTE, "mundo": mundo, "N": N, "K": SJI.K,
           "replicas": reps, "piloto": piloto, "ganho_verdadeiro_do_igualado": th,
           "ganho_medio_observado": soma / reps, "teste": "t de uma amostra, unilateral",
           "funcao_do_programa": "analise_do_estudo.ganho_frases_fixas",
           "sha256_programa": H.SHA256_DA_ANALISE_NA_IMPORTACAO,
           "taxas": taxas, "python": platform.python_version(), "numpy": np.__version__,
           "scipy": scipy.__version__, "segundos": time.time() - t0,
           "aviso": ("SINTÉTICO: vetores gaussianos e kNN-1; não é fala apráxica real; "
                     "mesmos estudos de h1b_igualado_dois_metodos.py")}
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(saida, "%.0fs" % out["segundos"], "th=%.4f" % th,
          {f: round(v["0.04"]["taxa"] * 100, 2) for f, v in taxas.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
         int(sys.argv[5]) if len(sys.argv) > 5 else 1000)
