#!/usr/bin/env python3
"""Falso positivo e poder da comparação com o Entre-pessoas igualado (dados sintéticos).

Uso: python h1b_igualado_dois_metodos.py <mundo> <N> <replicas> <saida.json> [piloto]
"""
import json
import os
import platform
import sys
import time

import numpy as np
import scipy

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(os.path.dirname(AQUI))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(RAIZ, "analise"))
sys.path.insert(0, RAIZ)

import analise_do_estudo as A
import gerador as G
import sim_jack_ig as SJI

with open(A.__file__, "rb") as _f:
    SHA256_DA_ANALISE_NA_IMPORTACAO = __import__("hashlib").sha256(_f.read()).hexdigest()

SEMENTE = 20261005
K = 20
ALFAS = (0.04, 0.05)
SJI.K = K


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def um_estudo(d2, N, rng):
    g = SJI.ganhos(d2, N, rng)
    gm = [SJI.ganhos(d2, N, rng, fora=k) for k in range(N)]
    return g, {k: float(gm[k].mean()) for k in range(N)}


def decide(g, medias, desloc):
    ganhos = {k: float(v - desloc) for k, v in enumerate(g)}
    med = {k: v - desloc for k, v in medias.items()}
    r = A.teste_da_h1b_dois_metodos(ganhos, med, alfa=A.ALFA_DA_H1B)
    if r["status"] != "calculado":
        return None
    return r


def main(mundo, N, reps, saida, piloto=1000):
    idm = list(G.MUNDOS).index(mundo)
    par = G.MUNDOS[mundo]
    mu = G.centros(np.random.SeedSequence([SEMENTE, 99, idm]))
    rng = np.random.default_rng(np.random.SeedSequence([SEMENTE, 1, idm, N, 0]))
    th = 0.0
    for _ in range(piloto):
        Xtr, Xte = G.falantes(par, mu, N, rng)
        th += SJI.ganhos(G.distancias(Xtr, Xte).astype(np.float32), N, rng).mean()
    th /= piloto
    rng = np.random.default_rng(np.random.SeedSequence([SEMENTE, 1, idm, N, 1]))
    nomes = ["ambos", "nadeau_bengio", "jackknife"]
    cont = {(f, n, a): 0 for f in ("centrado", "natural") for n in nomes for a in ALFAS}
    degenerados = {"centrado": 0, "natural": 0}
    t0 = time.time()
    for _ in range(reps):
        Xtr, Xte = G.falantes(par, mu, N, rng)
        d2 = G.distancias(Xtr, Xte).astype(np.float32)
        g, medias = um_estudo(d2, N, rng)
        for forma, desloc in (("centrado", th), ("natural", 0.0)):
            r = decide(g, medias, desloc)
            if r is None:
                degenerados[forma] += 1
                continue
            m = r["ganho_medio"]
            for a in ALFAS:
                cont[(forma, "ambos", a)] += int(m > 0 and r["p"] <= a)
                cont[(forma, "nadeau_bengio", a)] += int(m > 0 and r["nadeau_bengio"]["p"] <= a)
                cont[(forma, "jackknife", a)] += int(m > 0 and r["jackknife"]["p"] <= a)
    taxas = {}
    for (forma, nome, a), k in cont.items():
        taxas.setdefault(forma, {}).setdefault(nome, {})[str(a)] = {
            "taxa": k / reps, "wilson95": wilson(k, reps)}
    out = {"SINTETICO": True, "semente_raiz": SEMENTE, "mundo": mundo, "N": N, "K": K,
           "replicas": reps, "piloto": piloto, "ganho_verdadeiro_do_igualado": th,
           "funcao_do_programa": "analise_do_estudo.teste_da_h1b_dois_metodos",
           "sha256_programa": SHA256_DA_ANALISE_NA_IMPORTACAO,
           "sha256_calculado": "na importacao do modulo",
           "alfa_registrado": A.ALFA_DA_H1B, "taxas": taxas, "degenerados": degenerados,
           "python": platform.python_version(), "numpy": np.__version__,
           "scipy": scipy.__version__, "segundos": time.time() - t0,
           "aviso": ("SINTÉTICO: vetores gaussianos e kNN-1; não é fala apráxica real; "
                     "não cobre seleção de configuração, regressão logística, empate 1/k, "
                     "frases faltando, serviço comum nem o Genérico")}
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(saida, "%.0fs" % out["segundos"], "th=%.4f" % th,
          {f: {n: round(v["0.04"]["taxa"] * 100, 2) for n, v in d.items()}
           for f, d in taxas.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
         int(sys.argv[5]) if len(sys.argv) > 5 else 1000)

