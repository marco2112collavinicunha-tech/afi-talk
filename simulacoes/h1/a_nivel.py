"""Poder e falso positivo da primeira comparação em vários níveis (dados sintéticos).

Uso: python a_nivel.py <cenario> <replicas> <saida.json>
"""
import sys, json, time
import numpy as np
import a_h1
from comum import prog, meta, wilson, SEMENTE
ALFAS = (0.05, 0.045, 0.04, 0.035, 0.03, 0.025)


def main():
    cen, R, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    t0 = time.time(); cal = {a: a_h1.calibra(cen, a)[0] for a in a_h1.ALVOS}
    grades = {}
    if a_h1.CEN[cen].get("condicional"):
        for alvo in a_h1.ALVOS:
            rg = np.random.default_rng(np.random.SeedSequence([SEMENTE, 8, int(round(alvo * 100))]))
            grades[alvo] = a_h1.grade_m(cal[alvo], rg)
    res = []
    for N in a_h1.NS_PADRAO:
        for alvo in a_h1.ALVOS:
            rng = np.random.default_rng(np.random.SeedSequence([SEMENTE, 3, int(cen[1]), N, int(round(alvo * 100))]))
            ps = []; feito = 0
            while feito < R:
                r = min(1000, R - feito)
                D, cgrp, gr = a_h1.gera(rng, r, N, cen, cal[alvo])
                if a_h1.CEN[cen].get("condicional"):
                    cs, ms = grades[alvo]
                    D = D - (np.interp(cgrp, cs, ms).mean(1) - alvo)[:, None, None]
                for i in range(r):
                    rp = prog.ganho_frases_fixas(D[i])
                    ps.append(rp["p"] if rp["p"] is not None else 1.0)
                feito += r
            ps = np.array(ps)
            row = dict(N=N, alvo=alvo, R=R, rejeicao={str(a): dict(taxa=float((ps <= a).mean()),
                                                                     ic95_wilson=wilson(int((ps <= a).sum()), R)) for a in ALFAS})
            res.append(row)
            print(cen, N, alvo, " ".join("%s:%.4f" % (a, v["taxa"]) for a, v in row["rejeicao"].items()), flush=True)
    json.dump(dict(meta(), script="a_nivel.py", rotulo="pos-hoc", cenario=cen, replicas=R,
                   celulas=res, segundos=time.time() - t0), open(out, "w"), indent=1)


main()

