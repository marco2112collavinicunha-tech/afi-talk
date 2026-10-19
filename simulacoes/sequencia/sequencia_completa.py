#!/usr/bin/env python3
"""Sequência completa das três comparações: a primeira, depois Holm sobre a escolha forçada e o
Entre-pessoas igualado (dados sintéticos).

Uso:
  python sequencia_completa.py cal <calibracao.json>
  python sequencia_completa.py A <N> <replicas> <saida.json> <calibracao.json>

Mede a taxa de pelo menos uma conclusão falsa na sequência inteira (FWER), sob o nulo global e
sob nulos parciais, e o poder de cada componente, com 3, 5, 8 e 12 participantes. Quem decide é
o programa de análise: _decidir_pelas_frases_fixas (primeira comparação e cada versão da escolha
forçada), teste_da_h1b_dois_metodos (Entre-pessoas igualado) e aplicar_holm_da_familia (Holm).
Geradores: o da primeira comparação (../h1/a_h1.py, formas S1 e S2) e o do Entre-pessoas
(../entre_pessoas, mundo entre_alta centrado como nulo e entre_media como verdadeiro). A escolha
forçada não tem gerador próprio: para cada versão v, D2_v = mu_v + raiz(c)·(D1 - teta1) +
raiz(1 - c)·(raiz(r)·Z0 + raiz(1 - r)·Z_v), com c = 0,5 e r = 0,7. Não cobre fala apráxica, o
Whisper real nem a correlação real entre os testes.
"""
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import scipy

AQUI = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(AQUI)
RAIZ = os.path.dirname(SIM)
sys.path.insert(0, os.path.join(RAIZ, "analise"))
import analise_do_estudo as A  # noqa: E402

with open(A.__file__, "rb") as _f:
    SHA_PROGRAMA = hashlib.sha256(_f.read()).hexdigest()

sys.path.insert(0, os.path.join(SIM, "h1"))
import a_h1  # noqa: E402
sys.modules.pop("comum", None)
sys.path.remove(os.path.join(SIM, "h1"))
sys.path.insert(0, os.path.join(SIM, "entre_pessoas"))
import gerador as G  # noqa: E402
import sim_jack_ig as SJI  # noqa: E402
assert a_h1.prog is A and SJI.U.prog is A

SEMENTE = 20261006
K = 20
SJI.K = K
FORMAS = ("S1", "S2")
VERSOES = ("soma", "media_por_token", "calibrada_silencio")
C_LIGA, R_VERS = 0.5, 0.7
ALFA = A.ALFA_DA_H1
TETAS = (0.0, 0.10, 0.15)
D2_COMBOS = {"0_000": (0.0, (0, 0, 0)), "15_000": (0.15, (0, 0, 0)),
             "15_lfc": (0.15, (0.15, 0.15, 0.0)), "15_t15": (0.15, (0.15, 0.15, 0.15)),
             "0_t15": (0.0, (0.15, 0.15, 0.15)), "10_t10": (0.10, (0.10, 0.10, 0.10))}
# célula: (escolha forçada, Entre-pessoas igualado). G0 é o nulo global; P1 a P5, nulos parciais;
# W10 e W15, poder.
FAMILIAS = {"G0": ("0_000", "nula"), "P1": ("15_000", "nula"), "P2": ("15_lfc", "nula"),
            "P3": ("15_lfc", "verdadeira"), "P4": ("15_t15", "nula"),
            "P5": ("0_t15", "verdadeira"), "W10": ("10_t10", "verdadeira"),
            "W15": ("15_t15", "verdadeira")}
THETA1 = {"G0": 0.0, "P1": 0.15, "P2": 0.15, "P3": 0.15, "P4": 0.15, "P5": 0.0,
          "W10": 0.10, "W15": 0.15}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def decide(D):
    r, dec = A._decidir_pelas_frases_fixas(D)
    return (dec.get("teste") or {}).get("p"), dec["veredito"] is True, r, dec


def familia(testes_d2, teste_b):
    degrau_2 = {"estado": "passou", "motivo": None,
                "veredito": {"testes": {v: {"p": testes_d2[v][0]} for v in VERSOES},
                             "por_versao": {v: {"teste": testes_d2[v][2]} for v in VERSOES},
                             "comparadores": list(VERSOES), "falharam": []}}
    calc = teste_b.get("status") == "calculado"
    h1b = {"papel": "confirmatoria", "teste": teste_b,
           "estado": "passou" if calc else "nao_calculavel",
           "veredito": True if calc else None}
    f = A.aplicar_holm_da_familia(degrau_2, h1b, A.CHAVES)
    return (degrau_2["estado"] == "passou", h1b["estado"] == "passou",
            f["p_do_degrau_2"]["p"], f["p_da_h1b"]["p"])


def calibracao(arq):
    with open(arq) as fh:
        return {(k.split("_")[0], float(k.split("_")[1])): v for k, v in json.load(fh).items()}


def mundos_h1b():
    nomes = list(G.MUNDOS)
    par_n, par_v = G.MUNDOS["entre_alta"], G.MUNDOS["entre_media"]
    mu_n = G.centros(np.random.SeedSequence([SEMENTE, 99, nomes.index("entre_alta")]))
    mu_v = G.centros(np.random.SeedSequence([SEMENTE, 99, nomes.index("entre_media")]))
    return par_n, mu_n, par_v, mu_v


def h1b_teste(par, mu, N, rng, desloc):
    Xtr, Xte = G.falantes(par, mu, N, rng)
    d2 = G.distancias(Xtr, Xte).astype(np.float32)
    g = SJI.ganhos(d2, N, rng)
    gm = [SJI.ganhos(d2, N, rng, fora=k) for k in range(N)]
    ganhos = {k: float(v - desloc) for k, v in enumerate(g)}
    medias = {k: float(gm[k].mean() - desloc) for k in range(N)}
    return A.teste_da_h1b_dois_metodos(ganhos, medias, alfa=ALFA)


def piloto_th(par_n, mu_n, N, rng, piloto):
    return float(np.mean([SJI.ganhos(G.distancias(*G.falantes(par_n, mu_n, N, rng))
                                     .astype(np.float32), N, rng).mean()
                          for _ in range(piloto)]))


def parte_a(N, R, saida, arq_cal, piloto=1000):
    t0 = time.time()
    cal = calibracao(arq_cal)
    par_n, mu_n, par_v, mu_v = mundos_h1b()
    rp = np.random.default_rng(np.random.SeedSequence([SEMENTE, 1, N, 0]))
    th = piloto_th(par_n, mu_n, N, rp, piloto)
    rng_b = np.random.default_rng(np.random.SeedSequence([SEMENTE, 2, N]))
    rng_a = {f: np.random.default_rng(np.random.SeedSequence([SEMENTE, 3, N, i]))
             for i, f in enumerate(FORMAS)}
    chaves = ["d1", "fwer", "fwer_forcado", "d2_familia", "b_familia", "d2_familia_forcado",
              "b_familia_forcado", "d2_4", "b_4", "d2_4_forcado", "b_4_forcado",
              "fwer_regra_anterior", "fwer_regra_anterior_forcado",
              "d1_e_d2", "d1_e_b", "d1_e_d2_e_b", "d1_e_d2_regra_anterior",
              "d1_e_b_regra_anterior", "d1_e_d2_e_b_regra_anterior", "b_sem_p", "d2_sem_p"]
    cont = {(f, c, k): 0 for f in FORMAS for c in FAMILIAS for k in chaves}
    h1b_deg = {"nula": 0, "verdadeira": 0}
    feito, lote = 0, 200
    while feito < R:
        r = min(lote, R - feito)
        D1 = {(f, t): a_h1.gera(rng_a[f], r, N, f, cal[(f, t)])[0] for f in FORMAS for t in TETAS}
        Z = {f: [a_h1.gera(rng_a[f], r, N, f, cal[(f, 0.0)])[0] for _ in range(4)]
             for f in FORMAS}
        for i in range(r):
            tb = {"nula": h1b_teste(par_n, mu_n, N, rng_b, th),
                  "verdadeira": h1b_teste(par_v, mu_v, N, rng_b, 0.0)}
            for k, t in tb.items():
                h1b_deg[k] += int(t.get("status") != "calculado")
            pb4 = {k: (t.get("status") == "calculado" and t["passou"]) for k, t in tb.items()}
            for f in FORMAS:
                d1 = {t: decide(D1[(f, t)][i])[1] for t in TETAS}
                comum = np.sqrt(R_VERS) * Z[f][0][i]
                d2t = {}
                for nome, (t1, mus) in D2_COMBOS.items():
                    base = np.sqrt(C_LIGA) * (D1[(f, t1)][i] - t1)
                    d2t[nome] = {v: decide(m + base + np.sqrt(1 - C_LIGA) * (
                        comum + np.sqrt(1 - R_VERS) * Z[f][1 + j][i]))
                        for j, (v, m) in enumerate(zip(VERSOES, mus))}
                for cel, (combo, hb) in FAMILIAS.items():
                    rej2, rejb, p2, pb = familia(d2t[combo], tb[hb])
                    d2_4 = all(d2t[combo][v][1] for v in VERSOES)
                    b_4 = pb4[hb]
                    passa1 = d1[THETA1[cel]]
                    nulo1 = THETA1[cel] == 0.0
                    nulo2 = min(D2_COMBOS[combo][1]) == 0
                    nulob = hb == "nula"
                    err_fam = (rej2 and nulo2) or (rejb and nulob)
                    err_ant = (d2_4 and nulo2) or (b_4 and nulob)
                    x = cont
                    x[(f, cel, "d1")] += passa1
                    x[(f, cel, "fwer")] += (passa1 and nulo1) or (passa1 and err_fam)
                    x[(f, cel, "fwer_forcado")] += err_fam
                    x[(f, cel, "fwer_regra_anterior")] += (passa1 and nulo1) or (passa1 and err_ant)
                    x[(f, cel, "fwer_regra_anterior_forcado")] += err_ant
                    x[(f, cel, "d2_familia")] += passa1 and rej2
                    x[(f, cel, "b_familia")] += passa1 and rejb
                    x[(f, cel, "d2_familia_forcado")] += rej2
                    x[(f, cel, "b_familia_forcado")] += rejb
                    x[(f, cel, "d2_4")] += passa1 and d2_4
                    x[(f, cel, "b_4")] += passa1 and b_4
                    x[(f, cel, "d2_4_forcado")] += d2_4
                    x[(f, cel, "b_4_forcado")] += b_4
                    x[(f, cel, "d1_e_d2")] += passa1 and rej2
                    x[(f, cel, "d1_e_b")] += passa1 and rejb
                    x[(f, cel, "d1_e_d2_e_b")] += passa1 and rej2 and rejb
                    x[(f, cel, "d1_e_d2_regra_anterior")] += passa1 and d2_4
                    x[(f, cel, "d1_e_b_regra_anterior")] += passa1 and b_4
                    x[(f, cel, "d1_e_d2_e_b_regra_anterior")] += passa1 and d2_4 and b_4
                    x[(f, cel, "b_sem_p")] += pb is None
                    x[(f, cel, "d2_sem_p")] += p2 is None
        feito += r
        print("A N=%d %d/%d %.0fs" % (N, feito, R, time.time() - t0), flush=True)
    celulas = {}
    for (f, c, k), v in cont.items():
        celulas.setdefault(f, {}).setdefault(c, {})[k] = {"n": int(v), "taxa": v / R,
                                                          "wilson95": wilson(int(v), R)}
    out = {"SINTETICO": True, "semente_raiz": SEMENTE, "K": K, "c_ligacao_degrau_1_degrau_2": C_LIGA,
           "r_entre_versoes": R_VERS, "programa": "analise/analise_do_estudo.py",
           "sha256_programa": SHA_PROGRAMA, "python": platform.python_version(),
           "numpy": np.__version__, "scipy": scipy.__version__, "parte": "A", "N": N,
           "replicas": R, "piloto": piloto, "ganho_verdadeiro_entre_alta_centrado": th,
           "calibracao_a_h1": {"{}_{}".format(f, t): v for (f, t), v in cal.items()},
           "h1b_sem_teste_calculado": h1b_deg, "celulas": celulas,
           "familias": {k: list(v) for k, v in FAMILIAS.items()}, "theta1": THETA1,
           "d2_combos": {k: [v[0], list(v[1])] for k, v in D2_COMBOS.items()},
           "segundos": time.time() - t0}
    with open(saida, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(saida, "%.0fs" % out["segundos"], flush=True)


def main(argv):
    if argv[1] == "cal":
        cal = {(f, t): a_h1.calibra(f, t)[0] for f in FORMAS for t in TETAS}
        with open(argv[2], "w") as fh:
            json.dump({"{}_{}".format(f, t): v for (f, t), v in cal.items()}, fh, indent=1)
        print(argv[2])
    elif argv[1] == "A":
        parte_a(int(argv[2]), int(argv[3]), argv[4], argv[5])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
