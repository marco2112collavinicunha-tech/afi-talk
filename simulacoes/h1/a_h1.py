"""Gerador e testes da primeira comparação, Personalizado contra Genérico (dados sintéticos).

Uso: python a_h1.py <cenario> <replicas> <saida.json> [N separados por vírgula] [nivel]

Todos os testes usam o mesmo nível unilateral; sem o último argumento, o do estudo (0,04).
"""
import sys, json, time
import numpy as np
from scipy import stats
from comum import (prog, meta, wilson, logistica, SEMENTE, ALFA, t_johnson, bootstrap_t,
                   wilcoxon_exato, aparada, t_por_grupo)

ID = 1
F, SES, REP = 12, 2, 3
AG, BZ, SDU, SDW = -0.5, -0.9, 0.5, 0.4
SD_NU, SD_E = 0.3, 0.8
SD_TOT = np.sqrt(SD_NU ** 2 + SD_E ** 2)
GH_X, GH_W = np.polynomial.hermite.hermgauss(40)
GH_X = GH_X * np.sqrt(2) * SD_TOT; GH_W = GH_W / np.sqrt(np.pi)
P_EMPATE_P = 0.04
SD_C = 0.6
CEN = {"S1": dict(), "S2": dict(esq=(0.20, -2.0)), "S3": dict(esq=(0.10, -3.0)),
       "S4": dict(dir=(0.20, -2.0)), "S5": dict(grupo="trios"), "S6": dict(grupo="metade"),
       "S7": dict(grupo="trios", condicional=True), "S8": dict(esq=(0.20, -2.0), grupo="trios"),
       "S9": dict(estresse=True)}
NOMES = {"S1": "simétrico", "S2": "cauda à esquerda (20%, P falha)", "S3": "cauda à esquerda rara e forte (10%)",
         "S4": "cauda à direita (20%, G colapsa)", "S5": "agrupado em trios (população de serviços)",
         "S6": "agrupado, metade num serviço (população de serviços)",
         "S7": "agrupado em trios (condicional aos serviços)", "S8": "pior caso: S2 + trios",
         "S9": "estresse (exponencial à esquerda)"}
NS_PADRAO = (3, 4, 5, 6, 8, 10, 12)
ALVOS = (0.0, 0.10, 0.15)

rf = np.random.default_rng(np.random.SeedSequence([SEMENTE, 0]))
_z = rf.normal(size=(2, F))
PHI_G = 0.6 * _z[0]
PHI_P = 0.6 * (0.5 * _z[0] + np.sqrt(1 - 0.25) * _z[1])


def EL(eta):
    """E[logística(eta + gravação + sessão)] por quadratura."""
    return (logistica(eta[..., None] + GH_X) * GH_W).sum(-1)


def grupos_de(N, tipo):
    if tipo == "trios":
        return np.arange(N) // 3
    if tipo == "metade":
        h = int(np.ceil(N / 2))
        return np.concatenate([np.zeros(h, int), np.arange(1, N - h + 1)])
    return np.arange(N)


def pessoas(rng, shape, cen):
    """Sorteia os efeitos de pessoa (shape = (R, N) ou (M,)) e devolve dict com os termos."""
    c = CEN[cen]
    p = dict(z=rng.normal(size=shape), uG=rng.normal(0, SDU, shape), uP=rng.normal(0, SDU, shape),
             wG=rng.normal(0, SDW, shape + (F,)), wP=rng.normal(0, SDW, shape + (F,)))
    p["extraP"] = np.zeros(shape); p["extraG"] = np.zeros(shape)
    if "esq" in c:
        pr, k = c["esq"]; p["extraP"] = p["extraP"] + k * (rng.random(shape) < pr)
    if "dir" in c:
        pr, k = c["dir"]; p["extraG"] = p["extraG"] + k * (rng.random(shape) < pr)
    return p


def etas(p, aP, cgrp=None):
    eG = AG + BZ * p["z"][..., None] + p["uG"][..., None] + PHI_G + p["wG"] + p["extraG"][..., None]
    eP = aP + BZ * p["z"][..., None] + p["uP"][..., None] + PHI_P + p["wP"] + p["extraP"][..., None]
    if cgrp is not None:
        eP = eP + cgrp[..., None]
    piE = logistica(-2.5 + 0.8 * p["z"])
    return eG, eP, piE


def ganho_esperado(p, aP, cgrp=None):
    eG, eP, piE = etas(p, aP, cgrp)
    cG = piE[..., None] / 12 + (1 - piE[..., None]) * EL(eG)
    cP = EL(eP) * (1 - P_EMPATE_P * 0.5)
    return (cP - cG).mean(-1)


def ganho_esperado_s9(p, a, gam_exp):
    eG, _, piE = etas(p, 0.0)
    cG = piE[..., None] / 12 + (1 - piE[..., None]) * EL(eG)
    g = a - (gam_exp - 0.10)
    return (np.clip(cG + g[..., None], 0, 1) - cG).mean(-1)


def _fatia(p, a, b):
    return {k: v[a:b] for k, v in p.items()}


def _media_em_fatias(fn, p, M, *extra, passo=20000):
    tot = np.empty(M)
    for a in range(0, M, passo):
        b = min(M, a + passo)
        tot[a:b] = fn(_fatia(p, a, b), *[None if x is None else x[a:b] if np.ndim(x) else x for x in extra])
    return tot


def calibra(cen, alvo, M=200000):
    rng = np.random.default_rng(np.random.SeedSequence([SEMENTE, 9, int(cen[1]), int(round(alvo * 100))]))
    p = pessoas(rng, (M,), cen)
    if CEN[cen].get("grupo"):
        cg = rng.normal(0, SD_C, M)
    else:
        cg = None
    if CEN[cen].get("estresse"):
        ge = rng.exponential(0.10, M)
        g9 = lambda pp, a, gg: ganho_esperado_s9(pp, a, gg)
        f = lambda a: _media_em_fatias(lambda pp, gg: g9(pp, a, gg), p, M, ge).mean() - alvo
    else:
        f = lambda a: _media_em_fatias(lambda pp, cc: ganho_esperado(pp, a, cc), p, M, cg).mean() - alvo
    from scipy.optimize import brentq
    a = brentq(f, -4.0, 4.0, xtol=1e-9)
    if CEN[cen].get("estresse"):
        gv = _media_em_fatias(lambda pp, gg: ganho_esperado_s9(pp, a, gg), p, M, ge)
    else:
        gv = _media_em_fatias(lambda pp, cc: ganho_esperado(pp, a, cc), p, M, cg)
    return a, dict(media=float(gv.mean()), dp=float(gv.std()), assimetria=float(stats.skew(gv)),
                   p05=float(np.quantile(gv, .05)), mediana=float(np.median(gv)), p95=float(np.quantile(gv, .95)))


def grade_m(aP, rng, M=20000):
    """m(c): ganho esperado dado o efeito de grupo c (para S7)."""
    cs = np.linspace(-4 * SD_C, 4 * SD_C, 161)
    p = pessoas(rng, (M,), "S5")
    return cs, np.array([ganho_esperado(p, aP, np.full(M, c)).mean() for c in cs])


def gera(rng, R, N, cen, a):
    c = CEN[cen]
    p = pessoas(rng, (R, N), cen)
    cgrp = None; gr = grupos_de(N, c.get("grupo"))
    if c.get("grupo"):
        cg = rng.normal(0, SD_C, (R, gr.max() + 1)); cgrp = cg[:, gr]
    sh = (R, N, F, SES, REP)
    e = rng.normal(0, SD_E, sh)
    nuG = rng.normal(0, SD_NU, (R, N, 1, SES, 1)); nuP = rng.normal(0, SD_NU, (R, N, 1, SES, 1))
    if c.get("estresse"):
        eG, _, piE = etas(p, 0.0)
        cGbar = piE[..., None] / 12 + (1 - piE[..., None]) * EL(eG)
        g = a - (rng.exponential(0.10, (R, N)) - 0.10)
        pP = np.clip(cGbar + g[..., None], 0, 1)
        credP = (rng.random(sh) < pP[..., None, None]).astype(float)
    else:
        eG, eP, piE = etas(p, a, cgrp)
        acP = rng.random(sh) < logistica(eP[..., None, None] + nuP + e)
        credP = np.where(acP, np.where(rng.random(sh) < P_EMPATE_P, 0.5, 1.0), 0.0)
    vazio = rng.random(sh) < piE[..., None, None, None]
    acG = rng.random(sh) < logistica(eG[..., None, None] + nuG + e)
    credG = np.where(vazio, 1.0 / 12, acG.astype(float))
    D = credP.mean((3, 4)) - credG.mean((3, 4))
    return D, cgrp, gr


TESTES = ("t_programa", "johnson", "bootstrap_t", "sinais_programa", "wilcoxon", "aparada_a", "aparada_b", "t_por_grupo")


def main():
    cen, R, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    Ns = tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else NS_PADRAO
    alfa = float(sys.argv[5]) if len(sys.argv) > 5 else ALFA
    t0 = time.time()
    cal = {}
    for alvo in ALVOS:
        a, desc = calibra(cen, alvo); cal[alvo] = dict(a=a, ganho_verdadeiro=desc)
        print(cen, "calibrado alvo", alvo, "a=%.4f" % a, desc, flush=True)
    grades = {}
    if CEN[cen].get("condicional"):
        for alvo in ALVOS:
            rg = np.random.default_rng(np.random.SeedSequence([SEMENTE, 8, int(round(alvo * 100))]))
            grades[alvo] = grade_m(cal[alvo]["a"], rg)
    res = []
    for N in Ns:
        for alvo in ALVOS:
            rng = np.random.default_rng(np.random.SeedSequence([SEMENTE, ID, int(cen[1]), N, int(round(alvo * 100))]))
            rej = {k: 0 for k in TESTES}
            ganhos_obs = []
            n_degen = 0; t1 = time.time(); feito = 0
            while feito < R:
                r = min(1000, R - feito)
                D, cgrp, gr = gera(rng, r, N, cen, cal[alvo]["a"])
                if CEN[cen].get("condicional"):
                    cs, ms = grades[alvo]
                    theta = np.interp(cgrp, cs, ms).mean(1) - alvo
                    D = D - theta[:, None, None]
                d = D.mean(2)
                ganhos_obs.append(d[:200].ravel())
                for i in range(r):
                    rp = prog.ganho_frases_fixas(D[i], alfa=alfa)
                    if rp["status"] == "passou": rej["t_programa"] += 1
                    if rp["status"] == "degenerado": n_degen += 1
                    ps = prog.permutacao_de_sinais(d[i])["p"]
                    if ps is not None and ps <= alfa: rej["sinais_programa"] += 1
                rej["johnson"] += int(t_johnson(d, alfa).sum())
                rej["bootstrap_t"] += int(bootstrap_t(d, rng, alfa=alfa).sum())
                rej["wilcoxon"] += int(wilcoxon_exato(d, alfa).sum())
                rej["aparada_a"] += int(aparada(d, "a", alfa).sum())
                rej["aparada_b"] += int(aparada(d, "b", alfa).sum())
                if CEN[cen].get("grupo"):
                    rej["t_por_grupo"] += int(t_por_grupo(d, gr, alfa).sum())
                feito += r
            go = np.concatenate(ganhos_obs)
            row = dict(N=N, alvo=alvo, R=R, degenerados=n_degen,
                       ganho_observado=dict(dp=float(go.std()), assimetria=float(stats.skew(go))),
                       taxas={k: dict(taxa=v / R, ic95_wilson=wilson(v, R)) for k, v in rej.items()
                              if k != "t_por_grupo" or CEN[cen].get("grupo")})
            res.append(row)
            print(cen, N, alvo, " ".join("%s %.4f" % (k, v["taxa"]) for k, v in row["taxas"].items()),
                  "| %.0fs" % (time.time() - t1), flush=True)
    json.dump(dict(meta(), script="a_h1.py", cenario=cen, nome=NOMES[cen], replicas=R, alfa_unilateral=alfa,
                   calibracao={str(k): v for k, v in cal.items()}, celulas=res,
                   segundos=time.time() - t0), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()

