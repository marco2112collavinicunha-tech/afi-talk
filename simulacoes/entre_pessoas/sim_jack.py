"""Jackknife por participante (t_jack, rejeita) e um ensaio anterior com o Entre-pessoas completo
e o de doador único (dados sintéticos, nível de 5%)."""
import sys, json, time
import numpy as np
from scipy import stats
import gerador as G
import comum as U


def blocos(par, mu, N, rng):
    Xtr, Xte = G.falantes(par, mu, N, rng)
    d2 = G.distancias(Xtr, Xte)
    idx = d2.argmin(axis=3)
    val = np.take_along_axis(d2, idx[..., None], 3)[..., 0]
    hit = G.YTR[idx] == G.YTE[None, :, None]
    return val, hit


def ganhos_completo(val, hit, fora=None):
    """ganho médio P - Entre-pessoas completo, opcionalmente sem o participante `fora`."""
    N = val.shape[0]
    alvos = [i for i in range(N) if i != fora]
    g = []
    for i in alvos:
        v = val[i].copy(); v[:, i] = np.inf
        if fora is not None:
            v[:, fora] = np.inf
        j = v.argmin(1)
        e = hit[i][np.arange(hit.shape[1]), j].mean()
        g.append(hit[i][:, i].mean() - e)
    return np.array(g)


def ganhos_doador(A, fora=None):
    N = A.shape[0]
    keep = [i for i in range(N) if i != fora]
    B = A[np.ix_(keep, keep)]
    n = B.shape[0]
    d = np.diag(B)
    return d - (B.sum(1) - d) / (n - 1)


def t_jack(g_all, g_menos):
    """g_all: ganhos com todos (N,); g_menos: lista de N vetores de ganhos sem k."""
    N = len(g_all)
    th = g_all.mean()
    tk = np.array([x.mean() for x in g_menos])
    v = (N - 1) / N * ((tk - tk.mean()) ** 2).sum()
    return th, v


def rejeita(th, v, N, alfa=U.ALFA):
    if not np.isfinite(v) or v <= U.prog.TOLERANCIA_DEGENERADO ** 2:
        return False
    return bool(th > 0 and stats.t.sf(th / np.sqrt(v), N - 1) <= alfa)


def main(mundo, N, reps, saida, piloto=4000):
    idm = list(G.MUNDOS).index(mundo); par = G.MUNDOS[mundo]
    mu = G.centros(np.random.SeedSequence([U.SEMENTE, 99, idm]))
    rng = np.random.default_rng(np.random.SeedSequence([U.SEMENTE, 7, idm, N, 0]))
    thc = thd = 0.0
    for _ in range(piloto):
        val, hit = blocos(par, mu, N, rng)
        thc += ganhos_completo(val, hit).mean(); thd += ganhos_doador(hit.mean(1)).mean()
    thc /= piloto; thd /= piloto
    rng = np.random.default_rng(np.random.SeedSequence([U.SEMENTE, 7, idm, N, 1]))
    nomes = ["co_jk_cen", "co_jk_p10", "co_jk_nat", "du_jk_cen", "du_jk_p10", "du_jk_nat",
             "co_nb_cen", "du_nb_cen"]
    rej = {x: np.zeros(reps, bool) for x in nomes}
    med = {"co": np.zeros(reps), "du": np.zeros(reps)}; vj = {"co": np.zeros(reps), "du": np.zeros(reps)}
    t0 = time.time()
    for r in range(reps):
        val, hit = blocos(par, mu, N, rng)
        A = hit.mean(1)
        gc = ganhos_completo(val, hit); gcm = [ganhos_completo(val, hit, k) for k in range(N)]
        gd = ganhos_doador(A); gdm = [ganhos_doador(A, k) for k in range(N)]
        for pre, g, gm, th in (("co", gc, gcm, thc), ("du", gd, gdm, thd)):
            m, v = t_jack(g, gm)
            med[pre][r] = m; vj[pre][r] = v
            rej[pre + "_jk_nat"][r] = rejeita(m, v, N)
            rej[pre + "_jk_cen"][r] = rejeita(m - th, v, N)
            rej[pre + "_jk_p10"][r] = rejeita(m - th + 0.10, v, N)
            rej[pre + "_nb_cen"][r] = U.t_nb(g - th)
    out = {"meta": U.meta(), "experimento": "J", "mundo": mundo, "N": N, "reps": reps,
           "segundos": time.time() - t0, "theta": {"completo": thc, "doador_unico": thd},
           "razao_var_vjack": {k: float(np.var(med[k], ddof=1) / np.mean(vj[k])) for k in med},
           "taxas": {x: {"taxa": float(v.mean()), "wilson": U.wilson(int(v.sum()), reps)} for x, v in rej.items()}}
    json.dump(out, open(saida, "w"), indent=1, ensure_ascii=False)
    print(saida, "%.0fs" % out["segundos"], {k: round(v["taxa"] * 100, 2) for k, v in out["taxas"].items()})


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])

