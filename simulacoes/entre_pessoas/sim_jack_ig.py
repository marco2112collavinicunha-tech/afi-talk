"""Ganhos contra o Entre-pessoas igualado: sorteio das gravações dos outros e acerto de cada
participante. h1b_igualado_dois_metodos.py, t_comum_igualado.py e ../sequencia/sequencia_completa.py
usam estas funções com K = 20 sorteios; o main() daqui é um ensaio anterior, com K = 10 e nível de 5% (dados sintéticos)."""
import sys, json, time
import numpy as np
import gerador as G
import comum as U
from sim_jack import t_jack, rejeita

K = 10


def sorteio(i, excluidos, N, rng):
    """K sorteios de 4 por frase e por sessão dos outros (fora i e excluidos)."""
    outros = np.array([j for j in range(N) if j != i and j not in excluidos])
    tam = len(outros) * G.REP_TR
    u = rng.random((K, 2, G.C, tam))
    esc = np.argpartition(u, G.REP_TR - 1, axis=-1)[..., :G.REP_TR]
    jj = outros[esc // G.REP_TR]; r = esc % G.REP_TR
    ses = np.arange(2)[None, :, None, None]; cc = np.arange(G.C)[None, None, :, None]
    return jj.reshape(K, -1), (ses * G.C * G.REP_TR + cc * G.REP_TR + r).reshape(K, -1)


def ganho_i(d2, i, J, T):
    P = (G.YTR[d2[i, :, i, :].argmin(1)] == G.YTE).mean()
    flat = d2[i].reshape(d2.shape[1], -1)
    sub = flat[:, J * G.NTR + T]
    lab = G.YTR[np.take_along_axis(T, sub.argmin(2).T, 1)]
    return P - (lab == G.YTE[None]).mean()


def ganhos(d2, N, rng, fora=None):
    exc = () if fora is None else (fora,)
    return np.array([ganho_i(d2, i, *sorteio(i, exc, N, rng)) for i in range(N) if i != fora])


def main(mundo, N, reps, saida, piloto=2000):
    idm = list(G.MUNDOS).index(mundo); par = G.MUNDOS[mundo]
    mu = G.centros(np.random.SeedSequence([U.SEMENTE, 99, idm]))
    rng = np.random.default_rng(np.random.SeedSequence([U.SEMENTE, 8, idm, N, 0]))
    th = 0.0
    for _ in range(piloto):
        Xtr, Xte = G.falantes(par, mu, N, rng)
        th += ganhos(G.distancias(Xtr, Xte), N, rng).mean()
    th /= piloto
    rng = np.random.default_rng(np.random.SeedSequence([U.SEMENTE, 8, idm, N, 1]))
    nomes = ["ig_jk_cen", "ig_jk_p10", "ig_jk_nat", "ig_nb_cen", "ig_t_cen", "ig_nb_nat", "ig_ambos_cen", "ig_ambos_nat"]
    rej = {x: np.zeros(reps, bool) for x in nomes}
    med = np.zeros(reps); vj = np.zeros(reps)
    t0 = time.time()
    for r in range(reps):
        Xtr, Xte = G.falantes(par, mu, N, rng)
        d2 = G.distancias(Xtr, Xte).astype(np.float32)
        g = ganhos(d2, N, rng)
        gm = [ganhos(d2, N, rng, fora=k) for k in range(N)]
        m, v = t_jack(g, gm); med[r] = m; vj[r] = v
        rej["ig_jk_nat"][r] = rejeita(m, v, N)
        rej["ig_jk_cen"][r] = rejeita(m - th, v, N)
        rej["ig_jk_p10"][r] = rejeita(m - th + 0.10, v, N)
        rej["ig_nb_cen"][r] = U.t_nb(g - th)
        rej["ig_t_cen"][r] = U.t_simples((g - th)[:, None])[0]
        rej["ig_nb_nat"][r] = U.t_nb(g)
        rej["ig_ambos_cen"][r] = rej["ig_jk_cen"][r] and rej["ig_nb_cen"][r]
        rej["ig_ambos_nat"][r] = rej["ig_jk_nat"][r] and rej["ig_nb_nat"][r]
    out = {"meta": U.meta(), "experimento": "JI", "mundo": mundo, "N": N, "K": K, "reps": reps,
           "segundos": time.time() - t0, "theta_igualado": th,
           "razao_var_vjack": float(np.var(med, ddof=1) / np.mean(vj)),
           "taxas": {x: {"taxa": float(v.mean()), "wilson": U.wilson(int(v.sum()), reps)} for x, v in rej.items()}}
    json.dump(out, open(saida, "w"), indent=1, ensure_ascii=False)
    print(saida, "%.0fs" % out["segundos"], {k: round(v["taxa"] * 100, 2) for k, v in out["taxas"].items()})


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
         int(sys.argv[5]) if len(sys.argv) > 5 else 2000)

