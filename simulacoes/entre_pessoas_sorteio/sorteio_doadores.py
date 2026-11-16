"""Compara o sorteio atual do Entre-pessoas igualado (por frase e sessão, entre todos os doadores)
com um sorteio equilibrado (os mesmos doadores em todas as frases). Dados sintéticos do gerador
do projeto (simulacoes/entre_pessoas/gerador.py, commit ad973ae)."""
import sys, json
import numpy as np
sys.path.insert(0, sys.argv[1])
import gerador as G
C, REP = G.C, G.REP_TR

def sorteio_equilibrado(N, K, rng, n=REP, mesmos_nas_sessoes=False):
    M = C * 2 * n
    J = np.empty((N, K, M), int); T = np.empty((N, K, M), int)
    for i in range(N):
        outros = np.array([j for j in range(N) if j != i])
        for k in range(K):
            jj, tt = [], []
            d0 = rng.permutation(outros)[:n]
            for s in range(2):
                d = d0 if mesmos_nas_sessoes else rng.permutation(outros)[:n]
                for c in range(C):
                    for doador in d:
                        r = rng.integers(REP)
                        jj.append(doador); tt.append(s * C * REP + c * REP + r)
            J[i, k] = jj; T[i, k] = tt
    return J, T

def main(pasta, reps=150, N=12, K=20, semente=20261116):
    out = {}
    for mundo in ["nulo_exato", "entre_baixa", "entre_media", "entre_alta", "pior_caso"]:
        par = G.MUNDOS[mundo]
        idm = list(G.MUNDOS).index(mundo)
        mu = G.centros(np.random.SeedSequence([semente, 99, idm]))
        rng = np.random.default_rng(np.random.SeedSequence([semente, idm]))
        acc = {"personalizado": [], "igualado_atual": [], "igualado_equilibrado": [], "igualado_equilibrado_mesmos": []}
        for _ in range(reps):
            Xtr, Xte = G.falantes(par, mu, N, rng)
            d2 = G.distancias(Xtr, Xte)
            Ja, Ta = G.sorteio_igualado(N, K, rng)
            Jb, Tb = sorteio_equilibrado(N, K, rng)
            Jc, Tc = sorteio_equilibrado(N, K, rng, mesmos_nas_sessoes=True)
            M, Ef, Ea = G.avaliar_knn(d2, Ja, Ta)
            _, _, Eb = G.avaliar_knn(d2, Jb, Tb)
            _, _, Ec = G.avaliar_knn(d2, Jc, Tc)
            acc["personalizado"].append(M[np.arange(N), np.arange(N)].mean())
            acc["igualado_atual"].append(Ea.mean())
            acc["igualado_equilibrado"].append(Eb.mean())
            acc["igualado_equilibrado_mesmos"].append(Ec.mean())
        out[mundo] = {k: round(100 * float(np.mean(v)), 2) for k, v in acc.items()}
        out[mundo]["ganho_1b_atual"] = round(out[mundo]["personalizado"] - out[mundo]["igualado_atual"], 2)
        out[mundo]["ganho_1b_equilibrado"] = round(out[mundo]["personalizado"] - out[mundo]["igualado_equilibrado"], 2)
        print(mundo, out[mundo], flush=True)
    json.dump({"semente": semente, "reps": reps, "N": N, "K": K, "resultados": out}, open("sorteio_doadores.json", "w"), indent=1)

main(sys.argv[1])
