"""Teste t, permutação de sinais e Wilcoxon quando os ganhos têm média nula e distribuição
assimétrica (dados sintéticos).

Uso: python misturas_media_nula.py <saida.json> [replicas]

Cada participante tem ganho verdadeiro em pontos de UAR: com probabilidade q, perde muito
(-L, desvio 0,05); senão, ganha pouco (+g, desvio s), com g = qL/(1-q), de modo que a média
da população é zero. É o caso em que poucos perdem muito e muitos ganham pouco, com menos
dispersão entre os que ganham. A cada ganho soma-se o erro de medida do participante
(desvio 0,06, cerca do de 72 gravações de teste). Testes unilaterais (ganho > 0) a 4 %:
t de uma amostra; permutação de sinais exata; Wilcoxon dos postos sinalizados (SciPy, padrão).
"""
import sys, json, time, itertools
import numpy as np
from scipy import stats
from comum import meta, wilson

SEMENTE = 202611161
ALFA = 0.04
NS = (3, 5, 8, 12)
QS = (0.1, 0.2, 0.3)
LS = (0.20, 0.40)
SW = (0.02, 0.05)


def sinais(n):
    return np.array(list(itertools.product([-1, 1], repeat=n)), float)


def celula(n, q, L, s, R, semente):
    rng = np.random.default_rng(semente)
    g = q * L / (1 - q)
    perde = rng.random((R, n)) < q
    d = np.where(perde, rng.normal(-L, 0.05, (R, n)), rng.normal(g, s, (R, n)))
    d = d + rng.normal(0, 0.06, (R, n))
    m = d.mean(1); sd = d.std(1, ddof=1)
    t = m / (sd / np.sqrt(n))
    rej_t = t > stats.t.ppf(1 - ALFA, n - 1)
    S = sinais(n)
    rej_p = np.empty(R, bool)
    for a in range(0, R, 2000):
        b = min(R, a + 2000)
        medias = np.abs(d[a:b]) @ S.T / n
        p = (medias >= m[a:b, None] - 1e-12).mean(1)
        rej_p[a:b] = p <= ALFA
    pw = np.array([stats.wilcoxon(x, alternative="greater").pvalue for x in d])
    rej_w = pw <= ALFA
    out = dict(N=n, q=q, L=L, desvio_dos_que_ganham=s, R=R)
    for nome, r in (("t", rej_t), ("permutacao_de_sinais", rej_p), ("wilcoxon", rej_w)):
        k = int(r.sum()); out[nome] = k / R; out[nome + "_ic95"] = wilson(k, R)
    return out


def main():
    out = sys.argv[1]
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    t0 = time.time()
    grade = [(n, q, L, s) for n in NS for q in QS for L in LS for s in SW]
    filhos = np.random.SeedSequence(SEMENTE).spawn(len(grade))
    res = []
    for (n, q, L, s), f in zip(grade, filhos):
        res.append(celula(n, q, L, s, R, int(f.generate_state(1)[0])))
        c = res[-1]
        print("N=%2d q=%.1f L=%.2f s=%.2f  t %5.2f%%  perm %5.2f%%  wilcoxon %5.2f%%" % (
            n, q, L, s, 100 * c["t"], 100 * c["permutacao_de_sinais"], 100 * c["wilcoxon"]), flush=True)
    m = meta(); m["semente_raiz"] = SEMENTE
    json.dump(dict(m, script="misturas_media_nula.py", replicas=R, alfa_unilateral=ALFA,
                   celulas=res, segundos=time.time() - t0), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
