"""Primeira comparação com efeitos de frase: as doze frases fixas e as frases lidas como amostra
(dados sintéticos).

Uso: python frases_como_amostra.py <saida.json> [replicas]

Cada sistema recebe um desvio por frase, igual para todos os participantes, N(0, tau/raiz(2)),
com tau de 5, 10 e 20 pontos. Em cada estudo, quem decide é a função do programa que decide a
primeira comparação (h1a_frases_fixas), no nível do estudo.
  - estas_frases: desvios centrados nas doze frases, com ganho médio nulo nestas frases.
    A taxa é o falso positivo da conclusão do estudo, que vale para estas doze frases.
  - outras_frases: desvios como foram sorteados. As doze frases são tratadas como amostra de
    outras frases, e a taxa diz quanto o teste acusaria se o resultado fosse lido como válido
    para frases parecidas com elas.
  - ganho_igual: sem desvio por frase e ganho de 10 pontos igual em todos (poder).
Gerador: 12 frases x 6 gravações de teste, acerto-base de 45% nos dois sistemas,
gravações independentes. Não cobre fala apráxica, Whisper real, frases faltando, seleção de
configuração, efeito de sessão nem correlação entre sistemas no mesmo áudio.
"""
import sys, json, time
import numpy as np
from comum import prog, meta, wilson

SEMENTE = 20261008
F, T, BASE, CLIP = 12, 6, 0.45, (0.02, 0.98)
NS = (3, 4, 5, 6, 8, 10, 12)
TAUS = (0.05, 0.10, 0.20)
GANHO_IGUAL = 0.10
VARIANTES = ("estas_frases", "outras_frases", "ganho_igual")
BLOCO = prog.CONTRASTES_DA_H1["H1a"]["bloco"]


def desvio(rng, tau, centrado):
    e = rng.normal(0.0, tau / np.sqrt(2.0), (3, F)) if tau > 0 else np.zeros((3, F))
    if centrado:
        e = e - e.mean(axis=1, keepdims=True)
    return e


def sortear(rng, n, ganho, e=None):
    """Acertos 0/1 do Personalizado e do Genérico, (n, F, T)."""
    rng.normal(0.0, 1.0, n)  # contribuição ao banco do Entre-pessoas, que aqui não entra
    eP, eG = (e[0], e[1]) if e is not None else (0.0, 0.0)
    pG = np.clip(np.full((n, F), BASE) + eG, *CLIP)
    pP = np.clip(BASE + ganho + eP, *CLIP)
    u = rng.random((3, n, F, T))
    return (u[0] < pP[..., None]).astype(int), (u[1] < pG[..., None]).astype(int)


def por_participante(P, G):
    return {"P{:02d}".format(i + 1): {BLOCO: {"delta_por_frase": {
        "f{:02d}".format(j + 1): float(P[i, j].mean() - G[i, j].mean()) for j in range(F)}}}
        for i in range(P.shape[0])}


def celula(variante, n, tau, R):
    sim = sem = 0
    for r in range(R):
        rng = np.random.default_rng([SEMENTE, n, int(round(1000 * tau)), VARIANTES.index(variante), r])
        if variante == "ganho_igual":
            P, G = sortear(rng, n, np.full((n, F), GANHO_IGUAL))
        else:
            e = desvio(rng, tau, variante == "estas_frases")
            P, G = sortear(rng, n, np.zeros((n, F)), e)
        v = prog.h1a_frases_fixas(por_participante(P, G))["veredito"]
        sim += v is True
        sem += v is None
    return dict(variante=variante, N=n, tau=tau, R=R, sustentada=sim / R,
                ic95_wilson=wilson(sim, R), sem_veredito=sem / R)


def main():
    out = sys.argv[1]
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    t0 = time.time()
    grade = [(v, n, t) for v in VARIANTES[:2] for t in TAUS for n in NS] + [("ganho_igual", n, 0.0) for n in NS]
    res = []
    for v, n, t in grade:
        res.append(celula(v, n, t, R))
        c = res[-1]
        print("%-13s N=%2d tau=%.2f sustentada %6.2f%% [%.2f; %.2f]" % (
            v, n, t, 100 * c["sustentada"], 100 * c["ic95_wilson"][0], 100 * c["ic95_wilson"][1]), flush=True)
    json.dump(dict(meta(), script="frases_como_amostra.py", semente_raiz=SEMENTE, replicas=R,
                   alfa_unilateral=prog.ALFA_DA_H1A, funcao_que_decide="h1a_frases_fixas",
                   F=F, T=T, base=BASE, ganho_igual=GANHO_IGUAL, celulas=res,
                   segundos=time.time() - t0), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
