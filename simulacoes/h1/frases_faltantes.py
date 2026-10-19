"""Primeira comparação quando faltam frases a alguns participantes (dados sintéticos).

Uso: python frases_faltantes.py <saida.json> [replicas]

A regra do estudo: o ganho de cada participante é a média das frases que ele tem, com o mesmo
peso para cada uma, sem imputação, e o teste é o do programa (ganho_frases_fixas), no nível do
estudo. A tabela completa participante x frase é o ganho médio + efeito do participante + efeito
da frase (centrado nas doze) + ruído, com ou sem interação participante x frase. Depois somem
células de dois jeitos:
  - aleatoria: cada célula some com chance de 25%;
  - ligada_ao_ganho: células de ganho baixo somem mais (chance de 5% + 55%/(1 + exp(ganho/0,10))),
    o caso desfavorável para a regra.
Cada participante fica com pelo menos quatro das doze frases. Ganho médio verdadeiro de 0
(falso positivo) e de 10 pontos (poder). Não cobre fala apráxica, Whisper real, seleção de
configuração nem outros mecanismos de perda.
"""
import sys, json, time
import numpy as np
from comum import prog, meta, wilson

SEMENTE = 20261009
F = 12
EFEITO = 0.10
NS = (3, 5, 8, 12)
PERDAS = ("aleatoria", "ligada_ao_ganho")
ESTRUTURAS = ("aditiva", "com_interacao")


def tabela_completa(rng, n, efeito, estrutura):
    pessoa = rng.normal(0, 0.12, size=n)
    frase = rng.normal(0, 0.08, size=F)
    frase -= frase.mean()
    erro = rng.normal(0, 0.055, size=(n, F))
    inter = 0.0
    if estrutura == "com_interacao":
        inter = rng.normal(0, 0.08, size=(n, F))
        inter -= inter.mean(axis=1, keepdims=True)
    return np.clip(efeito + pessoa[:, None] + frase[None, :] + erro + inter, -1.0, 1.0)


def tirar_celulas(rng, completa, perda):
    if perda == "aleatoria":
        prob = np.full(completa.shape, 0.25)
    else:
        prob = 0.05 + 0.55 / (1 + np.exp(completa / 0.10))
    for _ in range(1000):
        falta = rng.random(completa.shape) < prob
        if np.all((~falta).sum(axis=1) >= 4):
            obs = completa.copy()
            obs[falta] = np.nan
            return obs
    raise RuntimeError("não saiu tabela com pelo menos quatro frases por participante")


def celula(n, perda, estrutura, efeito, R, semente):
    rng = np.random.default_rng(semente)
    rej = cobre = vereditos = 0
    soma_est = soma_vies_falta = soma_faltando = 0.0
    for _ in range(R):
        completa = tabela_completa(rng, n, efeito, estrutura)
        obs = tirar_celulas(rng, completa, perda)
        soma_faltando += np.isnan(obs).mean()
        r = prog.ganho_frases_fixas(obs)
        if r["status"] not in ("passou", "negativo"):
            continue
        vereditos += 1
        rej += r["status"] == "passou"
        lo, hi = r["intervalo_do_teste"]["limites"]
        cobre += lo <= efeito <= hi
        soma_est += r["ganho"]
        soma_vies_falta += r["ganho"] - completa.mean()
    v = max(vereditos, 1)
    return dict(N=n, perda=perda, estrutura=estrutura, efeito=efeito, R=R,
                fracao_de_celulas_faltando=soma_faltando / R,
                sem_veredito=(R - vereditos) / R,
                rejeicao=rej / v, ic95_wilson=wilson(rej, v),
                cobertura_do_intervalo_do_teste=cobre / v,
                vies=soma_est / v - efeito, vies_pelas_faltantes=soma_vies_falta / v)


def main():
    out = sys.argv[1]
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    t0 = time.time()
    grade = [(n, p, e, ef) for n in NS for p in PERDAS for e in ESTRUTURAS for ef in (0.0, EFEITO)]
    filhos = np.random.SeedSequence(SEMENTE).spawn(len(grade))
    res = []
    for (n, p, e, ef), f in zip(grade, filhos):
        res.append(celula(n, p, e, ef, R, int(f.generate_state(1)[0])))
        c = res[-1]
        print("N=%2d %-15s %-13s efeito %.2f  rejeição %6.2f%% [%.2f; %.2f]  faltando %.1f%%" % (
            n, p, e, ef, 100 * c["rejeicao"], 100 * c["ic95_wilson"][0], 100 * c["ic95_wilson"][1],
            100 * c["fracao_de_celulas_faltando"]), flush=True)
    json.dump(dict(meta(), script="frases_faltantes.py", semente_raiz=SEMENTE, replicas=R,
                   alfa_unilateral=prog.ALFA_DA_H1A, funcao_que_decide="ganho_frases_fixas",
                   regra="média das frases presentes de cada participante, sem imputação",
                   celulas=res, segundos=time.time() - t0), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
