"""Sensibilidade "só participantes com seis ou mais frases" quando faltam frases (dados sintéticos).

Uso: python frases_seis_ou_mais.py <saida.json> [replicas]

Mesmo gerador e mesmas perdas de frases_faltantes.py (semente própria). Em cada estudo, o teste
do programa (ganho_frases_fixas) é aplicado de dois modos: a todos os participantes (regra
registrada) e só aos que têm seis ou mais frases avaliáveis (sensibilidade do item 5 do campo de
análises exploratórias). Com menos de três participantes restantes, o estudo fica sem veredito na
sensibilidade. Ganho médio verdadeiro zero nas doze frases (falso positivo).
"""
import sys, json, time
import numpy as np
from comum import prog, meta, wilson
import frases_faltantes as FF

SEMENTE = 202611162
NS = (5, 8, 12)


def celula(n, perda, estrutura, R, semente):
    rng = np.random.default_rng(semente)
    rej_todos = rej_seis = v_todos = v_seis = 0
    soma_restantes = 0.0
    for _ in range(R):
        completa = FF.tabela_completa(rng, n, 0.0, estrutura)
        obs = FF.tirar_celulas(rng, completa, perda)
        r = prog.ganho_frases_fixas(obs)
        if r["status"] in ("passou", "negativo"):
            v_todos += 1; rej_todos += r["status"] == "passou"
        manter = np.isfinite(obs).sum(axis=1) >= 6
        soma_restantes += manter.sum()
        if manter.sum() >= 3:
            r6 = prog.ganho_frases_fixas(obs[manter])
            if r6["status"] in ("passou", "negativo"):
                v_seis += 1; rej_seis += r6["status"] == "passou"
    return dict(N=n, perda=perda, estrutura=estrutura, R=R,
                rejeicao_todos=rej_todos / max(v_todos, 1), ic95_todos=wilson(rej_todos, max(v_todos, 1)),
                rejeicao_seis_ou_mais=rej_seis / max(v_seis, 1), ic95_seis=wilson(rej_seis, max(v_seis, 1)),
                sem_veredito_seis=(R - v_seis) / R, participantes_restantes_medio=soma_restantes / R)


def main():
    out = sys.argv[1]
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    t0 = time.time()
    grade = [(n, p, e) for n in NS for p in FF.PERDAS for e in FF.ESTRUTURAS]
    filhos = np.random.SeedSequence(SEMENTE).spawn(len(grade))
    res = []
    for (n, p, e), f in zip(grade, filhos):
        res.append(celula(n, p, e, R, int(f.generate_state(1)[0])))
        c = res[-1]
        print("N=%2d %-15s %-13s todos %5.2f%%  seis ou mais %5.2f%% [%.2f; %.2f]  restantes %.1f" % (
            n, p, e, 100 * c["rejeicao_todos"], 100 * c["rejeicao_seis_ou_mais"],
            100 * c["ic95_seis"][0], 100 * c["ic95_seis"][1], c["participantes_restantes_medio"]), flush=True)
    m = meta(); m["semente_raiz"] = SEMENTE
    json.dump(dict(m, script="frases_seis_ou_mais.py", replicas=R, alfa_unilateral=prog.ALFA_DA_H1A,
                   celulas=res, segundos=time.time() - t0), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
