#!/usr/bin/env python3
"""Tabelas dos resultados de h1b_igualado_dois_metodos.py e de t_comum_igualado.py."""
import glob
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(os.path.dirname(AQUI))
PASTA = os.path.join(AQUI, "resultados")
MUNDOS = ["nulo_exato", "nulo_heterogeneo", "entre_baixa", "entre_media", "entre_alta",
          "pior_caso"]


def carregar():
    dados = {}
    for p in glob.glob(os.path.join(PASTA, "*_N*.json")):
        d = json.load(open(p, encoding="utf-8"))
        dados[(d["mundo"], d["N"])] = d
    return dados


def celula(d, forma, nome, alfa="0.04"):
    t = d["taxas"][forma][nome][alfa]
    lo, hi = t["wilson95"]
    return "{:.2f} [{:.2f}; {:.2f}]".format(100 * t["taxa"], 100 * lo, 100 * hi)


def celula_t(d, alfa):
    t = d["taxas"]["centrado"][alfa]
    lo, hi = t["wilson95"]
    return "{:.2f} [{:.2f}; {:.2f}]".format(100 * t["taxa"], 100 * lo, 100 * hi)


def main():
    dados = carregar()
    ns = sorted({n for _, n in dados})
    linhas = ["# Comparação com o Entre-pessoas igualado (dados sintéticos)", "",
              "Gerado por `tabela.py` a partir dos "
              "JSON desta pasta. Semente raiz 20261005, K = 20 sorteios, nível 4%. Porcentagem "
              "de estudos com \"ganho > 0\" e, entre colchetes, o intervalo de Wilson de 95%. "
              "n = réplicas por célula (coluna própria). SINTÉTICO: vetores gaussianos e kNN-1; "
              "não cobre fala apráxica real, seleção de configuração, regressão logística, "
              "empate 1/k, frases faltando, serviço comum nem o Genérico.", ""]
    for forma, titulo in (("centrado", "Falso positivo (ganho verdadeiro subtraído)"),
                          ("natural", "Poder (ganho verdadeiro do mundo)")):
        linhas += ["## " + titulo, "",
                   "| mundo | N | n | ganho verdadeiro do igualado | os dois (regra) | "
                   "só Nadeau e Bengio | só jackknife | os dois a 5% |",
                   "|---|---|---|---|---|---|---|---|"]
        for m in MUNDOS:
            for n in ns:
                d = dados.get((m, n))
                if d is None:
                    continue
                linhas.append("| {} | {} | {} | {:+.3f} | {} | {} | {} | {} |".format(
                    m, n, d["replicas"], d["ganho_verdadeiro_do_igualado"],
                    celula(d, forma, "ambos"), celula(d, forma, "nadeau_bengio"),
                    celula(d, forma, "jackknife"), celula(d, forma, "ambos", "0.05")))
        linhas.append("")
    tc = {}
    for p in glob.glob(os.path.join(PASTA, "t_comum", "*_N*.json")):
        d = json.load(open(p, encoding="utf-8"))
        tc[(d["mundo"], d["N"])] = d
    if tc:
        linhas += ["## Teste t comum, que trata os ganhos como independentes (`t_comum_igualado.py`)", "",
                   "Mesmos estudos das tabelas acima. Falso positivo, com o ganho verdadeiro subtraído.", "",
                   "| mundo | N | n | t comum a 4% | t comum a 5% |", "|---|---|---|---|---|"]
        for m in MUNDOS:
            for n in ns:
                d = tc.get((m, n))
                if d is not None:
                    linhas.append("| {} | {} | {} | {} | {} |".format(
                        m, n, d["replicas"], celula_t(d, "0.04"), celula_t(d, "0.05")))
        linhas.append("")
    shas = sorted({d["sha256_programa"] for d in list(dados.values()) + list(tc.values())})
    linhas += ["sha256 de `analise/analise_do_estudo.py` registrado nas células"
               ": " + ", ".join(s[:12] for s in shas), ""]
    with open(os.path.join(PASTA, "tabelas.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    print(os.path.join(PASTA, "tabelas.md"), len(dados), "células")


if __name__ == "__main__":
    main()

