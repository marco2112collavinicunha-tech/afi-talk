#!/usr/bin/env python3
"""Tabelas da sequência completa (resultados/tabelas.md)."""
import glob
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
PASTA = os.path.join(AQUI, "resultados")


def pct(x):
    return "{:.2f}".format(100 * x).replace(".", ",")


def ic(w):
    return "[{}; {}]".format(pct(w[0]), pct(w[1]))


def main():
    js = sorted((json.load(open(f, encoding="utf-8")) for f in glob.glob(os.path.join(PASTA, "a*.json"))),
                key=lambda d: d["N"])
    out = ["# Sequência completa das três comparações (dados sintéticos)", "",
           "Gerado por `tabela.py` a partir dos JSON desta pasta. Semente raiz 20261006; "
           + ", ".join("N = {}: {} réplicas".format(d["N"], d["replicas"]) for d in js)
           + ". Porcentagens com o intervalo de Wilson de 95%. FWER: pelo menos uma conclusão falsa "
           "na sequência inteira. G0: nenhuma das três comparações tem efeito. P1 a P5: só algumas têm "
           "(\"forçado\": a primeira comparação dada como sustentada, o pior caso). \"Regra anterior\": "
           "escolha forçada e Entre-pessoas igualado cada um a 4%, sem o Holm. S1: ganho simétrico; "
           "S2: o Personalizado falha em 20% dos participantes.", ""]
    for forma in ("S1", "S2"):
        out += ["## FWER, forma {}".format(forma), "",
                "| célula | N | primeira comparação sustentada | FWER da sequência (Holm) | FWER, forçado | "
                "FWER, regra anterior | regra anterior, forçado |", "|---|---|---|---|---|---|---|"]
        for c in ("G0", "P1", "P2", "P3", "P4", "P5"):
            for d in js:
                x = d["celulas"][forma][c]
                sem = c in ("G0", "P5")
                f1 = "—" if sem else "{} {}".format(pct(x["fwer_forcado"]["taxa"]), ic(x["fwer_forcado"]["wilson95"]))
                f2 = "—" if sem else "{} {}".format(pct(x["fwer_regra_anterior_forcado"]["taxa"]),
                                                    ic(x["fwer_regra_anterior_forcado"]["wilson95"]))
                out.append("| {} | {} | {} | {} {} | {} | {} {} | {} |".format(
                    c, d["N"], pct(x["d1"]["taxa"]), pct(x["fwer"]["taxa"]), ic(x["fwer"]["wilson95"]), f1,
                    pct(x["fwer_regra_anterior"]["taxa"]), ic(x["fwer_regra_anterior"]["wilson95"]), f2))
        out.append("")
    for forma in ("S1", "S2"):
        out += ["## Poder de cada componente, forma {}".format(forma), "",
                "| cenário | N | primeira | primeira e escolha forçada | primeira e Entre-pessoas | as três | "
                "as três, regra anterior |", "|---|---|---|---|---|---|---|"]
        for c in ("W10", "W15"):
            for d in js:
                x = d["celulas"][forma][c]
                out.append("| {} | {} | {} {} | {} | {} | {} {} | {} |".format(
                    c, d["N"], pct(x["d1"]["taxa"]), ic(x["d1"]["wilson95"]), pct(x["d1_e_d2"]["taxa"]),
                    pct(x["d1_e_b"]["taxa"]), pct(x["d1_e_d2_e_b"]["taxa"]),
                    ic(x["d1_e_d2_e_b"]["wilson95"]), pct(x["d1_e_d2_e_b_regra_anterior"]["taxa"])))
        out.append("")
    shas = sorted({d["sha256_programa"] for d in js})
    out += ["sha256 de `analise/analise_do_estudo.py` registrado: " + ", ".join(s[:12] for s in shas), ""]
    with open(os.path.join(PASTA, "tabelas.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(os.path.join(PASTA, "tabelas.md"), len(js), "arquivos")


if __name__ == "__main__":
    main()
