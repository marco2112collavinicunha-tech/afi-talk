#!/usr/bin/env python3
"""Tabelas dos resultados da primeira comparação (resultados/tabelas.md)."""
import glob
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
PASTA = os.path.join(AQUI, "resultados")
NOMES = {"S1": "simétrico", "S2": "o Personalizado falha em 20%", "S3": "o Personalizado falha muito em 10%",
         "S4": "o Genérico falha em 20%", "S5": "trios por serviço, outros serviços",
         "S6": "metade num serviço, outros serviços", "S7": "trios por serviço, estes serviços",
         "S8": "S2 e S5 juntos", "S9": "assimetria forte"}
ROTULO = {"aleatoria": "aleatória", "ligada_ao_ganho": "ligada ao ganho", "aditiva": "aditiva",
          "com_interacao": "com interação"}
TESTES = [("t_programa", "t do programa"), ("johnson", "t de Johnson"), ("bootstrap_t", "bootstrap-t"),
          ("sinais_programa", "permutação de sinais"), ("wilcoxon", "Wilcoxon"),
          ("aparada_a", "média aparada (a)"), ("aparada_b", "média aparada (b)")]


def pc(x):
    return "{:.2f}".format(100 * x).replace(".", ",")


def ler(nome):
    p = os.path.join(PASTA, nome)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def nivel():
    L = ["## Nível do teste (`a_nivel.py`)", "",
         "Falso positivo (ganho verdadeiro 0) e poder (ganho de 10 e 15 pontos) do t do programa. "
         "Pior falso positivo entre 3 e 12 participantes, com o N entre parênteses, e poder com 12 "
         "participantes. Em porcentagem.", "",
         "| cenário | pior FP a 5% | pior FP a 4% | poder +10 a 5% | poder +10 a 4% | poder +15 a 5% | poder +15 a 4% |",
         "|---|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(PASTA, "nivel_S*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        c = d["celulas"]

        def pior(a):
            x = max((k for k in c if k["alvo"] == 0.0), key=lambda k: k["rejeicao"][a]["taxa"])
            return "{} ({})".format(pc(x["rejeicao"][a]["taxa"]), x["N"])

        def poder(alvo, a):
            x = [k for k in c if k["alvo"] == alvo and k["N"] == 12][0]
            return pc(x["rejeicao"][a]["taxa"])
        L.append("| {} {} | {} | {} | {} | {} | {} | {} |".format(
            d["cenario"], NOMES[d["cenario"]], pior("0.05"), pior("0.04"), poder(0.10, "0.05"),
            poder(0.10, "0.04"), poder(0.15, "0.05"), poder(0.15, "0.04")))
    L += ["", "Poder do t do programa a 4% pelo número de participantes, no cenário simétrico (Figura 6):", "",
          "| N | +10 | +15 |", "|---|---|---|"]
    d = ler("nivel_S1.json")
    if d:
        for n in sorted({k["N"] for k in d["celulas"]}):
            v = {k["alvo"]: k["rejeicao"]["0.04"]["taxa"] for k in d["celulas"] if k["N"] == n}
            L.append("| {} | {} | {} |".format(n, pc(v[0.10]), pc(v[0.15])))
    return L + [""]


def alternativas():
    L = ["## Alternativas ao t (`a_h1.py`)", "",
         "Todos os testes no mesmo nível. Pior falso positivo entre 3 e 12 participantes (N entre "
         "parênteses) e poder com ganho de 10 pontos com 3, 8 e 12 participantes. Em porcentagem.", ""]
    for p in sorted(glob.glob(os.path.join(PASTA, "alternativas_S*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        c = d["celulas"]
        L += ["### {} {} (nível de {}%, {} réplicas)".format(
            d["cenario"], NOMES[d["cenario"]], pc(d["alfa_unilateral"]).replace(",00", ""), d["replicas"]), "",
              "| teste | pior FP | poder +10, N = 3 | poder +10, N = 8 | poder +10, N = 12 |",
              "|---|---|---|---|---|"]
        for t, nome in TESTES:
            x = max((k for k in c if k["alvo"] == 0.0), key=lambda k: k["taxas"][t]["taxa"])
            pw = {k["N"]: k["taxas"][t]["taxa"] for k in c if k["alvo"] == 0.10}
            L.append("| {} | {} ({}) | {} | {} | {} |".format(
                nome, pc(x["taxas"][t]["taxa"]), x["N"], pc(pw.get(3, float("nan"))),
                pc(pw.get(8, float("nan"))), pc(pw.get(12, float("nan")))))
        L.append("")
    return L


def frases_como_amostra():
    d = ler("frases_como_amostra.json")
    if not d:
        return []
    c = d["celulas"]
    L = ["## Frases fixas e frases lidas como amostra (`frases_como_amostra.py`)", "",
         "Porcentagem de estudos em que a primeira comparação sai sustentada, a 4%, sem ganho médio. "
         "tau é o tamanho dos efeitos de frase. {} réplicas por célula.".format(d["replicas"]), "",
         "| tau | N | estas doze frases (falso positivo) | lido para outras frases |", "|---|---|---|---|"]
    for t in sorted({k["tau"] for k in c if k["variante"] != "ganho_igual"}):
        for n in sorted({k["N"] for k in c}):
            v = {k["variante"]: k["sustentada"] for k in c if k["tau"] == t and k["N"] == n}
            L.append("| {} | {} | {} | {} |".format(("%.2f" % t).replace(".", ","), n, pc(v["estas_frases"]),
                                                   pc(v["outras_frases"])))
    L += ["", "Poder com ganho de 10 pontos igual em todos os participantes e sem efeito de frase:", "",
          "| N | poder |", "|---|---|"]
    for k in [k for k in c if k["variante"] == "ganho_igual"]:
        L.append("| {} | {} |".format(k["N"], pc(k["sustentada"])))
    return L + [""]


def frases_faltantes():
    d = ler("frases_faltantes.json")
    if not d:
        return []
    L = ["## Frases faltantes (`frases_faltantes.py`)", "",
         "Regra das frases presentes, a 4%. FP = falso positivo (ganho 0); poder com ganho de 10 pontos; "
         "cobertura do intervalo de 92% sem ganho. {} réplicas por célula.".format(d["replicas"]), "",
         "| N | perda | estrutura | células faltando | FP | poder | cobertura |", "|---|---|---|---|---|---|---|"]
    c = d["celulas"]
    for k in [k for k in c if k["efeito"] == 0.0]:
        pw = [x for x in c if x["efeito"] > 0 and x["N"] == k["N"] and x["perda"] == k["perda"]
              and x["estrutura"] == k["estrutura"]][0]
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            k["N"], ROTULO[k["perda"]], ROTULO[k["estrutura"]],
            pc(k["fracao_de_celulas_faltando"]), pc(k["rejeicao"]), pc(pw["rejeicao"]),
            pc(k["cobertura_do_intervalo_do_teste"])))
    return L + [""]


def main():
    L = ["# Primeira comparação (dados sintéticos)", "",
         "Gerado por `tabela.py` a partir dos JSON desta pasta. SINTÉTICO: não cobre fala apráxica real.", ""]
    L += nivel() + alternativas() + frases_como_amostra() + frases_faltantes()
    with open(os.path.join(PASTA, "tabelas.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(os.path.join(PASTA, "tabelas.md"))


if __name__ == "__main__":
    main()
