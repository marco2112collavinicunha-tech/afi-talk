"""O ganho do personalizado sobre as demais condições: motor, não interface."""

PAPEIS = {"personalizado": "interesse", "generico": "piso", "entre_pessoas": "controle"}


def ganhos_do_personalizado(comparacao):
    """Ganho = personalizado − condição, para cada outra condição do quadro."""
    por_condicao = {linha["condicao"]: linha for linha in comparacao}
    base = por_condicao.get("personalizado")
    if base is None:
        return {"disponivel": False, "acuracia_personalizado": None, "n": None, "ganhos": []}

    acc_p = float(base["acuracia"])
    ganhos = []
    for linha in comparacao:
        cond = linha["condicao"]
        if cond == "personalizado":
            continue
        acc_c = float(linha["acuracia"])
        ganhos.append(
            {
                "condicao": cond,
                "papel": linha.get("papel", PAPEIS.get(cond, "-")),
                "metrica": linha.get("metrica", linha.get("esquema", "-")),
                "acuracia_condicao": acc_c,
                "n_condicao": linha.get("n"),
                "ganho": acc_p - acc_c,
            }
        )
    return {
        "disponivel": True,
        "acuracia_personalizado": acc_p,
        "n": base.get("n"),
        "ganhos": ganhos,
    }

