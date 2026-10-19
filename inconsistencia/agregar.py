"""Medida de inconsistência, agregação: dos STIs por frase ao índice do participante."""
from __future__ import annotations

import numpy as np

from inconsistencia.sti import calcular_sti


def gravacoes_de_consistencia(config, pessoa):
    """Gravações do conjunto de consistência de uma pessoa, por frase."""
    from src.io_store import listar_exemplos

    por_frase = {}
    for exemplo in listar_exemplos(config, pessoa):
        if exemplo["conjunto"] == "consistencia":
            por_frase.setdefault(exemplo["rotulo"], []).append(exemplo["caminho"])
    if not por_frase:
        raise ValueError(
            "Nenhuma gravação de {!r} está no conjunto 'consistencia' — o "
            "STI lê SOMENTE esse conjunto (marque gravações como "
            "'consistencia' ao salvá-las no banco).".format(pessoa)
        )
    return por_frase


def stis_por_frase(gravacoes_por_frase, config):
    """Calcula o STI de cada frase: {frase: {"sti", "n_repeticoes"}}."""
    if not gravacoes_por_frase:
        raise ValueError("Não há frases para calcular o STI (dict vazio).")
    resultado = {}
    for frase in sorted(gravacoes_por_frase):
        gravacoes = gravacoes_por_frase[frase]
        if len(gravacoes) < 2:
            raise ValueError(
                "A frase {!r} tem {} gravação(ões) no conjunto de "
                "consistência — o STI exige 2+ repetições da mesma frase. "
                "Grave mais repetições ou deixe a frase fora desta análise "
                "explicitamente.".format(frase, len(gravacoes))
            )
        resultado[frase] = {
            "sti": calcular_sti(gravacoes, config),
            "n_repeticoes": len(gravacoes),
        }
    return resultado


def agregar_stis(stis, config):
    """Agrega os STIs por frase num índice único do participante."""
    if not stis:
        raise ValueError("Não há STIs para agregar (dict vazio).")
    ns = {frase: dados["n_repeticoes"] for frase, dados in stis.items()}
    if len(set(ns.values())) > 1:
        raise ValueError(
            "As frases têm números de repetições DIFERENTES ({}) — o valor "
            "do STI depende de N, então agregá-los misturaria escalas "
            "incomparáveis. Iguale o N entre as frases (ou calcule por frase, "
            "sem agregar).".format(ns)
        )

    metodo = config["inconsistencia"]["agregacao"]
    valores = [dados["sti"] for dados in stis.values()]
    if metodo == "media":
        return float(np.mean(valores))
    if metodo == "mediana":
        return float(np.median(valores))
    raise ValueError(
        "Método de agregação desconhecido: {!r} — use 'media' ou 'mediana' "
        "(inconsistencia.agregacao).".format(metodo)
    )


def indice_do_participante(gravacoes_por_frase, config):
    """Ponta a ponta: gravações por frase → STIs por frase → índice único."""
    stis = stis_por_frase(gravacoes_por_frase, config)
    indice = agregar_stis(stis, config)
    return {
        "indice": indice,
        "agregacao": config["inconsistencia"]["agregacao"],
        "n_repeticoes": next(iter(stis.values()))["n_repeticoes"],
        "stis_por_frase": stis,
    }

