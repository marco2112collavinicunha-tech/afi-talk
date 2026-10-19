"""Divisão anti-circularidade: três conjuntos sem sobreposição, por pessoa."""
from __future__ import annotations

import math

import numpy as np

CONJUNTOS = ("treino", "teste", "consistencia")


def validar_conjunto(conjunto):
    """Valida o rótulo explícito de conjunto de uma gravação."""
    if conjunto not in CONJUNTOS:
        raise ValueError(
            "Conjunto inválido: {!r} — cada gravação pertence a um de {}.".format(
                conjunto, list(CONJUNTOS)
            )
        )
    return conjunto


def montar_divisao(conjuntos):
    """Monta a divisão a partir dos conjuntos explícitos por gravação."""
    conjuntos = list(conjuntos)
    if not conjuntos:
        raise ValueError("Não há gravações para montar a divisão (lista de conjuntos vazia).")
    divisao = {conjunto: [] for conjunto in CONJUNTOS}
    for indice, conjunto in enumerate(conjuntos):
        validar_conjunto(conjunto)
        divisao[conjunto].append(indice)
    return divisao


def sugerir_conjunto(fracoes, contagens, conjunto_da_sessao=None):
    """Sugere o conjunto do próximo exemplo de uma classe (modo auxiliar)."""
    _validar_fracoes(fracoes)
    if conjunto_da_sessao is not None:
        validar_conjunto(conjunto_da_sessao)
        return conjunto_da_sessao
    atuais = {conjunto: int(contagens.get(conjunto, 0)) for conjunto in CONJUNTOS}
    if sum(atuais.values()) == 0:
        return "treino"
    total_novo = sum(atuais.values()) + 1
    deficits = {
        conjunto: float(fracoes[conjunto]) * total_novo - atuais[conjunto]
        for conjunto in CONJUNTOS
    }
    return max(CONJUNTOS, key=lambda c: (deficits[c], -CONJUNTOS.index(c)))


def sugerir_conjuntos(fracoes, rotulos):
    contagens = {}
    conjuntos = []
    for rotulo in rotulos:
        da_classe = contagens.setdefault(rotulo, {c: 0 for c in CONJUNTOS})
        conjunto = sugerir_conjunto(fracoes, da_classe)
        da_classe[conjunto] += 1
        conjuntos.append(conjunto)
    return conjuntos


def _validar_fracoes(fracoes):
    """Config errada deve falhar com barulho, nunca dividir errado em silêncio."""
    if set(fracoes) != set(CONJUNTOS):
        raise ValueError(
            "splits.fracoes deve ter exatamente as chaves {}; vieram {}.".format(
                sorted(CONJUNTOS), sorted(fracoes)
            )
        )
    for conjunto, fracao in fracoes.items():
        if not 0.0 <= float(fracao) <= 1.0:
            raise ValueError(
                "splits.fracoes.{} deve estar entre 0 e 1; veio {!r}.".format(conjunto, fracao)
            )
    soma = sum(float(f) for f in fracoes.values())
    if abs(soma - 1.0) > 1e-9:
        raise ValueError("splits.fracoes deve somar 1.0; soma {}.".format(soma))


def _repartir(n, fracoes):
    """Decide quantos itens vão para cada conjunto, somando exatamente n."""
    alvos = {c: float(fracoes[c]) * n for c in CONJUNTOS}
    contagens = {c: int(math.floor(alvos[c])) for c in CONJUNTOS}
    sobra = n - sum(contagens.values())

    por_maior_resto = sorted(
        CONJUNTOS, key=lambda c: (-(alvos[c] - contagens[c]), CONJUNTOS.index(c))
    )
    for conjunto in por_maior_resto[:sobra]:
        contagens[conjunto] += 1

    if n >= 1 and contagens["treino"] == 0:
        doador = max(
            (c for c in CONJUNTOS if c != "treino"), key=lambda c: contagens[c]
        )
        contagens[doador] -= 1
        contagens["treino"] += 1

    return contagens


def dividir_tres_vias(rotulos, fracoes, seed, estratificado_por_classe=True):
    """Divide os índices [0..n-1] das gravações de uma pessoa em 3 conjuntos."""
    rotulos = list(rotulos)
    if not rotulos:
        raise ValueError("Não há gravações para dividir (lista de rótulos vazia).")
    _validar_fracoes(fracoes)

    rng = np.random.default_rng(seed)
    divisao = {conjunto: [] for conjunto in CONJUNTOS}

    if estratificado_por_classe:
        grupos = [
            [i for i, r in enumerate(rotulos) if r == classe]
            for classe in sorted(set(rotulos), key=str)
        ]
    else:
        grupos = [list(range(len(rotulos)))]

    for indices in grupos:
        embaralhados = rng.permutation(np.asarray(indices, dtype=int))
        contagens = _repartir(len(embaralhados), fracoes)
        cursor = 0
        for conjunto in CONJUNTOS:
            fatia = embaralhados[cursor : cursor + contagens[conjunto]]
            divisao[conjunto].extend(int(i) for i in fatia)
            cursor += contagens[conjunto]

    for conjunto in CONJUNTOS:
        divisao[conjunto].sort()
    return divisao


def criar_divisao(config, rotulos):
    """Monta a divisão a partir do config.yaml (porta de entrada usada pelas fases seguintes)."""
    cfg = config["splits"]
    if cfg["esquema"] != "tres_vias":
        raise ValueError(
            "Requisito metodológico fixo: splits.esquema deve ser 'tres_vias'; "
            "veio {!r}.".format(cfg["esquema"])
        )
    return dividir_tres_vias(
        rotulos,
        fracoes=cfg["fracoes"],
        seed=config["reprodutibilidade"]["seed"],
        estratificado_por_classe=cfg["estratificado_por_classe"],
    )


def resumo_da_divisao(divisao, rotulos):
    """Contagem por conjunto e por classe — para relatórios e para a interface."""
    rotulos = list(rotulos)
    resumo = {}
    for conjunto in CONJUNTOS:
        contagem = {}
        for indice in divisao[conjunto]:
            classe = rotulos[indice]
            contagem[classe] = contagem.get(classe, 0) + 1
        resumo[conjunto] = contagem
    return resumo

