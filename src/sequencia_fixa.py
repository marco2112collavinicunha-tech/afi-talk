"""Teste em sequência fixa para os dois degraus da H1a."""
from __future__ import annotations

PASSOU = "passou"
NAO_PASSOU = "nao_passou"
NAO_TESTADO = "nao_testado"
NAO_CALCULAVEL = "nao_calculavel"
ESTADOS = (PASSOU, NAO_PASSOU, NAO_TESTADO, NAO_CALCULAVEL)


def estado_do_veredito(veredito):
    """Estado de um degrau a partir do dicionário de veredito da H1a."""
    if veredito is None:
        return NAO_CALCULAVEL, "o degrau não produziu veredito"
    valor = veredito.get("veredito")
    if valor is True:
        return PASSOU, None
    if valor is False:
        return NAO_PASSOU, None
    if valor is None:
        return NAO_CALCULAVEL, str(veredito.get("motivo") or "veredito ausente, sem motivo registrado")
    raise ValueError("veredito deve ser True, False ou None; veio {!r}".format(valor))


def sequencia_fixa(veredito_do_degrau_1, calcular_degrau_2):
    """Os dois degraus em ordem; o segundo só é calculado se o primeiro passar."""
    estado_1, motivo_1 = estado_do_veredito(veredito_do_degrau_1)
    saida = {
        "degrau_1": {"comparador": "generico_livre", "estado": estado_1,
                     "motivo": motivo_1, "veredito": veredito_do_degrau_1},
    }
    if estado_1 != PASSOU:
        saida["degrau_2"] = {
            "comparador": "whisper_escolha_forcada", "estado": NAO_TESTADO,
            "motivo": "o degrau 1 não passou ({}); pela sequência fixa, o degrau 2 "
                      "não é testado".format(estado_1),
            "veredito": None,
        }
        return saida
    try:
        veredito_2 = calcular_degrau_2()
    except (ValueError, RuntimeError) as erro:
        saida["degrau_2"] = {"comparador": "whisper_escolha_forcada",
                             "estado": NAO_CALCULAVEL, "motivo": str(erro),
                             "veredito": None}
        return saida
    estado_2, motivo_2 = estado_do_veredito(veredito_2)
    saida["degrau_2"] = {"comparador": "whisper_escolha_forcada", "estado": estado_2,
                         "motivo": motivo_2, "veredito": veredito_2}
    return saida

