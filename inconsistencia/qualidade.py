"""A qualidade da gravação que o STI não consegue separar da fala."""
from __future__ import annotations

import numpy as np

LIMIAR_SEGURO_S = 0.050
LIMIAR_INFLADO_S = 0.100

FRACAO_DO_PICO = 0.05


def _onset(sinal, fracao=FRACAO_DO_PICO):
    """Primeira amostra acima de `fracao` do pico."""
    envoltoria = np.abs(np.asarray(sinal, dtype=float))
    if envoltoria.size == 0 or envoltoria.max() <= 0:
        return 0
    return int(np.argmax(envoltoria > fracao * envoltoria.max()))


def _offset(sinal, fracao=FRACAO_DO_PICO):
    """Última amostra acima de `fracao` do pico."""
    envoltoria = np.abs(np.asarray(sinal, dtype=float))
    if envoltoria.size == 0 or envoltoria.max() <= 0:
        return len(envoltoria)
    acima = np.nonzero(envoltoria > fracao * envoltoria.max())[0]
    return int(acima[-1]) + 1 if acima.size else len(envoltoria)


def dispersao_de_inicio(sinais, taxa):
    """Desvio-padrão, em segundos, do instante em que a fala começa."""
    usaveis = [s for s in sinais if np.asarray(s).size > 0]
    if len(usaveis) < 2 or taxa <= 0:
        return None
    return float(np.std([_onset(s) / float(taxa) for s in usaveis]))


def fracao_de_silencio(sinal):
    """Quanto da gravação está fora do trecho com fala, entre 0 e 1."""
    x = np.asarray(sinal, dtype=float)
    if x.size == 0:
        return None
    inicio, fim = _onset(x), _offset(x)
    return float(1.0 - max(0, fim - inicio) / x.size)


def veredito_do_alinhamento(dispersao_s):
    """'nao_medido' | 'seguro' | 'limitrofe' | 'inflado' — nunca 'reprovado'."""
    if dispersao_s is None:
        return "nao_medido"
    if dispersao_s < LIMIAR_SEGURO_S:
        return "seguro"
    if dispersao_s < LIMIAR_INFLADO_S:
        return "limitrofe"
    return "inflado"


def qualidade_de_uma_frase(sinais, taxa):
    """O quadro de qualidade das repetições de uma frase."""
    dispersao = dispersao_de_inicio(sinais, taxa)
    fracoes = [f for f in (fracao_de_silencio(s) for s in sinais) if f is not None]
    return {
        "n_repeticoes": len(sinais),
        "dispersao_de_inicio_s": dispersao,
        "veredito_alinhamento": veredito_do_alinhamento(dispersao),
        "fracao_de_silencio_mediana": float(np.median(fracoes)) if fracoes else None,
        "fracao_de_silencio_maxima": float(max(fracoes)) if fracoes else None,
        "aviso": _aviso(dispersao, fracoes),
    }


def _aviso(dispersao, fracoes):
    """A frase que acompanha o número, e ela diz a direção do viés."""
    partes = []
    veredito = veredito_do_alinhamento(dispersao)
    if veredito == "inflado":
        partes.append(
            "As repetições começam em instantes muito diferentes ({:.0f} ms de "
            "dispersão). Acima de 100 ms o índice fica inflado em cerca de um "
            "quinto por razão que NÃO é fala.".format(dispersao * 1000)
        )
    elif veredito == "limitrofe":
        partes.append(
            "Dispersão de início de {:.0f} ms — entre o limiar seguro (50 ms) e "
            "o de inflação (100 ms).".format(dispersao * 1000)
        )
    if fracoes and max(fracoes) > 0.5:
        partes.append(
            "Uma das repetições é {:.0f}% silêncio. Silêncio DILUI o índice: o "
            "viés é para BAIXO, e a gravação parece MAIS consistente do que é."
            .format(max(fracoes) * 100)
        )
    if not partes:
        return None
    return " ".join(partes)

