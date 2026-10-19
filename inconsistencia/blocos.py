"""Medida de inconsistência: o bloco de repetições, sua ordem de apresentação e sua elegibilidade."""
from __future__ import annotations

from itertools import combinations

import numpy as np

ABBA_BAAB = ("A", "B", "B", "A", "B", "A", "A", "B")

ORDENS = ("sequencial", "abba_baab")

CAMPO_DA_POSICAO = "posicao"
CAMPOS_DO_CANAL = ("dispositivo", "ambiente")


class OrdemDesconhecida(ValueError):
    """A ordem pedida não é uma das declaradas em `ORDENS`."""


class BlocosIncompletos(ValueError):
    """As gravações não dividem exatamente em blocos do tamanho pedido."""


def sequencia_de_posicoes(n_blocos, ordem="sequencial", posicoes=("A", "B")):
    """As posições, na ordem em que os blocos são gravados."""
    n_blocos = int(n_blocos)
    if n_blocos <= 0:
        raise ValueError(
            "Número de blocos tem de ser positivo; veio {}.".format(n_blocos))
    if len(posicoes) != 2:
        raise ValueError(
            "A ordem balanceada é definida para DUAS posições; vieram {}: {}. "
            "Com três ou mais, ABBA BAAB não existe e o balanceamento precisa "
            "de outro desenho (quadrado latino).".format(len(posicoes),
                                                         list(posicoes)))
    if ordem == "sequencial":
        if n_blocos % 2:
            raise BlocosIncompletos(
                "A ordem 'sequencial' divide os blocos em duas metades iguais, "
                "e {} é ímpar. Use um número par de blocos.".format(n_blocos))
        metade = n_blocos // 2
        return [posicoes[0]] * metade + [posicoes[1]] * metade
    if ordem == "abba_baab":
        if n_blocos % len(ABBA_BAAB):
            raise BlocosIncompletos(
                "A ordem 'abba_baab' só é BALANCEADA em múltiplos de {} blocos, "
                "e vieram {}. Com um múltiplo quebrado uma das posições passa a "
                "vir sistematicamente antes — que é o defeito que esta ordem "
                "existe para corrigir.".format(len(ABBA_BAAB), n_blocos))
        mapa = {"A": posicoes[0], "B": posicoes[1]}
        repeticoes = n_blocos // len(ABBA_BAAB)
        return [mapa[p] for _ in range(repeticoes) for p in ABBA_BAAB]
    raise OrdemDesconhecida(
        "Ordem de apresentação desconhecida: {!r}. As declaradas são {}. A "
        "'abba_baab' é PROPOSTA (depende da escolha do pesquisador "
        "responsável): ela não é o padrão de lugar nenhum.".format(ordem, list(ORDENS)))


def desequilibrio_temporal(posicoes):
    """Quanto uma posição vem, em média, depois da outra, em número de blocos."""
    rotulos = list(posicoes)
    unicos = sorted(set(rotulos))
    if len(unicos) != 2:
        raise ValueError(
            "O desequilíbrio temporal compara DUAS posições; a sequência traz "
            "{}: {}.".format(len(unicos), unicos))
    indices = {u: [i for i, r in enumerate(rotulos) if r == u] for u in unicos}
    medias = {u: float(np.mean(v)) for u, v in indices.items()}
    return {
        "posicoes": unicos,
        "indice_medio": medias,
        "desequilibrio": abs(medias[unicos[0]] - medias[unicos[1]]),
        "blocos_por_posicao": {u: len(v) for u, v in indices.items()},
    }


def dividir_em_blocos(itens, tamanho):
    """Divide a lista na ordem de gravação, em blocos de `tamanho`."""
    itens = list(itens)
    tamanho = int(tamanho)
    if tamanho < 2:
        raise ValueError(
            "Um bloco precisa de pelo menos 2 repetições para ter STI; veio "
            "tamanho {}.".format(tamanho))
    if not itens:
        raise BlocosIncompletos("Não há gravações para dividir em blocos.")
    if len(itens) % tamanho:
        raise BlocosIncompletos(
            "{} gravações não dividem em blocos de {} — sobram {}. O STI "
            "depende de N, então um bloco curto não é comparável aos outros. "
            "Grave o bloco inteiro ou declare o tamanho real."
            .format(len(itens), tamanho, len(itens) % tamanho))
    return [itens[i:i + tamanho] for i in range(0, len(itens), tamanho)]


def _p_por_permutacao_exata(valores, posicoes):
    """p bilateral por enumeração de todas as atribuições de rótulo."""
    rotulos = list(posicoes)
    unicos = sorted(set(rotulos))
    valores = np.asarray(valores, dtype=float)
    a = [i for i, r in enumerate(rotulos) if r == unicos[0]]
    observado = abs(valores[a].mean() - np.delete(valores, a).mean())
    n, na = len(rotulos), len(a)
    extremos = 0
    total = 0
    for combinacao in combinations(range(n), na):
        grupo = np.array(combinacao)
        outro = np.setdiff1d(np.arange(n), grupo)
        total += 1
        if abs(valores[grupo].mean() - valores[outro].mean()) >= observado - 1e-12:
            extremos += 1
    return {
        "p_bilateral": extremos / total,
        "atribuicoes": total,
        "menor_p_atingivel": 2.0 / total,
    }


def _spearman(x, y):
    """Correlação de postos, sem scipy — só numpy (e sem empate aqui)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    px, py = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    px, py = px - px.mean(), py - py.mean()
    den = float(np.sqrt((px ** 2).sum() * (py ** 2).sum()))
    return float((px * py).sum() / den) if den > 0 else 0.0


def resumo_por_posicao(stis, posicoes):
    """A leitura do piloto: STI por bloco, variação dentro e entre posições."""
    stis = [float(s) for s in stis]
    rotulos = list(posicoes)
    if len(stis) != len(rotulos):
        raise ValueError(
            "{} STIs para {} posições — cada bloco tem exatamente um STI e uma "
            "posição.".format(len(stis), len(rotulos)))
    unicos = sorted(set(rotulos))
    if len(unicos) != 2:
        raise ValueError(
            "O resumo compara DUAS posições; vieram {}: {}.".format(
                len(unicos), unicos))
    valores = np.asarray(stis, dtype=float)
    dentro = {}
    for u in unicos:
        v = valores[[i for i, r in enumerate(rotulos) if r == u]]
        dentro[u] = {
            "n_blocos": int(v.size),
            "media": float(v.mean()),
            "desvio": float(v.std(ddof=1)) if v.size > 1 else None,
            "minimo": float(v.min()),
            "maximo": float(v.max()),
        }
    diferenca = dentro[unicos[0]]["media"] - dentro[unicos[1]]["media"]
    return {
        "sti_por_bloco": stis,
        "posicao_por_bloco": rotulos,
        "dentro_da_posicao": dentro,
        "entre_posicoes": {
            "diferenca": float(diferenca),
            "sentido": "{} − {}".format(unicos[0], unicos[1]),
            **_p_por_permutacao_exata(valores, rotulos),
        },
        "efeito_de_ordem": {
            "spearman_sti_x_ordem": _spearman(np.arange(len(stis)), valores),
            "aviso": ("A ordem balanceada impede que uma posição venha sempre "
                      "depois da outra; ela NÃO remove cansaço nem aquecimento. "
                      "Esta correlação é o que sobra deles."),
        },
        "desequilibrio_temporal": desequilibrio_temporal(rotulos),
        "aviso": ("Piloto de ENGENHARIA com a voz do pesquisador: mede o que o "
                  "caminho de gravação inventa, não fala de participante. Não é "
                  "prova de que a posição do celular importa (ou não)."),
    }


_PADRAO = {
    "ativa": False,
    "minimo_de_repeticoes": 12,
    "exigir_posicao_estavel": True,
    "exigir_canal_estavel": True,
}


def opcoes_de_elegibilidade(config):
    """As quatro chaves de `inconsistencia.elegibilidade_de_blocos`."""
    bruto = ((config or {}).get("inconsistencia") or {}).get(
        "elegibilidade_de_blocos")
    if bruto is None:
        raise KeyError(
            "Falta `inconsistencia.elegibilidade_de_blocos` no config.yaml. "
            "Não há valor padrão embutido: a ausência do padrão é a decisão, "
            "como no `AFI_DADOS`. As chaves são {}.".format(sorted(_PADRAO)))
    if not isinstance(bruto, dict):
        raise TypeError(
            "`inconsistencia.elegibilidade_de_blocos` tem de ser um "
            "dicionário; veio {}.".format(type(bruto).__name__))
    desconhecidas = sorted(set(bruto) - set(_PADRAO))
    if desconhecidas:
        raise KeyError(
            "Chave(s) que o código NÃO lê em "
            "`inconsistencia.elegibilidade_de_blocos`: {}. As lidas são {}."
            .format(desconhecidas, sorted(_PADRAO)))
    faltando = sorted(set(_PADRAO) - set(bruto))
    if faltando:
        raise KeyError(
            "Falta(m) {} em `inconsistencia.elegibilidade_de_blocos`. Todas "
            "as quatro chaves são obrigatórias: um padrão implícito aqui seria "
            "uma regra de análise que ninguém escreveu.".format(faltando))
    opcoes = {
        "ativa": bool(bruto["ativa"]),
        "minimo_de_repeticoes": int(bruto["minimo_de_repeticoes"]),
        "exigir_posicao_estavel": bool(bruto["exigir_posicao_estavel"]),
        "exigir_canal_estavel": bool(bruto["exigir_canal_estavel"]),
    }
    if opcoes["minimo_de_repeticoes"] < 2:
        raise ValueError(
            "`minimo_de_repeticoes` = {} — o STI exige 2+ repetições da mesma "
            "frase, então qualquer valor abaixo de 2 não é uma regra."
            .format(opcoes["minimo_de_repeticoes"]))
    return opcoes


def _valores_distintos(gravacoes, campos):
    """Os conjuntos de valores não vazios que o bloco traz, por campo."""
    vistos = set()
    sem_registro = 0
    for gravacao in gravacoes:
        condicoes = (gravacao or {}).get("condicoes") or {}
        valores = tuple(str(condicoes.get(c, "") or "").strip() for c in campos)
        if not all(valores):
            sem_registro += 1
            continue
        vistos.add(valores)
    return vistos, sem_registro


def elegibilidade_do_bloco(gravacoes, config, rotulo=None):
    """Diz se um bloco é elegível para a análise principal da H2."""
    opcoes = opcoes_de_elegibilidade(config)
    gravacoes = list(gravacoes)
    motivos = []

    n = len(gravacoes)
    if n < opcoes["minimo_de_repeticoes"]:
        motivos.append(
            "poucas repetições válidas: {} de {} exigidas — repetição "
            "descartada na hora não é guardada, então "
            "ela chega aqui como repetição AUSENTE."
            .format(n, opcoes["minimo_de_repeticoes"]))

    if opcoes["exigir_posicao_estavel"]:
        posicoes, sem_posicao = _valores_distintos(gravacoes,
                                                   (CAMPO_DA_POSICAO,))
        if len(posicoes) > 1:
            motivos.append(
                "a posição do aparelho mudou dentro do bloco ({} valores: {}) "
                "— parte da variação pode vir da captação, não da fala."
                .format(len(posicoes), sorted(p[0] for p in posicoes)))
        elif sem_posicao and not posicoes:
            motivos.append(
                "nenhuma repetição do bloco registra a posição do aparelho — "
                "não é 'não mudou', é 'não se sabe'.")

    if opcoes["exigir_canal_estavel"]:
        canais, sem_canal = _valores_distintos(gravacoes, CAMPOS_DO_CANAL)
        if len(canais) > 1:
            motivos.append(
                "o canal (dispositivo, ambiente) mudou dentro do bloco ({} "
                "canais distintos) — o deslocamento infla o índice."
                .format(len(canais)))
        elif sem_canal and not canais:
            motivos.append(
                "nenhuma repetição do bloco registra o canal completo — é o "
                "caso da bancada de engenharia, não da coleta real.")

    return {
        "rotulo": rotulo,
        "n_repeticoes": n,
        "regra_ativa": opcoes["ativa"],
        "motivos_observados": motivos,
        "elegivel": (not motivos) if opcoes["ativa"] else True,
        "excluido_pela_regra": bool(motivos) and opcoes["ativa"],
        "aviso": ("O áudio é preservado sempre. Isto decide se o NÚMERO entra "
                  "na análise principal da H2 — e, com a regra desligada, nem "
                  "isso: só informa."),
    }


def elegibilidade_dos_blocos(blocos, config):
    """A regra aplicada a vários blocos, com o quadro de quantos e de quem."""
    opcoes = opcoes_de_elegibilidade(config)
    por_bloco = {rotulo: elegibilidade_do_bloco(gravacoes, config, rotulo=rotulo)
                 for rotulo, gravacoes in sorted(blocos.items())}
    com_motivo = sorted(r for r, d in por_bloco.items() if d["motivos_observados"])
    excluidos = sorted(r for r, d in por_bloco.items() if d["excluido_pela_regra"])
    total = len(por_bloco)
    return {
        "regra_ativa": opcoes["ativa"],
        "opcoes": opcoes,
        "por_bloco": por_bloco,
        "total_de_blocos": total,
        "com_motivo": com_motivo,
        "excluidos": excluidos,
        "taxa_de_inelegibilidade_observada": (len(com_motivo) / total
                                              if total else None),
        "elegiveis": sorted(r for r, d in por_bloco.items() if d["elegivel"]),
        "aviso": (
            "A regra de elegibilidade é PROPOSTA e depende da escolha do "
            "pesquisador responsável e da pesquisadora responsável pela ética "
            "(muda procedimento de coleta). Com `ativa: false` nada é "
            "excluído — os motivos saem só para o relatório poder dizer "
            "quantos e de quem."),
    }


def filtrar_stis_elegiveis(stis, elegibilidade):
    """(elegíveis, excluídos) a partir da saída de `stis_por_frase`."""
    por_bloco = elegibilidade["por_bloco"]
    faltando = sorted(set(stis) - set(por_bloco))
    if faltando:
        raise KeyError(
            "Sem veredito de elegibilidade para {} — omissão não vira "
            "'elegível'. Passe todos os blocos para `elegibilidade_dos_blocos`."
            .format(faltando))
    elegiveis, excluidos = {}, {}
    for frase, dados in stis.items():
        veredito = por_bloco[frase]
        if veredito["elegivel"]:
            elegiveis[frase] = dados
        else:
            excluidos[frase] = {**dados,
                                "motivos": veredito["motivos_observados"]}
    return elegiveis, excluidos


MINIMO_DE_REPETICOES_DO_BLOCO_INTERROMPIDO = 6
REPETICOES_DO_BLOCO = 12

ACAO_REFAZER_BLOCO = "refazer o bloco inteiro em OUTRO dia, nunca no mesmo dia"


def _veredito_do_bloco(n, repeticoes_do_bloco):
    """(entra, completo, motivo, acao) para um bloco com `n` repetições."""
    if n > repeticoes_do_bloco:
        return (False, False,
                "o bloco tem {} repetições, mais que as {} previstas — provável "
                "mistura da tentativa interrompida com a refeita; não "
                "entra até a leitura do banco separar as duas."
                .format(n, repeticoes_do_bloco),
                "conferir no banco quais gravações pertencem a este bloco")
    if n < MINIMO_DE_REPETICOES_DO_BLOCO_INTERROMPIDO:
        return (False, False,
                "bloco interrompido com {} de {} repetições — menos que as {} "
                "exigidas: não entra em nenhuma análise; o áudio é "
                "preservado. O bloco é refeito em outro dia, nunca no mesmo."
                .format(n, repeticoes_do_bloco, MINIMO_DE_REPETICOES_DO_BLOCO_INTERROMPIDO),
                ACAO_REFAZER_BLOCO)
    if n < repeticoes_do_bloco:
        return (True, False,
                "bloco interrompido com {} de {} repetições — vale (6 ou "
                "mais); o índice é calculado com N = {} e sai da sensibilidade "
                "com só blocos completos.".format(n, repeticoes_do_bloco, n),
                None)
    return (True, True, "", None)


def sti_dos_blocos_com_minimo(blocos, config, repeticoes_do_bloco=REPETICOES_DO_BLOCO,
                       calcular=None):
    """O STI de cada bloco sob a regra do bloco interrompido, com o N obtido registrado."""
    if calcular is None:
        from inconsistencia.sti import calcular_sti as calcular
    if not blocos:
        raise ValueError("Não há blocos para avaliar (dict vazio).")
    por_bloco = {}
    for chave in sorted(blocos, key=lambda c: tuple(str(x) for x in c)):
        if not (isinstance(chave, tuple) and len(chave) == 3):
            raise KeyError(
                "Chave de bloco {!r} inválida — use (pessoa, frase, bloco), para "
                "que o N fique registrado por pessoa, frase e bloco.".format(chave))
        repeticoes = list(blocos[chave])
        n = len(repeticoes)
        entra, completo, motivo, acao = _veredito_do_bloco(n, repeticoes_do_bloco)
        por_bloco[chave] = {
            "pessoa": chave[0],
            "frase": chave[1],
            "bloco": chave[2],
            "n_repeticoes": n,
            "n_previsto": repeticoes_do_bloco,
            "completo": completo,
            "entra": entra,
            "sti": float(calcular(repeticoes, config)) if entra else None,
            "motivo": motivo,
            "acao": acao,
        }
    entram = [c for c, d in por_bloco.items() if d["entra"]]
    nao_entram = [c for c, d in por_bloco.items() if not d["entra"]]
    return {
        "regra": ("Bloco interrompido vale com {} ou "
                  "mais repetições; índice com o N obtido; N < {} → refazer em "
                  "outro dia.").format(MINIMO_DE_REPETICOES_DO_BLOCO_INTERROMPIDO,
                                       MINIMO_DE_REPETICOES_DO_BLOCO_INTERROMPIDO),
        "minimo_de_repeticoes": MINIMO_DE_REPETICOES_DO_BLOCO_INTERROMPIDO,
        "por_bloco": por_bloco,
        "entram": entram,
        "nao_entram": nao_entram,
        "incompletos_que_entram": [c for c in entram
                                   if not por_bloco[c]["completo"]],
        "a_refazer_em_outro_dia": [c for c in nao_entram
                                   if por_bloco[c]["acao"] == ACAO_REFAZER_BLOCO],
    }


def sensibilidade_so_blocos_completos(resultado):
    """A sensibilidade do bloco interrompido: só blocos com as doze repetições previstas."""
    por_bloco = resultado["por_bloco"]
    mantidos, excluidos = {}, {}
    for chave, dados in por_bloco.items():
        if not dados["entra"]:
            continue
        if dados["completo"]:
            mantidos[chave] = dados
        else:
            excluidos[chave] = {
                "n_repeticoes": dados["n_repeticoes"],
                "motivo": ("bloco incompleto (N = {} de {}): fora da "
                           "sensibilidade com só blocos completos."
                           .format(dados["n_repeticoes"], dados["n_previsto"])),
            }
    return {
        "sensibilidade": "so_blocos_completos",
        "por_bloco": mantidos,
        "entram": sorted(mantidos, key=lambda c: tuple(str(x) for x in c)),
        "excluidos": excluidos,
        "n_excluidos": len(excluidos),
        "declaracao": ("A sensibilidade com só blocos completos excluiu {} "
                       "bloco(s) incompleto(s) que entram na análise principal: "
                       "{}.".format(len(excluidos),
                                    ", ".join("{}/{}/{} (N = {})".format(
                                        c[0], c[1], c[2], d["n_repeticoes"])
                                        for c, d in sorted(
                                            excluidos.items(),
                                            key=lambda i: tuple(str(x) for x in i[0])))
                                    or "nenhum")),
    }


def sti_por_celula_dos_blocos(resultado):
    """{(pessoa, frase): {sti, n_repeticoes, bloco}} para a H2, e os conflitos."""
    candidatos = {}
    for chave, dados in resultado["por_bloco"].items():
        if dados["entra"]:
            candidatos.setdefault((chave[0], chave[1]), []).append(dados)
    celulas, conflitos = {}, {}
    for celula, lista in candidatos.items():
        if len(lista) > 1:
            conflitos[celula] = (
                "{} blocos válidos para a mesma pessoa e frase ({}) — qual vale "
                "não está decidido (depende da escolha do pesquisador "
                "responsável); a célula fica sem índice até essa escolha."
                .format(len(lista), sorted(str(d["bloco"]) for d in lista)))
            continue
        d = lista[0]
        celulas[celula] = {"sti": d["sti"], "n_repeticoes": d["n_repeticoes"],
                           "bloco": d["bloco"], "completo": d["completo"]}
    return celulas, conflitos

