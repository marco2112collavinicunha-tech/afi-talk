"""Avaliação do pipeline: validação cruzada, métricas e resultados rastreáveis."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

from src.config import RAIZ_PROJETO


class PaneDoTranscritor(Exception):
    """O transcritor falhou em todas as gravações de teste de uma pessoa."""


class PaneDoTranscritorNoAudio(PaneDoTranscritor, ValueError):
    """Pane em que o transcritor rejeitou todos os áudios (todas as falhas ValueError)."""


class PaneDoTranscritorNaInfraestrutura(PaneDoTranscritor, RuntimeError):
    """Pane com alguma falha que não é rejeição de áudio (ponte, modelo, ambiente)."""


def postos_pessimistas(verdadeiros, probabilidades, classes):
    """Posto do rótulo verdadeiro em cada gravação, com empate contando contra."""
    posicao = {classe: i for i, classe in enumerate(classes)}
    postos = []
    for rotulo, probs in zip(verdadeiros, np.asarray(probabilidades)):
        p_verdadeiro = probs[posicao[rotulo]]
        postos.append(int(np.sum(probs >= p_verdadeiro) - 1))
    return postos


def acertos_no_top1(verdadeiros, probabilidades, classes):
    """Acerto 0/1 no top-1 com empate contando como erro (posto pessimista)."""
    return [posto == 0 for posto in postos_pessimistas(verdadeiros, probabilidades, classes)]


def marcar_sem_resposta(predicoes, classes):
    """Por gravação, se a predição está fora do universo de classes."""
    universo = {str(c) for c in classes}
    return [str(p) not in universo for p in predicoes]


def creditos_no_top1(verdadeiros, probabilidades, classes, sem_resposta=None):
    """Acerto de cada gravação no top-1, com o empate valendo o sorteio."""
    probabilidades = np.asarray(probabilidades, dtype=float)
    verdadeiros = list(verdadeiros)
    if sem_resposta is None:
        sem_resposta = [False] * len(verdadeiros)
    if len(sem_resposta) != len(verdadeiros):
        raise ValueError("sem_resposta tem {} marcas para {} gravações.".format(
            len(sem_resposta), len(verdadeiros)))
    posicao = {classe: i for i, classe in enumerate(classes)}
    creditos = []
    for rotulo, probs, calado in zip(verdadeiros, probabilidades, sem_resposta):
        p_verdadeiro = probs[posicao[rotulo]]
        if calado or np.any(probs > p_verdadeiro):
            creditos.append(0.0)
            continue
        creditos.append(1.0 / int(np.sum(probs == p_verdadeiro)))
    return creditos


def empatadas_no_topo(probabilidades, classes, sem_resposta=None):
    """Por gravação, as classes empatadas na maior nota (lista vazia sem empate)."""
    probabilidades = np.asarray(probabilidades, dtype=float)
    if probabilidades.size == 0:
        return []
    if sem_resposta is None:
        sem_resposta = [False] * len(probabilidades)
    saida = []
    for probs, calado in zip(probabilidades, sem_resposta):
        topo = np.flatnonzero(probs == probs.max())
        saida.append([] if calado or len(topo) < 2
                     else [str(classes[i]) for i in topo])
    return saida


def empates_no_topo(probabilidades):
    """Em quais gravações o máximo de probabilidade não é único."""
    probabilidades = np.asarray(probabilidades, dtype=float)
    if probabilidades.size == 0:
        return []
    topo = probabilidades.max(axis=1)
    return [bool(n > 1) for n in (probabilidades == topo[:, None]).sum(axis=1)]


def taxa_topk(verdadeiros, probabilidades, classes, k=3):
    """Fração das gravações cujo rótulo verdadeiro está entre os k mais prováveis."""
    verdadeiros = list(verdadeiros)
    if not verdadeiros:
        raise ValueError("Não há gravações para calcular o top-{}.".format(k))
    k_efetivo = min(k, len(classes))
    postos = postos_pessimistas(verdadeiros, probabilidades, classes)
    return sum(1 for posto in postos if posto < k_efetivo) / len(verdadeiros)


def matriz_de_confusao_esperada(verdadeiros, predicoes, probabilidades, classes,
                                sem_resposta=None):
    """Matriz de confusão com o empate distribuído: 1/t para cada empatada."""
    classes = [str(c) for c in classes]
    posicao = {c: i for i, c in enumerate(classes)}
    matriz = np.zeros((len(classes), len(classes)), dtype=float)
    empatadas = empatadas_no_topo(probabilidades, classes, sem_resposta)
    if sem_resposta is None:
        sem_resposta = [False] * len(verdadeiros)
    for verdadeiro, predicao, grupo, calado in zip(
            verdadeiros, predicoes, empatadas, sem_resposta):
        if calado or str(predicao) not in posicao:
            continue
        linha = posicao[str(verdadeiro)]
        if grupo:
            for c in grupo:
                matriz[linha, posicao[c]] += 1.0 / len(grupo)
        else:
            matriz[linha, posicao[str(predicao)]] += 1.0
    return matriz


def _alinhar_probabilidades(classes_globais, classes_do_modelo, probs):
    """Reordena as probabilidades de um modelo para a ordem global de classes."""
    alinhado = np.zeros(len(classes_globais), dtype=float)
    posicao = {classe: i for i, classe in enumerate(classes_globais)}
    for classe, p in zip(classes_do_modelo, probs):
        alinhado[posicao[classe]] = float(p)
    return alinhado


def _empates_de_votos(modelo, X):
    """`(havia_empate_de_votos, desfeito)` por linha, ou `None`."""
    estimador = modelo.steps[-1][1] if hasattr(modelo, "steps") else modelo
    if not hasattr(estimador, "empates_de_votos"):
        return None
    Xt = modelo[:-1].transform(X) if hasattr(modelo, "steps") else X
    havia, desfeito = estimador.empates_de_votos(Xt)
    return [bool(h) for h in havia], [bool(d) for d in desfeito]


def _juntar_empates(acumulado, novo):
    if novo is None:
        return None
    if acumulado is None:
        return [list(novo[0]), list(novo[1])]
    acumulado[0].extend(novo[0])
    acumulado[1].extend(novo[1])
    return acumulado


def _bloco_do_desempate(empates_de_votos, n):
    """Taxa de empate de votos do kNN antes e depois do desempate."""
    havia, desfeito = empates_de_votos
    n_havia = int(sum(havia))
    n_desfeitos = int(sum(desfeito))
    return {
        "regra": ("empate de votos do kNN "
                  "desempatado pelo vizinho mais próximo; o que sobra empatado "
                  "vale 1/t no acerto (t = número de empatadas)"),
        "n_empates_de_votos": n_havia,
        "n_desfeitos_pelo_vizinho": n_desfeitos,
        "n_residuais": n_havia - n_desfeitos,
        "taxa_de_empate_antes": float(n_havia / n) if n else 0.0,
        "taxa_de_empate_depois": float((n_havia - n_desfeitos) / n) if n else 0.0,
    }


def _validar_knn_para_n(classificador, n_treino):
    """kNN com k maior que o treino disponível deve falhar antes, com explicação."""
    estimador = classificador.steps[-1][1] if hasattr(classificador, "steps") else classificador
    k = getattr(estimador, "n_neighbors", None)
    if k is not None and n_treino < k:
        raise ValueError(
            "kNN com k={} precisa de pelo menos {} gravações de treino por dobra; "
            "há {}. Grave mais exemplos ou use um k menor.".format(k, k, n_treino)
        )


def _resultado(esquema, verdadeiros, predicoes, probabilidades, classes,
               reportar_abstencao=False, empates_de_votos=None):
    """Empacota as métricas num dicionário JSON-compatível."""
    verdadeiros = [str(v) for v in verdadeiros]
    predicoes = [str(p) for p in predicoes]
    classes = list(classes)
    probabilidades = np.asarray(probabilidades, dtype=float)
    from src.transcritor import FALHA_DO_TRANSCRITOR, SEM_CORRESPONDENCIA
    calados = marcar_sem_resposta(predicoes, classes)
    creditos = creditos_no_top1(verdadeiros, probabilidades, classes,
                                sem_resposta=calados)
    n_aval = len(verdadeiros)
    acuracia = float(np.mean(creditos)) if n_aval else 0.0
    matriz = matriz_de_confusao_esperada(verdadeiros, predicoes, probabilidades,
                                         classes, sem_resposta=calados)

    def _uar_de(valores):
        return float(np.mean([
            np.mean([v for rotulo, v in zip(verdadeiros, valores) if rotulo == classe])
            for classe in sorted(set(verdadeiros))
        ]))

    uar = _uar_de(creditos)
    extra = {}
    if empates_de_votos is not None:
        extra["desempate_pelo_vizinho"] = _bloco_do_desempate(
            empates_de_votos, len(verdadeiros))
    if reportar_abstencao:
        extra["abstencao"] = {
            "curva": curva_de_abstencao(verdadeiros, probabilidades, classes,
                                        sem_resposta=calados),
            "escore": ("similaridade de texto (NÃO é probabilidade calibrada)"
                       if esquema == "transcricao" else "probabilidade do classificador"),
            "limiar_escolhido": None,
            "aviso": (
                "Curva de reporte. O limiar NÃO foi escolhido — depende da "
                "escolha do pesquisador responsável. E sobre frases-alvo apenas esta curva mostra "
                "o CUSTO de se calar, não o ganho: o ganho exige gravações de "
                "fala NÃO-ALVO."
            ),
        }
    empatadas = empatadas_no_topo(probabilidades, classes, sem_resposta=calados)
    com_empate = [bool(g) for g in empatadas]
    com_a_verdadeira = [bool(g) and v in g for g, v in zip(empatadas, verdadeiros)]
    fora_do_empate = {
        "sem_correspondencia": int(sum(
            1 for p in predicoes if p == SEM_CORRESPONDENCIA)),
        "falha_do_transcritor": int(sum(
            1 for p in predicoes if p == FALHA_DO_TRANSCRITOR)),
        "nota": ("Predições-sentinela: valem 0 no acerto e NÃO entram em "
                 "n_empates, taxa_de_empate, empates_com_a_verdadeira_dentro "
                 "nem nas sensibilidades do empate."),
    }
    pessimistas = [0.0 if g else c for c, g in zip(creditos, com_empate)]
    otimistas = [1.0 if dentro else c for c, dentro in zip(creditos, com_a_verdadeira)]
    sensibilidade_pessimista = {
        "papel": ("SENSIBILIDADE — não decide nada. A regra do estudo é o "
                  "crédito 1/t no empate"),
        "acuracia": float(np.mean(pessimistas)) if n_aval else 0.0,
        "uar": _uar_de(pessimistas),
        "nota": ("LIMITE INFERIOR: empate conta como erro, mesmo com o rótulo "
                 "verdadeiro entre as empatadas"),
    }
    sensibilidade_otimista = {
        "papel": ("SENSIBILIDADE — não decide nada. A regra do estudo é o "
                  "crédito 1/t no empate"),
        "acuracia": float(np.mean(otimistas)) if n_aval else 0.0,
        "uar": _uar_de(otimistas),
        "nota": ("LIMITE SUPERIOR: supõe que todo empate com o rótulo "
                 "verdadeiro dentro seria resolvido a favor dele"),
    }
    return {
        "esquema": esquema,
        "n": len(verdadeiros),
        "classes": [str(c) for c in classes],
        "acuracia": acuracia,
        "uar": uar,
        "top3": float(taxa_topk(verdadeiros, probabilidades, classes, k=3)),
        "regra_do_empate": ("crédito 1/t: com t frases empatadas no topo, a "
                            "gravação vale 1/t se a verdadeira está entre elas "
                            "e 0 se não está"),
        "taxa_de_empate": float(np.mean(com_empate)) if com_empate else 0.0,
        "n_empates": int(sum(com_empate)),
        "empates_com_a_verdadeira_dentro": int(sum(com_a_verdadeira)),
        "sentinelas_fora_do_empate": fora_do_empate,
        "sensibilidade_pessimista_do_empate": sensibilidade_pessimista,
        "sensibilidade_otimista_do_empate": sensibilidade_otimista,
        "matriz_confusao": matriz.tolist(),
        "verdadeiros": verdadeiros,
        "predicoes": predicoes,
        "acertos": [float(c) for c in creditos],
        "empatadas_no_topo": empatadas,
        **extra,
    }


LIMIARES_DE_ABSTENCAO = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def curva_de_abstencao(verdadeiros, probabilidades, classes,
                       limiares=LIMIARES_DE_ABSTENCAO, sem_resposta=None):
    """Cobertura × acerto quando o sistema pode se calar."""
    verdadeiros = [str(v) for v in verdadeiros]
    probabilidades = np.asarray(probabilidades, dtype=float)
    if probabilidades.size == 0:
        return []
    posicao = {classe: i for i, classe in enumerate(classes)}
    faltando = sorted(set(verdadeiros) - set(posicao))
    if faltando:
        raise ValueError(
            "curva_de_abstencao: os rótulos {} não estão entre as classes "
            "conhecidas {}.".format(faltando, sorted(posicao))
        )
    topos = probabilidades.max(axis=1)
    acertos = creditos_no_top1(verdadeiros, probabilidades, classes,
                               sem_resposta=sem_resposta)

    curva = []
    for limiar in limiares:
        cobertos = [i for i, topo in enumerate(topos) if topo >= limiar]
        n = len(cobertos)
        if n == 0:
            curva.append({
                "limiar": float(limiar), "cobertura": 0.0, "n_coberto": 0,
                "acuracia_coberta": None, "uar_coberta": None,
            })
            continue
        acuracia = float(np.mean([acertos[i] for i in cobertos]))
        rotulos_cobertos = sorted({verdadeiros[i] for i in cobertos})
        recalls = [
            float(np.mean([acertos[i] for i in cobertos if verdadeiros[i] == classe]))
            for classe in rotulos_cobertos
        ]
        curva.append({
            "limiar": float(limiar),
            "cobertura": float(n / len(verdadeiros)),
            "n_coberto": n,
            "acuracia_coberta": acuracia,
            "uar_coberta": float(np.mean(recalls)),
        })
    return curva

def avaliar_loo(classificador, X, rotulos):
    """Leave-one-out: treina em n-1 gravações, testa na restante, n vezes."""
    X = np.asarray(X)
    rotulos = [str(r) for r in rotulos]
    n = len(rotulos)
    if n < 2:
        raise ValueError("LOO precisa de pelo menos 2 gravações; há {}.".format(n))
    if X.shape[0] != n:
        raise ValueError(
            "Quantidade de embeddings ({}) difere da de rótulos ({}).".format(X.shape[0], n)
        )
    _validar_knn_para_n(classificador, n - 1)

    classes = sorted(set(rotulos))
    if len(classes) < 2:
        raise ValueError(
            "A avaliação precisa de pelo menos 2 intenções diferentes; há apenas "
            "{!r}. Um classificador de uma classe só não discrimina nada.".format(classes[0])
        )

    predicoes, probabilidades = [], []
    empates = None
    for i in range(n):
        mascara = np.ones(n, dtype=bool)
        mascara[i] = False

        modelo = clone(classificador)
        try:
            modelo.fit(X[mascara], np.asarray(rotulos, dtype=object)[mascara])
        except ValueError as erro:
            contagem = {classe: rotulos.count(classe) for classe in classes}
            raise ValueError(
                "A dobra {} do LOO falhou ao treinar. Contagem por intenção: {}. "
                "Isso costuma acontecer quando uma intenção tem 1 única gravação "
                "e o classificador exige 2+ classes no treino (ex.: regressão "
                "logística) — grave mais exemplos dessa intenção ou use kNN. "
                "Erro original: {}".format(i, contagem, erro)
            ) from erro

        predicoes.append(modelo.predict(X[i : i + 1])[0])
        probabilidades.append(
            _alinhar_probabilidades(classes, modelo.classes_, modelo.predict_proba(X[i : i + 1])[0])
        )
        empates = _juntar_empates(empates, _empates_de_votos(modelo, X[i : i + 1]))

    return _resultado("loo", rotulos, predicoes, probabilidades, classes,
                      empates_de_votos=empates)


def avaliar_loso(classificador, X, rotulos, sessoes):
    """Leave-one-session-out: deixa de fora uma sessão inteira por vez."""
    X = np.asarray(X)
    rotulos = [str(r) for r in rotulos]
    sessoes = [str(s) for s in sessoes]
    n = len(rotulos)

    if X.shape[0] != n:
        raise ValueError(
            "Quantidade de embeddings ({}) difere da de rótulos ({}).".format(X.shape[0], n)
        )
    if len(sessoes) != n:
        raise ValueError(
            "Quantidade de sessões ({}) difere da de gravações ({}). O vetor de "
            "sessões precisa estar na MESMA ordem das gravações.".format(len(sessoes), n)
        )

    distintas = sorted(set(sessoes))
    if len(distintas) < 2:
        raise ValueError(
            "O LOSO precisa de pelo menos 2 sessões DIFERENTES entre as gravações "
            "avaliadas; há {} ({!r}). Não se deixa uma sessão de fora de um "
            "conjunto que tem uma sessão só — é aritmeticamente impossível, não é "
            "falta de implementação. Grave em outro dia/ambiente e registre a "
            "sessão (campo `sessao` do sidecar) antes de usar "
            "este esquema.".format(
                len(distintas), distintas
            )
        )

    classes = sorted(set(rotulos))
    if len(classes) < 2:
        raise ValueError(
            "A avaliação precisa de pelo menos 2 intenções diferentes; há apenas "
            "{!r}. Um classificador de uma classe só não discrimina nada.".format(classes[0])
        )

    rotulos_np = np.asarray(rotulos, dtype=object)
    sessoes_np = np.asarray(sessoes, dtype=object)

    predicoes, probabilidades, verdadeiros = [], [], []
    empates = None
    for sessao in distintas:
        fora = sessoes_np == sessao
        dentro = ~fora
        n_treino = int(dentro.sum())
        if n_treino == 0:
            raise ValueError(
                "A dobra da sessão {!r} deixaria o treino VAZIO — todas as "
                "gravações são dessa sessão.".format(sessao)
            )

        classes_treino = sorted(set(rotulos_np[dentro]))
        if len(classes_treino) < 2:
            raise ValueError(
                "Ao deixar a sessão {!r} de fora, o treino fica com uma intenção só "
                "({!r}). Isso quer dizer que as intenções não estão distribuídas "
                "entre as sessões: cada sessão gravou intenções diferentes. O LOSO "
                "não é aplicável a esse desenho — cada sessão precisa conter "
                "TODAS as intenções.".format(sessao, classes_treino[0])
            )

        _validar_knn_para_n(classificador, n_treino)

        modelo = clone(classificador)
        try:
            modelo.fit(X[dentro], rotulos_np[dentro])
        except ValueError as erro:
            contagem = {c: list(rotulos_np[dentro]).count(c) for c in classes_treino}
            raise ValueError(
                "A dobra da sessão {!r} falhou ao treinar. Contagem por intenção no "
                "treino: {}. Erro original: {}".format(sessao, contagem, erro)
            ) from erro

        for indice in np.flatnonzero(fora):
            predicoes.append(modelo.predict(X[indice : indice + 1])[0])
            probabilidades.append(
                _alinhar_probabilidades(
                    classes, modelo.classes_, modelo.predict_proba(X[indice : indice + 1])[0]
                )
            )
            verdadeiros.append(rotulos[indice])
        empates = _juntar_empates(empates, _empates_de_votos(modelo, X[fora]))

    resultado = _resultado("loso", verdadeiros, predicoes, probabilidades, classes,
                           empates_de_votos=empates)
    resultado["n_sessoes"] = len(distintas)
    resultado["sessoes"] = distintas
    return resultado


def avaliar_entre_os_outros(classificador, X, rotulos, pessoas, agregacao="agrupada",
                            treino_da_dobra=None):
    """Seleção do Entre-pessoas sem o alvo."""
    X = np.asarray(X)
    rotulos = [str(r) for r in rotulos]
    pessoas = [str(p) for p in pessoas]
    n = len(rotulos)
    if X.shape[0] != n or len(pessoas) != n:
        raise ValueError(
            "Embeddings ({}), rótulos ({}) e pessoas ({}) descrevem as MESMAS "
            "gravações e precisam do mesmo comprimento.".format(X.shape[0], n, len(pessoas))
        )

    distintas = sorted(set(pessoas))
    if len(distintas) < 2:
        raise ValueError(
            "A seleção do Entre-pessoas sem o alvo deixa uma OUTRA pessoa "
            "de fora por vez e precisa de pelo menos 2 outras pessoas do grupo; "
            "há {} ({!r}).".format(len(distintas), distintas)
        )

    classes = sorted(set(rotulos))
    if len(classes) < 2:
        raise ValueError(
            "A avaliação precisa de pelo menos 2 intenções diferentes; há apenas "
            "{!r}. Um classificador de uma classe só não discrimina nada.".format(classes[0])
        )

    rotulos_np = np.asarray(rotulos, dtype=object)
    pessoas_np = np.asarray(pessoas, dtype=object)

    predicoes, probabilidades, verdadeiros = [], [], []
    empates = None
    for pessoa in distintas:
        fora = pessoas_np == pessoa
        dentro = ~fora
        if treino_da_dobra is not None:
            indices = [int(i) for i in treino_da_dobra[pessoa]]
            if any(not 0 <= i < n or not dentro[i] for i in indices):
                raise ValueError(
                    "O treino da dobra que deixa {!r} de fora traz gravação fora das "
                    "outras pessoas.".format(pessoa))
            dentro = np.zeros(n, dtype=bool)
            dentro[indices] = True
        n_treino = int(dentro.sum())
        classes_treino = sorted(set(rotulos_np[dentro]))
        if len(classes_treino) < 2:
            raise ValueError(
                "Ao deixar {!r} de fora, o treino da seleção do Entre-pessoas fica "
                "com uma intenção só ({!r}).".format(pessoa, classes_treino)
            )

        _validar_knn_para_n(classificador, n_treino)

        modelo = clone(classificador)
        try:
            modelo.fit(X[dentro], rotulos_np[dentro])
        except ValueError as erro:
            contagem = {c: list(rotulos_np[dentro]).count(c) for c in classes_treino}
            raise ValueError(
                "A dobra que deixa {!r} de fora falhou ao treinar. Contagem por "
                "intenção no treino: {}. Erro original: {}".format(pessoa, contagem, erro)
            ) from erro

        for indice in np.flatnonzero(fora):
            predicoes.append(modelo.predict(X[indice : indice + 1])[0])
            probabilidades.append(
                _alinhar_probabilidades(
                    classes, modelo.classes_, modelo.predict_proba(X[indice : indice + 1])[0]
                )
            )
            verdadeiros.append(rotulos[indice])
        empates = _juntar_empates(empates, _empates_de_votos(modelo, X[fora]))

    resultado = _resultado("entre_os_outros", verdadeiros, predicoes, probabilidades,
                           classes, empates_de_votos=empates)
    resultado["n_pessoas_da_selecao"] = len(distintas)
    resultado["pessoas_da_selecao"] = distintas
    if agregacao == "media_por_pessoa":
        pessoa_de = [p for p in distintas for _ in range(int((pessoas_np == p).sum()))]
        uars = {}
        for p in distintas:
            idx = [i for i, q in enumerate(pessoa_de) if q == p]
            v = [verdadeiros[i] for i in idx]
            pr = [predicoes[i] for i in idx]
            recalls = [np.mean([a == c for a, c in zip(pr, v) if c == classe])
                       for classe in sorted(set(v))]
            uars[p] = float(np.mean(recalls))
        resultado["uar_agrupada"] = resultado["uar"]
        resultado["uar_por_pessoa"] = uars
        resultado["uar"] = float(np.mean(list(uars.values())))
        resultado["agregacao_da_selecao"] = "media_por_pessoa"
    elif agregacao != "agrupada":
        raise ValueError("agregacao desconhecida: {!r}".format(agregacao))
    return resultado


def avaliar_treino_teste(classificador, X_treino, y_treino, X_aval, y_aval,
                         reportar_abstencao=False):
    """Treina uma vez no conjunto de treino e avalia noutro conjunto disjunto."""
    y_treino = [str(y) for y in y_treino]
    y_aval = [str(y) for y in y_aval]
    if not y_treino or not y_aval:
        raise ValueError("Treino e avaliação precisam ambos de ao menos 1 gravação.")
    if len(set(y_treino)) < 2:
        raise ValueError(
            "O treino precisa de pelo menos 2 intenções diferentes; há apenas "
            "{!r}. Um classificador de uma classe só não discrimina nada.".format(
                sorted(set(y_treino))[0]
            )
        )
    _validar_knn_para_n(classificador, len(y_treino))

    classes = sorted(set(y_treino) | set(y_aval))
    modelo = clone(classificador)
    try:
        modelo.fit(np.asarray(X_treino), np.asarray(y_treino, dtype=object))
    except ValueError as erro:
        contagem = {classe: y_treino.count(classe) for classe in sorted(set(y_treino))}
        raise ValueError(
            "O treino da condição falhou. Contagem por intenção no treino: {}. "
            "Erro original: {}".format(contagem, erro)
        ) from erro

    predicoes = list(modelo.predict(np.asarray(X_aval)))
    probabilidades = [
        _alinhar_probabilidades(classes, modelo.classes_, linha)
        for linha in modelo.predict_proba(np.asarray(X_aval))
    ]
    return _resultado("treino_teste", y_aval, predicoes, probabilidades, classes,
                      reportar_abstencao=reportar_abstencao,
                      empates_de_votos=_empates_de_votos(modelo, np.asarray(X_aval)))


def avaliar_so_pela_duracao(sinais_treino, y_treino, sinais_aval, y_aval, taxa):
    """Linha de base de referência: decide a frase só pela duração da gravação."""
    y_treino = [str(y) for y in y_treino]
    y_aval = [str(y) for y in y_aval]
    if not y_treino or not y_aval:
        raise ValueError("Treino e avaliação precisam ambos de ao menos 1 gravação.")
    if len(set(y_treino)) < 2:
        raise ValueError(
            "O treino precisa de pelo menos 2 intenções diferentes para a "
            "referência só pela duração.")
    if not taxa or taxa <= 0:
        raise ValueError("Taxa de amostragem inválida: {!r}.".format(taxa))
    d_treino = [len(np.asarray(s)) / float(taxa) for s in sinais_treino]
    d_aval = [len(np.asarray(s)) / float(taxa) for s in sinais_aval]
    classes_treino = sorted(set(y_treino))
    mediana = {c: float(np.median([d for d, y in zip(d_treino, y_treino) if y == c]))
               for c in classes_treino}
    classes = sorted(set(y_treino) | set(y_aval))
    probabilidades, predicoes = [], []
    for d in d_aval:
        pont = np.array([1.0 / (1.0 + abs(d - mediana[c])) if c in mediana else 0.0
                         for c in classes])
        probabilidades.append(pont / pont.sum())
        predicoes.append(classes[int(np.argmax(pont))])
    saida = _resultado("so_pela_duracao", y_aval, predicoes, probabilidades, classes)
    saida["mediana_da_duracao_no_treino_s"] = mediana
    saida["papel"] = ("referencia descritiva: sem veredito, fora da "
                      "H1 e dos contrastes")
    return saida


def avaliar_transcricao(sinais, rotulos, frases_alvo, transcritor, taxa,
                        reportar_abstencao=False, regua=None):
    from src.transcritor import (
        FALHA_DO_TRANSCRITOR, REGUA_PADRAO, SEM_CORRESPONDENCIA,
        casar_com_frase_alvo, taxa_de_erro_de_palavra,
    )

    regua = REGUA_PADRAO if regua is None else regua

    rotulos = [str(r) for r in rotulos]
    if not rotulos:
        raise ValueError("Não há gravações para avaliar na condição genérica.")
    if len(sinais) != len(rotulos):
        raise ValueError(
            "Quantidade de sinais ({}) difere da de rótulos ({}).".format(
                len(sinais), len(rotulos)
            )
        )
    if len(set(rotulos)) < 2:
        raise ValueError(
            "A avaliação precisa de pelo menos 2 intenções diferentes; há apenas "
            "{!r}. Uma condição avaliada numa única intenção não discrimina "
            "nada.".format(sorted(set(rotulos))[0])
        )

    classes = sorted(set(rotulos) | set(str(r) for r in frases_alvo))
    predicoes, probabilidades = [], []
    transcricoes, similaridades, rankings, wers = [], [], [], []
    falhas = []
    for indice, (sinal, rotulo_verdadeiro) in enumerate(zip(sinais, rotulos)):
        transcreveu, texto, motivos = _transcrever_com_repeticao(
            transcritor, sinal, taxa)
        if not transcreveu:
            falhas.append({
                "indice": indice,
                "tentativas": len(motivos),
                "tipos": [tipo for tipo, _ in motivos],
                "motivo": ("o transcritor falhou nas {} tentativas nesta gravação "
                           "(repete; se persistir, conta como erro). "
                           "Detalhe técnico só na auditoria protegida."
                           .format(len(motivos))),
                "detalhe_protegido": [msg for _, msg in motivos],
            })
            predicoes.append(FALHA_DO_TRANSCRITOR)
            transcricoes.append("")
            similaridades.append(0.0)
            rankings.append([])
            wers.append(None)
            probabilidades.append(np.zeros(len(classes), dtype=float))
            continue
        if not isinstance(texto, str):
            raise ValueError(
                "O transcritor devolveu {!r} — o contrato é "
                "transcrever(sinal, taxa) -> str (texto, possivelmente "
                "vazio).".format(texto)
            )
        predicao, similaridade, ranking = casar_com_frase_alvo(
            texto, frases_alvo, regua=regua
        )
        wers.append(taxa_de_erro_de_palavra(
            frases_alvo.get(str(rotulo_verdadeiro), ""), texto
        ))
        predicoes.append(predicao)
        transcricoes.append(texto)
        similaridades.append(float(similaridade))
        rankings.append([[rotulo, float(s)] for rotulo, s in ranking])
        do_universo = [(r, s) for r, s in ranking if r != SEM_CORRESPONDENCIA]
        probabilidades.append(
            _alinhar_probabilidades(
                classes, [r for r, _ in do_universo], [s for _, s in do_universo]
            )
        )

    if falhas and len(falhas) == len(rotulos):
        tipos = sorted({t for f in falhas for t in f["tipos"]})
        resumos = sorted({m.split(" Log:")[0]
                          for f in falhas for m in f["detalhe_protegido"]})
        classe = (PaneDoTranscritorNoAudio if tipos == ["ValueError"]
                  else PaneDoTranscritorNaInfraestrutura)
        raise classe(
            "O transcritor falhou em todas as {} gravações de teste (depois de "
            "repetir cada uma): nenhuma tentativa de reconhecimento rodou, e o "
            "Genérico fica indisponível para esta pessoa. Se a falha for da "
            "instalação, corrija-a e rode a avaliação de novo. Tipos de falha: {}. "
            "Falhas (sem o log): {}".format(
                len(rotulos), ", ".join(tipos), " | ".join(resumos[:3])))

    resultado = _resultado("transcricao", rotulos, predicoes, probabilidades, classes,
                           reportar_abstencao=reportar_abstencao)
    resultado["transcricoes"] = transcricoes
    resultado["similaridades"] = similaridades
    resultado["sem_correspondencia"] = int(
        sum(1 for p in predicoes if p == SEM_CORRESPONDENCIA)
    )
    resultado["rankings"] = rankings
    resultado["n_falhas_do_transcritor"] = len(falhas)
    resultado["falhas_do_transcritor"] = falhas
    medidos = [w for w in wers if w is not None]
    resultado["wer"] = {
        "media": float(np.mean(medidos)) if medidos else None,
        "por_gravacao": wers,
        "regua_do_casamento": regua,
        "aviso": (
            "Diagnóstico do Genérico — não é métrica comum, não entra em "
            "ganho e não se compara com a UAR das outras condições."
        ),
    }
    return resultado


TENTATIVAS_DO_TRANSCRITOR = 2

FALHAS_DO_TRANSCRITOR = (RuntimeError, ValueError, OSError)


def _transcrever_com_repeticao(transcritor, sinal, taxa):
    motivos = []
    for _ in range(TENTATIVAS_DO_TRANSCRITOR):
        try:
            return True, transcritor.transcrever(sinal, taxa), []
        except FALHAS_DO_TRANSCRITOR as erro:
            motivos.append((type(erro).__name__, str(erro)))
    return False, None, motivos


def avaliar_preparo(preparo, classificador=None):
    """Porta única: recebe o preparo de uma condição (conditions.py) e avalia."""
    if preparo["esquema"] == "loo":
        resultado = avaliar_loo(classificador, preparo["X"], preparo["y"])
    elif preparo["esquema"] == "loso":
        resultado = avaliar_loso(
            classificador, preparo["X"], preparo["y"], preparo["sessoes"]
        )
    elif preparo["esquema"] == "entre_os_outros":
        resultado = avaliar_entre_os_outros(
            classificador, preparo["X"], preparo["y"], preparo["pessoas"],
            agregacao=preparo.get("agregacao", "agrupada"),
        )
    elif preparo["esquema"] == "treino_teste":
        resultado = avaliar_treino_teste(
            classificador,
            preparo["X_treino"],
            preparo["y_treino"],
            preparo["X_aval"],
            preparo["y_aval"],
        )
    elif preparo["esquema"] == "transcricao":
        resultado = avaliar_transcricao(
            preparo["sinais"],
            preparo["rotulos"],
            preparo["frases_alvo"],
            preparo["transcritor"],
            preparo["taxa"],
        )
    else:
        raise ValueError("Esquema de avaliação desconhecido: {!r}.".format(preparo["esquema"]))

    resultado["condicao"] = preparo.get("condicao")
    resultado["alvo"] = preparo.get("alvo")
    return resultado


def _rotulo_da_celula(valor):
    """Inteiro sem casas; fração com duas casas."""
    return ("{:d}".format(int(round(valor))) if abs(valor - round(valor)) < 1e-9
            else "{:.2f}".format(valor))


def figura_matriz_confusao(matriz, classes, caminho):
    """Grava a matriz de confusão como figura (contagens anotadas, PT-BR)."""
    matriz = np.asarray(matriz, dtype=float)
    figura, eixo = plt.subplots(figsize=(1.2 * len(classes) + 3, 1.0 * len(classes) + 2.5))
    imagem = eixo.imshow(matriz, cmap="Blues")
    figura.colorbar(imagem, ax=eixo, fraction=0.046)

    eixo.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    eixo.set_yticks(range(len(classes)), classes)
    eixo.set_xlabel("Intenção prevista")
    eixo.set_ylabel("Intenção verdadeira")
    eixo.set_title("Matriz de confusão")

    limiar = matriz.max() / 2 if matriz.max() > 0 else 0.5
    for linha in range(matriz.shape[0]):
        for coluna in range(matriz.shape[1]):
            eixo.text(
                coluna,
                linha,
                _rotulo_da_celula(matriz[linha, coluna]),
                ha="center",
                va="center",
                color="white" if matriz[linha, coluna] > limiar else "black",
            )

    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)


def criar_pasta_resultados(config, nome=None):
    """Cria results/<timestamp>[_nome]/ sem sobrescrever nada (sufixo em colisão)."""
    base = Path(config["caminhos"]["resultados"])
    if not base.is_absolute():
        base = RAIZ_PROJETO / base

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nome_base = timestamp + ("_" + nome if nome else "")
    pasta = base / nome_base
    contador = 1
    while pasta.exists():
        pasta = base / "{}_{}".format(nome_base, contador)
        contador += 1
    pasta.mkdir(parents=True)
    return pasta


def gravar_config_usado(config, pasta):
    """Grava a cópia exata do config na pasta de resultados (auditabilidade)."""
    with open(pasta / "config_usado.yaml", "w", encoding="utf-8") as arquivo:
        yaml.safe_dump(config, arquivo, allow_unicode=True, sort_keys=False)


def gravar_resultados(config, resultado, metadados=None, nome=None):
    """Cria results/<timestamp>[_nome]/ com config + métricas + figura."""
    pasta = criar_pasta_resultados(config, nome)
    gravar_config_usado(config, pasta)

    conteudo = {"metadados": metadados or {}, **resultado}
    with open(pasta / "metricas.json", "w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo, ensure_ascii=False, indent=2)

    figura_matriz_confusao(
        resultado["matriz_confusao"], resultado["classes"], pasta / "matriz_confusao.png"
    )
    return pasta

