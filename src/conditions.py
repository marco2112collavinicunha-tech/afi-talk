"""Etapa 4 do pipeline: as três condições do estudo, como abstração de primeira classe."""
from __future__ import annotations

import numpy as np

from src.transcritor import frases_alvo_para


class CondicaoIndisponivel(RuntimeError):
    """A condição existe, mas o banco atual não tem os dados que ela exige."""


def _treino_do_participante(dados):
    """Aplica o split: devolve (X, y) somente do conjunto de treino da pessoa."""
    indices = dados["divisao"]["treino"]
    if not indices:
        raise ValueError("O conjunto de treino deste participante está vazio.")
    X = np.asarray(dados["embeddings"])[indices]
    y = np.asarray([str(dados["rotulos"][i]) for i in indices], dtype=object)
    return X, y


def _sessoes_do_treino(dados):
    """Sessões das gravações de TREINO, na mesma ordem de `_treino_do_participante`."""
    sessoes = dados.get("sessoes")
    if not sessoes:
        return None
    indices = dados["divisao"]["treino"]
    if len(sessoes) < max(indices) + 1:
        return None
    return [sessoes[i] for i in indices]


class Condicao:
    nome = "?"
    via = "embeddings"

    def montar(self, banco, alvo):
        raise NotImplementedError


class CondicaoPersonalizado(Condicao):
    """A condição de interesse: conhece a própria pessoa."""

    nome = "personalizado"
    via = "embeddings"

    def __init__(self, esquema="loo"):
        """`esquema` é "loo" ou "loso" (leave-one-session-out)."""
        if esquema not in ("loo", "loso"):
            raise ValueError(
                "Esquema do personalizado desconhecido: {!r}. Use \"loo\" (padrão) "
                "ou \"loso\" (leave-one-session-out).".format(esquema)
            )
        self.esquema = esquema

    def montar(self, banco, alvo):
        if alvo not in banco:
            raise ValueError(
                "Participante {!r} não está no banco (há: {}).".format(alvo, sorted(banco))
            )
        dados = banco[alvo]
        X, y = _treino_do_participante(dados)
        preparo = {"condicao": self.nome, "alvo": alvo, "esquema": self.esquema, "X": X, "y": y}

        if self.esquema == "loso":
            sessoes = _sessoes_do_treino(dados)
            if sessoes is None:
                raise CondicaoIndisponivel(
                    "O personalizado foi pedido com esquema LOSO, mas o banco de "
                    "{!r} não traz o número da sessão de cada gravação. O campo "
                    "`sessao` nem sempre existiu; gravações anteriores a ele "
                    "não o têm, e inventar sessão inventaria o resultado. Regrave "
                    "ou volte ao esquema LOO.".format(alvo)
                )
            if len(set(sessoes)) < 2:
                raise CondicaoIndisponivel(
                    "O personalizado foi pedido com esquema LOSO, mas as gravações "
                    "de TREINO de {!r} vêm de {} sessão(ões) — o LOSO precisa de 2 "
                    "ou mais. Não é falta de implementação: não se deixa uma sessão "
                    "de fora de um conjunto que tem uma sessão.".format(alvo, len(set(sessoes)))
                )
            preparo["sessoes"] = sessoes
        return preparo


class CondicaoEntrePessoas(Condicao):
    """O controle: conhece "a AFI em geral", não a pessoa."""

    nome = "entre_pessoas"
    via = "embeddings"

    def __init__(self, agregacao_da_selecao="agrupada"):
        if agregacao_da_selecao not in AGREGACOES_DA_SELECAO_ENTRE_PESSOAS:
            raise ValueError(
                "condicoes.agregacao_da_selecao_entre_pessoas precisa ser um de {}; "
                "veio {!r}.".format(AGREGACOES_DA_SELECAO_ENTRE_PESSOAS,
                                    agregacao_da_selecao))
        self.agregacao_da_selecao = agregacao_da_selecao

    def montar(self, banco, alvo):
        if alvo not in banco:
            raise ValueError(
                "Participante {!r} não está no banco (há: {}).".format(alvo, sorted(banco))
            )
        grupo_do_alvo = banco[alvo].get("grupo")
        if grupo_do_alvo is None:
            raise ValueError(
                "O participante {!r} está sem GRUPO no banco. O entre-pessoas "
                "precisa saber o grupo para treinar só dentro dele — regrave o "
                "exemplo ou corrija o sidecar.".format(alvo)
            )

        outros = [
            pessoa for pessoa in sorted(banco)
            if pessoa != alvo
            and banco[pessoa].get("grupo") == grupo_do_alvo
            and banco[pessoa].get("grupo") != "engenharia"
        ]
        if grupo_do_alvo == "engenharia":
            raise CondicaoIndisponivel(
                "O participante-alvo {!r} é do grupo 'engenharia' (voz de "
                "bancada). O entre-pessoas é uma condição do ESTUDO e não roda "
                "sobre vozes de engenharia — elas serviriam de treino umas para "
                "as outras e produziriam um número sem significado "
                "científico.".format(alvo)
            )
        if not outros:
            fora_do_grupo = [
                pessoa for pessoa in sorted(banco)
                if pessoa != alvo and banco[pessoa].get("grupo") != grupo_do_alvo
            ]
            detalhe = ""
            if fora_do_grupo:
                detalhe = (
                    " Há {} outro(s) participante(s) no banco, mas de grupo "
                    "diferente — e o treino do entre-pessoas fica DENTRO do "
                    "grupo do alvo ({!r}), por desenho.".format(
                        len(fora_do_grupo), grupo_do_alvo
                    )
                )
            raise CondicaoIndisponivel(
                "A condição 'entre_pessoas' treina na fala de OUTROS "
                "participantes DO MESMO GRUPO e por isso exige 2+ pessoas do "
                "grupo {!r} no banco.{} Com uma única voz, apenas "
                "'personalizado' e 'generico' rodam — os dados de "
                "vários participantes só existem depois da coleta do "
                "estudo.".format(grupo_do_alvo, detalhe)
            )

        if len(outros) < 2:
            raise CondicaoIndisponivel(
                "A configuração do Entre-pessoas é escolhida SÓ com os outros "
                "participantes do grupo {!r}, nunca com as gravações do alvo. "
                "Isso deixa uma outra pessoa de fora por vez e exige 2+ "
                "outras pessoas do grupo; há {} ({}).".format(
                    grupo_do_alvo, len(outros), ", ".join(outros)
                )
            )

        blocos_X, blocos_y, pessoas = [], [], []
        for pessoa in outros:
            X, y = _treino_do_participante(banco[pessoa])
            blocos_X.append(X)
            blocos_y.append(y)
            pessoas.extend([pessoa] * len(y))

        return {
            "condicao": self.nome,
            "alvo": alvo,
            "esquema": "entre_os_outros",
            "X": np.vstack(blocos_X),
            "y": np.concatenate(blocos_y),
            "pessoas": pessoas,
            "agregacao": self.agregacao_da_selecao,
        }


class CondicaoGenerico(Condicao):
    """O piso: a ferramenta de prateleira, sem adaptação nenhuma."""

    nome = "generico"
    via = "transcricao"

    def __init__(self, transcritor=None, frases_definidas=None, taxa=16000):
        self.transcritor = transcritor
        self.frases_definidas = frases_definidas or {}
        self.taxa = taxa

    def montar(self, banco, alvo):
        if alvo not in banco:
            raise ValueError(
                "Participante {!r} não está no banco (há: {}).".format(alvo, sorted(banco))
            )
        if self.transcritor is None:
            raise ValueError(
                "A condição genérica precisa de um transcritor — na varredura, "
                "executar_experimento resolve a ponte do WhisperX sozinho "
                "(criar_transcritor); em testes/engenharia, injete um mock com "
                "transcrever(sinal, taxa) -> str."
            )

        dados = banco[alvo]
        indices = dados["divisao"]["treino"]
        if not indices:
            raise ValueError("O conjunto de treino deste participante está vazio.")

        rotulos = [str(dados["rotulos"][i]) for i in indices]
        frases_alvo = frases_alvo_para({"frases_alvo": self.frases_definidas}, rotulos,
                                       grupo=dados.get("grupo"))
        frases_por_fallback = sorted(
            {r for r in rotulos if r not in (self.frases_definidas or {})}
        )
        return {
            "condicao": self.nome,
            "alvo": alvo,
            "esquema": "transcricao",
            "sinais": [dados["sinais"][i] for i in indices],
            "rotulos": rotulos,
            "frases_alvo": frases_alvo,
            "frases_por_fallback": frases_por_fallback,
            "transcritor": self.transcritor,
            "taxa": self.taxa,
        }


AGREGACOES_DA_SELECAO_ENTRE_PESSOAS = ("agrupada", "media_por_pessoa")

CONDICOES = {
    CondicaoPersonalizado.nome: CondicaoPersonalizado,
    CondicaoEntrePessoas.nome: CondicaoEntrePessoas,
    CondicaoGenerico.nome: CondicaoGenerico,
}


def _esquema_de_selecao(config):
    """O esquema que seleciona a configuração do personalizado."""
    esquema = (config.get("condicoes") or {}).get("esquema_de_selecao", "loso")
    if esquema not in ("loo", "loso"):
        raise ValueError(
            "condicoes.esquema_de_selecao precisa ser \"loso\" ou \"loo\" "
            "(diagnóstico); veio {!r}.".format(esquema)
        )
    return esquema


def criar_condicoes(config, transcritor=None):
    """Instancia as condições ativas no config.yaml (condicoes.ativas), na ordem."""
    ativas = config["condicoes"]["ativas"]
    if not ativas:
        raise ValueError("condicoes.ativas está vazio no config.yaml — ative ao menos uma.")
    instancias = []
    for nome in ativas:
        if nome not in CONDICOES:
            raise ValueError(
                "Condição desconhecida: {!r}. Disponíveis: {}.".format(nome, sorted(CONDICOES))
            )
        if nome == CondicaoGenerico.nome:
            instancias.append(
                CondicaoGenerico(
                    transcritor=transcritor,
                    frases_definidas=config.get("frases_alvo"),
                    taxa=config["audio"]["taxa_amostragem"],
                )
            )
        elif nome == CondicaoPersonalizado.nome:
            instancias.append(
                CondicaoPersonalizado(esquema=_esquema_de_selecao(config))
            )
        elif nome == CondicaoEntrePessoas.nome:
            instancias.append(CondicaoEntrePessoas(
                (config.get("condicoes") or {}).get(
                    "agregacao_da_selecao_entre_pessoas", "agrupada")))
        else:
            instancias.append(CONDICOES[nome]())
    return instancias

