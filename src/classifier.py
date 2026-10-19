"""Etapa 3 do pipeline: classificadores pessoais (scikit-learn)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DELTA_DO_DESEMPATE = 1e-6
TOLERANCIA_DA_DISTANCIA = 1e-9


class KNNDesempatePeloMaisProximo(KNeighborsClassifier):
    """kNN de votos uniformes com desempate pelo vizinho mais próximo."""

    def fit(self, X, y):
        super().fit(X, y)
        self._indice_de_classe_ = np.searchsorted(
            self.classes_, np.asarray(y, dtype=self.classes_.dtype))
        return self

    def _desempatar(self, X):
        """(votos desempatados, máscara 'havia empate de votos', máscara 'o empate foi desfeito pela distância')."""
        votos = np.array(super().predict_proba(X), dtype=float, copy=True)
        distancias, vizinhos = self.kneighbors(X)
        classes_viz = self._indice_de_classe_[vizinhos]
        havia = np.zeros(len(votos), dtype=bool)
        desfeito = np.zeros(len(votos), dtype=bool)
        for i, linha in enumerate(votos):
            topo = linha.max()
            empatadas = np.flatnonzero(linha == topo)
            if len(empatadas) < 2:
                continue
            havia[i] = True
            mais_perto = np.array([
                distancias[i][classes_viz[i] == c].min() for c in empatadas])
            menor = mais_perto.min()
            if np.sum(np.isclose(mais_perto, menor, rtol=TOLERANCIA_DA_DISTANCIA,
                                 atol=0.0)) > 1:
                continue
            vencedora = empatadas[int(np.argmin(mais_perto))]
            perdedoras = [c for c in empatadas if c != vencedora]
            linha[vencedora] += DELTA_DO_DESEMPATE
            linha[perdedoras] -= DELTA_DO_DESEMPATE / len(perdedoras)
            desfeito[i] = True
        return votos, havia, desfeito

    def predict_proba(self, X):
        return self._desempatar(X)[0]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def empates_de_votos(self, X):
        """Por gravação, `(havia_empate, desfeito)`."""
        _, havia, desfeito = self._desempatar(X)
        return havia, desfeito


def criar_knn(k):
    """kNN dentro do pipeline padrão."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(
            "k do kNN deve ser um inteiro >= 1; veio {!r} "
            "(confira classificadores.knn.valores_k no config.yaml).".format(k)
        )
    return Pipeline(
        [
            ("padronizador", StandardScaler()),
            ("classificador", KNNDesempatePeloMaisProximo(n_neighbors=k)),
        ]
    )


def criar_regressao_logistica(C, max_iter, seed):
    """Regressão logística dentro do pipeline padrão."""
    return Pipeline(
        [
            ("padronizador", StandardScaler()),
            (
                "classificador",
                LogisticRegression(C=C, max_iter=max_iter, random_state=seed),
            ),
        ]
    )


def criar_classificadores(config):
    """Monta o dicionário {nome: pipeline} com todos os classificadores ativos."""
    cfg = config["classificadores"]
    seed = config["reprodutibilidade"]["seed"]
    classificadores = {}

    if cfg["knn"]["ativo"]:
        valores_k = cfg["knn"]["valores_k"]
        if not valores_k:
            raise ValueError("classificadores.knn está ativo, mas valores_k está vazio.")
        for k in valores_k:
            classificadores["knn_k{}".format(k)] = criar_knn(k)

    if cfg["regressao_logistica"]["ativo"]:
        classificadores["regressao_logistica"] = criar_regressao_logistica(
            C=cfg["regressao_logistica"]["C"],
            max_iter=cfg["regressao_logistica"]["max_iter"],
            seed=seed,
        )

    if not classificadores:
        raise ValueError(
            "Nenhum classificador ativo no config.yaml — ative ao menos um em "
            "classificadores.* para o pipeline ter o que treinar."
        )
    return classificadores

