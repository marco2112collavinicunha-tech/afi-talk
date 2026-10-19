"""Etapa 2 do pipeline: embeddings — wav2vec 2.0 congelado → vetor por gravação."""
from __future__ import annotations

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


def detectar_dispositivo(pedido="auto"):
    """Resolve o dispositivo de cálculo."""
    if pedido == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    dispositivo = torch.device(pedido)
    if dispositivo.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("config pede dispositivo 'cuda', mas não há GPU CUDA nesta máquina.")
    if dispositivo.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("config pede dispositivo 'mps', mas esta máquina não tem GPU do Mac.")
    return dispositivo


class EstrategiaPooling:
    """Contrato: recebe os frames de uma camada, forma (T, H), e devolve a representação daquela camada como vetor 1-D."""

    nome = "?"

    def aplicar(self, frames):
        raise NotImplementedError


class PoolingMedia(EstrategiaPooling):
    """Baseline do projeto: média no tempo — (T, H) → (H,)."""

    nome = "media"

    def aplicar(self, frames):
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 2:
            raise ValueError(
                "pooling espera os frames de UMA camada, forma (T, H); veio {}.".format(frames.shape)
            )
        return frames.mean(axis=0)


ESTRATEGIAS_POOLING = {
    PoolingMedia.nome: PoolingMedia,
}


def obter_pooling(nome):
    """Instancia a estratégia de pooling pedida no config (erro claro se não existir)."""
    if nome not in ESTRATEGIAS_POOLING:
        raise ValueError(
            "Estratégia de pooling desconhecida: {!r}. Disponíveis: {}.".format(
                nome, sorted(ESTRATEGIAS_POOLING)
            )
        )
    return ESTRATEGIAS_POOLING[nome]()


def resolver_camadas(espec, n_estados):
    """Converte a chave embeddings.camadas do config em índices concretos."""
    if espec == "all":
        return list(range(n_estados))
    if isinstance(espec, str):
        raise ValueError(
            "embeddings.camadas deve ser 'all' ou uma LISTA de índices "
            "(ex.: [0, 6, 12, 24]); veio a string {!r}.".format(espec)
        )
    try:
        camadas = [int(c) for c in espec]
    except (TypeError, ValueError):
        raise ValueError(
            "embeddings.camadas deve ser 'all' ou uma lista de índices; veio {!r}.".format(espec)
        )
    for c in camadas:
        if not 0 <= c < n_estados:
            raise ValueError(
                "Índice de camada inválido: {} (este modelo tem {} estados: 0 a {}).".format(
                    c, n_estados, n_estados - 1
                )
            )
    return camadas


class ExtratorEmbeddings:
    """wav2vec 2.0 congelado que transforma sinal 16 kHz em embeddings por camada."""

    def __init__(self, checkpoint, dispositivo="auto", congelado=True,
                 modelo=None, extrator_features=None, revisao=None):
        if congelado is not True:
            raise ValueError(
                "Decisão metodológica FIXA: o extrator é congelado. "
                "embeddings.congelado deve ser true (ver README — congelar é o que "
                "torna a comparação entre condições válida)."
            )

        self.checkpoint = checkpoint
        self.revisao = None if revisao is None else str(revisao)
        self._hash_dos_pesos = None
        self.dispositivo = detectar_dispositivo(dispositivo)
        pedido_de_revisao = {} if self.revisao is None else {"revision": self.revisao}

        self.modelo = (
            modelo
            if modelo is not None
            else Wav2Vec2Model.from_pretrained(
                checkpoint, dtype=torch.float32, **pedido_de_revisao
            )
        )
        self.extrator_features = (
            extrator_features
            if extrator_features is not None
            else Wav2Vec2FeatureExtractor.from_pretrained(checkpoint, **pedido_de_revisao)
        )

        for parametro in self.modelo.parameters():
            parametro.requires_grad = False
        self.modelo.eval()
        self.modelo.to(self.dispositivo)

    @property
    def n_estados(self):
        """Quantos estados ocultos o modelo devolve (camadas do transformer + 1)."""
        return self.modelo.config.num_hidden_layers + 1

    @property
    def dim(self):
        """Tamanho H de cada vetor de embedding."""
        return self.modelo.config.hidden_size

    @property
    def esta_congelado(self):
        """Verificação real do congelamento (usada em teste e em auditoria)."""
        return (not self.modelo.training) and all(
            not p.requires_grad for p in self.modelo.parameters()
        )

    def hash_dos_pesos(self):
        """sha256 dos tensores efetivamente carregados (nome, tipo, forma, bytes)."""
        if self._hash_dos_pesos is None:
            import hashlib

            resumo = hashlib.sha256()
            for nome, tensor in sorted(self.modelo.state_dict().items()):
                dado = tensor.detach().to("cpu").contiguous()
                resumo.update(nome.encode("utf-8"))
                resumo.update(str(dado.dtype).encode("utf-8"))
                resumo.update(str(tuple(dado.shape)).encode("utf-8"))
                if dado.numel():
                    resumo.update(dado.reshape(-1).view(torch.uint8).numpy().tobytes())
            self._hash_dos_pesos = resumo.hexdigest()
        return self._hash_dos_pesos

    @property
    def procedencia(self):
        """Identidade dos pesos para o JSON de resultado, para auditoria."""
        commit = getattr(getattr(self.modelo, "config", None), "_commit_hash", None)
        return {
            "checkpoint": self.checkpoint,
            "revisao_pedida": self.revisao if self.revisao is not None else (
                "NAO_FIXADA: sem `revision=`, vale a versão corrente do Hub/cache"
            ),
            "commit_do_config_carregado": commit if commit else (
                "NAO_LIDO: o modelo carregado não expõe config._commit_hash "
                "(modelo injetado ou local)"
            ),
            "sha256_dos_tensores": self.hash_dos_pesos(),
            "identidade_conferida": getattr(
                self, "identidade_conferida",
                "NAO_CONFERIDA: extrator montado fora de criar_extrator"),
            "dispositivo": str(self.dispositivo),
        }

    def extrair_estados(self, sinal, taxa):
        """Todos os estados ocultos de uma gravação: array (L, T, H) float32."""
        taxa_esperada = self.extrator_features.sampling_rate
        if taxa != taxa_esperada:
            raise ValueError(
                "O extrator espera áudio a {} Hz, mas recebeu {} Hz — "
                "passe o sinal pela padronização de áudio (src/preprocess.py) antes.".format(taxa_esperada, taxa)
            )

        sinal = np.asarray(sinal, dtype=np.float32)
        if sinal.ndim != 1:
            raise ValueError(
                "Esperava sinal 1-D (saída da padronização de áudio); veio forma {}.".format(sinal.shape)
            )

        self.modelo.eval()

        entradas = self.extrator_features(sinal, sampling_rate=taxa, return_tensors="pt")
        valores = entradas.input_values.to(self.dispositivo)

        with torch.inference_mode():
            saida = self.modelo(valores, output_hidden_states=True)

        return np.stack(
            [estado.squeeze(0).float().cpu().numpy() for estado in saida.hidden_states]
        )

    def extrair(self, sinal, taxa, pooling, camadas="all"):
        """Embedding pronto para o classificador: array (n_camadas, H) float32."""
        estados = self.extrair_estados(sinal, taxa)
        indices = resolver_camadas(camadas, len(estados))
        vetores = [pooling.aplicar(estados[i]) for i in indices]
        return np.stack(vetores).astype(np.float32)


def criar_extrator(config, checkpoint=None):
    """Monta (extrator, pooling) a partir do config.yaml."""
    emb = config["embeddings"]
    if not emb["checkpoints"]:
        raise ValueError("config.yaml: a lista embeddings.checkpoints está vazia.")
    escolhido = checkpoint if checkpoint is not None else emb["checkpoints"][0]
    if emb["congelado"] is not True:
        ExtratorEmbeddings(escolhido, congelado=emb["congelado"])
    from src.identidade_dos_pesos import conferir_tensores, identidade_do_xlsr
    revisao, sha_esperado = identidade_do_xlsr(config, escolhido)
    extrator = ExtratorEmbeddings(
        escolhido,
        dispositivo=emb["dispositivo"],
        congelado=emb["congelado"],
        revisao=revisao,
    )
    extrator.identidade_conferida = conferir_tensores(extrator, sha_esperado)
    pooling = obter_pooling(emb["pooling"])
    return extrator, pooling

