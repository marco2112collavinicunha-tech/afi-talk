"""Etapa 1 do pipeline: pré-processamento (áudio bruto → sinal padronizado)."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

FORMATOS_PELO_AV = (".m4a", ".aac", ".mp4", ".m4b", ".3gp", ".amr", ".wma")


TIPOS_DE_FALHA = frozenset({
    "vazio", "valores invalidos", "formato nao suportado", "silencio",
    "curta demais", "nao abre",
    "saturada", "fala cortada", "longa demais",
})


class AudioInvalido(ValueError):
    """Áudio que não pode seguir no pipeline (vazio, silencioso, corrompido ou curto)."""

    def __init__(self, mensagem, tipo=None):
        super().__init__(mensagem)
        self.tipo = tipo if tipo in TIPOS_DE_FALHA else None


def para_float_mono(sinal, mono=True):
    """Converte o sinal para float32 em [-1, 1] e, se pedido, para um canal só."""
    s = np.asarray(sinal)
    if s.size == 0:
        raise AudioInvalido("O áudio está vazio (nenhuma amostra).", tipo="vazio")

    if np.issubdtype(s.dtype, np.integer):
        s = s.astype(np.float32) / abs(np.iinfo(s.dtype).min)
    else:
        s = s.astype(np.float32)

    if not np.isfinite(s).all():
        raise AudioInvalido("O áudio contém valores inválidos (NaN/infinito) — grave de novo.",
                           tipo="valores invalidos")

    if mono and s.ndim == 2:
        eixo_canais = 0 if s.shape[0] <= s.shape[1] else 1
        s = s.mean(axis=eixo_canais)

    if s.ndim != 1:
        raise AudioInvalido(
            "Formato de áudio não suportado: sinal com {} dimensões. O pipeline "
            "trabalha com sinal mono 1-D (estéreo 2-D é aceito apenas para "
            "conversão a mono).".format(s.ndim),
            tipo="formato nao suportado",
        )
    return s


def normalizar_volume(sinal, metodo, rms_alvo=None):
    """Normaliza o volume do sinal."""
    pico = float(np.abs(sinal).max())
    if pico == 0.0:
        raise AudioInvalido("O áudio é silêncio puro — não há o que normalizar.",
                           tipo="silencio")

    if metodo == "pico":
        return (sinal / pico).astype(np.float32)

    if metodo == "rms":
        if rms_alvo is None:
            raise ValueError("normalizacao 'rms' exige a chave audio.rms_alvo no config.yaml")
        rms = float(np.sqrt(np.mean(np.square(sinal, dtype=np.float64))))
        escalado = sinal * (rms_alvo / rms)
        return np.clip(escalado, -1.0, 1.0).astype(np.float32)

    raise ValueError("Estratégia de normalização desconhecida: {!r}".format(metodo))


def cortar_silencio(sinal, top_db):
    """Remove o silêncio das pontas da gravação (o miolo fica intacto)."""
    aparado, _indices = librosa.effects.trim(sinal, top_db=top_db)
    return aparado


def preprocessar_sinal(sinal, taxa_origem, config, normalizar=True):
    """Caminho único de pré-processamento: sinal em memória → sinal padronizado."""
    cfg = config["audio"]

    if cfg["mono"] is not True:
        raise ValueError(
            "config.yaml: audio.mono deve ser true — todas as etapas do pipeline "
            "(corte de silêncio, duração, embeddings) assumem sinal 1-D mono."
        )

    s = para_float_mono(sinal, mono=cfg["mono"])

    if float(np.abs(s).max()) == 0.0:
        raise AudioInvalido("A gravação é silêncio puro — grave de novo mais perto do microfone.",
                           tipo="silencio")

    taxa_alvo = cfg["taxa_amostragem"]
    if taxa_origem != taxa_alvo:
        s = librosa.resample(s, orig_sr=taxa_origem, target_sr=taxa_alvo)

    if cfg["cortar_silencio"]["ativo"]:
        s = cortar_silencio(s, top_db=cfg["cortar_silencio"]["top_db"])

    duracao = len(s) / taxa_alvo
    if duracao < cfg["duracao_minima_s"]:
        raise AudioInvalido(
            "A gravação ficou curta demais: quase não tem fala nela "
            "({} s de fala; o mínimo é {} s). Grave de novo, com ele falando a "
            "frase inteira perto do celular.".format(
                "{:.2f}".format(duracao).replace(".", ","),
                "{:.2f}".format(cfg["duracao_minima_s"]).replace(".", ","),
            ),
            tipo="curta demais",
        )

    if not normalizar:
        return s.astype(np.float32)

    metodo = cfg["normalizacao"]
    rms_alvo = cfg.get("rms_alvo") if metodo == "rms" else None
    s = normalizar_volume(s, metodo=metodo, rms_alvo=rms_alvo)

    return s.astype(np.float32)


CRITERIOS_DA_CLASSE_1 = ("saturacao", "fala_cortada", "duracao_maxima_s")
CRITERIOS_QUE_EXCLUEM = ("saturacao", "fala_cortada")
CRITERIOS_QUE_SO_SINALIZAM = ("duracao_maxima_s",)
SINAL_DE_DURACAO = "duracao_acima_do_limite"
PONTAS_DA_FALA_CORTADA = ("inicio_s", "fim_s")
TETO_DE_AMOSTRA = 32767.0 / 32768.0
TIPO_DO_CRITERIO = {"saturacao": "saturada", "fala_cortada": "fala cortada",
                    "duracao_maxima_s": "longa demais"}
PAPEL_DO_CRITERIO = {"saturacao": "exclui", "fala_cortada": "exclui",
                     "duracao_maxima_s": "sinal para a escuta; não exclui (08/10/2026)"}
MOTIVO_SEM_VALOR = (
    "valor não fixado em audio.criterios_da_classe_1 do config.yaml; o valor é do "
    "pesquisador responsável, medido nas gravações de engenharia antes do "
    "primeiro participante, e o programa não usa valor padrão")


class CriteriosDaClasse1SemValor(ValueError):
    """Trava: algum critério da classe 1 não tem valor fixado no config."""


def _numero_nao_negativo(valor, nome):
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError("audio.criterios_da_classe_1.{} precisa ser número ou nulo; "
                         "veio {!r}.".format(nome, valor))
    valor = float(valor)
    if not np.isfinite(valor) or valor < 0:
        raise ValueError("audio.criterios_da_classe_1.{} precisa ser número finito e "
                         "não negativo; veio {!r}.".format(nome, valor))
    return valor


def criterios_da_classe_1(config):
    """Os valores dos três critérios lidos do config, conferidos."""
    bloco = (config.get("audio") or {}).get("criterios_da_classe_1")
    if bloco is None:
        bloco = {}
    if not isinstance(bloco, dict):
        raise ValueError("audio.criterios_da_classe_1 precisa ser um mapa; veio {!r}."
                         .format(bloco))
    desconhecidas = sorted(set(bloco) - set(CRITERIOS_DA_CLASSE_1))
    if desconhecidas:
        raise ValueError("audio.criterios_da_classe_1 tem chave(s) desconhecida(s): {}. "
                         "Valem {}.".format(desconhecidas, list(CRITERIOS_DA_CLASSE_1)))
    saida = {}
    for nome in ("saturacao", "duracao_maxima_s"):
        v = bloco.get(nome)
        saida[nome] = None if v is None else _numero_nao_negativo(v, nome)
    corte = bloco.get("fala_cortada")
    if corte is None:
        saida["fala_cortada"] = None
    else:
        if not isinstance(corte, dict) or set(corte) != set(PONTAS_DA_FALA_CORTADA):
            raise ValueError("audio.criterios_da_classe_1.fala_cortada precisa ser "
                             "{{inicio_s, fim_s}}; veio {!r}.".format(corte))
        pontas = {p: (None if corte[p] is None
                      else _numero_nao_negativo(corte[p], "fala_cortada." + p))
                  for p in PONTAS_DA_FALA_CORTADA}
        saida["fala_cortada"] = None if any(v is None for v in pontas.values()) else pontas
    return saida


def criterios_sem_valor(config):
    """Os critérios da classe 1 ainda sem valor fixado, na ordem declarada."""
    valores = criterios_da_classe_1(config)
    return [nome for nome in CRITERIOS_DA_CLASSE_1 if valores[nome] is None]


def exigir_criterios_da_classe_1(config, onde="o lote oficial"):
    """Trava: recusa, com o motivo, enquanto algum critério estiver sem valor."""
    faltam = criterios_sem_valor(config)
    if faltam:
        raise CriteriosDaClasse1SemValor(
            "{} não roda: os critérios da classe 1 {} não têm valor em "
            "audio.criterios_da_classe_1 do config.yaml. Sem eles a exclusão "
            "registrada (plano de análise) não pode ser aplicada. Os valores "
            "são do pesquisador responsável, fixados nas gravações de engenharia "
            "antes do primeiro participante. O programa não usa valor "
            "padrão.".format(onde.capitalize(), ", ".join(faltam)))


def declaracao_dos_criterios_da_classe_1(config):
    """O que a leitura aplicou e o que não aplicou, para ir ao resultado."""
    valores = criterios_da_classe_1(config)
    return {
        "valores": valores,
        "aplicados": [n for n in CRITERIOS_DA_CLASSE_1 if valores[n] is not None],
        "nao_aplicados": {n: MOTIVO_SEM_VALOR for n in CRITERIOS_DA_CLASSE_1
                          if valores[n] is None},
        "papel": dict(PAPEL_DO_CRITERIO),
    }


def medir_criterios_da_classe_1(sinal, taxa_origem, config):
    """As três medidas de uma gravação crua (o que o banco guarda)."""
    cfg = config["audio"]
    s = para_float_mono(sinal, mono=True)
    fracao = float(np.count_nonzero(np.abs(s) >= TETO_DE_AMOSTRA) / s.size)
    taxa_alvo = cfg["taxa_amostragem"]
    r = librosa.resample(s, orig_sr=taxa_origem, target_sr=taxa_alvo) \
        if taxa_origem != taxa_alvo else s
    if float(np.abs(r).max()) == 0.0:
        return {"fracao_no_teto": fracao, "folga_inicio_s": 0.0, "folga_fim_s": 0.0,
                "duracao_apos_corte_s": 0.0}
    top_db = cfg["cortar_silencio"]["top_db"]
    _, (ini, fim) = librosa.effects.trim(r, top_db=top_db)
    return {"fracao_no_teto": fracao,
            "folga_inicio_s": float(ini) / taxa_alvo,
            "folga_fim_s": float(len(r) - fim) / taxa_alvo,
            "duracao_apos_corte_s": float(fim - ini) / taxa_alvo}


def reprovacoes_da_classe_1(medidas, valores):
    """[(critério, texto)] dos critérios com valor em que a gravação reprova."""
    saida = []
    if valores["saturacao"] is not None and medidas["fracao_no_teto"] > valores["saturacao"]:
        saida.append(("saturacao", "fração no teto {:.4f} acima de {:.4f}".format(
            medidas["fracao_no_teto"], valores["saturacao"])))
    corte = valores["fala_cortada"]
    if corte is not None and (medidas["folga_inicio_s"] < corte["inicio_s"]
                              or medidas["folga_fim_s"] < corte["fim_s"]):
        saida.append(("fala_cortada", "folga de {:.3f} s no início e {:.3f} s no fim, "
                      "mínimo {:.3f} s e {:.3f} s".format(
                          medidas["folga_inicio_s"], medidas["folga_fim_s"],
                          corte["inicio_s"], corte["fim_s"])))
    return saida


def sinais_para_a_escuta(medidas, valores):
    """{sinal: bool} dos critérios que só sinalizam (decisão de 08/10/2026, 10h45)."""
    limite = valores["duracao_maxima_s"]
    return {SINAL_DE_DURACAO: bool(limite is not None
                                   and medidas["duracao_apos_corte_s"] > limite)}


def aplicar_criterios_da_classe_1(sinal, taxa_origem, config):
    medidas = medir_criterios_da_classe_1(sinal, taxa_origem, config)
    valores = criterios_da_classe_1(config)
    reprovadas = reprovacoes_da_classe_1(medidas, valores)
    if reprovadas:
        criterio, texto = reprovadas[0]
        raise AudioInvalido("Classe 1 ({}): {}.".format(TIPO_DO_CRITERIO[criterio], texto),
                            tipo=TIPO_DO_CRITERIO[criterio])
    medidas.update(sinais_para_a_escuta(medidas, valores))
    return medidas


def carregar_audio_bruto(caminho):
    """(sinal, taxa) do arquivo, na taxa e nos canais originais, sem padronizar."""
    if not Path(caminho).exists():
        raise FileNotFoundError("Arquivo de áudio não encontrado: {}".format(caminho))
    sufixo = Path(caminho).suffix.lower()
    if sufixo in FORMATOS_PELO_AV:
        return _carregar_com_av(caminho)
    try:
        return librosa.load(caminho, sr=None, mono=False)
    except Exception as erro:
        if sufixo not in FORMATOS_DIRETOS:
            try:
                return _carregar_com_av(caminho)
            except AudioInvalido:
                raise AudioInvalido(_mensagem_de_leitura(caminho, erro), tipo="nao abre") from erro
        raise AudioInvalido(_mensagem_de_leitura(caminho, erro), tipo="nao abre") from erro


def preprocessar_arquivo(caminho, config, normalizar=True, classe_1=False,
                         medidas_da_classe_1=None):
    """Carrega um arquivo de áudio do disco e o padroniza."""
    if classe_1:
        sinal, taxa_origem = carregar_audio_bruto(caminho)
        preprocessado = preprocessar_sinal(sinal, taxa_origem, config, normalizar=normalizar)
        medidas = aplicar_criterios_da_classe_1(sinal, taxa_origem, config)
        if medidas_da_classe_1 is not None:
            medidas_da_classe_1.update(medidas)
        return preprocessado
    if not Path(caminho).exists():
        raise FileNotFoundError("Arquivo de áudio não encontrado: {}".format(caminho))
    sufixo = Path(caminho).suffix.lower()
    if sufixo in FORMATOS_PELO_AV:
        sinal, taxa_origem = _carregar_com_av(caminho)
    else:
        try:
            sinal, taxa_origem = librosa.load(caminho, sr=None, mono=False)
        except Exception as erro:
            if sufixo not in FORMATOS_DIRETOS:
                try:
                    sinal, taxa_origem = _carregar_com_av(caminho)
                except AudioInvalido:
                    raise AudioInvalido(_mensagem_de_leitura(caminho, erro), tipo="nao abre") from erro
            else:
                raise AudioInvalido(_mensagem_de_leitura(caminho, erro), tipo="nao abre") from erro
    return preprocessar_sinal(sinal, taxa_origem, config, normalizar=normalizar)


def _carregar_com_av(caminho):
    """Decodifica com `av` (FFmpeg embutido) e devolve (sinal, taxa_original)."""
    try:
        import av
    except ImportError as erro:
        raise AudioInvalido(
            "O arquivo {!r} está num formato que precisa do pacote `av` para ser "
            "lido (o leitor nativo não abre {}), e ele não está instalado neste "
            "ambiente. Instale com `pip install av` — ele já traz o FFmpeg "
            "embutido, não é preciso instalar mais nada no sistema.".format(
                Path(caminho).name, Path(caminho).suffix.lower() or "este formato"
            ),
            tipo="nao abre",
        ) from erro

    quadros = []
    try:
        with av.open(str(caminho)) as recipiente:
            if not recipiente.streams.audio:
                raise AudioInvalido(
                    "O arquivo {!r} foi aberto, mas não tem NENHUMA trilha de áudio "
                    "dentro. Se for um vídeo sem som, ou um arquivo que só tem capa/"
                    "imagem, regrave o áudio e reenvie.".format(Path(caminho).name),
                    tipo="nao abre",
                )
            fluxo = recipiente.streams.audio[0]
            taxa_origem = int(fluxo.rate)
            reamostrador = av.audio.resampler.AudioResampler(
                format="fltp", layout=fluxo.layout.name, rate=taxa_origem
            )
            for quadro in recipiente.decode(audio=0):
                for convertido in reamostrador.resample(quadro):
                    quadros.append(convertido.to_ndarray())
    except AudioInvalido:
        raise
    except Exception as erro:
        raise AudioInvalido(
            "Não consegui ler o arquivo de áudio {!r}: o arquivo parece "
            "corrompido ou truncado — é o que acontece quando o envio é "
            "interrompido no meio. Peça para reenviar, ou regravar. "
            "(detalhe técnico: {})".format(Path(caminho).name, erro),
            tipo="nao abre",
        ) from erro

    if not quadros:
        raise AudioInvalido(
            "O arquivo de áudio {!r} abriu, mas não veio NENHUM som dele — "
            "provavelmente a gravação foi cortada logo no início ou o arquivo "
            "chegou incompleto. Peça para regravar e reenviar.".format(
                Path(caminho).name
            ),
            tipo="nao abre",
        )

    sinal = np.concatenate(quadros, axis=-1).astype(np.float32, copy=False)
    if sinal.shape[0] == 1:
        sinal = sinal[0]
    return sinal, taxa_origem


FORMATOS_DIRETOS = (".wav", ".flac", ".ogg", ".opus", ".aiff", ".aif", ".au",
                    ".caf", ".mp3", ".w64", ".aifc")


def _mensagem_de_leitura(caminho, erro):
    """Mensagem de falha de leitura que orienta o conserto."""
    nome = Path(caminho).name
    sufixo = Path(caminho).suffix.lower()
    partes = ["Não consegui ler o arquivo de áudio {!r}: {}".format(nome, erro)]
    if sufixo and sufixo not in FORMATOS_DIRETOS and sufixo not in FORMATOS_PELO_AV:
        partes.append(
            "O formato {} não está entre os que o sistema lê. Lidos diretamente: "
            "{}. Lidos pelo pacote `av`: {}.".format(
                sufixo, ", ".join(FORMATOS_DIRETOS), ", ".join(FORMATOS_PELO_AV)
            )
        )
        partes.append(
            "Saída mais simples: reenviar como nota de voz do WhatsApp (vira "
            ".opus, que o sistema lê) ou gravar em .wav."
        )
    else:
        partes.append(
            "O formato é reconhecido, então o mais provável é o arquivo estar "
            "truncado ou corrompido — envio interrompido no meio, por exemplo. "
            "Peça para reenviar, ou regravar."
        )
    return " ".join(partes)

