#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

N_INTENCOES = 12
ACASO = 1.0 / N_INTENCOES
SEMENTE = 20260921

NIVEL_DO_IC = 0.95

MIN_FRASES_DA_SENSIBILIDADE_DO_GRUPO = 6

MAXIMO_DE_FRASES_SEM_TREINO_SEM_MARCA = 3

CHAVES = {
    "n_min_por_frase": 3,
    "treino_minimo_por_sessao_de_treino": 2,
    "min_frases": 1,
    "min_frases_no_grupo": 4,
    "exige_duas_sessoes": True,
    "curva_aprendizagem": True,
    "selecao": "por_participante",
    "grade_de_configuracoes": "12:knn_k1,12:regressao_logistica,18:knn_k1,18:regressao_logistica",
    "replicas_permutacao": 0,
    "replicas_bootstrap": 2000,
    "replicas_bootstrap_duplo": 600,
    "semente_bootstrap_duplo": 20260925,
    "replicas_permutacao_h2": 20000,
    "regra_de_faltantes": True,
    "regra_de_sucesso": "k_fixo_3",
    "metodo_do_intervalo": "wald_gl_efetivos",
    "metodo_do_intervalo_sensibilidade": "bootstrap_duplo",
    "frases_de_consistencia": 8,
    "metodo_da_h2": "freedman_lane_proporcao",
    "h2_minimo_de_frases_com_indice": 4,
    "h2_minimo_de_frases_maximo_da_sensibilidade": 6,
    "h2_escala_de_sensibilidade": "freedman_lane_logito",
    "h2_gravacoes_minimas_da_sensibilidade_de_escala": 5,
    "sensibilidade_h2_qualidade_tecnica": None,
    "sensibilidade_h2_duracao": None,
    "alcance_da_h1a": "frases_fixas_peso_igual",
    "degrau_2_comparadores": "soma,media_por_token,calibrada_silencio",
    "degrau_2_regra": "vencer_todos",
    "degrau_2_referencia_de_silencio": "zeros_mesma_duracao",
    "degrau_2_whisper_forcado_ligado": True,
    "h1_alfa_unilateral": 0.04,
    "h1_n_minimo_para_veredito": 3,
    "h1b_comparador": "entre_pessoas_igualado",
    "h1b_igualado_contagem": "por_frase_e_sessao_igual_ao_personalizado",
    "h1b_igualado_sorteios": 20,
    "h1b_igualado_semente": 20260921,
    "h1b_regra": "nadeau_bengio_e_jackknife_por_participante",
    "h1b_fator_nadeau_bengio": "1/N+1/(N-1)",
    "h1b_igualado_faltante": "reduz_os_dois_sem_reposicao",
    "h1b_igualado_minimo_por_celula": "treino_minimo_por_sessao_de_treino",
    "h1b_igualado_configuracao": "selecao_com_a_quantidade_do_personalizado_sem_o_alvo",
    "h1b_ordem": "sempre_calculada_confirmatoria_so_com_h1a",
    "entre_pessoas_completo_papel": "exploratorio_sem_veredito",
    "h1_intervalos_relatados": "92_decide_e_95_ao_lado",
    "h1b_sorteios_relatados": "cada_sorteio_e_a_media",
    "empate": "credito_fracionario_1_sobre_k",
    "h1a_degraus": "degrau_1_principal_degrau_2_qualifica_sem_apagar",
    "degrau_2_grafias": "como_escrita,maiuscula_e_ponto",
    "degrau_2_agregacao_das_grafias": "maior_nota_por_rotulo_e_gravacao",
    "degrau_2_sha256_das_sequencias":
        "93a05e4d52a6b1230e8c5d34d344f774a4df9f741b714e23fae7b1184d6a6bbf",
    "h1_testes_de_apoio": "permutacao_sensibilidade_wilcoxon_apoio_sem_decidir",
    "sensibilidade_seis_frases": "todos_os_contrastes_pareados_sem_decidir",
    "sensibilidade_por_servico": "medias_por_servico_3_servicos_algum_com_2_sem_decidir",
    "desvio_de_producao": "registrado_a_parte_sem_efeito_na_analise",
    "aparelho": "todos_na_h2_estratificada_descritiva_com_3_por_plataforma",
    "ordem_das_frases": "random_Random_texto_shuffle_rotulos_alfabeticos_repeticoes_juntas",
    "ordem_das_frases_semente": "{semente_do_sorteio}|{pessoa}|sessao{N}",
    "consistencia_divisao_entre_sessoes":
        "escolha_clinica_ou_shuffle_{semente_do_sorteio}|{pessoa}_metades_na_ordem_das_sessoes",
    "ordem_retomada": "mesma_ordem_primeira_pendente_contada_por_sessao_e_frase_no_banco",
    "conferencia_20_sementes": "afi-2026|escuta20|{codigo};afi-2026|onset20|{codigo}",
    "conferencia_20_derivacao": "sha256_utf8_8_primeiros_bytes_big_endian_random_Random_sample",
    "conferencia_20_amostra": "por_participante_ordem_lexicografica_sem_reposicao_teto_20pct_min_1",
    "h1_multiplicidade": "degrau_1_a_4pct_depois_holm_4pct_degrau_2_e_h1b",
    "h1_p_do_degrau_2_na_familia": "maior_dos_tres_p_unilaterais",
    "h1_p_da_h1b_na_familia": "maior_entre_nadeau_bengio_e_jackknife",
    "h1_intervalo_no_nivel_de_holm": "degrau_2_e_h1b_no_limiar_efetivamente_usado",
    "h1_rotulo_de_n_pequeno": "preliminar_nesta_amostra_de_3_a_7_plena_com_8",
    "selecao_sem_cobertura":
        "primeira_da_lista_com_motivo_se_alguma_frase_tiver_menos_de_2_validas_numa_sessao_de_treino",
    "sensibilidade_configuracao_padrao": "degrau_1_com_a_primeira_da_lista_para_todos_sem_decidir",
    "escuta_classe_2":
        "checklist_3_itens_decide_o_pesquisador_segunda_ouvinte_20pct_so_relata",
    "classe_1_saturacao_fracao_maxima": 0.001,
    "classe_1_fala_cortada_folga_minima_inicio_s": 0.1,
    "classe_1_fala_cortada_folga_minima_fim_s": 0.1,
    "classe_1_duracao_maxima_s": 10,
    "sensibilidade_barulho": "sem_as_marcadas_pela_familia_nas_tres_condicoes_sem_decidir",
    "pesos_xlsr53_revisao": "c3f9d884181a224a6ac87bf8885c84d1cff3384f",
    "pesos_xlsr53_sha256_dos_tensores": "9f7863587e4a2a77e40d66745e41f6820bbad95320466b0e07fdb07095901dff",
    "pesos_whisper_revisao": "08e178d48790749d25932bbc082711ddcfdfbc4f",
    "pesos_whisper_sha256_dos_arquivos": "b4ffdc08b46c4f51f369a7b3af94d7408b0ae898451f6c5e7a5798143ef7171c",
    "pesos_vad_sha256_dos_tensores": "6d939b3e09ac660b64b825ba75d2271e52c48df240ddc454d6bb25c7111ac50a",
}

REGRAS_DE_06_10 = {k: CHAVES[k] for k in (
    "h1a_degraus", "degrau_2_grafias", "degrau_2_agregacao_das_grafias",
    "h1_testes_de_apoio", "sensibilidade_seis_frases", "sensibilidade_por_servico",
    "desvio_de_producao", "aparelho", "ordem_das_frases", "ordem_das_frases_semente",
    "consistencia_divisao_entre_sessoes", "ordem_retomada", "conferencia_20_sementes",
    "conferencia_20_derivacao", "conferencia_20_amostra")}

REGRAS_DA_NOITE_DE_06_10 = {k: CHAVES[k] for k in (
    "h1_multiplicidade", "h1_p_do_degrau_2_na_familia", "h1_p_da_h1b_na_familia",
    "h1_intervalo_no_nivel_de_holm", "h1_rotulo_de_n_pequeno")}

REGRAS_DE_07_10 = {k: CHAVES[k] for k in (
    "selecao_sem_cobertura", "sensibilidade_configuracao_padrao", "escuta_classe_2",
    "sensibilidade_barulho")}

MARCA_A_PREENCHER = "A PREENCHER"
CHAVES_COM_VALOR_A_PREENCHER = (
    "classe_1_saturacao_fracao_maxima", "classe_1_fala_cortada_folga_minima_inicio_s",
    "classe_1_fala_cortada_folga_minima_fim_s", "classe_1_duracao_maxima_s",
    "pesos_xlsr53_revisao", "pesos_xlsr53_sha256_dos_tensores", "pesos_whisper_revisao",
    "pesos_whisper_sha256_dos_arquivos", "pesos_vad_sha256_dos_tensores")


def marcas_a_preencher(chaves):
    """As chaves de valor ainda marcadas como a preencher, na ordem declarada."""
    return [k for k in CHAVES_COM_VALOR_A_PREENCHER
            if isinstance(chaves.get(k), str) and MARCA_A_PREENCHER in chaves[k]]


def conferir_regras_de_07_10(chaves):
    """Recusa chave das regras de 07/10/2026 com valor que o programa não implementa."""
    for k, v in REGRAS_DE_07_10.items():
        if chaves.get(k) != v:
            raise ValueError("{} = {!r} não implementado (o programa implementa {!r})"
                             .format(k, chaves.get(k), v))

POLITICAS_DE_EMPATE = ("credito_fracionario_1_sobre_k",)

ALCANCES_DA_H1A = ("frases_fixas_peso_igual", "frases_como_amostra")

METODOS_DA_H2 = ("freedman_lane_proporcao", "freedman_lane_logito")
METODOS_DO_INTERVALO = ("wald_gl_efetivos", "bootstrap_duplo", "delta_var_entre")

MINIMO_DE_PARTICIPANTES = 8

ARQUIVO_DAS_CHAVES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "chaves.json")
MODOS = ("estudo", "em_verificacao", "exploratorio")


def chaves_congeladas(caminho=ARQUIVO_DAS_CHAVES):
    """(chaves do arquivo, sha256 do arquivo)."""
    with open(caminho, "rb") as f:
        bruto = f.read()
    congeladas = json.loads(bruto.decode("utf-8"))
    faltam = sorted(set(CHAVES) - set(congeladas))
    sobram = sorted(set(congeladas) - set(CHAVES))
    diferentes = sorted(k for k in set(CHAVES) & set(congeladas)
                        if congeladas[k] != CHAVES[k]
                        or type(congeladas[k]) is not type(CHAVES[k]))
    if faltam or sobram or diferentes:
        raise ValueError(
            "o arquivo de chaves congeladas ({}) discorda de CHAVES no script: "
            "faltam {}, sobram {}, valores diferentes em {}. Um dos dois foi "
            "mudado sem o outro; a analise nao escolhe qual vale.".format(
                caminho, faltam, sobram, diferentes))
    return congeladas, hashlib.sha256(bruto).hexdigest()

REGRAS_DE_SUCESSO = {
    "k_fixo_3": {
        "pergunta": "o ganho se replica?",
        "criterio": "pelo menos 3 participantes com o intervalo do ganho "
                    "inteiro acima de zero E mais participantes com ganho "
                    "demonstrado do que com piora demonstrada",
        "prova": ("rejeita a hipotese de que nenhum participante se beneficia; "
                  "nao estabelece que a maioria se beneficia"),
        "estado": ("contagem descritiva com a trava; decide a H1a so' com "
                   "alcance_da_h1a = frases_como_amostra"),
    },
    "metade_ou_mais": {
        "pergunta": "a maioria dos adolescentes ganha?",
        "criterio": "metade ou mais dos participantes com o intervalo do ganho "
                    "inteiro acima de zero",
        "estado": "alternativa comparada; nao e' a regra do estudo",
    },
}

REGRA_DE_FALTANTES = {
    "gravacao_ausente": "exclui o par",
    "reconhecedor_sem_saida": "conta como erro",
    "pane_tecnica": "repete; se persistir, conta como erro",
    "frase_sem_gravacao_valida": "sai das tres condicoes",
}

CAUSAS = {
    "acertou": "o sistema respondeu a frase-alvo",
    "predicao_errada": "o sistema respondeu outra frase do banco",
    "empate_no_topo": "t frases empataram na maior nota; vale 1/t se a frase-alvo esta' entre elas",
    "sem_saida": "o reconhecedor nao devolveu nada, ou devolveu fora das doze",
    "falha_tecnica": "pane tecnica que persistiu depois de repetir",
    "falta_de_treino": "a frase nao tem o treino minimo naquele participante",
    "gravacao_ausente": "nao ha gravacao valida",
}

CAUSAS_DE_ERRO = ("predicao_errada", "sem_saida", "falha_tecnica")

CAUSAS_QUE_NAO_SAO_GRAVACAO = ("falta_de_treino", "gravacao_ausente")

CAUSAS_SEM_RESPOSTA_DO_GENERICO = ("sem_saida", "falha_tecnica")

CAUSA_AUSENTE = None


SEPARADOR_DO_EMPATE = "|"


def ler_acerto(bruto, n_frases=N_INTENCOES):
    """O acerto gravado ("0", "1" ou "1/t") como número."""
    if isinstance(bruto, (int, float, np.integer, np.floating)) and not isinstance(
            bruto, bool):
        valor = float(bruto)
        if valor in (0.0, 1.0):
            return valor
        t = int(round(1.0 / valor)) if valor > 0 else 0
        if 2 <= t <= n_frases and abs(valor - 1.0 / t) < 1e-12:
            return 1.0 / t
        raise ValueError("acerto {!r}: vale 0, 1 ou 1/t com t entre 2 e {}".format(
            bruto, n_frases))
    texto = str(bruto).strip()
    if texto in ("0", "1"):
        return float(texto)
    if texto.startswith("1/") and texto[2:].isdigit():
        t = int(texto[2:])
        if 2 <= t <= n_frases:
            return 1.0 / t
    raise ValueError("acerto {!r}: vale \"0\", \"1\" ou \"1/t\" com t entre 2 e "
                     "{}".format(bruto, n_frases))


def empatadas_do_predito(predito):
    """As frases empatadas listadas em `predito` ("a|b|c"), ou [] sem empate."""
    texto = str(predito or "")
    if SEPARADOR_DO_EMPATE not in texto:
        return []
    return texto.split(SEPARADOR_DO_EMPATE)


def conferir_acerto_com_o_empate(l, numero_da_linha):
    """Recusa a linha cujo acerto não bate com o empate registrado em `predito`."""
    acerto = ler_acerto(l["acerto"])
    empatadas = empatadas_do_predito(l.get("predito"))
    if empatadas:
        if len(set(empatadas)) != len(empatadas) or len(empatadas) < 2:
            raise ValueError("linha {}: `predito` {!r} não é uma lista de frases "
                             "empatadas distintas".format(numero_da_linha,
                                                          l.get("predito")))
        esperado = 1.0 / len(empatadas) if l["frase"] in empatadas else 0.0
        if abs(acerto - esperado) > 1e-12:
            raise ValueError(
                "linha {}: {} frases empatadas em `predito` ({}) e acerto {!r}; o "
                "acerto teria de ser {}".format(
                    numero_da_linha, len(empatadas), l.get("predito"), l["acerto"],
                    "1/{}".format(len(empatadas)) if esperado else "0"))
    elif acerto not in (0.0, 1.0):
        raise ValueError(
            "linha {}: acerto {!r} fracionário sem as frases empatadas em "
            "`predito`".format(numero_da_linha, l["acerto"]))
    return acerto


def _normalizar_causa(linhas, tem_coluna):
    """Põe `causa` em toda linha: um valor do vocabulário, ou `None`."""
    if not tem_coluna:
        for l in linhas:
            l["causa"] = CAUSA_AUSENTE
        return False

    def _e_reconhecimento(l):
        return l.get("condicao") in ("personalizado", "entre_pessoas",
                                     "generico")

    vazias, preenchidas = [], 0
    for i, l in enumerate(linhas, 2):
        if not _e_reconhecimento(l):
            l["causa"] = CAUSA_AUSENTE
            continue
        valor = (l.get("causa") or "").strip()
        if not valor:
            vazias.append(i)
        else:
            preenchidas += 1
    if vazias and preenchidas:
        raise SystemExit(
            "O acertos.csv esta MEIO MIGRADO na coluna `causa`: {} linha(s) "
            "tem causa e {} nao tem (a primeira sem causa e a linha {}).\n\n"
            "Somar os dois esquemas no mesmo arquivo faria as clausulas da "
            "regra de faltantes rodarem em parte das linhas e nao em outra, e o "
            "relatorio nao teria como dizer em quais.\n"
            "Ou TODAS as linhas tem a causa, ou NENHUMA tem.".format(
                preenchidas, len(vazias), vazias[0]))
    if vazias or not preenchidas:
        for l in linhas:
            l["causa"] = CAUSA_AUSENTE
        return False

    for i, l in enumerate(linhas, 2):
        if not _e_reconhecimento(l):
            continue
        causa = (l["causa"] or "").strip()
        if causa not in CAUSAS:
            raise SystemExit(
                "Causa desconhecida {!r} na linha {} do acertos.csv — "
                "participante {!r}, frase {!r}, condicao {!r}.\n\n"
                "As causas aceitas sao: {}.\n"
                "Causa fora do vocabulario NAO vira 'predicao_errada': a "
                "analise para, porque tratar um estado desconhecido como erro "
                "de predicao seria afirmar algo que ninguem mediu.".format(
                    causa, i, l.get("pessoa"), l.get("frase"),
                    l.get("condicao"), ", ".join(sorted(CAUSAS))))
        l["causa"] = causa
        try:
            valor = ler_acerto(l.get("acerto", ""))
        except ValueError as e:
            raise SystemExit("Linha {} do acertos.csv: {} (participante {!r}, "
                             "frase {!r}, condicao {!r}).".format(
                                 i, e, l.get("pessoa"), l.get("frase"),
                                 l.get("condicao")))
        acerto = {0.0: "0", 1.0: "1"}.get(valor, "fracao")
        tem_empate = bool(empatadas_do_predito(l.get("predito")))
        if (causa == "empate_no_topo") != tem_empate:
            raise SystemExit(
                "Linha {} do acertos.csv: causa {!r} e `predito` {!r} — a causa "
                "'empate_no_topo' vai junto com as frases empatadas em `predito`, "
                "e so' com ela (participante {!r}, frase {!r}, condicao "
                "{!r}).".format(i, causa, l.get("predito"), l.get("pessoa"),
                                l.get("frase"), l.get("condicao")))
        if acerto == "fracao" and causa != "empate_no_topo":
            raise SystemExit(
                "Linha {} do acertos.csv diz `acerto = {}` e causa {!r}: acerto "
                "fracionario so' existe no empate ('empate_no_topo').".format(
                    i, l.get("acerto"), causa))
        if acerto == "1" and causa != "acertou":
            raise SystemExit(
                "Linha {} do acertos.csv diz `acerto = 1` e causa {!r} ao "
                "mesmo tempo — participante {!r}, frase {!r}, condicao {!r}.\n"
                "Acerto so' aceita a causa 'acertou'.".format(
                    i, causa, l.get("pessoa"), l.get("frase"), l.get("condicao")))
        if acerto == "0" and causa == "acertou":
            raise SystemExit(
                "Linha {} do acertos.csv diz `acerto = 0` e causa 'acertou' "
                "ao mesmo tempo — participante {!r}, frase {!r}, condicao "
                "{!r}.".format(i, l.get("pessoa"), l.get("frase"),
                               l.get("condicao")))
    return True


def causa_disponivel(linhas):
    """True quando toda linha de reconhecimento traz causa do vocabulário; nunca por linha."""
    reconhecimento = [l for l in linhas
                      if l.get("condicao") in ("personalizado",
                                               "entre_pessoas", "generico")]
    return bool(reconhecimento) and all(l.get("causa") in CAUSAS
                                        for l in reconhecimento)


def _sha256_do_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 16), b""):
            h.update(bloco)
    return h.hexdigest()


def hashes_das_entradas(diretorio, nomes):
    """sha256 completo de cada arquivo de entrada lido."""
    saida = {}
    for nome in nomes:
        caminho = os.path.join(diretorio, nome)
        if os.path.exists(caminho):
            saida[nome] = {"sha256": _sha256_do_arquivo(caminho),
                           "bytes": os.path.getsize(caminho)}
        else:
            saida[nome] = {"sha256": None, "motivo": "arquivo nao encontrado"}
    return saida


def _hash_de_src_analise():
    """Hash do `src/analise.py` que este processo importa (o mesmo módulo usado no IC do escore pareado)."""
    try:
        import src.analise as _m
        return _sha256_do_arquivo(_m.__file__)[:16]
    except (ImportError, OSError) as e:
        return "indisponivel: {}".format(e)


def estado_do_git(raiz=None):
    """Commit e estado da árvore em que este script rodou."""
    import subprocess
    raiz = raiz or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _git(*args):
        r = subprocess.run(["git", "-C", raiz] + list(args),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise OSError(r.stderr.strip() or "git falhou")
        return r.stdout

    try:
        commit = _git("rev-parse", "HEAD").strip()
        sujos = [l for l in _git("status", "--porcelain",
                                 "--untracked-files=all").splitlines() if l]
    except (OSError, FileNotFoundError) as e:
        return {"commit": None, "arvore_limpa": None,
                "arquivos_alterados": None,
                "motivo": "estado do git indisponivel: {}".format(e)}
    return {"commit": commit, "arvore_limpa": not sujos,
            "arquivos_alterados": len(sujos)}


def ler(diretorio):
    caminho = os.path.join(diretorio, "acertos.csv")
    if not os.path.exists(caminho):
        raise SystemExit("Nao achei {}.".format(caminho))
    with open(caminho, encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        colunas = list(leitor.fieldnames or [])
        linhas = list(leitor)
    _normalizar_causa(linhas, "causa" in colunas)
    for i, l in enumerate(linhas, 2):
        try:
            l["acerto"] = conferir_acerto_com_o_empate(l, i)
        except (TypeError, ValueError) as e:
            raise SystemExit(
                "Linha {} do acertos.csv: 'acerto' invalido -- {} (pessoa {!r}, "
                "frase {!r}, condicao {!r}).".format(
                    i, e, l.get("pessoa"), l.get("frase"), l.get("condicao")))
        try:
            l["sessao"] = int(l["sessao"])
        except (TypeError, ValueError):
            raise SystemExit(
                "Linha {} do acertos.csv: 'sessao' nao e' inteiro "
                "(pessoa {!r}, frase {!r}, condicao {!r}).".format(
                    i, l.get("pessoa"), l.get("frase"), l.get("condicao")))
        bruto = l.get("sti")
        if bruto is None or str(bruto).strip() == "":
            l["sti"] = None
            continue
        try:
            l["sti"] = float(bruto)
        except (TypeError, ValueError):
            raise SystemExit(
                "STI NAO NUMERICO na linha {} do acertos.csv — "
                "participante {!r}, frase {!r}, valor {!r}.\n\n"
                "Vazio quer dizer 'frase sem indice' e e' aceito "
                "(a H2 usa so' as 8 do bloco de consistencia). "
                "Um valor que EXISTE e nao e' numero e' outra coisa: e' "
                "arquivo corrompido ou coluna trocada, e o script para.".format(
                    i, l.get("pessoa"), l.get("frase"), bruto))
    _exigir_separacao_entre_o_sti_e_o_acerto(linhas)
    return linhas


def _exigir_indice_nas_frases_de_consistencia(linhas, chaves):
    """Confere, por participante, o número de frases com índice contra o que o protocolo promete."""
    esperado = chaves.get("frases_de_consistencia")
    if not esperado:
        return {}
    minimo = chaves["h2_minimo_de_frases_com_indice"]
    com_indice = defaultdict(set)
    todas = defaultdict(set)
    for l in linhas:
        todas[l["pessoa"]].add(l["frase"])
        if l.get("sti") is not None:
            com_indice[l["pessoa"]].add(l["frase"])
    fora_da_h2 = {}
    for pessoa in sorted(todas):
        n = len(com_indice.get(pessoa, ()))
        if minimo <= n <= esperado:
            continue
        if n < minimo:
            fora_da_h2[pessoa] = (
                "fora so' da H2: indice em {} de {} frase(s) do bloco de "
                "consistencia, abaixo do minimo de {} (plano de analise). "
                "{} Continua na H1, que nao usa o indice.".format(
                    n, esperado, minimo,
                    "Sem indice nenhum: sessoes de consistencia (3 e 4) "
                    "ausentes ou sem bloco valido." if n == 0 else
                    "Indice faltou onde deveria haver (STI "
                    "incalculavel ou bloco invalido)."))
            continue
        raise SystemExit(
            "O participante {!r} tem indice em {} frase(s), e o bloco de "
            "consistencia tem {}.\n\n"
            "Indice a MAIS quer dizer que ele foi calculado sobre gravacao "
            "que nao e' de consistencia. A H2 passaria a correlacionar o "
            "indice com o acerto medido na MESMA fala que o originou, e a "
            "correlacao subiria por circularidade, nao por efeito.".format(
                pessoa, n, esperado))
    return fora_da_h2


def _linhas_da_h2(linhas, fora_da_h2):
    """As linhas que a H2 recebe: as de quem saiu da H2 vão sem índice."""
    if not fora_da_h2:
        return linhas
    return [dict(l, sti=None) if l["pessoa"] in fora_da_h2 else l
            for l in linhas]


SESSOES_DE_CONSISTENCIA = (3, 4)
SESSOES_DE_TESTE = (5, 6)


def _exigir_separacao_entre_o_sti_e_o_acerto(linhas):
    """A H2 correlaciona o STI com o acerto."""
    presentes = sorted({int(l["sessao"]) for l in linhas})
    if not presentes:
        return
    fora = [s for s in presentes if s not in SESSOES_DE_TESTE]
    if fora:
        raise SystemExit(
            "O acertos.csv traz sessoes que nao sao de teste: {}.\n\n"
            "A analise do estudo exige que o ACERTO venha SO' das sessoes "
            "de teste {} e que o STI venha das sessoes de consistencia {}.\n"
            "Se as duas medidas vierem das mesmas gravacoes, a H2 correlaciona "
            "o indice com um acerto medido sobre a propria fala que o originou -- "
            "e a correlacao sobe por circularidade, nao por efeito.\n\n"
            "Se a inclusao de outras sessoes for deliberada, ela e' MUDANCA DE "
            "ANALISE DO ESTUDO e precisa de decisao registrada, nao de um "
            "arquivo diferente.".format(
                fora, list(SESSOES_DE_TESTE), list(SESSOES_DE_CONSISTENCIA)))
    sobreposicao = set(presentes) & set(SESSOES_DE_CONSISTENCIA)
    if sobreposicao:
        raise SystemExit(
            "As sessoes do acerto e do STI se sobrepoem em {}. "
            "A H2 ficaria circular.".format(sorted(sobreposicao)))


def frases_sem_treino_minimo(linhas):
    """{pessoa: set(frases)} marcadas `falta_de_treino` (cláusula 4)."""
    fora = defaultdict(set)
    for l in linhas:
        if l.get("causa") == "falta_de_treino":
            fora[l["pessoa"]].add(l["frase"])
    return dict(fora)


def gravacoes_ausentes(linhas):
    """set de identificadores marcados `gravacao_ausente` (cláusula 1)."""
    return {(l["pessoa"], l["gravacao"]) for l in linhas
            if l.get("causa") == "gravacao_ausente" and l.get("gravacao")}


def _e_gravacao(l):
    """A linha corresponde a uma tentativa real de reconhecimento?"""
    return l.get("causa") not in CAUSAS_QUE_NAO_SAO_GRAVACAO


EM_VERIFICACAO_PADRAO = {
    "metodo_da_h1": "k_fixo_3",
    "h1b_confirmatoria": False,
    "holm_h1a_h1b": True,
    "n_minimo_dois_fatores": 3,
    "min_gravacoes_validas_por_frase": None,
    "metodo_da_h2_em_verificacao": None,
    "h2_minimo_de_frases_com_indice": 4,
    "sensibilidade_barulho": False,
    "sensibilidade_6_primeiras": False,
    "sensibilidade_letras_palavras": False,
    "sensibilidade_sessoes_completas": False,
}
_VALORES_EM_VERIFICACAO = {
    "metodo_da_h1": ("k_fixo_3", "dois_fatores"),
    "metodo_da_h2_em_verificacao": (None, "freedman_lane_proporcao_com_logito"),
}
_BOOLEANAS_EM_VERIFICACAO = ("h1b_confirmatoria", "holm_h1a_h1b", "sensibilidade_barulho",
                             "sensibilidade_6_primeiras", "sensibilidade_letras_palavras",
                             "sensibilidade_sessoes_completas")


ARQUIVO_DAS_DATAS = "datas_das_sessoes.csv"


def sessoes_no_mesmo_dia(diretorio):
    """Quantas sessões abriram no mesmo dia em que a anterior terminou."""
    caminho = os.path.join(diretorio, ARQUIVO_DAS_DATAS)
    if not os.path.exists(caminho):
        return {"disponivel": False,
                "motivo": "{} ausente: a contagem não é feita (não se supõe zero)".format(
                    ARQUIVO_DAS_DATAS)}
    por = defaultdict(dict)
    with open(caminho, encoding="utf-8", newline="") as f:
        for l in csv.DictReader(f):
            if l.get("banco_ilegivel"):
                continue
            por[l["pessoa"]][int(l["sessao"])] = (l["primeira_gravacao_em"] or "",
                                                 l["ultima_gravacao_em"] or "")
    no_mesmo_dia, sem_data = {}, 0
    for pessoa, sessoes in sorted(por.items()):
        ordem = sorted(sessoes)
        for anterior, atual in zip(ordem, ordem[1:]):
            fim_anterior = sessoes[anterior][1][:10]
            inicio = sessoes[atual][0][:10]
            if not fim_anterior or not inicio:
                sem_data += 1
            elif inicio == fim_anterior:
                no_mesmo_dia.setdefault(pessoa, []).append(atual)
    return {"disponivel": True,
            "total": sum(len(v) for v in no_mesmo_dia.values()),
            "por_participante": no_mesmo_dia,
            "pares_de_sessoes_sem_data": sem_data,
            "papel": ("DESCRITIVO: sessão aberta no mesmo dia em que a anterior terminou "
                      "(a página recusa, o receptor não). Não exclui nem muda veredito.")}


SESSOES_DO_PARTICIPANTE_COMPLETO = {"h1": (1, 2, 5, 6), "h2": (1, 2, 3, 4, 5, 6)}


def fluxo_de_participantes(diretorio, linhas, grupos, motivos_da_elegibilidade,
                           saidas):
    """Uma entrada por participante: sessões feitas, se é completo em cada hipótese e de onde saiu, com o motivo."""
    caminho = os.path.join(diretorio, ARQUIVO_DAS_DATAS)
    sessoes = defaultdict(set)
    ilegivel = {}
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8", newline="") as f:
            for l in csv.DictReader(f):
                if l.get("banco_ilegivel"):
                    ilegivel[l["pessoa"]] = l["banco_ilegivel"]
                    continue
                sessoes[l["pessoa"]].add(int(l["sessao"]))
        motivo_sem_sessoes = None
    else:
        motivo_sem_sessoes = ("{} ausente: sessões feitas e completude não "
                              "conferidas".format(ARQUIVO_DAS_DATAS))
    com_linha = {l["pessoa"] for l in linhas}
    participantes = {}
    for pessoa in sorted(com_linha | set(sessoes) | set(ilegivel)):
        feitas = sorted(sessoes.get(pessoa, ()))
        sem_sessoes = motivo_sem_sessoes or ilegivel.get(pessoa)
        if sem_sessoes:
            completo = {h: None for h in SESSOES_DO_PARTICIPANTE_COMPLETO}
        else:
            completo = {h: set(exigidas) <= set(feitas)
                        for h, exigidas in SESSOES_DO_PARTICIPANTE_COMPLETO.items()}
        saiu = {}
        if pessoa in ilegivel and pessoa not in com_linha:
            saiu["todas"] = ilegivel[pessoa] + ": não entra em nenhuma análise"
        elif pessoa not in com_linha:
            saiu["todas"] = ("gravou, mas não tem linha no acertos.csv: não entra "
                             "em nenhuma análise")
        if pessoa in motivos_da_elegibilidade:
            saiu["analise_principal"] = motivos_da_elegibilidade[pessoa]
        for nome, por_pessoa in sorted(saidas.items()):
            if pessoa in (por_pessoa or {}):
                saiu[nome] = por_pessoa[pessoa]
        participantes[pessoa] = {
            "grupo": grupos.get(pessoa),
            "sessoes_feitas": None if sem_sessoes else feitas,
            "completo": completo,
            "saiu_de": saiu,
        }
    return {
        "sessoes_disponiveis": motivo_sem_sessoes is None,
        "motivo": motivo_sem_sessoes,
        "definicao_de_completo": {h: list(v) for h, v in
                                  SESSOES_DO_PARTICIPANTE_COMPLETO.items()},
        "participantes": participantes,
        "papel": ("DESCRITIVO: quem fez quais sessões e de onde saiu, com o motivo. "
                  "Não exclui nem muda veredito."),
    }


ROTULOS_DO_MODO = {
    "estudo": "análise do estudo",
    "em_verificacao": "análise em verificação",
    "exploratoria": "análise exploratória",
}


def chaves_em_verificacao_fora_do_padrao(verif):
    return sorted(k for k in verif if verif[k] != EM_VERIFICACAO_PADRAO[k])


def rotulo_do_modo(chaves_fora_do_arquivo, verif_fora_do_padrao, modo="estudo"):
    """O nome que a execução leva no JSON e na tela."""
    if chaves_fora_do_arquivo:
        return ROTULOS_DO_MODO["exploratoria"]
    if verif_fora_do_padrao or modo == "em_verificacao":
        return ROTULOS_DO_MODO["em_verificacao"]
    return ROTULOS_DO_MODO["estudo"]


def hash_das_chaves_em_verificacao(verif):
    """sha256 dos valores em verificação que valeram (JSON canônico, chaves ordenadas)."""
    return hashlib.sha256(json.dumps(verif, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def caminho_do_config():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config.yaml")


def ler_em_verificacao(config=None):
    """As chaves em verificação: padrão + seção `analise_em_verificacao` do config."""
    if config is None:
        import yaml
        caminho = caminho_do_config()
        if not os.path.exists(caminho):
            return dict(EM_VERIFICACAO_PADRAO)
        with open(caminho, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    secao = config.get("analise_em_verificacao") or {}
    desconhecidas = sorted(set(secao) - set(EM_VERIFICACAO_PADRAO))
    if desconhecidas:
        raise ValueError("analise_em_verificacao: chave(s) desconhecida(s) {}; as que "
                         "existem são {}".format(desconhecidas, sorted(EM_VERIFICACAO_PADRAO)))
    saida = dict(EM_VERIFICACAO_PADRAO, **secao)
    for k, validos in _VALORES_EM_VERIFICACAO.items():
        if saida[k] not in validos:
            raise ValueError("analise_em_verificacao.{} = {!r}; vale um de {}".format(
                k, saida[k], validos))
    for k in _BOOLEANAS_EM_VERIFICACAO:
        if not isinstance(saida[k], bool):
            raise ValueError("analise_em_verificacao.{} tem de ser true ou false".format(k))
    h = saida["h2_minimo_de_frases_com_indice"]
    if not (isinstance(h, int) and not isinstance(h, bool) and 1 <= h <= 8):
        raise ValueError("analise_em_verificacao.h2_minimo_de_frases_com_indice: "
                         "inteiro de 1 a 8")
    if not (isinstance(saida["n_minimo_dois_fatores"], int)
            and saida["n_minimo_dois_fatores"] >= 2):
        raise ValueError("analise_em_verificacao.n_minimo_dois_fatores tem de ser inteiro >= 2")
    m = saida["min_gravacoes_validas_por_frase"]
    if m is not None and not (isinstance(m, int) and m >= 1):
        raise ValueError("analise_em_verificacao.min_gravacoes_validas_por_frase: "
                         "inteiro >= 1, ou null")
    return saida


def frases_avaliaveis(linhas, chaves, verif=None):
    """{pessoa: set(frases)} com pelo menos `n_min_por_frase` gravações válidas."""
    contagem = defaultdict(Counter)
    for l in linhas:
        if l["condicao"] == "personalizado" and _e_gravacao(l):
            contagem[l["pessoa"]][l["frase"]] += 1
    minimo = chaves["n_min_por_frase"]
    if verif and verif.get("min_gravacoes_validas_por_frase") is not None:
        minimo = verif["min_gravacoes_validas_por_frase"]
    saida = {p: {f for f, n in c.items() if n >= minimo}
             for p, c in contagem.items()}
    if chaves.get("regra_de_faltantes"):
        sem_treino = frases_sem_treino_minimo(linhas)
        for pessoa, frases in sem_treino.items():
            saida.setdefault(pessoa, set())
            saida[pessoa] = saida[pessoa] - frases
    return saida


def elegiveis(linhas, chaves, verif=None):
    """(primario, sensibilidade, motivos) -- quem entra em que, e por que não."""
    avaliaveis = frases_avaliaveis(linhas, chaves, verif)
    sem_treino_de = frases_sem_treino_minimo(linhas)
    sessoes = defaultdict(set)
    for l in linhas:
        if _e_gravacao(l):
            sessoes[l["pessoa"]].add(l["sessao"])

    primario, sensibilidade, motivos = [], [], {}
    for pessoa in sorted({l["pessoa"] for l in linhas}):
        if pessoa not in avaliaveis:
            motivos[pessoa] = ("sem linhas de gravacao do Personalizado no "
                               "acertos.csv -- fica fora, sem frase avaliavel")
            continue
        n_frases = len(avaliaveis[pessoa])
        duas = len(sessoes[pessoa]) >= 2
        if n_frases == 0:
            sem_treino = sem_treino_de.get(pessoa, set())
            if chaves.get("regra_de_faltantes") and sem_treino:
                motivos[pessoa] = (
                    "nenhuma frase avaliavel -- fica fora: {} frase(s) sem o "
                    "treino minimo ({} gravacoes de treino validas em cada uma "
                    "das duas sessoes de treino; sem uma sessao de treino "
                    "inteira, nenhuma frase o alcanca)".format(
                        len(sem_treino),
                        chaves.get("treino_minimo_por_sessao_de_treino")))
            else:
                motivos[pessoa] = "nenhuma frase avaliavel -- fica fora"
            continue
        if n_frases < chaves["min_frases"]:
            motivos[pessoa] = "so {} frase(s) avaliavel(is), minimo {}".format(
                n_frases, chaves["min_frases"])
            continue
        if chaves["exige_duas_sessoes"] and not duas:
            motivos[pessoa] = ("uma sessao de teste so -- fica fora da analise "
                               "principal; nenhuma sensibilidade usa este grupo")
            sensibilidade.append(pessoa)
            continue
        primario.append(pessoa)
    return primario, sensibilidade, motivos


GRUPO_CONFIRMATORIO = "afi"
GRUPOS_VALIDOS = ("afi", "tipico")


def grupos_por_pessoa(linhas):
    """{pessoa: grupo}, recusando rótulo ausente, fora do vocabulário ou trocado."""
    grupos = {}
    for i, l in enumerate(linhas, 2):
        g = (l.get("grupo") or "").strip()
        if g not in GRUPOS_VALIDOS:
            raise SystemExit(
                "GRUPO INVALIDO na linha {} do acertos.csv — participante "
                "{!r}, valor {!r}. Os valores aceitos sao {}. Sem o grupo "
                "nao se sabe quem entra na H1, na H2 e na regra de sucesso "
                "(so' o grupo AFI), e o script nao supoe.".format(
                    i, l.get("pessoa"), l.get("grupo"), GRUPOS_VALIDOS))
        anterior = grupos.setdefault(l["pessoa"], g)
        if anterior != g:
            raise SystemExit(
                "O participante {!r} aparece em DOIS grupos ({!r} e {!r}; "
                "linha {}). Um participante tem um grupo so'.".format(
                    l["pessoa"], anterior, g, i))
    return grupos


def so_o_grupo_confirmatorio(pessoas, grupos):
    """As pessoas de `pessoas` que pertencem ao grupo AFI, na mesma ordem."""
    return [p for p in pessoas if grupos[p] == GRUPO_CONFIRMATORIO]


def _ic_newcombe(a_cel, b_cel, c_cel, d_cel, nivel=0.95):
    """IC de escore pareado para a diferença de proporções em dados pareados."""
    from src.analise import _ic_escore_pareado
    return _ic_escore_pareado(a_cel, b_cel, c_cel, d_cel, nivel)


SEMENTE_DO_BOOTSTRAP = CHAVES["semente_bootstrap_duplo"]
B_BOOTSTRAP = CHAVES["replicas_bootstrap_duplo"]


def semente_do_participante(pessoa, semente_base=SEMENTE_DO_BOOTSTRAP):
    """A semente da reamostragem dupla de um participante."""
    h = hashlib.sha256(str(pessoa).encode("utf-8")).digest()
    return [int(semente_base), int.from_bytes(h[:8], "big")]


def _delta_por_frase(pares_por_frase):
    """A média das diferenças dentro de cada frase, uma por frase."""
    return np.array([float(np.mean(a) - np.mean(b))
                     for a, b in pares_por_frase])


TOLERANCIA_DE_LARGURA_ZERO = 1e-9


def _ic_delta_var_entre(pares_por_frase):
    """Dispersão das diferenças entre frases, t de Student."""
    d = _delta_por_frase(pares_por_frase)
    k = len(d)
    if k < 2:
        return None, None
    ep = float(np.std(d, ddof=1) / np.sqrt(k))
    t = float(stats.t.ppf(1 - (1 - NIVEL_DO_IC) / 2, k - 1))
    m = float(np.mean(d))
    if ep <= TOLERANCIA_DE_LARGURA_ZERO:
        return m, m
    return m - t * ep, m + t * ep


TOLERANCIA_DA_VARIANCIA_NA_FRASE = 1e-12


def _wald_gl_efetivos_detalhe(pares_por_frase, nivel=None):
    """O intervalo de Wald com graus de liberdade efetivos, com as partes da conta."""
    if nivel is None:
        nivel = NIVEL_DO_IC
    difs = [np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
            for a, b in pares_por_frase]
    difs = [d for d in difs if len(d) > 0]
    J = len(difs)
    if J == 0:
        return None
    n = np.array([len(d) for d in difs], dtype=float)
    m = np.array([float(d.mean()) for d in difs])
    centro = float(m.mean())
    estimavel = n >= 2
    s2 = np.zeros(J)
    for j, d in enumerate(difs):
        if estimavel[j]:
            bruto = (float(np.sum(d ** 2)) - n[j] * m[j] ** 2) / (n[j] - 1)
            s2[j] = 0.0 if bruto <= TOLERANCIA_DA_VARIANCIA_NA_FRASE else bruto
    piso = 1.0 / n
    com_variacao = bool(np.any(s2[estimavel] > 0)) if estimavel.any() else False
    z = float(stats.norm.ppf(1 - (1 - nivel) / 2))
    if com_variacao:
        v = np.where(estimavel, s2 / n, piso / n)
        soma = float(v.sum())
        ve = v[estimavel]
        denom = float(np.sum(ve ** 2 / (n[estimavel] - 1)))
        gl = soma ** 2 / denom
        quantil = float(stats.t.ppf(1 - (1 - nivel) / 2, max(gl, 1.0)))
        piso_aplicado = False
    else:
        soma = float(np.sum(piso / n))
        gl = None
        quantil = z
        piso_aplicado = True
    ep = float(np.sqrt(soma) / J)
    lo = float(np.clip(centro - quantil * ep, -1.0, 1.0))
    hi = float(np.clip(centro + quantil * ep, -1.0, 1.0))
    return {"ic": [lo, hi], "centro": centro, "ep": ep,
            "gl_efetivos": (float(gl) if gl is not None else None),
            "quantil": quantil, "piso_aplicado": piso_aplicado,
            "frases": J, "frases_com_um_par": int((~estimavel).sum()),
            "nivel": nivel}


MIN_FRASES_DO_INTERVALO_INDIVIDUAL = 2


def _ic_wald_gl_efetivos(pares_por_frase):
    """(inferior, superior) do Wald com graus de liberdade efetivos."""
    det = _wald_gl_efetivos_detalhe(pares_por_frase)
    if det is None:
        return None, None
    return det["ic"][0], det["ic"][1]


def _ic_bootstrap_duplo(pares_por_frase, semente, replicas=None):
    """Reamostra frases e, dentro de cada uma, as gravações."""
    k = len(pares_por_frase)
    if k < 2:
        return None, None
    B = int(replicas) if replicas else B_BOOTSTRAP
    rng = np.random.default_rng(semente)
    diffs = [np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
             for a, b in pares_por_frase]
    m = B * k
    internas = np.empty((k, m), dtype=float)
    for f, d in enumerate(diffs):
        n = len(d)
        if n == 0:
            internas[f, :] = np.nan
            continue
        idx = rng.integers(0, n, size=(m, n))
        internas[f, :] = d[idx].mean(axis=1)
    if not np.all(np.isfinite(internas)):
        return None, None
    escolhidas = rng.integers(0, k, size=(B, k))
    posicoes = np.arange(B * k).reshape(B, k)
    amostras = internas[escolhidas, posicoes].mean(axis=1)
    return float(np.percentile(amostras, 2.5)), float(np.percentile(amostras, 97.5))


def _pares_por_frase(chaves_comuns, a, b):
    """{frase: (acertos do alvo, acertos da referencia)} -> lista de pares."""
    por = defaultdict(lambda: ([], []))
    for k in chaves_comuns:
        frase = k[0]
        por[frase][0].append(a[k])
        por[frase][1].append(b[k])
    return [(np.array(va), np.array(vb)) for _, (va, vb) in sorted(por.items())]


def por_participante(linhas, pessoas, chaves, verif=None):
    """Newcombe, McNemar exato, dUAR -- e por sessão."""
    saida = {}
    avaliaveis = frases_avaliaveis(linhas, chaves, verif)
    tem_causa = causa_disponivel(linhas)
    ausentes = gravacoes_ausentes(linhas) if tem_causa else set()
    for pessoa in pessoas:
        dele = [l for l in linhas
                if l["pessoa"] == pessoa and l["frase"] in avaliaveis[pessoa]
                and _e_gravacao(l)
                and (l["pessoa"], l.get("gravacao")) not in ausentes]
        por_cond = defaultdict(list)
        for l in dele:
            por_cond[l["condicao"]].append(l)

        gravadas = {l["frase"] for l in linhas
                    if l["pessoa"] == pessoa and l["condicao"] == "personalizado"}
        res = {"n_frases": len(avaliaveis[pessoa]),
               "frases_fora_da_conta": sorted(gravadas - avaliaveis[pessoa]),
               "piso_de_gravacoes_por_frase": (
                   verif["min_gravacoes_validas_por_frase"]
                   if verif and verif.get("min_gravacoes_validas_por_frase") is not None
                   else chaves["n_min_por_frase"]),
               "causa_no_csv": tem_causa,
               "causas": (Counter(l["causa"] for l in dele) if tem_causa
                          else None),
               "frases_sem_treino_minimo": (
                   sorted(frases_sem_treino_minimo(linhas).get(pessoa, set()))
                   if tem_causa else None),
               "resultado_parcial": (
                   len(frases_sem_treino_minimo(linhas).get(pessoa, set()))
                   > MAXIMO_DE_FRASES_SEM_TREINO_SEM_MARCA
                   if tem_causa else None),
               "gravacoes_marcadas_ausentes": (
                   len([g for (pp, g) in ausentes if pp == pessoa])
                   if tem_causa else None)}
        uar_nos_pares = {}
        for alvo, ref in (("personalizado", "generico"),
                          ("personalizado", "entre_pessoas")):
            def _por_grupo(linhas_da_cond):
                fora, contador = {}, defaultdict(int)
                for l in linhas_da_cond:
                    g = (l["frase"], l["sessao"])
                    chave = ((g[0], g[1], l["gravacao"]) if tem_coluna
                             else (g[0], g[1], contador[g]))
                    fora[chave] = l["acerto"]
                    contador[g] += 1
                return fora, dict(contador)

            if por_cond[alvo] and not por_cond[ref]:
                res["{}_x_{}".format(alvo, ref)] = {"erro": "sem pares",
                                                    "condicao_ausente": ref}
                continue
            com = sum(1 for l in por_cond[alvo] + por_cond[ref]
                      if l.get("gravacao"))
            total_linhas = len(por_cond[alvo]) + len(por_cond[ref])
            if 0 < com < total_linhas:
                res["{}_x_{}".format(alvo, ref)] = {"erro": (
                    "o acertos.csv esta' MEIO MIGRADO no participante {}: {} de "
                    "{} linhas de {} e {} tem a coluna `gravacao` e o resto nao. "
                    "Parear assim misturaria dois esquemas no mesmo "
                    "participante. Ou todas as linhas tem o identificador, ou "
                    "nenhuma tem.".format(pessoa, com, total_linhas, alvo, ref))}
                continue
            tem_coluna = com > 0

            a, cont_a = _por_grupo(por_cond[alvo])
            b, cont_b = _por_grupo(por_cond[ref])
            desiguais = [] if tem_coluna else sorted(
                g for g in set(cont_a) | set(cont_b)
                if cont_a.get(g, 0) != cont_b.get(g, 0))
            if desiguais:
                f0, s0 = desiguais[0]
                res["{}_x_{}".format(alvo, ref)] = {"erro": (
                    "contagem de gravacoes DIFERENTE entre {} e {} em {} "
                    "grupo(s) (frase, sessao) do participante {}. O primeiro e' "
                    "frase {!r}, sessao {}: {} contra {}. O acertos.csv nao "
                    "tem identificador de gravacao, entao nao da' para saber "
                    "QUAL faltou -- parear assim mesmo fabricaria discordantes. "
                    "Acrescente a coluna `gravacao` ao CSV, ou decida com "
                    "o pesquisador responsavel como tratar a gravacao faltante.".format(
                        alvo, ref, len(desiguais), pessoa, f0, s0,
                        cont_a.get((f0, s0), 0), cont_b.get((f0, s0), 0)))}
                continue
            motivos = {}
            contadas_como_erro = {}
            if tem_causa:
                linhas_pareadas = [l for l in por_cond[alvo] + por_cond[ref]
                                   if l["causa"] in ("sem_saida", "falha_tecnica")]
                contadas_como_erro = dict(Counter(
                    l["causa"] for l in linhas_pareadas))
            if chaves.get("regra_de_faltantes"):
                so_no_alvo = set(a) - set(b)
                so_na_ref = set(b) - set(a)
                for k in sorted(so_no_alvo | so_na_ref):
                    ausente_em = ref if k in so_no_alvo else alvo
                    motivos[k] = "gravacao_ausente"
                    if REGRA_DE_FALTANTES["gravacao_ausente"] == "exclui o par":
                        continue
                    if ausente_em == ref:
                        b[k] = 0
                    else:
                        a[k] = 0

            chaves_comuns = sorted(set(a) & set(b))
            if not chaves_comuns:
                res["{}_x_{}".format(alvo, ref)] = {"erro": "sem pares"}
                continue
            va = np.array([a[k] for k in chaves_comuns])
            vb = np.array([b[k] for k in chaves_comuns])
            n = len(va)
            uar_nos_pares[(alvo, ref)] = (
                _uar([{"frase": k[0], "acerto": a[k]} for k in chaves_comuns]),
                _uar([{"frase": k[0], "acerto": b[k]} for k in chaves_comuns]))
            regra_usada = ("regra de faltantes APLICADA"
                           if chaves.get("regra_de_faltantes")
                           else "sem regra de faltantes: par que nao se forma "
                                "nao entra, e o n cai (padrao)")
            inteiros = (((va == 0) | (va == 1)) & ((vb == 0) | (vb == 1)))
            n_com_fracao = int(np.sum(~inteiros))
            va0, vb0 = va[inteiros], vb[inteiros]
            cel_a = int(np.sum((va0 == 1) & (vb0 == 1)))
            disc_b = int(np.sum((va0 == 1) & (vb0 == 0)))
            disc_c = int(np.sum((va0 == 0) & (vb0 == 1)))
            cel_d = int(np.sum((va0 == 0) & (vb0 == 0)))
            dif = float(va.mean() - vb.mean())
            motivo_sem_escore = None
            if len(va0) == 0:
                lo = hi = None
                p_mc = None
                motivo_sem_escore = (
                    "nao calculavel: os {} pares tem acerto fracionario (empate "
                    "na maior nota) em pelo menos uma das condicoes, e o escore "
                    "somado e o McNemar so' existem na tabela 2x2 de acertos 0 "
                    "ou 1".format(n))
            else:
                _dif_ic, lo, hi = _ic_newcombe(
                    cel_a, disc_b, disc_c, cel_d, NIVEL_DO_IC)
                p_mc = float(stats.binomtest(disc_b, disc_b + disc_c, 0.5).pvalue
                             if disc_b + disc_c else 1.0)
            pares_f = _pares_por_frase(chaves_comuns, a, b)
            det_w = _wald_gl_efetivos_detalhe(pares_f)
            lo_e, hi_e = _ic_delta_var_entre(pares_f)
            semente_dele = semente_do_participante(
                pessoa, chaves.get("semente_bootstrap_duplo",
                                   SEMENTE_DO_BOOTSTRAP))
            b_dele = chaves.get("replicas_bootstrap_duplo", B_BOOTSTRAP)
            lo_b, hi_b = (_ic_bootstrap_duplo(pares_f, semente_dele)
                          if b_dele == B_BOOTSTRAP
                          else _ic_bootstrap_duplo(pares_f, semente_dele, b_dele))
            ic_por_metodo = {
                "escore_somado": ([lo, hi] if lo is not None else None),
                "wald_gl_efetivos": (
                    list(det_w["ic"])
                    if det_w is not None
                    and det_w["frases"] >= MIN_FRASES_DO_INTERVALO_INDIVIDUAL
                    else None),
                "delta_var_entre": ([lo_e, hi_e] if lo_e is not None else None),
                "bootstrap_duplo": ([lo_b, hi_b] if lo_b is not None else None),
            }
            res["{}_x_{}".format(alvo, ref)] = {
                "n": n, "dif": dif, "ic": [lo, hi],
                "metodo_do_ic": "escore_somado",
                "ic_por_metodo": ic_por_metodo,
                "wald_gl_efetivos_detalhe": det_w,
                "pares_no_escore_e_no_mcnemar": int(len(va0)),
                "pares_com_fracao_fora_do_escore_e_do_mcnemar": n_com_fracao,
                "motivo_sem_escore_e_mcnemar": motivo_sem_escore,
                "frases_no_ic_por_frase": len(pares_f),
                "delta_por_frase": dict(zip(
                    sorted({k[0] for k in chaves_comuns}),
                    [float(x) for x in _delta_por_frase(pares_f)])),
                "gravacoes_por_frase": dict(zip(
                    sorted({k[0] for k in chaves_comuns}),
                    [int(len(pa)) for pa, _pb in pares_f])),
                "B_do_bootstrap": b_dele,
                "semente_do_bootstrap": semente_dele,
                "discordantes": [disc_b, disc_c], "mcnemar_p": p_mc,
                "regra_de_faltantes": regra_usada,
                "pares_tratados_pela_regra": len(motivos),
                "causa_no_csv": tem_causa,
                "contadas_como_erro_por_causa": (contadas_como_erro
                                                 if tem_causa else None),
                "clausulas_aplicaveis": (
                    sorted(REGRA_DE_FALTANTES) if tem_causa
                    else ["gravacao_ausente"])}

        _nan = (float("nan"), float("nan"))
        pg = uar_nos_pares.get(("personalizado", "generico"), _nan)
        pe = uar_nos_pares.get(("personalizado", "entre_pessoas"), _nan)
        uar_p = pg[0]
        origem_p = "pares de personalizado_x_generico"
        if not np.isfinite(uar_p) and np.isfinite(pe[0]):
            uar_p = pe[0]
            origem_p = ("pares de personalizado_x_entre_pessoas: o Generico "
                        "nao formou par com o Personalizado")
        res["uar"] = {"personalizado": uar_p, "entre_pessoas": pe[1],
                      "generico": pg[1]}
        res["uar_base"] = ("pares do contraste: personalizado e generico nos "
                           "pares de personalizado_x_generico; entre_pessoas "
                           "nos de personalizado_x_entre_pessoas")
        res["origem_da_uar_personalizado"] = origem_p
        res["duar_pg"] = pg[0] - pg[1]
        res["duar_pe"] = pe[0] - pe[1]
        por_sessao = {}
        for sessao in sorted({l["sessao"] for l in dele}):
            desta = [l for l in dele if l["sessao"] == sessao]
            pc = defaultdict(list)
            for l in desta:
                pc[l["condicao"]].append(l)
            por_sessao[str(sessao)] = (_uar(pc["personalizado"]) - _uar(pc["generico"])
                                       if pc["generico"] else None)
        res["duar_pg_por_sessao"] = por_sessao
        valores = [v for v in por_sessao.values() if v is not None]
        res["sessoes_divergem_em_sinal"] = (
            len(valores) == 2 and valores[0] * valores[1] < 0)
        saida[pessoa] = res
    return saida


def _uar(linhas):
    """Acurácia média por intenção -- não acurácia bruta."""
    if not linhas:
        return float("nan")
    por_frase = defaultdict(list)
    for l in linhas:
        por_frase[l["frase"]].append(l["acerto"])
    return float(np.mean([np.mean(v) for v in por_frase.values()]))


CONTRASTES_DA_H1 = {
    "H1a": {"bloco": "personalizado_x_generico",
            "nome": "H1a (Personalizado x Generico)",
            "referencia": "o Generico"},
    "H1b": {"bloco": "personalizado_x_entre_pessoas",
            "nome": "Entre-pessoas completo (descritivo), diferencas por participante",
            "referencia": "o Entre-pessoas completo"},
}
N_MINIMO_DA_H1 = 3
N_PLENO_DA_H1 = 8
ROTULO_PRELIMINAR = "preliminar, nesta amostra"
ROTULO_PLENO = "conclusao plena"

ALFA_DA_H1 = 0.04


def _motivo_sem_intervalo(bloco, metodo):
    """Por que esta pessoa não tem intervalo individual."""
    ausente = bloco.get("condicao_ausente")
    if ausente:
        return ("nao calculavel: a condicao {} nao esta' no acertos.csv deste "
                "participante, e sem ela o contraste nao forma par".format(
                    ausente))
    k = bloco.get("frases_no_ic_por_frase")
    if metodo in ("delta_var_entre", "bootstrap_duplo") and isinstance(k, int) and k < 2:
        return ("nao calculavel: {} frase com dado neste contraste; o intervalo "
                "entre frases ({}) exige pelo menos 2".format(k, metodo))
    if (isinstance(k, int)
            and k < MIN_FRASES_DO_INTERVALO_INDIVIDUAL):
        return ("nao calculavel: {} frase com dado neste contraste; o intervalo "
                "individual ({}) sai com pelo menos {} frases".format(
                    k, metodo, MIN_FRASES_DO_INTERVALO_INDIVIDUAL))
    return "nao calculavel: o bloco nao traz o intervalo {}".format(metodo)


def veredito_da_h1(por_part, chaves, contraste="personalizado_x_generico"):
    """Quantos participantes têm o intervalo inteiro acima de zero, e o veredito."""
    metodo = chaves.get("metodo_do_intervalo")
    if metodo is None:
        return {
            "veredito": None,
            "n_participantes": None,
            "motivo": (
                "A chave `metodo_do_intervalo` veio nula, e sem ela o "
                "veredito da H1 nao sai. O registrado e' "
                "'wald_gl_efetivos', com 'bootstrap_duplo' como sensibilidade "
                "pre-declarada; o estimando trata as frases como fixas. O "
                "metodo de escore somado NAO serve: ele "
                "estima a diferenca ponderada por gravacao, e no cenario de "
                "perdas desiguais declara a H1 sustentada em 31,4% das "
                "replicas com efeito verdadeiro zero (SINTETICO: 4000 "
                "replicas, 12 participantes)."),
            "candidatos": ["wald_gl_efetivos", "bootstrap_duplo"],
            "sensibilidade_rotulada": ["escore_somado"],
            "onde_decidir": "CHAVES['metodo_do_intervalo'], em analise/analise_do_estudo.py",
        }
    if metodo not in METODOS_DO_INTERVALO:
        raise SystemExit(
            "metodo_do_intervalo invalido: {!r}. A contagem aceita so' "
            "'wald_gl_efetivos', 'bootstrap_duplo' ou 'delta_var_entre'. "
            "'escore_somado' e a "
            "regra hibrida ficam como SENSIBILIDADE rotulada, nunca como "
            "decisao: os dois estimam a diferenca ponderada por gravacao, e o "
            "estimando declarado da' peso igual a cada frase.".format(metodo))

    if contraste not in ("personalizado_x_generico",
                         "personalizado_x_entre_pessoas"):
        raise SystemExit("contraste da H1 invalido: {!r}".format(contraste))

    def _finita(f):
        return bool(f) and len(f) == 2 and all(
            x is not None and np.isfinite(x) for x in f)

    acima, abaixo, degenerados, sem_intervalo = [], [], [], []
    substituidos = []
    motivos_sem_intervalo = {}
    for pessoa, r in sorted(por_part.items()):
        bloco = r.get(contraste) or {}
        por_metodo = bloco.get("ic_por_metodo") or {}
        faixa = (bloco.get("ic_por_metodo") or {}).get(metodo)
        if not _finita(faixa):
            sem_intervalo.append(pessoa)
            motivos_sem_intervalo[pessoa] = _motivo_sem_intervalo(bloco, metodo)
            continue
        lo, hi = faixa
        if hi - lo <= TOLERANCIA_DE_LARGURA_ZERO:
            degenerados.append(pessoa)
            if metodo != "bootstrap_duplo":
                reserva = por_metodo.get("bootstrap_duplo")
                if not _finita(reserva):
                    sem_intervalo.append(pessoa)
                    motivos_sem_intervalo[pessoa] = (
                        "nao calculavel: intervalo de largura zero e sem a "
                        "reamostragem dupla no bloco para substitui-lo")
                    continue
                lo, hi = reserva
                substituidos.append(pessoa)
        acima.append((pessoa, lo > 0.0))
        abaixo.append((pessoa, hi < 0.0))
    n = len(acima)
    k = sum(1 for _, v in acima if v)
    k_pioras = sum(1 for _, v in abaixo if v)
    base = {"n_participantes": n, "acima_de_zero": k,
            "abaixo_de_zero": k_pioras,
            "contraste": contraste,
            "n_analisados": len(por_part),
            "n_degenerados": len(degenerados),
            "n_sem_intervalo": len(sem_intervalo),
            "participantes_sem_intervalo": sem_intervalo,
            "motivos_sem_intervalo": motivos_sem_intervalo,
            "metodo_do_intervalo": metodo,
            "participantes": [p for p, v in acima if v],
            "participantes_com_piora": [p for p, v in abaixo if v],
            "intervalos_degenerados": degenerados,
            "substituidos_pela_reamostragem_dupla": substituidos,
            "medido_em": (
                "results/medicoes/custo_da_trava.json (regra k=3 com a trava, "
                "intervalo delta_var_entre, anterior ao wald_gl_efetivos; "
                "SINTETICO: nao cobre a troca da largura zero pela "
                "reamostragem dupla). Cobertura do wald_gl_efetivos: "
                "results/medicoes/intervalo_individual_frases_fixas.json, "
                "results/medicoes/s2_intervalo_fora_da_amostra.json e "
                "results/medicoes/wald_gl_efetivos_com_empate.json")}
    base["proporcao_de_ajudados"] = (k / n) if n else None
    base["papel_da_proporcao"] = (
        "DESCRITIVA: relata quantos ganharam, nao decide a H1")
    regra = chaves.get("regra_de_sucesso")
    if regra is None:
        base["veredito"] = None
        base["motivo"] = (
            "regra_de_sucesso veio nula. O padrao do script e' 'k_fixo_3'"
            "; o veredito fica em branco em vez de ser "
            "arbitrado aqui.")
        return base
    if regra not in REGRAS_DE_SUCESSO:
        raise SystemExit(
            "regra_de_sucesso desconhecida: {!r}. Use uma de {}.".format(
                regra, sorted(REGRAS_DE_SUCESSO)))
    limiar = (n + 1) // 2 if regra == "metade_ou_mais" else 3
    if n < N_MINIMO_DA_H1:
        base.update({"regra": regra, "limiar": limiar,
                     "pergunta": REGRAS_DE_SUCESSO[regra]["pergunta"],
                     "criterio": REGRAS_DE_SUCESSO[regra]["criterio"],
                     "prova": REGRAS_DE_SUCESSO[regra].get("prova"),
                     "passa_contagem": None,
                     "passa_trava_ganhos_maior_que_pioras": None,
                     "veredito": None,
                     "resultado": "sem veredito",
                     "motivo": (
                         "sem veredito: N = {} participante(s) com intervalo, "
                         "menos que {}; a regra k=3 nao pode ser cumprida e "
                         "nao houve teste possivel (N < 3)".format(
                             n, N_MINIMO_DA_H1))})
        return base
    passa_contagem = k >= limiar
    passa_trava = k > k_pioras
    base.update({"regra": regra, "limiar": limiar,
                 "pergunta": REGRAS_DE_SUCESSO[regra]["pergunta"],
                 "criterio": REGRAS_DE_SUCESSO[regra]["criterio"],
                 "prova": REGRAS_DE_SUCESSO[regra].get("prova"),
                 "passa_contagem": passa_contagem,
                 "passa_trava_ganhos_maior_que_pioras": passa_trava,
                 "veredito": bool(passa_contagem and passa_trava)})
    base["resultado"] = ("sustentada" if base["veredito"]
                         else "nao sustentada")
    return base


def veredito_da_h1_com_sensibilidade(por_part, chaves,
                                     contraste="personalizado_x_generico"):
    """O veredito da H1 pelo método que decide, com a sensibilidade anexada."""
    decisao = veredito_da_h1(por_part, chaves, contraste)
    metodo = chaves.get("metodo_do_intervalo")
    sens = chaves.get("metodo_do_intervalo_sensibilidade")
    if sens is None:
        decisao["sensibilidade_do_intervalo"] = {
            "executada": False,
            "motivo": ("`metodo_do_intervalo_sensibilidade` nulo: nenhuma "
                       "sensibilidade do intervalo foi pre-declarada")}
        return decisao
    if sens not in METODOS_DO_INTERVALO:
        raise SystemExit(
            "metodo_do_intervalo_sensibilidade invalido: {!r}. Aceita so' "
            "{}.".format(sens, " ou ".join(repr(m) for m in METODOS_DO_INTERVALO)))
    if sens == metodo:
        decisao["sensibilidade_do_intervalo"] = {
            "executada": False,
            "motivo": ("metodo_do_intervalo_sensibilidade = metodo_do_intervalo "
                       "({!r}): a sensibilidade que repete o metodo que decide "
                       "nao confere nada".format(sens))}
        return decisao
    if metodo is None:
        decisao["sensibilidade_do_intervalo"] = {
            "executada": False,
            "motivo": ("o metodo que decide nao foi escolhido; sem decisao, "
                       "nao ha' o que a sensibilidade confira")}
        return decisao
    outro = veredito_da_h1(por_part, dict(chaves, metodo_do_intervalo=sens),
                           contraste)
    campos = ("veredito", "n_participantes", "acima_de_zero", "abaixo_de_zero",
              "participantes", "participantes_com_piora",
              "intervalos_degenerados")
    rotulos = {
        "veredito": "o resultado da regra",
        "n_participantes": "o N que entra na regra",
        "acima_de_zero": "quantos tem ganho demonstrado",
        "abaixo_de_zero": "quantos tem piora demonstrada",
        "participantes": "quem tem ganho demonstrado",
        "participantes_com_piora": "quem tem piora demonstrada",
        "intervalos_degenerados": "quem tem intervalo de largura zero",
    }
    discordancia = [
        "{} difere: {} {!r} contra {} {!r}".format(
            rotulos[c], metodo, decisao.get(c), sens, outro.get(c))
        for c in campos if decisao.get(c) != outro.get(c)]
    decisao["sensibilidade_do_intervalo"] = {
        "executada": True,
        "papel": ("SENSIBILIDADE PRE-DECLARADA: NAO decide a H1; a "
                  "discordancia com o metodo que decide e' relatada"),
        "metodo": sens,
        "resultado": {("resultado_da_regra" if c == "veredito" else c):
                      outro.get(c) for c in campos},
        "concorda": not discordancia,
        "discordancia": discordancia,
    }
    return decisao


def leitura_combinada_da_h1(v_a, v_b):
    """A leitura da H1 inteira a partir dos vereditos da H1a e da H1b."""
    a, b = v_a.get("veredito"), v_b.get("veredito")
    if a is None and b is None:
        return {"leitura": "sem veredito", "passaram": [],
                "motivo": "nenhuma das duas teve veredito (H1a: {}; H1b: {})".format(
                    v_a.get("motivo"), v_b.get("motivo"))}
    if a is None or b is None:
        qual = "H1a" if a is None else "H1b"
        return {"leitura": None, "passaram": [],
                "motivo": ("{} sem veredito e a outra com veredito: combinacao "
                           "nao prevista na regra; a leitura combinada nao e' "
                           "arbitrada aqui (depende da escolha do pesquisador "
                           "responsavel)".format(qual))}
    if a and b:
        return {"leitura": "sustentada", "passaram": ["H1a", "H1b"],
                "texto": ("o Personalizado venceu o Generico (H1a) e o "
                          "Entre-pessoas (H1b)")}
    if a or b:
        qual = "H1a" if a else "H1b"
        return {"leitura": "sustentada em parte", "passaram": [qual],
                "texto": ("so' a {} passou: o Personalizado venceu {} e nao "
                          "venceu {}".format(
                              qual,
                              CONTRASTES_DA_H1[qual]["referencia"],
                              CONTRASTES_DA_H1["H1b" if a else "H1a"]
                              ["referencia"]))}
    return {"leitura": "nao sustentada", "passaram": [],
            "texto": "nem a H1a nem a H1b passaram"}


MOTIVO_DA_H1B_DESCRITIVA = (
    "DESCRITIVA (plano de analise): os ganhos contra o Entre-pessoas "
    "completo nao sao independentes entre participantes, porque o Entre-pessoas "
    "de cada um e' treinado com os dados dos outros; este bloco traz so' as "
    "diferencas observadas, sem valor p nem veredito (o completo exploratorio, com "
    "intervalos e p exploratorio, esta' em entre_pessoas_completo_exploratorio, e a "
    "H1b, contra o Entre-pessoas igualado, em h1b)")

ALFA_DA_H1B = ALFA_DA_H1
N_MINIMO_DA_H1B = N_MINIMO_DA_H1
FATOR_DA_CORRECAO_H1B = "1/N + 1/(N-1)"


def descricao_da_h1b(por_part):
    """Resumo descritivo Personalizado x Entre-pessoas, sempre publicado."""
    bloco = CONTRASTES_DA_H1["H1b"]["bloco"]
    ganhos, fora = {}, {}
    for pessoa, r in sorted(por_part.items()):
        g = r.get("duar_pe")
        if g is not None and np.isfinite(g):
            ganhos[pessoa] = float(g)
        else:
            erro = (r.get(bloco) or {}).get("erro")
            fora[pessoa] = ("sem ganho observado contra o Entre-pessoas" +
                            (": {}".format(erro) if erro else ""))
    valores = list(ganhos.values())
    return {
        "hipotese": CONTRASTES_DA_H1["H1b"]["nome"],
        "contraste": bloco,
        "papel": "descritivo",
        "veredito": None,
        "resultado": "descritivo",
        "motivo": MOTIVO_DA_H1B_DESCRITIVA,
        "n_participantes": len(valores),
        "n_analisados": len(por_part),
        "ganhos_por_participante": ganhos,
        "ganho_medio": float(np.mean(valores)) if valores else None,
        "ganho_mediano": float(np.median(valores)) if valores else None,
        "menor_ganho": min(valores) if valores else None,
        "maior_ganho": max(valores) if valores else None,
        "melhoras": sum(1 for g in valores if g > 0),
        "pioras": sum(1 for g in valores if g < 0),
        "empates": sum(1 for g in valores if g == 0),
        "fora_da_descricao": fora,
    }


def teste_nadeau_bengio_leave_one_out(ganhos, alfa=ALFA_DA_H1B):
    """Teste t corrigido para ganhos dependentes de leave-one-participant-out."""
    x = np.asarray(list(ganhos), dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    base = {
        "metodo": "t corrigido de Nadeau-Bengio para leave-one-participant-out",
        "alfa_unilateral": float(alfa),
        "n": n,
        "gl": n - 1 if n else None,
        "fator_da_variancia": FATOR_DA_CORRECAO_H1B,
    }
    if n < 2:
        return dict(base, status="nao_calculavel", ganho_medio=(float(np.mean(x)) if n else None),
                    motivo="menos de 2 ganhos finitos")
    media = float(np.mean(x))
    desvio = float(np.std(x, ddof=1))
    fator = 1.0 / n + 1.0 / (n - 1)
    ep = float(desvio * np.sqrt(fator))
    base.update(ganho_medio=media, desvio_dos_ganhos=desvio,
                fator_numerico_da_variancia=fator, erro_padrao_corrigido=ep)
    if not np.isfinite(ep) or desvio <= TOLERANCIA_DEGENERADO:
        return dict(base, status="degenerado", p=None, t=None, intervalo_do_teste=None,
                    motivo=("os ganhos não têm dispersão; o procedimento não transforma "
                            "variância zero em evidência infinita"))
    gl = n - 1
    t = float(media / ep)
    p = float(stats.t.sf(t, gl))
    q = float(stats.t.ppf(1 - alfa, gl))
    return dict(base, status="calculado", t=t, p=p,
                intervalo_do_teste=_intervalo_do_teste(media, ep, q, alfa),
                intervalo_95=_intervalo_de_95(media, ep, gl), motivo="")


ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO = "entre_pessoas_igualado.csv"
COLUNAS_DO_ENTRE_PESSOAS_IGUALADO = ("pessoa", "versao", "sem", "sorteio", "frase",
                                     "sessao", "gravacao", "acerto", "calculavel")
VERSOES_DO_ENTRE_PESSOAS = ("igualado", "completo", "personalizado_reduzido")
VERSOES_COM_SORTEIOS = ("igualado", "personalizado_reduzido")
ARQUIVO_DOS_SORTEIOS_DO_IGUALADO = "entre_pessoas_igualado_sorteios.csv"
COLUNAS_DOS_SORTEIOS_DO_IGUALADO = ("pessoa", "sem", "braco", "sorteio", "doadora",
                                    "arquivo_doado", "frase_doada", "sessao_doada",
                                    "semente")
BRACOS_DOS_SORTEIOS = ("igualado", "personalizado_reduzido")
REGRAS_DA_H1B = ("nadeau_bengio_e_jackknife_por_participante",)
COMPARADORES_DA_H1B = ("entre_pessoas_igualado",)
ORDENS_DA_H1B = ("sempre_calculada_confirmatoria_so_com_h1a",)
FALTANTES_DO_IGUALADO = ("reduz_os_dois_sem_reposicao",)
CONFIGURACOES_DO_IGUALADO = ("selecao_com_a_quantidade_do_personalizado_sem_o_alvo",)
PAPEIS_DO_COMPLETO = ("exploratorio_sem_veredito",)
SEMENTE_DO_PERSONALIZADO_REDUZIDO = "personalizado_reduzido"


PROCEDIMENTO_DOS_DOIS_METODOS = (
    "concordancia do t corrigido de Nadeau e Bengio (2003) com o jackknife por "
    "participante (o maior dos dois erros-padrao): procedimento do projeto, validado "
    "por simulacao SINTETICA (relatorio de 04/10/2026 e medicao de 05/10/2026); nao "
    "e' metodo estabelecido na literatura")


def conferir_entre_pessoas_igualado(linhas_ep, K):
    """As linhas do `entre_pessoas_igualado.csv`, conferidas e lidas."""
    if not linhas_ep:
        raise ValueError("{} está vazio".format(ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO))
    faltam = [c for c in COLUNAS_DO_ENTRE_PESSOAS_IGUALADO if c not in linhas_ep[0]]
    if faltam:
        raise ValueError("{} sem as colunas {}".format(
            ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO, faltam))
    vistas, saida = set(), []
    for i, l in enumerate(linhas_ep, 2):
        onde = "linha {} de {}".format(i, ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO)
        versao = str(l["versao"]).strip()
        if versao not in VERSOES_DO_ENTRE_PESSOAS:
            raise ValueError("{}: versão {!r}; vale {}".format(
                onde, l["versao"], ", ".join(VERSOES_DO_ENTRE_PESSOAS)))
        sem = str(l.get("sem") or "").strip()
        try:
            sorteio = int(str(l["sorteio"]).strip())
            sessao = int(str(l["sessao"]).strip())
        except ValueError:
            raise ValueError("{}: sorteio {!r} e sessão {!r} têm de ser inteiros".format(
                onde, l["sorteio"], l["sessao"]))
        try:
            acerto = ler_acerto(l["acerto"])
        except ValueError as e:
            raise ValueError("{}: {}".format(onde, e))
        calculavel = str(l["calculavel"]).strip()
        if calculavel not in ("0", "1"):
            raise ValueError("{}: calculavel {!r}; vale \"0\" ou \"1\"".format(
                onde, l["calculavel"]))
        if versao in VERSOES_COM_SORTEIOS and not 1 <= sorteio <= K:
            raise ValueError("{}: sorteio {} do {} fora de 1 a {} (chave "
                             "h1b_igualado_sorteios)".format(onde, sorteio, versao, K))
        if versao == "completo" and (not sem or sorteio != 0):
            raise ValueError("{}: o completo só aparece nas rodadas do jackknife (sem "
                             "preenchido) e com sorteio 0; o da rodada principal é a "
                             "condição entre_pessoas do acertos.csv".format(onde))
        chave = (l["pessoa"], versao, sem, sorteio, l["gravacao"])
        if chave in vistas:
            raise ValueError("{}: gravação {} de {} repetida (versão {}, sem {!r}, "
                             "sorteio {})".format(onde, l["gravacao"], l["pessoa"],
                                                  versao, sem, sorteio))
        vistas.add(chave)
        saida.append(dict(l, versao=versao, sem=sem, sorteio=sorteio, sessao=sessao,
                          acerto=acerto, calculavel=calculavel == "1"))
    return saida


def acerto_medio_por_gravacao(linhas_ep, versao, sem, K):
    """{(pessoa, gravação): (frase, acerto médio)} de uma versão numa rodada."""
    por_gravacao = defaultdict(dict)
    frase_de = {}
    for l in linhas_ep:
        if l["versao"] != versao or l["sem"] != sem:
            continue
        chave = (l["pessoa"], l["gravacao"])
        if frase_de.setdefault(chave, l["frase"]) != l["frase"]:
            raise ValueError("a gravação {} de {} aparece com as frases {!r} e {!r} no "
                             "{}".format(l["gravacao"], l["pessoa"], frase_de[chave],
                                         l["frase"], ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO))
        por_gravacao[chave][l["sorteio"]] = l["acerto"]
    esperados = set(range(1, K + 1)) if versao in VERSOES_COM_SORTEIOS else {0}
    saida = {}
    for chave, por_sorteio in por_gravacao.items():
        if set(por_sorteio) != esperados:
            raise ValueError(
                "a gravação {} de {} tem os sorteios {} no {} ({}, sem {!r}); a regra "
                "exige {}".format(chave[1], chave[0], sorted(por_sorteio),
                                  ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO, versao, sem,
                                  "1 a {}".format(K) if versao in VERSOES_COM_SORTEIOS
                                  else "0"))
        saida[chave] = (frase_de[chave], float(np.mean(list(por_sorteio.values()))))
    return saida


def acerto_de_um_sorteio(linhas_ep, versao, sem, sorteio):
    """{(pessoa, gravação): (frase, acerto)} de um sorteio de uma versão numa rodada."""
    return {(l["pessoa"], l["gravacao"]): (l["frase"], l["acerto"]) for l in linhas_ep
            if l["versao"] == versao and l["sem"] == sem and l["sorteio"] == sorteio}


def frases_fora_da_h1b(linhas_ep):
    """{pessoa: [frases]} que saem da H1b daquele participante."""
    fora = defaultdict(set)
    for l in linhas_ep:
        if l["versao"] == "igualado" and not l["calculavel"]:
            fora[l["pessoa"]].add(l["frase"])
    return {p: sorted(f) for p, f in sorted(fora.items())}


def acerto_do_completo_no_acertos_csv(linhas, pessoas):
    """{(pessoa, gravação): (frase, acerto)} do Entre-pessoas completo no acertos.csv."""
    saida = {}
    for l in linhas:
        if (l["pessoa"] in pessoas and l["condicao"] == "entre_pessoas"
                and _e_gravacao(l)):
            saida[(l["pessoa"], l.get("gravacao"))] = (l["frase"], float(l["acerto"]))
    return saida


def tabela_contra_o_entre_pessoas(linhas_personalizado, acertos_da_referencia, pessoas,
                                  rotulo, acerto_do_personalizado=None, fora=None):
    pessoas = set(pessoas)
    fora = fora or {}
    reduzido = acerto_do_personalizado or {}
    com_reduzido = {p for (p, _g) in reduzido}
    por_frase = defaultdict(lambda: ([], []))
    faltando = []
    for l in linhas_personalizado:
        if l["pessoa"] not in pessoas or l["frase"] in fora.get(l["pessoa"], ()):
            continue
        par = acertos_da_referencia.get((l["pessoa"], l["gravacao"]))
        if par is None:
            faltando.append((l["pessoa"], l["gravacao"]))
            continue
        if par[0] != l["frase"]:
            raise ValueError("a gravação {} de {} é {!r} no Personalizado e {!r} no {}".format(
                l["gravacao"], l["pessoa"], l["frase"], par[0], rotulo))
        acerto_p = ler_acerto(l["acerto"])
        if l["pessoa"] in com_reduzido:
            r = reduzido.get((l["pessoa"], l["gravacao"]))
            if r is None or r[0] != l["frase"]:
                raise ValueError("a gravação {} de {} não tem o Personalizado reduzido no "
                                 "{} (ou tem outra frase)".format(l["gravacao"], l["pessoa"],
                                                                 rotulo))
            acerto_p = float(r[1])
        a, b = por_frase[(l["pessoa"], l["frase"])]
        a.append(acerto_p)
        b.append(par[1])
    if faltando:
        raise ValueError("{} gravação(ões) do Personalizado do degrau 1 sem linha no {} "
                         "(a primeira: {} {}); a hipótese 1 usa as mesmas gravações em "
                         "todos os passos".format(len(faltando), rotulo, *faltando[0]))
    com_dado = sorted(p for p in pessoas if any(k[0] == p for k in por_frase))
    frases = sorted({f for (_p, f) in por_frase})
    D = np.full((len(com_dado), len(frases)), np.nan)
    for i, p in enumerate(com_dado):
        for j, f in enumerate(frases):
            if (p, f) in por_frase:
                a, b = por_frase[(p, f)]
                D[i, j] = float(np.mean(a) - np.mean(b))
    return com_dado, frases, D


def ganhos_da_tabela(pessoas, D):
    """{pessoa: ganho}: a média das frases que cada um tem, com peso igual."""
    saida = {}
    for i, p in enumerate(pessoas):
        presente = np.isfinite(D[i])
        if presente.any():
            saida[p] = float(D[i, presente].mean())
    return saida


def erro_padrao_jackknife(medias_sem_k):
    """EP do jackknife por participante a partir de {k: ganho médio dos outros sem k}."""
    valores = np.array([float(v) for _k, v in sorted(medias_sem_k.items())])
    n = int(valores.size)
    if n < 2 or not np.all(np.isfinite(valores)):
        raise ValueError("o jackknife precisa de pelo menos 2 rodadas com ganho finito")
    centro = float(valores.mean())
    variancia = float((n - 1) / n * np.sum((valores - centro) ** 2))
    return float(np.sqrt(variancia)), {
        "formula": "EP^2 = (N-1)/N * soma_k (g_(-k) - media dos g_(-k))^2",
        "ganho_medio_sem_cada_participante": {k: float(v) for k, v in
                                              sorted(medias_sem_k.items())},
        "media_das_rodadas": centro, "n": n}


def teste_da_h1b_dois_metodos(ganhos, medias_sem_k, alfa=ALFA_DA_H1B):
    """O teste da H1b: passa só se o t de Nadeau e Bengio E o jackknife rejeitarem."""
    if set(ganhos) != set(medias_sem_k):
        raise ValueError("as pessoas dos ganhos e as do jackknife diferem: {} contra "
                         "{}".format(sorted(ganhos), sorted(medias_sem_k)))
    x = np.array([ganhos[p] for p in sorted(ganhos)], dtype=float)
    n = int(x.size)
    nb = teste_nadeau_bengio_leave_one_out(x, alfa=alfa)
    base = {"regra": REGRAS_DA_H1B[0],
            "descricao_da_regra": (
                "t = ganho medio / max(EP de Nadeau e Bengio, EP do jackknife por "
                "participante), gl = N - 1, unilateral; equivale a exigir que os "
                "dois metodos rejeitem"),
            "alfa_unilateral": float(alfa), "n": n, "gl": n - 1 if n else None,
            "ganho_medio": float(x.mean()) if n else None,
            "nadeau_bengio": nb}
    if n < 2:
        return dict(base, status="nao_calculavel", motivo="menos de 2 ganhos")
    if nb["status"] != "calculado":
        return dict(base, status=nb["status"], motivo=nb["motivo"], t=None, p=None,
                    intervalo_do_teste=None)
    ep_jk, detalhe = erro_padrao_jackknife(medias_sem_k)
    base["jackknife"] = dict(detalhe, erro_padrao=ep_jk)
    if not np.isfinite(ep_jk) or ep_jk <= TOLERANCIA_DEGENERADO:
        return dict(base, status="degenerado", t=None, p=None, intervalo_do_teste=None,
                    motivo=("o erro-padrão do jackknife é zero: as rodadas sem cada "
                            "participante dão o mesmo ganho médio, e o procedimento não "
                            "transforma variância zero em evidência infinita"))
    media, gl = float(x.mean()), n - 1
    ep_nb = float(nb["erro_padrao_corrigido"])
    t_jk = media / ep_jk
    p_jk = float(stats.t.sf(t_jk, gl))
    base["jackknife"].update(t=float(t_jk), p=p_jk)
    ep = max(ep_nb, ep_jk)
    t = media / ep
    p = float(stats.t.sf(t, gl))
    q = float(stats.t.ppf(1 - alfa, gl))
    rejeita_nb = bool(media > 0 and nb["p"] <= alfa)
    rejeita_jk = bool(media > 0 and p_jk <= alfa)
    passou = bool(media > 0 and p <= alfa)
    if passou != (rejeita_nb and rejeita_jk):
        raise RuntimeError("a regra do maior erro-padrão discordou de 'os dois "
                           "rejeitam' (p = {}, NB {}, jackknife {})".format(p, nb["p"], p_jk))
    return dict(base, status="calculado", erro_padrao_usado=float(ep),
                metodo_do_erro_padrao_usado=("jackknife" if ep_jk >= ep_nb
                                             else "nadeau_bengio"),
                t=float(t), p=p, intervalo_do_teste=_intervalo_do_teste(media, ep, q, alfa),
                intervalo_95=_intervalo_de_95(media, ep, gl),
                rejeita_nadeau_bengio=rejeita_nb, rejeita_jackknife=rejeita_jk,
                passou=passou, motivo="")


def _descricao_dos_ganhos(ganhos):
    """Média, mediana, menor, maior e contagem de melhoras, pioras e empates."""
    v = list(ganhos.values())
    return {"ganho_medio": float(np.mean(v)) if v else None,
            "ganho_mediano": float(np.median(v)) if v else None,
            "menor_ganho": min(v) if v else None, "maior_ganho": max(v) if v else None,
            "melhoras": sum(1 for g in v if g > 0), "pioras": sum(1 for g in v if g < 0),
            "empates": sum(1 for g in v if g == 0)}


def _ganhos_contra(linhas_p, referencia, pessoas, rotulo, acerto_do_personalizado=None,
                   fora=None):
    usados, frases, D = tabela_contra_o_entre_pessoas(
        linhas_p, referencia, pessoas, rotulo, acerto_do_personalizado, fora)
    return usados, frases, ganhos_da_tabela(usados, D)


def _ganhos_da_h1b(linhas_p, linhas_ep, pessoas, sem, K, fora, sorteio=None):
    if sorteio is None:
        ref = acerto_medio_por_gravacao(linhas_ep, "igualado", sem, K)
        reduzido = acerto_medio_por_gravacao(linhas_ep, "personalizado_reduzido", sem, K)
    else:
        ref = acerto_de_um_sorteio(linhas_ep, "igualado", sem, sorteio)
        reduzido = acerto_de_um_sorteio(linhas_ep, "personalizado_reduzido", sem, sorteio)
    rotulo = "igualado" + (" sem {}".format(sem) if sem else "") + (
        " (sorteio {})".format(sorteio) if sorteio is not None else "")
    return _ganhos_contra(linhas_p, ref, pessoas, rotulo, reduzido, fora)


def _medias_sem_cada_um(linhas_p, linhas_ep, pessoas, versao, K, fora=None):
    """{k: ganho médio dos outros} com a referência da rodada sem k."""
    medias = {}
    for k in sorted(pessoas):
        outros = [p for p in pessoas if p != k]
        if versao == "igualado":
            usados, _f, g = _ganhos_da_h1b(linhas_p, linhas_ep, outros, k, K, fora)
        else:
            ref = acerto_medio_por_gravacao(linhas_ep, versao, k, K)
            usados, _f, g = _ganhos_contra(linhas_p, ref, outros,
                                           "{} sem {}".format(versao, k))
        if sorted(usados) != sorted(outros):
            raise ValueError("na rodada sem {}, {} ficou sem ganho contra o {}".format(
                k, sorted(set(outros) - set(usados)), versao))
        medias[k] = float(np.mean([g[p] for p in outros]))
    return medias


def distribuicao_dos_sorteios(linhas_p, linhas_ep, pessoas, K, fora, ganho_medio):
    """O resultado de cada um dos K sorteios da rodada principal (decisão de 05/10/2026, tarde: "guardar os 20 sorteios")."""
    por_sorteio = []
    for s in range(1, K + 1):
        usados, _f, g = _ganhos_da_h1b(linhas_p, linhas_ep, pessoas, "", K, fora,
                                       sorteio=s)
        if sorted(usados) != sorted(pessoas):
            raise ValueError("no sorteio {}, {} ficou sem ganho".format(
                s, sorted(set(pessoas) - set(usados))))
        por_sorteio.append({"sorteio": s,
                            "ganho_medio": float(np.mean([g[p] for p in pessoas])),
                            "ganho_por_participante": {p: float(g[p]) for p in pessoas}})
    medias = np.array([x["ganho_medio"] for x in por_sorteio])
    if abs(float(medias.mean()) - float(ganho_medio)) > 1e-9:
        raise RuntimeError("a media dos ganhos dos {} sorteios ({}) difere do ganho medio "
                           "({})".format(K, medias.mean(), ganho_medio))
    return {"papel": ("DESCRITIVO: o ganho medio de cada um dos {} sorteios do igualado "
                      "(rodada principal); a decisao usa a media dos sorteios em cada "
                      "gravacao".format(K)),
            "sorteios": por_sorteio,
            "resumo": {"minimo": float(medias.min()),
                       "quartil_1": float(np.percentile(medias, 25)),
                       "mediana": float(np.median(medias)),
                       "quartil_3": float(np.percentile(medias, 75)),
                       "maximo": float(medias.max()),
                       "media_dos_sorteios": float(medias.mean())},
            "media_dos_sorteios_igual_ao_ganho_medio": True}


def semente_do_sorteio_do_igualado(semente_base, alvo, sem=""):
    """A entropia dos K sorteios de um alvo numa rodada, pela fórmula registrada."""
    def _int(texto):
        return int.from_bytes(hashlib.sha256(str(texto).encode("utf-8")).digest()[:8], "big")
    return [int(semente_base), _int(alvo), _int("sem:" + (sem or ""))]


def semente_do_personalizado_reduzido(semente_base, alvo, sem=""):
    return semente_do_sorteio_do_igualado(semente_base, alvo, sem) + [int.from_bytes(
        hashlib.sha256(SEMENTE_DO_PERSONALIZADO_REDUZIDO.encode("utf-8")).digest()[:8],
        "big")]


def conferir_sorteios_do_igualado(diretorio, chaves, linhas_ep):
    """Confere que a avaliação usou a semente, o K e a regra de sorteio congelados."""
    caminho_av = os.path.join(diretorio, ARQUIVO_DA_AVALIACAO)
    if not os.path.exists(caminho_av):
        raise FileNotFoundError("{} ausente".format(ARQUIVO_DA_AVALIACAO))
    brutas = _ler_csv_opcional(diretorio, ARQUIVO_DOS_SORTEIOS_DO_IGUALADO)
    if brutas is None:
        raise FileNotFoundError("{} ausente".format(ARQUIVO_DOS_SORTEIOS_DO_IGUALADO))
    with open(caminho_av, encoding="utf-8") as f:
        avaliacao = json.load(f)
    meta = (avaliacao.get("metadados") or {}).get("entre_pessoas_igualado") or {}
    for nome, campo in ((ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO, "sha256"),
                        (ARQUIVO_DOS_SORTEIOS_DO_IGUALADO, "sha256_dos_sorteios")):
        declarado = meta.get(campo)
        if not declarado:
            raise ValueError("{} sem o sha256 de {} (metadados.entre_pessoas_igualado."
                             "{})".format(ARQUIVO_DA_AVALIACAO, nome, campo))
        lido = _sha256_do_arquivo(os.path.join(diretorio, nome))
        if lido != declarado:
            raise ValueError("o sha256 de {} ({}) difere do declarado no {} ({})".format(
                nome, lido[:12], ARQUIVO_DA_AVALIACAO, declarado[:12]))
    K = int(chaves["h1b_igualado_sorteios"])
    base = int(chaves["h1b_igualado_semente"])
    minimo = int(chaves["treino_minimo_por_sessao_de_treino"])
    if brutas:
        faltam = [c for c in COLUNAS_DOS_SORTEIOS_DO_IGUALADO if c not in brutas[0]]
        if faltam:
            raise ValueError("{} sem as colunas {}".format(
                ARQUIVO_DOS_SORTEIOS_DO_IGUALADO, faltam))
    rodadas_declaradas, esperada_de, semente_p_de = {}, {}, {}
    nao_calculaveis_de, reduzidas_de = {}, {}
    for registro in avaliacao.get("registros", []):
        bloco = registro.get("entre_pessoas_igualado")
        if not isinstance(bloco, dict) or "indisponivel" in bloco:
            continue
        alvo = str(registro["participante"])
        if int(bloco["K"]) != K:
            raise ValueError("o igualado de {} usou K = {}; a chave congelada pede "
                             "{}".format(alvo, bloco["K"], K))
        if int(bloco["semente_base"]) != base:
            raise ValueError("o igualado de {} usou a semente-base {}; a chave congelada "
                             "e' {}".format(alvo, bloco["semente_base"], base))
        if bloco.get("faltante") != chaves["h1b_igualado_faltante"]:
            raise ValueError("o igualado de {} usou a regra de faltante {!r}; a chave "
                             "congelada e' {!r}".format(alvo, bloco.get("faltante"),
                                                       chaves["h1b_igualado_faltante"]))
        if bloco.get("minimo_por_celula") != minimo:
            raise ValueError("o igualado de {} usou o minimo por celula {!r}; a chave "
                             "congelada e' {}".format(alvo, bloco.get("minimo_por_celula"),
                                                     minimo))
        contagem = Counter({(str(c["frase"]), int(c["sessao"])): int(c["n"])
                            for c in bloco["contagem_do_alvo"]})
        for rodada in bloco["rodadas"]:
            sem = str(rodada["sem"] or "")
            esperada = semente_do_sorteio_do_igualado(base, alvo, sem)
            if [int(v) for v in rodada["semente"]] != esperada:
                raise ValueError("a rodada sem {!r} de {} usou a entropia {}; a formula "
                                 "registrada da' {}".format(sem, alvo, rodada["semente"],
                                                            esperada))
            if int(rodada["sorteios"]) != K:
                raise ValueError("a rodada sem {!r} de {} declara {} sorteios; a chave "
                                 "pede {}".format(sem, alvo, rodada["sorteios"], K))
            if rodada.get("completadas"):
                raise ValueError("a rodada sem {!r} de {} declara celulas completadas por "
                                 "repeticao; a regra de completar por repeticao foi "
                                 "removida em 05/10/2026 (tarde): so' gravacoes "
                                 "unicas".format(sem, alvo))
            usadas = Counter(contagem)
            reduzidas = {}
            for c in rodada.get("celulas_reduzidas") or []:
                cel = (str(c["frase"]), int(c["sessao"]))
                u = int(c["usadas"])
                if (cel not in contagem or int(c["personalizado"]) != contagem[cel]
                        or not 0 <= u < contagem[cel] or int(c["disponiveis"]) != u):
                    raise ValueError("a rodada sem {!r} de {} declara uma reducao "
                                     "incoerente na frase {!r}, sessao {}: so' se reduz ate' "
                                     "as gravacoes disponiveis dos outros".format(
                                         sem, alvo, cel[0], cel[1]))
                usadas[cel] = u
                reduzidas[cel] = u
            abaixo = {cel for cel, n in usadas.items() if n < minimo}
            declaradas = {(str(c["frase"]), int(c["sessao"]))
                          for c in rodada.get("celulas_nao_calculaveis") or []}
            if declaradas != abaixo:
                raise ValueError("a rodada sem {!r} de {} declara as celulas nao "
                                 "calculaveis {}; abaixo do minimo de {} ficam {}".format(
                                     sem, alvo, sorted(declaradas), minimo, sorted(abaixo)))
            reduzido = rodada.get("personalizado_reduzido")
            if bool(reduzido) != bool(reduzidas):
                raise ValueError("a rodada sem {!r} de {} {} o Personalizado reduzido, mas "
                                 "{} celula reduzida".format(
                                     sem, alvo, "traz" if reduzido else "nao traz",
                                     "nao tem" if reduzido else "tem"))
            if reduzido:
                esperada_p = semente_do_personalizado_reduzido(base, alvo, sem)
                if [int(v) for v in reduzido["semente"]] != esperada_p:
                    raise ValueError("o Personalizado reduzido da rodada sem {!r} de {} "
                                     "usou a entropia {}; a formula registrada da' "
                                     "{}".format(sem, alvo, reduzido["semente"], esperada_p))
                if int(reduzido["sorteios"]) != K:
                    raise ValueError("o Personalizado reduzido da rodada sem {!r} de {} "
                                     "declara {} sorteios; a chave pede {}".format(
                                         sem, alvo, reduzido["sorteios"], K))
                semente_p_de[(alvo, sem)] = "-".join(str(v) for v in esperada_p)
            rodadas_declaradas[(alvo, sem)] = "-".join(str(v) for v in esperada)
            esperada_de[(alvo, sem)] = +usadas
            nao_calculaveis_de[(alvo, sem)] = {f for f, _s in abaixo}
            reduzidas_de[(alvo, sem)] = reduzidas
    por_sorteio = defaultdict(list)
    for i, l in enumerate(brutas, 2):
        alvo, sem = str(l["pessoa"]), str(l.get("sem") or "").strip()
        braco = str(l.get("braco") or "").strip()
        onde = "linha {} de {}".format(i, ARQUIVO_DOS_SORTEIOS_DO_IGUALADO)
        if braco not in BRACOS_DOS_SORTEIOS:
            raise ValueError("{}: braco {!r}; vale {}".format(
                onde, l.get("braco"), ", ".join(BRACOS_DOS_SORTEIOS)))
        if (alvo, sem) not in rodadas_declaradas:
            raise ValueError("{}: rodada sem {!r} de {} ausente do {}".format(
                onde, sem, alvo, ARQUIVO_DA_AVALIACAO))
        if braco == "igualado":
            if str(l["semente"]).strip() != rodadas_declaradas[(alvo, sem)]:
                raise ValueError("{}: semente {!r} diferente da formula registrada".format(
                    onde, l["semente"]))
            if str(l["doadora"]) in ({alvo} | ({sem} if sem else set())):
                raise ValueError("{}: a doadora {} e' o alvo ou o participante deixado "
                                 "de fora".format(onde, l["doadora"]))
        else:
            if (alvo, sem) not in semente_p_de:
                raise ValueError("{}: Personalizado reduzido numa rodada sem celula "
                                 "reduzida".format(onde))
            if str(l["semente"]).strip() != semente_p_de[(alvo, sem)]:
                raise ValueError("{}: semente {!r} do Personalizado reduzido diferente da "
                                 "formula registrada".format(onde, l["semente"]))
            if str(l["doadora"]) != alvo:
                raise ValueError("{}: o Personalizado reduzido de {} usa so' gravacoes do "
                                 "proprio alvo; veio {}".format(onde, alvo, l["doadora"]))
        por_sorteio[(alvo, sem, braco, int(l["sorteio"]))].append(
            (str(l["doadora"]), str(l["arquivo_doado"]),
             (str(l["frase_doada"]), int(l["sessao_doada"]))))
    for (alvo, sem) in rodadas_declaradas:
        for braco in BRACOS_DOS_SORTEIOS:
            if braco == "personalizado_reduzido" and (alvo, sem) not in semente_p_de:
                continue
            vistos = sorted(s for (a, k, b, s) in por_sorteio if (a, k, b) == (alvo, sem, braco))
            if vistos != list(range(1, K + 1)):
                raise ValueError("a rodada sem {!r} de {} tem os sorteios {} do {} na "
                                 "auditoria; a regra exige 1 a {}".format(
                                     sem, alvo, vistos, braco, K))
    for (alvo, sem, braco, s), doadas in por_sorteio.items():
        conta = Counter(c for _d, _a, c in doadas)
        if conta != esperada_de[(alvo, sem)]:
            raise ValueError("o sorteio {} do {} da rodada sem {!r} de {} tem contagem por "
                             "frase e sessao diferente da regra (b) (a do Personalizado, "
                             "com as celulas reduzidas)".format(s, braco, sem, alvo))
        arquivos = [(d, a) for d, a, _c in doadas]
        if len(set(arquivos)) != len(arquivos):
            raise ValueError("o sorteio {} do {} da rodada sem {!r} de {} repete "
                             "gravacao; a regra so' aceita gravacoes unicas, sem "
                             "reposicao".format(s, braco, sem, alvo))
    rodadas_no_csv = {(l["pessoa"], l["sem"]) for l in linhas_ep
                      if l["versao"] == "igualado"}
    if rodadas_no_csv != set(rodadas_declaradas):
        raise ValueError("as rodadas do {} ({}) e as da auditoria dos sorteios ({}) "
                         "diferem".format(ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO,
                                          len(rodadas_no_csv), len(rodadas_declaradas)))
    reduzidas_no_csv = {(l["pessoa"], l["sem"]) for l in linhas_ep
                        if l["versao"] == "personalizado_reduzido"}
    if reduzidas_no_csv != set(semente_p_de):
        raise ValueError("as rodadas com o Personalizado reduzido no {} ({}) e as "
                         "declaradas ({}) diferem".format(
                             ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO, sorted(reduzidas_no_csv),
                             sorted(semente_p_de)))
    marcadas = defaultdict(set)
    for l in linhas_ep:
        if l["versao"] == "igualado" and not l["calculavel"]:
            marcadas[(l["pessoa"], l["sem"])].add(l["frase"])
    for chave, frases in nao_calculaveis_de.items():
        com_teste = {l["frase"] for l in linhas_ep
                     if l["versao"] == "igualado" and (l["pessoa"], l["sem"]) == chave}
        if marcadas.get(chave, set()) != frases & com_teste:
            raise ValueError("a rodada sem {!r} de {} marca como nao calculaveis no {} as "
                             "frases {}; a avaliacao declara {}".format(
                                 chave[1], chave[0], ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO,
                                 sorted(marcadas.get(chave, set())), sorted(frases)))
    return {"conferida": True, "K": K, "semente_base": base, "minimo_por_celula": minimo,
            "rodadas": len(rodadas_declaradas),
            "sorteios": sum(1 for k in por_sorteio if k[2] == "igualado"),
            "sorteios_do_personalizado_reduzido": sum(
                1 for k in por_sorteio if k[2] == "personalizado_reduzido"),
            "celulas_reduzidas": {"{} sem {}".format(a, k): [
                {"frase": c[0], "sessao": c[1], "usadas": n} for c, n in sorted(v.items())]
                for (a, k), v in sorted(reduzidas_de.items()) if v},
            "frases_nao_calculaveis": {"{} sem {}".format(a, k): sorted(v)
                                       for (a, k), v in sorted(nao_calculaveis_de.items())
                                       if v},
            "versao_do_numpy_na_avaliacao": meta.get("versao_do_numpy")}


def _procedencia_dos_sorteios(diretorio, chaves, linhas_ep, modo):
    """(bloco, motivo_de_recusa)."""
    try:
        return conferir_sorteios_do_igualado(diretorio, chaves, linhas_ep), None
    except FileNotFoundError as erro:
        bloco = {"conferida": False, "motivo": "nao conferida: {}".format(erro)}
        if modo == "estudo":
            return bloco, ("procedencia dos sorteios do igualado nao conferida ({}); no "
                           "modo do estudo a H1b exige a semente e os sorteios da "
                           "avaliacao".format(erro))
        return bloco, None
    except (ValueError, KeyError, TypeError, OSError) as erro:
        motivo = "procedencia dos sorteios do igualado recusada: {}".format(erro)
        return {"conferida": False, "motivo": motivo}, motivo


def h1b_na_analise(veredito_da_h1a, pessoas, linhas, chaves, verif, diretorio,
                   modo=None):
    """A H1b: Personalizado contra o Entre-pessoas igualado."""
    if chaves["h1b_comparador"] not in COMPARADORES_DA_H1B:
        raise ValueError("h1b_comparador não implementado: {!r}".format(
            chaves["h1b_comparador"]))
    if chaves["h1b_regra"] not in REGRAS_DA_H1B:
        raise ValueError("h1b_regra não implementada: {!r}".format(chaves["h1b_regra"]))
    if chaves["h1b_fator_nadeau_bengio"] != "1/N+1/(N-1)":
        raise ValueError("h1b_fator_nadeau_bengio divergente do procedimento")
    if chaves["h1b_ordem"] not in ORDENS_DA_H1B:
        raise ValueError("h1b_ordem não implementada: {!r}".format(chaves["h1b_ordem"]))
    if chaves["h1b_igualado_faltante"] not in FALTANTES_DO_IGUALADO:
        raise ValueError("h1b_igualado_faltante não implementada: {!r}".format(
            chaves["h1b_igualado_faltante"]))
    if chaves["h1b_igualado_configuracao"] not in CONFIGURACOES_DO_IGUALADO:
        raise ValueError("h1b_igualado_configuracao não implementada: {!r}".format(
            chaves["h1b_igualado_configuracao"]))
    alfa = float(chaves["h1_alfa_unilateral"])
    n_minimo = int(chaves["h1_n_minimo_para_veredito"])
    K = int(chaves["h1b_igualado_sorteios"])
    confirmatoria = (veredito_da_h1a or {}).get("veredito") is True
    base = {"hipotese": "H1b (Personalizado x Entre-pessoas igualado)",
            "comparador": "entre_pessoas_igualado",
            "papel": "confirmatoria" if confirmatoria else "exploratoria",
            "rotulo": ("CONFIRMATORIA: a H1a foi sustentada, e a H1b recebe veredito"
                       if confirmatoria else
                       "EXPLORATORIA: a H1a nao foi sustentada ({}); a H1b sai com "
                       "estimativa, intervalos e p, sem rotulo de sustentada ou nao "
                       "sustentada".format((veredito_da_h1a or {}).get("resultado",
                                                                       "ausente"))),
            "alfa_unilateral": alfa, "n_minimo_para_veredito": n_minimo,
            "sorteios": K, "semente": chaves["h1b_igualado_semente"],
            "contagem": chaves["h1b_igualado_contagem"],
            "faltante": chaves["h1b_igualado_faltante"],
            "configuracao_do_igualado": chaves["h1b_igualado_configuracao"],
            "metrica": ("ganho de cada participante = media, nas frases dele, com peso "
                        "igual, de (acerto do Personalizado - media dos K sorteios do "
                        "igualado), nas mesmas gravacoes do degrau 1, sem as frases nao "
                        "calculaveis; onde a regra (b) reduziu celulas, o Personalizado "
                        "e' o reduzido (media dos K sorteios dele)"),
            "procedimento": PROCEDIMENTO_DOS_DOIS_METODOS,
            "fonte": "plano de analise; decisoes de 05/10/2026"}

    def _nao_calculavel(motivo, **extra):
        return dict(base, estado="nao_calculavel", veredito=None,
                    resultado=("sem veredito" if confirmatoria
                               else "exploratoria, nao calculada"),
                    motivo=motivo, **extra)

    try:
        brutas = _ler_csv_opcional(diretorio, ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO)
        if brutas is None:
            raise ValueError("sem o arquivo {}: o Entre-pessoas igualado nao "
                             "rodou".format(ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO))
        linhas_ep = conferir_entre_pessoas_igualado(brutas, K)
        procedencia, recusa = _procedencia_dos_sorteios(diretorio, chaves, linhas_ep, modo)
        base["procedencia_dos_sorteios"] = procedencia
        if recusa:
            raise ValueError(recusa)
        linhas_p = linhas_do_personalizado_do_degrau_1(linhas, pessoas, chaves, verif)
        fora = frases_fora_da_h1b(linhas_ep)
        usados, frases, ganhos = _ganhos_da_h1b(linhas_p, linhas_ep, pessoas, "", K, fora)
        base.update(participantes_usados=usados, frases_no_grupo=frases,
                    frases_fora_da_h1b={p: f for p, f in fora.items() if p in set(pessoas)},
                    ganho_por_participante=ganhos, **_descricao_dos_ganhos(ganhos))
        if usados:
            base["distribuicao_dos_sorteios"] = distribuicao_dos_sorteios(
                linhas_p, linhas_ep, usados, K, fora, base["ganho_medio"])
        if len(usados) < n_minimo:
            return _nao_calculavel("sem veredito: {} participante(s) elegivel(is), menos "
                                   "que {}; sai so' a descricao".format(len(usados),
                                                                       n_minimo))
        medias = _medias_sem_cada_um(linhas_p, linhas_ep, usados, "igualado", K, fora=fora)
        teste = teste_da_h1b_dois_metodos(ganhos, medias, alfa=alfa)
    except (ValueError, RuntimeError) as erro:
        return _nao_calculavel(str(erro))
    if teste["status"] != "calculado":
        return _nao_calculavel("sem veredito: " + teste["motivo"], teste=teste)
    if not confirmatoria:
        return dict(base, estado="exploratoria", veredito=None, resultado="exploratoria",
                    teste=teste,
                    motivo=("a H1a nao foi sustentada: a H1b e' relatada como exploratoria, "
                            "com estimativa, intervalos e p (maior dos dois erros-padrao); "
                            "o p nao decide nada"))
    passou = teste["passou"]
    return dict(base, estado="passou" if passou else "nao_passou", veredito=passou,
                resultado="sustentada" if passou else "nao sustentada", teste=teste,
                motivo=("" if passou else
                        "o ganho medio foi {:+.4f}; p de Nadeau e Bengio {:.4g}, p do "
                        "jackknife {:.4g}; a H1b exige ganho medio positivo e os dois "
                        "p <= {:.3g}".format(teste["ganho_medio"],
                                             teste["nadeau_bengio"]["p"],
                                             teste["jackknife"]["p"], alfa)))


def entre_pessoas_completo_exploratorio(pessoas, linhas, chaves, verif, diretorio):
    """O Entre-pessoas COMPLETO (todos os outros participantes): exploratório."""
    if chaves["entre_pessoas_completo_papel"] not in PAPEIS_DO_COMPLETO:
        raise ValueError("entre_pessoas_completo_papel não implementado: {!r}".format(
            chaves["entre_pessoas_completo_papel"]))
    alfa = float(chaves["h1_alfa_unilateral"])
    K = int(chaves["h1b_igualado_sorteios"])
    base = {"nome": "Entre-pessoas completo (exploratorio)",
            "papel": ("EXPLORATORIO, sem veredito: Personalizado contra o Entre-pessoas "
                      "completo (treinado com todas as gravacoes de treino dos outros "
                      "participantes); estimativa, intervalos de 92% e 95% e p "
                      "exploratorio (unilateral, maior dos dois erros-padrao), que nao "
                      "decide nada; nao e' a H1b"),
            "nivel_do_intervalo": 1 - 2 * alfa,
            "procedimento": PROCEDIMENTO_DOS_DOIS_METODOS}
    try:
        linhas_p = linhas_do_personalizado_do_degrau_1(linhas, pessoas, chaves, verif)
        ref = acerto_do_completo_no_acertos_csv(linhas, set(pessoas))
        usados, _frases, ganhos = _ganhos_contra(linhas_p, ref, pessoas,
                                                 "Entre-pessoas completo do acertos.csv")
    except ValueError as erro:
        return dict(base, intervalo=None, intervalo_95=None, p_exploratorio=None,
                    motivo=str(erro))
    base.update(participantes_usados=usados, ganho_por_participante=ganhos,
                **_descricao_dos_ganhos(ganhos))
    try:
        brutas = _ler_csv_opcional(diretorio, ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO)
        if brutas is None:
            raise ValueError("sem o arquivo {}: sem as rodadas do jackknife".format(
                ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO))
        if len(usados) < 2:
            raise ValueError("menos de 2 participantes com ganho")
        linhas_ep = conferir_entre_pessoas_igualado(brutas, K)
        medias = _medias_sem_cada_um(linhas_p, linhas_ep, usados, "completo", K)
        teste = teste_da_h1b_dois_metodos(ganhos, medias, alfa=alfa)
    except (ValueError, RuntimeError) as erro:
        return dict(base, intervalo=None, intervalo_95=None, p_exploratorio=None,
                    motivo="intervalo nao calculado: " + str(erro))
    if teste["status"] != "calculado":
        return dict(base, intervalo=None, intervalo_95=None, p_exploratorio=None,
                    motivo="intervalo nao calculado: " + teste["motivo"])
    return dict(base, intervalo=_intervalo_exploratorio(teste["intervalo_do_teste"]),
                intervalo_95=_intervalo_exploratorio(teste["intervalo_95"]),
                p_exploratorio=teste["p"], t_exploratorio=teste["t"], gl=teste["gl"],
                erro_padrao_nadeau_bengio=teste["nadeau_bengio"]["erro_padrao_corrigido"],
                erro_padrao_jackknife=teste["jackknife"]["erro_padrao"],
                erro_padrao_usado=teste["erro_padrao_usado"], motivo="")


ARQUIVO_DA_CONFIGURACAO_PADRAO = "personalizado_configuracao_padrao.csv"
COLUNAS_DA_CONFIGURACAO_PADRAO = ("pessoa", "frase", "sessao", "gravacao", "acerto",
                                  "configuracao", "igual_a_selecionada")
PAPEL_DA_SENSIBILIDADE_DA_PADRAO = (
    "SENSIBILIDADE -- nao decide: o degrau 1 refeito com o Personalizado de todos na "
    "configuracao padrao (a primeira da lista), nas mesmas gravacoes e frases; nao "
    "muda o resultado principal e nao resgata um resultado negativo")


def acerto_do_generico_no_acertos_csv(linhas, pessoas):
    """{(pessoa, gravação): (frase, acerto)} do Genérico no acertos.csv."""
    return {(l["pessoa"], l.get("gravacao")): (l["frase"], float(l["acerto"]))
            for l in linhas
            if l["pessoa"] in pessoas and l["condicao"] == "generico" and _e_gravacao(l)}


def sensibilidade_configuracao_padrao_na_analise(fixas, linhas, chaves, verif,
                                                 diretorio):
    """O bloco `sensibilidade_configuracao_padrao` do JSON."""
    if chaves["sensibilidade_configuracao_padrao"] != \
            REGRAS_DE_07_10["sensibilidade_configuracao_padrao"]:
        raise ValueError("sensibilidade_configuracao_padrao não implementada: {!r}"
                         .format(chaves["sensibilidade_configuracao_padrao"]))
    pessoas = list(fixas.get("participantes_usados") or [])
    base = {"papel": PAPEL_DA_SENSIBILIDADE_DA_PADRAO,
            "fonte": "decisao de 26/09/2026; regra 1 de 07/10/2026; plano de analise",
            "resultado_principal": fixas.get("status"),
            "nao_altera_o_resultado_principal": True}
    brutas = _ler_csv_opcional(diretorio, ARQUIVO_DA_CONFIGURACAO_PADRAO)
    if brutas is None:
        return dict(base, disponivel=False,
                    motivo="sem o arquivo {}: a avaliacao nao rodou a configuracao "
                           "padrao".format(ARQUIVO_DA_CONFIGURACAO_PADRAO))
    if brutas and any(c not in brutas[0] for c in COLUNAS_DA_CONFIGURACAO_PADRAO):
        return dict(base, disponivel=False, motivo="{} sem as colunas {}".format(
            ARQUIVO_DA_CONFIGURACAO_PADRAO, list(COLUNAS_DA_CONFIGURACAO_PADRAO)))
    padrao = {}
    for i, l in enumerate(brutas, 2):
        try:
            padrao[(l["pessoa"], l["gravacao"])] = (l["frase"], ler_acerto(l["acerto"]))
        except ValueError as e:
            return dict(base, disponivel=False, motivo="linha {} de {}: {}".format(
                i, ARQUIVO_DA_CONFIGURACAO_PADRAO, e))
    configuracoes = sorted({l["configuracao"] for l in brutas})
    iguais = sorted({l["pessoa"] for l in brutas if l["igual_a_selecionada"] == "1"})
    try:
        linhas_p = linhas_do_personalizado_do_degrau_1(linhas, set(pessoas), chaves, verif)
        ref = acerto_do_generico_no_acertos_csv(linhas, set(pessoas))
        usados, frases, D = tabela_contra_o_entre_pessoas(
            linhas_p, ref, pessoas, "Genérico (configuracao padrao)",
            acerto_do_personalizado=padrao)
    except ValueError as e:
        return dict(base, disponivel=False, motivo=str(e))
    faltam = sorted(set(pessoas) - {p for (p, _g) in padrao})
    if faltam:
        return dict(base, disponivel=False,
                    motivo="sem a configuracao padrao para {}".format(faltam))
    ganhos = ganhos_da_tabela(usados, D)
    saida = dict(base, disponivel=True, configuracao=configuracoes,
                 participantes=usados, n=len(usados),
                 ja_usavam_a_padrao=iguais, frases_no_grupo=frases,
                 ganho_por_participante=ganhos, **_descricao_dos_ganhos(ganhos))
    if len(usados) < N_MINIMO_DA_H1:
        return dict(saida, so_estimativa=True,
                    motivo="{} participante(s), menos que {}: so' a estimativa".format(
                        len(usados), N_MINIMO_DA_H1))
    r = ganho_frases_fixas(D)
    return dict(saida, so_estimativa=False, t=r["t"], gl=r["gl"], p=r["p"],
                intervalo_do_teste=r["intervalo_do_teste"],
                intervalo_95=r["intervalo_95"],
                acusaria_ganho_a_este_nivel=bool(r["p"] is not None
                                                 and r["p"] <= ALFA_DA_H1A),
                motivo=r.get("motivo", ""))


PAPEL_DA_SENSIBILIDADE_SEIS = (
    "SENSIBILIDADE -- nao decide: refeita so' com quem tem {} ou mais frases "
    "distintas e avaliaveis, pareadas nos dois sistemas deste contraste; nao muda "
    "quem entrou na analise principal nem o resultado dela, e nao resgata um "
    "resultado principal negativo".format(MIN_FRASES_DA_SENSIBILIDADE_DO_GRUPO))


def _quem_tem_seis(pessoas, D):
    """(ficam, fora): pelo número de frases pareadas (células finitas) de cada um."""
    ficam, fora = [], {}
    for i, p in enumerate(pessoas):
        n = int(np.isfinite(D[i]).sum())
        if n >= MIN_FRASES_DA_SENSIBILIDADE_DO_GRUPO:
            ficam.append(p)
        else:
            fora[p] = "{} frase(s) pareada(s) neste contraste, abaixo de {}".format(
                n, MIN_FRASES_DA_SENSIBILIDADE_DO_GRUPO)
    return ficam, fora


def _base_da_sensibilidade_seis(contraste, pessoas, ficam, fora, principal):
    return {"papel": PAPEL_DA_SENSIBILIDADE_SEIS, "contraste": contraste,
            "minimo_de_frases": MIN_FRASES_DA_SENSIBILIDADE_DO_GRUPO,
            "n_da_analise_principal": len(pessoas), "participantes": list(ficam),
            "n_restante": len(ficam), "fora": fora,
            "resultado_principal": principal,
            "nao_altera_o_resultado_principal": True}


def _sensibilidade_seis_pelo_t(contraste, pessoas, D, principal):
    """O teste das frases fixas refeito com quem tem seis frases ou mais."""
    ficam, fora = _quem_tem_seis(pessoas, D)
    base = _base_da_sensibilidade_seis(contraste, pessoas, ficam, fora, principal)
    sub = D[[pessoas.index(p) for p in ficam]] if ficam else np.empty((0, D.shape[1]))
    ganhos = ganhos_da_tabela(ficam, sub)
    if len(ficam) < N_MINIMO_DA_H1:
        return dict(base, so_estimativa=True,
                    ganho_medio=(float(np.mean(list(ganhos.values()))) if ganhos else None),
                    motivo="{} participante(s) restante(s), menos que {}: so' a estimativa "
                           "e o N, sem teste".format(len(ficam), N_MINIMO_DA_H1))
    r = ganho_frases_fixas(sub)
    return dict(base, so_estimativa=False, ganho_medio=r["ganho"], t=r["t"], gl=r["gl"],
                p=r["p"], intervalo_do_teste=r["intervalo_do_teste"],
                intervalo_95=r["intervalo_95"],
                acusaria_ganho_a_este_nivel=bool(r["p"] is not None and r["p"] <= ALFA_DA_H1A),
                ganho_por_participante=ganhos, motivo=r.get("motivo", ""))


def sensibilidade_seis_frases_nos_contrastes(pp_conf, fixas, h1a, degrau_2, h1b,
                                             linhas, chaves, verif, diretorio):
    """O bloco `sensibilidade_seis_frases` do JSON: um item por contraste."""
    if chaves["sensibilidade_seis_frases"] != REGRAS_DE_06_10["sensibilidade_seis_frases"]:
        raise ValueError("sensibilidade_seis_frases não implementada: {!r}".format(
            chaves["sensibilidade_seis_frases"]))
    pessoas = list(fixas.get("participantes_usados") or [])
    saida = {"papel": PAPEL_DA_SENSIBILIDADE_SEIS,
             "fonte": "decisao de 06/10/2026; plano de analise"}
    sub = {p: pp_conf[p] for p in pessoas if p in pp_conf}
    pes, _fr, D = _tabela_do_contraste(sub, CONTRASTES_DA_H1["H1a"]["bloco"])
    saida["degrau_1"] = _sensibilidade_seis_pelo_t(
        "Personalizado x Whisper livre (degrau 1)", pes, D, h1a.get("resultado"))
    versoes = comparadores_do_degrau_2(chaves)
    try:
        tabelas = _tabelas_do_degrau_2(pessoas, linhas, chaves, verif, diretorio, versoes)
    except ValueError as erro:
        saida["degrau_2"] = {"disponivel": False, "motivo": str(erro)}
    else:
        principal_2 = degrau_2.get("estado")
        saida["degrau_2"] = {v: dict(_sensibilidade_seis_pelo_t(
            "Personalizado x Whisper em escolha forcada ({}) (degrau 2)".format(v),
            tabelas[v][0], tabelas[v][2], principal_2),
            **({"so_descricao": True, "nota": "o degrau 1 nao passou: no principal o "
                "degrau 2 e' so' descricao, e aqui tambem nao ha leitura de teste"}
               if h1a.get("veredito") is not True else {}))
            for v in versoes}
    saida["h1b"] = _sensibilidade_seis_da_h1b(pessoas, linhas, chaves, verif, diretorio,
                                              h1b)
    try:
        linhas_p = linhas_do_personalizado_do_degrau_1(linhas, pessoas, chaves, verif)
        ref = acerto_do_completo_no_acertos_csv(linhas, set(pessoas))
        usados, _f, Dc = tabela_contra_o_entre_pessoas(linhas_p, ref, pessoas,
                                                       "Entre-pessoas completo")
        ficam, fora = _quem_tem_seis(usados, Dc)
        g = ganhos_da_tabela(ficam, Dc[[usados.index(p) for p in ficam]]
                             if ficam else np.empty((0, Dc.shape[1])))
        saida["entre_pessoas_completo"] = dict(
            _base_da_sensibilidade_seis("Personalizado x Entre-pessoas completo "
                                        "(descritivo)", usados, ficam, fora,
                                        "exploratorio, sem veredito"),
            so_estimativa=True, **_descricao_dos_ganhos(g))
    except ValueError as erro:
        saida["entre_pessoas_completo"] = {"disponivel": False, "motivo": str(erro)}
    return saida


def _sensibilidade_seis_da_h1b(pessoas, linhas, chaves, verif, diretorio, h1b):
    """A H1b refeita com quem tem seis frases ou mais pareadas contra o igualado."""
    K = int(chaves["h1b_igualado_sorteios"])
    try:
        brutas = _ler_csv_opcional(diretorio, ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO)
        if brutas is None:
            raise ValueError("sem o arquivo {}".format(ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO))
        linhas_ep = conferir_entre_pessoas_igualado(brutas, K)
        linhas_p = linhas_do_personalizado_do_degrau_1(linhas, pessoas, chaves, verif)
        fora_b = frases_fora_da_h1b(linhas_ep)
        ref = acerto_medio_por_gravacao(linhas_ep, "igualado", "", K)
        reduzido = acerto_medio_por_gravacao(linhas_ep, "personalizado_reduzido", "", K)
        usados, _f, D = tabela_contra_o_entre_pessoas(linhas_p, ref, pessoas, "igualado",
                                                      reduzido, fora_b)
    except (ValueError, RuntimeError) as erro:
        return {"disponivel": False, "motivo": str(erro)}
    ficam, fora = _quem_tem_seis(usados, D)
    base = _base_da_sensibilidade_seis("Personalizado x Entre-pessoas igualado (H1b)",
                                       usados, ficam, fora, (h1b or {}).get("resultado"))
    ganhos = ganhos_da_tabela(ficam, D[[usados.index(p) for p in ficam]]
                              if ficam else np.empty((0, D.shape[1])))
    if len(ficam) < N_MINIMO_DA_H1:
        return dict(base, so_estimativa=True,
                    ganho_medio=(float(np.mean(list(ganhos.values()))) if ganhos else None),
                    motivo="{} participante(s) restante(s), menos que {}: so' a estimativa "
                           "e o N, sem teste".format(len(ficam), N_MINIMO_DA_H1))
    if (h1b or {}).get("papel") == "exploratoria" and len(ficam) < MINIMO_DE_PARTICIPANTES:
        return dict(base, so_estimativa=True,
                    ganho_medio=float(np.mean(list(ganhos.values()))),
                    motivo="a H1b saiu exploratoria e restaram {} participante(s), menos "
                           "que {}: so' a estimativa e o N".format(
                               len(ficam), MINIMO_DE_PARTICIPANTES))
    try:
        medias = _medias_sem_cada_um(linhas_p, linhas_ep, ficam, "igualado", K, fora=fora_b)
        teste = teste_da_h1b_dois_metodos(ganhos, medias,
                                          alfa=float(chaves["h1_alfa_unilateral"]))
    except (ValueError, RuntimeError) as erro:
        return dict(base, so_estimativa=True,
                    ganho_medio=float(np.mean(list(ganhos.values()))),
                    motivo="teste nao calculado: " + str(erro))
    return dict(base, so_estimativa=False, ganho_medio=teste["ganho_medio"],
                p=teste.get("p"), p_nadeau_bengio=(teste.get("nadeau_bengio") or {}).get("p"),
                p_jackknife=(teste.get("jackknife") or {}).get("p"),
                intervalo_do_teste=teste.get("intervalo_do_teste"),
                intervalo_95=teste.get("intervalo_95"),
                acusaria_ganho_a_este_nivel=bool(teste.get("passou")),
                ganho_por_participante=ganhos, motivo=teste.get("motivo", ""))


def _intervalo_exploratorio(intervalo):
    """Um intervalo do Entre-pessoas completo, com rótulo de exploratório."""
    return {"nivel": intervalo["nivel"], "limites": list(intervalo["limites"]),
            "rotulo": ("intervalo exploratorio de {:.0f}% (maior dos dois erros-padrao); "
                       "nao decide".format(100 * intervalo["nivel"]))}


FORMULACAO_H1A_SIM_H1B_NAO = (
    "há evidência de vantagem sobre o Whisper, porém não de vantagem sobre o "
    "sistema treinado com outros participantes sob quantidade de treino igualada")


ERRO_DA_FAMILIA_DA_H1 = (
    "a H1a e' decidida pelo degrau 1 a 4%; so' se ele passar, o degrau 2 e a H1b formam "
    "uma familia testada por Holm a 4% (o menor p tem de ser <= 2%; se for, o outro "
    "e' testado a 4%; se nao for, nenhum dos dois passa). Com p validos, a chance de "
    "pelo menos uma conclusao falsa na sequencia inteira nao passa de 4%, qualquer que "
    "seja a dependencia entre os testes (plano de analise)")

FAMILIA_DEPOIS_DO_DEGRAU_1 = ("degrau_2", "h1b")


def p_do_degrau_2_na_familia(testes, versoes):
    """O p do degrau 2 que entra no Holm: o MAIOR dos três p unilaterais."""
    ps = {v: (testes.get(v) or {}).get("p") for v in versoes}
    presentes = {v: float(p) for v, p in ps.items() if p is not None}
    faltam = sorted(v for v in versoes if ps.get(v) is None)
    base = {"regra": ("maior dos tres p unilaterais (intersecao-uniao: o degrau passa "
                      "a um nivel a so' se as tres versoes passarem a a)"),
            "p_por_versao": ps, "completo": not faltam, "sem_p": faltam}
    if not presentes:
        return dict(base, p=None, versao_do_maior_p=None,
                    motivo="nenhuma versao com p: {}".format(", ".join(faltam)))
    v_max = max(presentes, key=lambda v: (presentes[v], versoes.index(v)))
    if faltam and presentes[v_max] <= ALFA_DA_H1:
        return dict(base, p=None, versao_do_maior_p=None,
                    motivo=("sem p em {}: o degrau 2 nao pode ser dado como vencido "
                            "sem as tres versoes".format(", ".join(faltam))))
    return dict(base, p=presentes[v_max], versao_do_maior_p=v_max,
                motivo=("" if not faltam else
                        "sem p em {}; a versao {} ja' passa de {:.0%}, e o degrau nao "
                        "passaria a nivel nenhum da familia".format(
                            ", ".join(faltam), v_max, ALFA_DA_H1)))


def p_da_h1b_na_familia(teste):
    """O p da H1b que entra no Holm: o MAIOR entre o de Nadeau e Bengio e o do jackknife."""
    teste = teste or {}
    p_nb = (teste.get("nadeau_bengio") or {}).get("p")
    p_jk = (teste.get("jackknife") or {}).get("p")
    base = {"regra": ("maior entre o p de Nadeau e Bengio e o do jackknife por "
                      "participante (os dois metodos tem de rejeitar)"),
            "p_nadeau_bengio": p_nb, "p_jackknife": p_jk}
    if teste.get("status") != "calculado" or p_nb is None or p_jk is None:
        return dict(base, p=None, metodo_do_maior_p=None,
                    motivo=teste.get("motivo") or "teste da H1b sem p")
    p = max(float(p_nb), float(p_jk))
    media = teste.get("ganho_medio")
    if (media is not None and media > 0 and teste.get("p") is not None
            and not np.isclose(p, float(teste["p"]), rtol=1e-9, atol=1e-12)):
        raise RuntimeError("o p do maior erro-padrao ({}) difere do maior dos dois p "
                           "({})".format(teste["p"], p))
    return dict(base, p=p, metodo_do_maior_p=("jackknife" if float(p_jk) >= float(p_nb)
                                              else "nadeau_bengio"), motivo="")


def holm_da_familia(ps, alfa=ALFA_DA_H1):
    """Holm a `alfa` sobre a família {degrau 2, H1b}, aberta só depois do degrau 1."""
    nomes = [n for n in FAMILIA_DEPOIS_DO_DEGRAU_1 if n in ps]
    if sorted(nomes) != sorted(ps):
        raise ValueError("a familia depois do degrau 1 e' {}; veio {}".format(
            FAMILIA_DEPOIS_DO_DEGRAU_1, sorted(ps)))
    ordem = sorted(nomes, key=lambda n: (ps[n] is None,
                                         ps[n] if ps[n] is not None else 0.0,
                                         FAMILIA_DEPOIS_DO_DEGRAU_1.index(n)))
    m = len(ordem)
    saida, continua = {}, True
    for i, nome in enumerate(ordem):
        limiar = alfa / (m - i)
        p = ps[nome]
        if continua:
            rej = p is not None and p <= limiar
            saida[nome] = {"p": p, "posicao": i + 1, "limiar": limiar, "alcancado": True,
                           "rejeitada": bool(rej), "nivel_efetivo": limiar}
            continua = bool(rej)
        else:
            saida[nome] = {"p": p, "posicao": i + 1, "limiar": limiar, "alcancado": False,
                           "rejeitada": False, "nivel_efetivo": alfa / m}
    return saida


def intervalo_no_nivel_de_holm(media, ep, gl, nivel_unilateral):
    """O intervalo bilateral de 1 - 2·limiar, com o limiar efetivamente usado no Holm (96% para 2%, 92% para 4%)."""
    if media is None or ep is None or gl is None or not np.isfinite(ep) or gl < 1:
        return None
    q = float(stats.t.ppf(1 - nivel_unilateral, gl))
    return {"nivel": round(1 - 2 * nivel_unilateral, 10),
            "limites": [float(media - q * ep), float(media + q * ep)],
            "limiar_unilateral": float(nivel_unilateral),
            "rotulo": ("intervalo no nivel ajustado de Holm efetivamente usado ({:.0f}%, "
                       "limiar unilateral de {:.0%}); os de 92% e 95% continuam "
                       "relatados ao lado".format(100 * (1 - 2 * nivel_unilateral),
                                                  nivel_unilateral))}


def _rotulo_de_n(n):
    if n is None or n < N_MINIMO_DA_H1:
        return None
    return ROTULO_PRELIMINAR if n < N_PLENO_DA_H1 else ROTULO_PLENO


def aplicar_holm_da_familia(degrau_2, h1b, chaves, alfa=ALFA_DA_H1):
    """Aplica a família {degrau 2, H1b} depois do degrau 1 sustentado."""
    for k, v in ((k, REGRAS_DA_NOITE_DE_06_10[k]) for k in REGRAS_DA_NOITE_DE_06_10):
        if chaves.get(k) != v:
            raise ValueError("{} = {!r}: o programa so' implementa {!r}".format(
                k, chaves.get(k), v))
    base = {"regra": chaves["h1_multiplicidade"], "alfa_da_familia": alfa,
            "membros": list(FAMILIA_DEPOIS_DO_DEGRAU_1), "descricao": ERRO_DA_FAMILIA_DA_H1,
            "fonte": "decisao de 06/10/2026; plano de analise"}
    if degrau_2.get("estado") == "nao_testado":
        return dict(base, aberta=False,
                    motivo="o degrau 1 nao passou: a familia nao e' aberta, o degrau 2 "
                           "e' so' descricao e a H1b e' exploratoria")
    v2 = degrau_2.get("veredito") or {}
    versoes = list(v2.get("comparadores") or comparadores_do_degrau_2(chaves))
    if degrau_2.get("estado") == "nao_calculavel" and not v2.get("testes"):
        p2 = {"p": None, "motivo": degrau_2.get("motivo"), "p_por_versao": {},
              "completo": False, "regra": "degrau 2 nao calculavel"}
    else:
        p2 = p_do_degrau_2_na_familia(v2.get("testes") or {}, versoes)
    if h1b.get("papel") != "confirmatoria":
        raise ValueError("familia aberta com a H1b nao confirmatoria: o degrau 1 passou "
                         "e a H1b deveria ser confirmatoria")
    pb = p_da_h1b_na_familia(h1b.get("teste"))
    holm = holm_da_familia({"degrau_2": p2["p"], "h1b": pb["p"]}, alfa=alfa)
    r2 = holm["degrau_2"]
    degrau_2["sem_a_familia"] = {"estado": degrau_2.get("estado"),
                                 "veredito": v2.get("veredito"),
                                 "nota": "cada versao a {:.0%}, sem a familia; nao "
                                         "decide".format(alfa)}
    if r2["rejeitada"]:
        estado2, ver2, res2, mot2, falharam = "passou", True, "sustentada", "", []
    elif p2["p"] is None and not v2.get("falharam"):
        estado2, ver2, res2 = "nao_calculavel", None, "sem veredito"
        mot2, falharam = (p2.get("motivo") or degrau_2.get("motivo")), []
    else:
        estado2, ver2, res2 = "nao_passou", False, "nao sustentada"
        lim = r2["limiar"] if r2["alcancado"] else None
        falharam = ([v for v in versoes if (p2["p_por_versao"].get(v) is None
                                            or p2["p_por_versao"][v] > lim)]
                    if lim is not None else [])
        if lim is not None and p2["p"] is not None:
            mot2 = ("não venceu: {} (p do degrau 2, o maior dos tres, = {:.4g} > limiar "
                    "de Holm {:.0%})".format(", ".join(falharam), p2["p"], lim))
        elif not r2["alcancado"]:
            mot2 = ("o menor p da familia nao passou de {:.0%}: a familia parou e o "
                    "degrau 2 nao foi alcancado (p do degrau 2 = {})".format(
                        alfa / 2, "{:.4g}".format(p2["p"]) if p2["p"] is not None
                        else "sem p"))
        else:
            mot2 = str(p2.get("motivo"))
    degrau_2["estado"] = estado2
    degrau_2["motivo"] = mot2 or None
    if isinstance(degrau_2.get("veredito"), dict):
        degrau_2["veredito"].update(veredito=ver2, resultado=res2, falharam=falharam,
                                    motivo=mot2 or "")
    degrau_2["familia"] = dict(r2, p_da_familia=p2)
    for v in versoes:
        t = (v2.get("testes") or {}).get(v)
        if t is None:
            continue
        r = ((v2.get("por_versao") or {}).get(v) or {}).get("teste") or {}
        t["intervalo_no_nivel_de_holm"] = intervalo_no_nivel_de_holm(
            r.get("ganho"), r.get("ep"), r.get("gl"), r2["nivel_efetivo"])
        if r2["alcancado"]:
            t["passa_no_limiar_de_holm"] = bool(t.get("p") is not None
                                                and t["p"] <= r2["limiar"])
        t["nota_da_familia"] = ("`status` e `veredito` deste teste sao a {:.0%}, sem a "
                                "familia; quem decide e' o Holm (`degrau_2.familia`)"
                                .format(alfa))
    rb = holm["h1b"]
    h1b["sem_a_familia"] = {"estado": h1b.get("estado"), "veredito": h1b.get("veredito"),
                            "nota": "os dois metodos a {:.0%}, sem a familia; nao "
                                    "decide".format(alfa)}
    tb = h1b.get("teste") or {}
    if tb.get("status") == "calculado":
        tb["nota_da_familia"] = ("`passou`, `rejeita_nadeau_bengio` e `rejeita_jackknife` "
                                 "sao a {:.0%}, sem a familia; quem decide e' o Holm "
                                 "(`h1b.familia`)".format(alfa))
        tb["intervalo_no_nivel_de_holm"] = intervalo_no_nivel_de_holm(
            tb.get("ganho_medio"), tb.get("erro_padrao_usado"), tb.get("gl"),
            rb["nivel_efetivo"])
    if h1b.get("estado") in ("passou", "nao_passou"):
        passou = rb["rejeitada"]
        h1b.update(estado="passou" if passou else "nao_passou", veredito=passou,
                   resultado="sustentada" if passou else "nao sustentada",
                   motivo=("" if passou else
                           "p da H1b (maior entre Nadeau e Bengio e jackknife) = {:.4g} "
                           "> limiar de Holm {:.0%}".format(pb["p"], rb["limiar"])
                           if rb["alcancado"] else
                           "o menor p da familia nao passou de {:.0%}: a familia parou e "
                           "a H1b nao foi alcancada (p = {:.4g})".format(alfa / 2, pb["p"])))
    h1b["familia"] = dict(rb, p_da_familia=pb)
    return dict(base, aberta=True, holm=holm, p_do_degrau_2=p2, p_da_h1b=pb)


def _versoes_que_falharam(degrau_2):
    v = degrau_2.get("veredito") or {}
    return list(v.get("falharam") or [])


def estado_da_h1a(degrau_1, degrau_2):
    """O veredito da H1a a partir dos estados dos dois degraus."""
    e1, e2 = degrau_1.get("estado"), degrau_2.get("estado")
    if e1 == "passou":
        if e2 == "passou":
            return {"veredito": True, "resultado": "sustentada",
                    "alcance": "livre_e_escolha_forcada",
                    "qualificacao": None,
                    "leitura": "o Personalizado venceu o Whisper livre (degrau 1, a "
                               "afirmacao principal) e tambem o Whisper em escolha "
                               "forcada, nas tres versoes da nota (degrau 2)"}
        if e2 == "nao_passou":
            falharam = _versoes_que_falharam(degrau_2)
            return {"veredito": True, "resultado": "sustentada",
                    "alcance": "so_livre",
                    "qualificacao": (
                        "degrau 2 nao sustentado{}: a vantagem nao se manteve quando os "
                        "dois sistemas escolhem entre as mesmas doze frases; isso "
                        "qualifica o alcance e nao apaga o degrau 1".format(
                            " (nao venceu: {})".format(", ".join(falharam))
                            if falharam else "")),
                    "leitura": "o Personalizado venceu o Whisper livre (degrau 1, a "
                               "afirmacao principal); no degrau 2, em escolha forcada, "
                               "nao venceu as tres versoes: a conclusao vale contra o "
                               "Whisper como ele e' usado, e nao se estendeu a' escolha "
                               "forcada"}
        return {"veredito": True, "resultado": "sustentada",
                "alcance": "so_livre",
                "qualificacao": ("degrau 2 sem veredito ({}): o alcance na escolha forcada "
                                 "nao foi verificado; isso nao apaga o degrau 1".format(
                                     degrau_2.get("motivo") or e2)),
                "leitura": "o Personalizado venceu o Whisper livre (degrau 1, a "
                           "afirmacao principal); o degrau 2 ficou sem veredito"}
    if e1 == "nao_passou":
        return {"veredito": False, "resultado": "nao sustentada",
                "alcance": None, "qualificacao": None,
                "leitura": "o Personalizado nao venceu o Whisper livre (degrau 1); o "
                           "degrau 2 sai so' como descricao, sem teste"}
    return {"veredito": None, "resultado": "sem veredito", "alcance": None,
            "qualificacao": None,
            "leitura": "degrau 1: {}; degrau 2: {} (so' descricao)".format(e1, e2)}


def leitura_da_h1(h1a, h1b):
    """As combinações da H1 a partir dos vereditos próprios da H1a e da H1b."""
    r = _leitura_da_h1_sem_rotulo(h1a, h1b)
    rotulos = [x.get("rotulo_do_n") for x in (h1a, h1b) if x.get("veredito") is not None]
    if ROTULO_PRELIMINAR in rotulos:
        r = dict(r, rotulo_do_n=ROTULO_PRELIMINAR,
                 texto="[{}] {}".format(ROTULO_PRELIMINAR, r["texto"]))
    elif rotulos and all(x == ROTULO_PLENO for x in rotulos):
        r = dict(r, rotulo_do_n=ROTULO_PLENO)
    return r


def _leitura_da_h1_sem_rotulo(h1a, h1b):
    a, b = h1a.get("veredito"), h1b.get("veredito")
    qualificacao = h1a.get("qualificacao")
    extra = "; H1a qualificada: {}".format(qualificacao) if qualificacao else ""
    if a is None:
        return {"leitura": "sem veredito",
                "texto": "a H1a ficou sem veredito; a H1b sai como exploratoria"}
    if a is False:
        return {"leitura": "nao sustentada",
                "texto": ("a H1a nao foi sustentada; a H1b sai como exploratoria, sem "
                          "rotulo de sustentada ou nao sustentada")}
    if b is True:
        return {"leitura": "sustentada",
                "texto": "o Personalizado venceu o Whisper livre (H1a) e o "
                         "Entre-pessoas igualado (H1b)" + extra}
    if b is False:
        return {"leitura": "sustentada em parte (contra o Generico)",
                "texto": FORMULACAO_H1A_SIM_H1B_NAO + extra}
    return {"leitura": "H1a sustentada; H1b sem veredito",
            "texto": "a H1a foi sustentada e a H1b ficou sem veredito ({}){}".format(
                h1b.get("motivo"), extra)}


def conferir_constantes_da_h1(chaves):
    """Recusa a análise se as chaves da H1 discordam das constantes que a executam."""
    if float(chaves["h1_alfa_unilateral"]) != ALFA_DA_H1:
        raise ValueError("h1_alfa_unilateral = {} e ALFA_DA_H1 = {}: o nivel da H1 e' "
                         "constante; a chave so' viaja com o hash".format(
                             chaves["h1_alfa_unilateral"], ALFA_DA_H1))
    if int(chaves["h1_n_minimo_para_veredito"]) != N_MINIMO_DA_H1:
        raise ValueError("h1_n_minimo_para_veredito = {} e N_MINIMO_DA_H1 = {}".format(
            chaves["h1_n_minimo_para_veredito"], N_MINIMO_DA_H1))
    if int(chaves["h1b_igualado_semente"]) != SEMENTE:
        raise ValueError("h1b_igualado_semente = {} e a semente do estudo e' {}".format(
            chaves["h1b_igualado_semente"], SEMENTE))


def conferir_whisper_forcado_no_config(chaves, config, modo):
    """Confere `condicoes.whisper_forcado.ligado` do config contra a chave congelada."""
    if config is None:
        return {"config": None, "chave_congelada": bool(chaves["degrau_2_whisper_forcado_ligado"]),
                "concordam": None,
                "motivo": "config.yaml ausente ao lado do script: nao conferido"}
    ligado = bool((((config or {}).get("condicoes") or {}).get("whisper_forcado") or {})
                  .get("ligado", False))
    esperado = bool(chaves["degrau_2_whisper_forcado_ligado"])
    bloco = {"config": ligado, "chave_congelada": esperado, "concordam": ligado == esperado}
    if ligado != esperado and modo in ("estudo", "em_verificacao"):
        raise ValueError(
            "condicoes.whisper_forcado.ligado = {} no config.yaml, mas a chave congelada "
            "degrau_2_whisper_forcado_ligado = {}: o degrau 2 da H1a nao rodaria como "
            "registrado".format(ligado, esperado))
    return bloco


def conferir_regras_de_06_10(chaves):
    """Recusa a análise se uma chave das decisões de 06/10/2026 (18h14 a 18h21) pede uma regra que o código não implementa."""
    regras = dict(REGRAS_DE_06_10, **REGRAS_DA_NOITE_DE_06_10)
    diferentes = sorted(k for k, v in regras.items() if chaves.get(k) != v)
    if diferentes:
        raise ValueError(
            "chave(s) {} com valor que o programa nao implementa ({}); a unica "
            "implementacao e' {}".format(
                ", ".join(diferentes),
                "; ".join("{}={!r}".format(k, chaves.get(k)) for k in diferentes),
                "; ".join("{}={!r}".format(k, regras[k]) for k in diferentes)))


def conferir_grafias_no_config(chaves, config, modo):
    """Confere as grafias e as 24 sequências literais da escolha forçada."""
    esperadas = [g for g in str(chaves["degrau_2_grafias"]).split(",") if g]
    bloco = {"grafias_congeladas": esperadas,
             "agregacao": chaves["degrau_2_agregacao_das_grafias"],
             "sha256_congelado": chaves["degrau_2_sha256_das_sequencias"]}
    if config is None:
        return dict(bloco, concordam=None,
                    motivo="config.yaml ausente ao lado do script: nao conferido")
    from src.escolha_forcada import sha256_das_sequencias
    opcoes = ((config.get("condicoes") or {}).get("whisper_forcado") or {})
    grafias = list(opcoes.get("grafias") or [])
    try:
        sha, n = sha256_das_sequencias(config.get("frases_alvo") or {}, grafias)
    except ValueError as erro:
        sha, n = None, 0
        bloco["erro"] = str(erro)
    concordam = grafias == esperadas and sha == bloco["sha256_congelado"]
    bloco.update(grafias_no_config=grafias, sha256_do_config=sha, n_sequencias=n,
                 concordam=concordam)
    if not concordam and modo in ("estudo", "em_verificacao"):
        raise ValueError(
            "as sequencias literais da escolha forcada no config.yaml (grafias {}, "
            "sha256 {}) nao sao as congeladas (grafias {}, sha256 {}): o degrau 2 da "
            "H1a nao pontuaria os textos registrados".format(
                grafias, sha, esperadas, bloco["sha256_congelado"]))
    return bloco


def leitura_da_h1_com_a_h1b_descritiva(v_a):
    """Leitura legada do bloco `h1`, preservada para compatibilidade."""
    return {"leitura": None,
            "parte_confirmatoria": "H1a",
            "veredito_da_parte_confirmatoria": v_a.get("veredito"),
            "motivo": ("este bloco legado nao le a H1 inteira: o bloco h1.H1b traz "
                       "so' a parte descritiva do Entre-pessoas completo; a H1a e a H1b (contra o "
                       "Entre-pessoas igualado) tem vereditos proprios em h1a e h1b, "
                       "e a leitura das duas esta' em "
                       "h1_sequencia_confirmatoria.leitura (plano de analise)")}


def participantes_no_grupo(por_part, minimo):
    """(dentro, fora): quem entra no resultado do grupo pelo número de frases."""
    if not (isinstance(minimo, int) and minimo >= 1):
        raise ValueError("o minimo de frases do grupo tem de ser inteiro >= 1; "
                         "veio {!r}".format(minimo))
    dentro, fora = [], {}
    for pessoa in sorted(por_part):
        n = int(por_part[pessoa]["n_frases"])
        if n >= minimo:
            dentro.append(pessoa)
        else:
            fora[pessoa] = {
                "n_frases": n,
                "motivo": ("{} frase(s) avaliavel(is), abaixo do minimo de {} "
                           "para entrar no resultado do grupo; continua "
                           "descrito individualmente".format(n, minimo))}
    return dentro, fora


def resultado_do_grupo_da_h1(por_part, chaves):
    """A H1 do grupo com o mínimo de frases, e a sensibilidade com seis ou mais."""
    minimo = chaves["min_frases_no_grupo"]
    dentro, fora = participantes_no_grupo(por_part, minimo)
    h1 = veredito_da_h1ab({p: por_part[p] for p in dentro}, chaves)
    fixas = h1a_frases_fixas({p: por_part[p] for p in dentro})
    dentro_s, fora_s = participantes_no_grupo(
        por_part, max(minimo, MIN_FRASES_DA_SENSIBILIDADE_DO_GRUPO))
    fixas_s = h1a_frases_fixas({p: por_part[p] for p in dentro_s})
    fixas_s["papel"] = ("SENSIBILIDADE -- nao decide: o teste das frases fixas "
                        "refeito so' com quem tem {} ou mais frases "
                        "avaliaveis".format(MIN_FRASES_DA_SENSIBILIDADE_DO_GRUPO))
    sensibilidade = {
        "papel": ("SENSIBILIDADE -- nao decide: o resultado do grupo refeito so' "
                  "com quem tem {} ou mais frases avaliaveis".format(
                      MIN_FRASES_DA_SENSIBILIDADE_DO_GRUPO)),
        "minimo_de_frases": MIN_FRASES_DA_SENSIBILIDADE_DO_GRUPO,
        "participantes": dentro_s,
        "fora": fora_s,
        "h1": veredito_da_h1ab({p: por_part[p] for p in dentro_s}, chaves),
        "h1a_frases_fixas": fixas_s,
    }
    return h1, {"minimo_de_frases": minimo, "participantes": dentro,
                "fora": fora, "h1a_frases_fixas": fixas,
                "sensibilidade_seis_frases_ou_mais": sensibilidade,
                "fonte": "plano de analise"}


def veredito_da_h1ab(por_part, chaves):
    """Bloco legado da H1a e resumo descritivo da H1b."""
    saida = {"H1a": dict(veredito_da_h1_com_sensibilidade(
                             por_part, chaves, CONTRASTES_DA_H1["H1a"]["bloco"]),
                         hipotese=CONTRASTES_DA_H1["H1a"]["nome"]),
             "H1b": descricao_da_h1b(por_part)}
    saida["leitura_combinada"] = leitura_da_h1_com_a_h1b_descritiva(saida["H1a"])
    saida["fonte"] = ("H1a pela contagem k=3 + trava (decide so' com "
                      "alcance_da_h1a = frases_como_amostra); h1.H1b e' o bloco "
                      "legado do Entre-pessoas completo (descritivo), sem p, sem "
                      "intervalo e sem veredito; a H1b (contra o Entre-pessoas "
                      "igualado) esta' no bloco h1b")
    return saida


def contagem_k3_so_descritiva(bloco):
    """A contagem k=3 sem palavras de veredito, para quando ela não decide."""
    novo = {k: v for k, v in bloco.items() if k not in ("veredito", "resultado")}
    atingida = bloco.get("veredito")
    if atingida is None:
        leitura = "sem contagem: " + str(bloco.get("motivo"))
    elif atingida:
        leitura = ("contagem e trava atingidas (descritiva; nao decide a H1a)")
    else:
        leitura = ("contagem ou trava nao atingida (descritiva; nao decide a H1a)")
    novo["contagem_k3_descritiva"] = {"atingida": atingida, "leitura": leitura}
    return novo


TOLERANCIA_DO_EMPATE = 1e-14


def permutacao_de_sinais(d, unicaudal=True, replicas=0, semente=SEMENTE):
    """p exato por permutação de sinais quando n <= 20; amostrado acima."""
    d = np.asarray([x for x in d if np.isfinite(x)])
    n = len(d)
    if n == 0:
        return {"n": 0, "p": None, "erro": "sem observacoes"}
    obs = float(np.mean(d))
    if n <= 20 and not replicas:
        sinais = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n)
        medias = (sinais * np.abs(d)).mean(axis=1)
        exato = True
    else:
        rng = np.random.default_rng(semente)
        r = replicas or 10000
        sinais = rng.choice([-1, 1], size=(r, n))
        medias = (sinais * np.abs(d)).mean(axis=1)
        exato = False
    tol = TOLERANCIA_DO_EMPATE
    p = (float(np.mean(medias >= obs - tol)) if unicaudal
         else float(np.mean(np.abs(medias) >= abs(obs) - tol)))
    return {"n": n, "media": obs, "p": p, "exato": exato,
            "p_minimo": 1.0 / 2 ** n if exato else 1.0 / len(medias)}


def holm(ps, modo="simetrico", alfa=0.05):
    """(rejeitadas, limiares) de Holm, nos modos simétrico e hierárquico."""
    nomes = list(ps)
    if modo == "hierarquico":
        rej, limiar = {}, {}
        passou = True
        for nome in nomes:
            limiar[nome] = alfa
            rej[nome] = passou and ps[nome] is not None and ps[nome] < alfa
            passou = rej[nome]
        return rej, limiar
    ordem = sorted(nomes, key=lambda k: (ps[k] is None, ps[k]))
    rej, limiar = {}, {}
    passou = True
    for i, nome in enumerate(ordem):
        limiar[nome] = alfa / (len(ordem) - i)
        desta = ps[nome] is not None and ps[nome] < limiar[nome]
        rej[nome] = passou and desta
        passou = rej[nome]
    return rej, limiar


PAPEL_ENTRE_PARTICIPANTES = (
    "EXPLORATORIO -- nao decide a H1. A H1a e' decidida pelo teste das frases "
    "fixas ou pela regra k=3, conforme a chave alcance_da_h1a; a H1b "
    "(Personalizado x Entre-pessoas igualado) esta' no bloco h1b, com veredito "
    "proprio quando a H1a e' sustentada (plano de analise). Os p abaixo sao "
    "brutos: sem Holm e sem alfa alternativo, porque o plano de analise nao declara "
    "correcao de multiplicidade. Abaixo de 8 participantes, so' descricao "
    "(plano de analise).")


ROTULOS_EXPLORATORIOS = (("exploratoria_PxG", "duar_pg"),)


def entre_participantes(por_part, chaves):
    """Permutação de sinais dos contrastes PxG e PxE entre participantes."""
    saida = {"papel": PAPEL_ENTRE_PARTICIPANTES, "n": len(por_part)}
    if len(por_part) < MINIMO_DE_PARTICIPANTES:
        saida["abaixo_do_minimo"] = (
            "{} participante(s), abaixo do minimo de {}: esta analise "
            "fica so' descritiva; os p abaixo nao sao lidos como teste".format(
                len(por_part), MINIMO_DE_PARTICIPANTES))
    for rotulo, campo in ROTULOS_EXPLORATORIOS:
        d = [r[campo] for r in por_part.values() if np.isfinite(r.get(campo, np.nan))]
        saida[rotulo] = {
            "permutacao_de_sinais": permutacao_de_sinais(
                d, replicas=chaves["replicas_permutacao"]),
            "t_pareado": (float(stats.ttest_1samp(d, 0, alternative="greater").pvalue)
                          if len(d) > 1 else None),
            "wilcoxon": (float(stats.wilcoxon(d, alternative="greater").pvalue)
                         if len(d) > 1 and any(x != 0 for x in d) else None),
            "papel_de_cada_p": {
                "permutacao_de_sinais": SUPOSICAO_DA_PERMUTACAO,
                "wilcoxon": SUPOSICAO_DO_WILCOXON,
                "t_pareado": ("o mesmo t de uma amostra, sobre o ganho de UAR de "
                              "todos os participantes do grupo confirmatorio, sem o "
                              "minimo de frases do grupo; nao e' o teste que decide")},
        }
    d_pe = [r["duar_pe"] for r in por_part.values()
            if np.isfinite(r.get("duar_pe", np.nan))]
    saida["descritivo_PxE"] = {
        "papel": MOTIVO_DA_H1B_DESCRITIVA,
        "n": len(d_pe),
        "media": float(np.mean(d_pe)) if d_pe else None,
        "mediana": float(np.median(d_pe)) if d_pe else None,
    }
    return saida


PAPEL_DOS_TESTES_DE_APOIO = (
    "PRE-ESPECIFICADOS, NAO DECIDEM: o veredito da H1a sai so' do teste t do degrau 1. "
    "Estes valores p usam os mesmos ganhos e o mesmo nivel so' para mostrar se "
    "concordam; nenhum muda o veredito nem resgata um resultado negativo, e a "
    "divergencia e' mostrada (decisao de 06/10/2026, 18h14 a 18h21)")
SUPOSICAO_DA_PERMUTACAO = (
    "SENSIBILIDADE: relaxa a suposicao de que a media dos ganhos tem distribuicao "
    "aproximadamente normal (a do teste t); mantem a suposicao de que, sem ganho, a "
    "distribuicao dos ganhos e' simetrica em torno de zero; testa a mesma quantidade "
    "(a media dos ganhos)")
SUPOSICAO_DO_WILCOXON = (
    "APOIO DE ROBUSTEZ: o teste dos postos sinalizados de Wilcoxon testa outra "
    "quantidade (a pseudomediana dos ganhos, nao a media); usa os postos e tambem "
    "supoe simetria; nao substitui o teste principal")


def apoio_do_degrau_1(ganho_por_participante, teste_principal, alfa=ALFA_DA_H1):
    """Permutação de sinais e Wilcoxon sobre os ganhos do degrau 1, sem decidir."""
    d = [float(g) for _p, g in sorted(ganho_por_participante.items())
         if g is not None and np.isfinite(g)]
    p_principal = (teste_principal or {}).get("p")
    base = {"papel": PAPEL_DOS_TESTES_DE_APOIO, "n": len(d), "alfa_unilateral": alfa,
            "p_do_teste_principal": p_principal}
    if p_principal is None or len(d) < N_MINIMO_DA_H1:
        return dict(base, calculado=False, motivo=(
            "sem o teste principal do degrau 1 (ou com menos de {} ganhos): nao ha o "
            "que conferir".format(N_MINIMO_DA_H1)))
    p_principal = float(p_principal)
    principal_acusa = bool(p_principal <= alfa)
    perm = permutacao_de_sinais(d)
    wil = (float(stats.wilcoxon(d, alternative="greater").pvalue)
           if any(x != 0 for x in d) else None)
    saida = dict(base, calculado=True, principal_acusa_ganho=principal_acusa)
    for nome, p, papel in (("permutacao_de_sinais", perm.get("p"), SUPOSICAO_DA_PERMUTACAO),
                           ("wilcoxon", wil, SUPOSICAO_DO_WILCOXON)):
        if p is None:
            saida[nome] = {"papel": papel, "p": None,
                           "motivo": "todos os ganhos sao zero: sem postos"}
            continue
        p = float(p)
        acusa = bool(p <= alfa)
        saida[nome] = {"papel": papel, "p": p, "acusaria_ganho_a_este_nivel": acusa,
                       "concorda_com_o_principal": bool(acusa == principal_acusa),
                       "leitura": ("concorda com o teste principal" if acusa == principal_acusa
                                   else "DIVERGE do teste principal ({} aqui, {} no "
                                        "principal); o veredito continua o do teste "
                                        "principal".format(
                                            "acusaria" if acusa else "nao acusaria",
                                            "acusa" if principal_acusa else "nao acusa"))}
        if nome == "permutacao_de_sinais":
            saida[nome].update(exato=perm.get("exato"), p_minimo=perm.get("p_minimo"))
    saida["divergencias"] = [n for n in ("permutacao_de_sinais", "wilcoxon")
                             if saida[n].get("concorda_com_o_principal") is False]
    saida["nao_altera_o_veredito"] = True
    return saida


ALFA_DA_H1A = ALFA_DA_H1

ALCANCE_DAS_FRASES_FIXAS = (
    "estas frases: a conclusão vale para as frases do banco, com o mesmo peso "
    "para cada uma, nesta tarefa, nesta população")
LIMITACAO_DAS_FRASES_FIXAS = (
    "as frases são tratadas como fixas: a incerteza é só a de quais "
    "participantes; se o resultado for lido como válido para outras frases, o "
    "erro tipo I deixa de ser o nominal (para essa leitura, ver o ganho médio "
    "por dois fatores)")


def ganho_frases_fixas(D, alfa=ALFA_DA_H1A):
    """t de uma amostra, unilateral (ganho > 0), sobre o ganho de cada participante."""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2:
        raise ValueError("a tabela das diferenças tem de ser participante x frase")
    presente = np.isfinite(D)
    com_dado = presente.any(axis=1)
    ganhos = np.array([float(D[i, presente[i]].mean())
                       for i in range(D.shape[0]) if com_dado[i]])
    frases = [int(presente[i].sum()) for i in range(D.shape[0]) if com_dado[i]]
    N = len(ganhos)
    base = {"n_participantes": N, "gl": None, "ganho": None, "dp": None, "ep": None,
            "t": None, "p": None, "intervalo_do_teste": None, "intervalo_95": None,
            "alfa": alfa,
            "ganhos_por_participante": [float(g) for g in ganhos],
            "frases_por_participante": frases,
            "sem_frase_nenhuma": int((~com_dado).sum())}
    if N < 2:
        return dict(base, status="nao_calculavel",
                    ganho=float(ganhos[0]) if N == 1 else None,
                    motivo="não calculável: o teste entre participantes exige pelo "
                           "menos 2 participantes com dado (havia {})".format(N))
    gl = N - 1
    media = float(ganhos.mean())
    dp = float(ganhos.std(ddof=1))
    base.update(ganho=media, dp=dp, gl=gl)
    if dp <= TOLERANCIA_DEGENERADO:
        return dict(base, status="degenerado",
                    motivo="degenerado: os {} participantes têm o mesmo ganho ({:+.4f}); "
                           "sem variação entre participantes o t não existe".format(
                               N, media))
    ep = dp / np.sqrt(N)
    t = media / ep
    p = float(stats.t.sf(t, gl))
    q = float(stats.t.ppf(1 - alfa, gl))
    return dict(base, status="passou" if p <= alfa else "negativo", ep=float(ep),
                t=float(t), p=p, intervalo_do_teste=_intervalo_do_teste(media, ep, q, alfa),
                intervalo_95=_intervalo_de_95(media, ep, gl), motivo="")


def _intervalo_do_teste(media, ep, q, alfa):
    """O intervalo bilateral de nível 1 - 2·alfa, com o rótulo que diz o papel dele."""
    return {"nivel": round(1 - 2 * alfa, 10),
            "limites": [media - q * ep, media + q * ep],
            "rotulo": "intervalo do teste (equivale ao limite unilateral de "
                      "{:.0f}%)".format(100 * (1 - alfa))}


NIVEL_DO_INTERVALO_AO_LADO = 0.95


def _intervalo_de_95(media, ep, gl):
    q = float(stats.t.ppf(1 - (1 - NIVEL_DO_INTERVALO_AO_LADO) / 2, gl))
    return {"nivel": NIVEL_DO_INTERVALO_AO_LADO,
            "limites": [media - q * ep, media + q * ep],
            "rotulo": ("intervalo de 95% (relatado ao lado; nao decide: a regra e' a do "
                       "intervalo de 92%)")}


def h1a_frases_fixas(por_part, n_minimo=N_MINIMO_DA_H1, alfa=ALFA_DA_H1A):
    """A H1a com as frases fixas, sobre o grupo confirmatório já filtrado."""
    bloco = CONTRASTES_DA_H1["H1a"]["bloco"]
    pessoas, frases, D = _tabela_do_contraste(por_part, bloco)
    fora = {}
    for p in sorted(set(por_part) - set(pessoas)):
        b = por_part[p].get(bloco)
        fora[p] = ("sem o contraste {}: {}".format(bloco, b["erro"])
                   if isinstance(b, dict) and b.get("erro")
                   else "sem o contraste {} (nenhuma frase com par)".format(bloco))
    ausentes = {p: sorted(set(frases) - set(por_part[p][bloco]["delta_por_frase"]))
                for p in pessoas}
    r, decisao = _decidir_pelas_frases_fixas(D, n_minimo=n_minimo, alfa=alfa)
    saida = {"procedimento": "frases_fixas_peso_igual",
             "hipotese": CONTRASTES_DA_H1["H1a"]["nome"],
             "contraste": bloco,
             "alcance": ALCANCE_DAS_FRASES_FIXAS,
             "limitacao": LIMITACAO_DAS_FRASES_FIXAS,
             "alfa_unilateral": alfa,
             "n_minimo_para_veredito": n_minimo,
             "n_participantes": r["n_participantes"],
             "definicao_de_n": (
                 "participantes elegíveis: os do grupo confirmatório com o "
                 "mínimo de frases avaliáveis do grupo (aplicado antes, em "
                 "resultado_do_grupo_da_h1) e com o contraste (pelo menos uma "
                 "frase com par nas duas condições)"),
             "participantes_usados": pessoas,
             "fora_e_por_que": fora,
             "frases_no_grupo": frases,
             "frases_ausentes_por_participante": {p: f for p, f in ausentes.items() if f},
             "regra_das_frases_ausentes": (
                 "o ganho de cada participante é a média das frases que ele tem, "
                 "com o mesmo peso para cada uma; nada é imputado"),
             "ganho_por_participante": dict(zip(pessoas, r["ganhos_por_participante"]))}
    return dict(saida, **decisao)


def _decidir_pelas_frases_fixas(D, n_minimo=N_MINIMO_DA_H1, alfa=ALFA_DA_H1A):
    """(teste, decisão) do teste das frases fixas sobre a tabela participante x frase."""
    r = ganho_frases_fixas(D, alfa=alfa)
    if r["n_participantes"] < n_minimo or r["status"] == "nao_calculavel":
        n = r["n_participantes"]
        return r, dict(status="sem_veredito", veredito=None,
                       resultado="sem veredito", ganho_observado=r["ganho"],
                       motivo="sem veredito: {} participante(s) elegível(is) na análise, "
                              "menos que {}; sai só a descrição (ganho observado e ganho "
                              "de cada participante), sem teste, sem valor p e sem "
                              "intervalo".format(n, max(n_minimo, 2)))
    if r["status"] == "degenerado":
        return r, dict(status="degenerado", veredito=None,
                       resultado="sem veredito", ganho_observado=r["ganho"], teste=r,
                       motivo="sem veredito: " + r["motivo"])
    veredito = r["status"] == "passou"
    return r, dict(status=r["status"], veredito=veredito,
                   resultado="sustentada" if veredito else "nao sustentada",
                   teste=r, motivo="")


TOLERANCIA_DEGENERADO = 1e-9


def _pesos_iguais(presente):
    """w_if = 1/(N·F_i): cada frase pesa igual dentro da pessoa, cada pessoa no grupo."""
    por_pessoa = presente.sum(axis=1)
    n = int((por_pessoa > 0).sum())
    w = np.zeros(presente.shape)
    for i in range(presente.shape[0]):
        if por_pessoa[i] > 0:
            w[i, presente[i]] = 1.0 / (n * por_pessoa[i])
    return w


def _anova_completa(D):
    N, F = D.shape
    m = float(D.mean())
    rp, cf = D.mean(1), D.mean(0)
    qm_p = F * float(np.sum((rp - m) ** 2)) / (N - 1)
    qm_f = N * float(np.sum((cf - m) ** 2)) / (F - 1)
    gl_pf = (N - 1) * (F - 1)
    qm_pf = float(np.sum((D - rp[:, None] - cf[None, :] + m) ** 2)) / gl_pf
    s2_p = max(0.0, (qm_p - qm_pf) / F)
    s2_f = max(0.0, (qm_f - qm_pf) / N)
    var = s2_p / N + s2_f / F + qm_pf / (N * F)
    c_p = 1.0 if s2_p > 0 else 0.0
    c_f = 1.0 if s2_f > 0 else 0.0
    c_pf = 1.0 - c_p - c_f
    den = sum(x * x / g for x, g in ((c_p * qm_p, N - 1), (c_f * qm_f, F - 1),
                                     (c_pf * qm_pf, gl_pf))) / (N * F) ** 2
    gl = var * var / den if den > 0 else float("nan")
    return var, gl, {"s2_p": s2_p, "s2_f": s2_f, "s2_pf": qm_pf}


def _reml_com_ausencia(D, presente, w):
    """REML dos três componentes com células ausentes; var(w'y) e gl."""
    from scipy import optimize
    ii, ff = np.nonzero(presente)
    y = D[ii, ff]
    n = len(y)
    Zp = np.zeros((n, D.shape[0]))
    Zp[np.arange(n), ii] = 1.0
    Zf = np.zeros((n, D.shape[1]))
    Zf[np.arange(n), ff] = 1.0
    G = [Zp @ Zp.T, Zf @ Zf.T, np.eye(n)]
    X = np.ones((n, 1))
    wv = w[ii, ff]

    def partes(th):
        V = th[0] * G[0] + th[1] * G[1] + th[2] * G[2]
        Vi = np.linalg.inv(V)
        XtViX = X.T @ Vi @ X
        P = Vi - Vi @ X @ np.linalg.inv(XtViX) @ X.T @ Vi
        return V, Vi, XtViX, P

    def menos2ll(th):
        """−2 log-verossimilhança REML e o gradiente analítico tr(PGₖ) − y'PGₖPy."""
        try:
            with np.errstate(all="ignore"):
                V, Vi, XtViX, P = partes(th)
                s1, ld1 = np.linalg.slogdet(V)
                s2, ld2 = np.linalg.slogdet(XtViX)
            if s1 <= 0 or s2 <= 0 or not np.isfinite(ld1 + ld2):
                return 1e12, np.zeros(3)
            Py = P @ y
            grad = np.array([np.sum(P * Gk) - Py @ Gk @ Py for Gk in G])
            return float(ld1 + ld2 + y @ Py), grad
        except np.linalg.LinAlgError:
            return 1e12, np.zeros(3)

    escala = max(float(np.var(y)), 1e-6)
    inicio = np.array([escala / 3] * 3)
    res = optimize.minimize(menos2ll, inicio, jac=True, method="L-BFGS-B",
                            bounds=[(0.0, None), (0.0, None), (escala * 1e-8, None)])
    th = res.x
    V, Vi, XtViX, P = partes(th)
    var = float(wv @ V @ wv)
    livres = [k for k in range(3) if th[k] > escala * 1e-6]
    info = np.array([[0.5 * np.trace(P @ G[a] @ P @ G[b]) for b in livres] for a in livres])
    g = np.array([float(wv @ G[k] @ wv) for k in livres])
    try:
        cov = np.linalg.inv(info)
        den = float(g @ cov @ g)
        gl = 2 * var * var / den if den > 0 else float("nan")
    except np.linalg.LinAlgError:
        gl = float("nan")
    return var, gl, {"s2_p": float(th[0]), "s2_f": float(th[1]), "s2_pf": float(th[2]),
                     "reml_convergiu": bool(res.success)}


def ganho_dois_fatores(D, presente=None, alfa=0.05):
    """O ganho médio de peso igual e o teste unilateral (ganho > 0), com as quatro saídas."""
    D = np.asarray(D, dtype=float)
    if presente is None:
        presente = np.isfinite(D)
    presente = np.asarray(presente, dtype=bool) & np.isfinite(D)
    linhas = presente.any(axis=1)
    colunas = presente.any(axis=0)
    D, presente = D[linhas][:, colunas], presente[linhas][:, colunas]
    N, F = presente.shape
    base = {"n_participantes": int(N), "n_frases": int(F),
            "celulas": int(presente.sum()), "celulas_ausentes": int(N * F - presente.sum()),
            "ganho": None, "ep": None, "gl": None, "p": None,
            "ic_inferior": None, "ic95": None, "componentes": None, "metodo": None}
    if N < 2 or F < 2:
        return dict(base, status="nao_calculavel",
                    motivo="são precisos pelo menos 2 participantes e 2 frases com dado "
                           "(havia {} e {})".format(N, F))
    w = _pesos_iguais(presente)
    ganho = float(np.nansum(np.where(presente, D, 0.0) * w))
    base["ganho"] = ganho
    valores = D[presente]
    if float(np.ptp(valores)) <= TOLERANCIA_DEGENERADO:
        return dict(base, status="degenerado",
                    motivo="todas as diferenças são iguais: não há variação para medir")
    if presente.all():
        var, gl, comp = _anova_completa(D)
        metodo = "anova_dois_fatores_completa"
    else:
        var, gl, comp = _reml_com_ausencia(np.where(presente, D, 0.0), presente, w)
        metodo = "reml_dois_fatores_com_ausencia"
    base.update(componentes=comp, metodo=metodo)
    if not (np.isfinite(var) and var > 0 and np.isfinite(gl) and gl > 0):
        return dict(base, status="nao_calculavel",
                    motivo="a variância ou os graus de liberdade não saíram ({}, {})".format(
                        var, gl))
    ep = float(np.sqrt(var))
    t = ganho / ep
    p = float(stats.t.sf(t, gl))
    q1 = float(stats.t.ppf(1 - alfa, gl))
    q2 = float(stats.t.ppf(0.975, gl))
    return dict(base, status="passou" if p <= alfa else "negativo", ep=ep, gl=float(gl),
                p=p, ic_inferior=ganho - q1 * ep, ic95=[ganho - q2 * ep, ganho + q2 * ep],
                motivo="")


ALCANCE_DA_H1 = ("frases desse tipo, nesta tarefa, nesta população")
LIMITACAO_DAS_FRASES = ("as frases foram escolhidas por consulta, não sorteadas: nenhum "
                        "método transforma isso em amostra aleatória, e a generalização "
                        "vale para frases como estas")
LIMITACAO_DA_H1B = (
    "Entre-pessoas completo (descritivo; não é a H1b, que é contra o igualado): o "
    "Entre-pessoas de cada participante é treinado nos mesmos colegas; neste bloco "
    "legado não há p-valor nem intervalo, só as diferenças observadas, como "
    "descrição, e elas valem para ESTE Entre-pessoas")


def _tabela_do_contraste(por_part, bloco):
    """(pessoas, frases, D com NaN nas frases que a pessoa não tem)."""
    pessoas = sorted(p for p, r in por_part.items()
                     if isinstance(r.get(bloco), dict) and r[bloco].get("delta_por_frase"))
    frases = sorted({f for p in pessoas for f in por_part[p][bloco]["delta_por_frase"]})
    D = np.full((len(pessoas), len(frases)), np.nan)
    for i, p in enumerate(pessoas):
        for j, f in enumerate(frases):
            v = por_part[p][bloco]["delta_por_frase"].get(f)
            if v is not None:
                D[i, j] = v
    return pessoas, frases, D


def h1_dois_fatores(por_part, chaves, verif, alfa=0.05):
    """O candidato em verificação para a H1, ganho médio por dois fatores (não decide o estudo)."""
    saida = {"papel": ("CANDIDATO EM VERIFICAÇÃO (método "
                       "implementado para verificação, ainda não definitivo). "
                       "Não decide o estudo."),
             "alcance": ALCANCE_DA_H1, "limitacao_das_frases": LIMITACAO_DAS_FRASES,
             "em_verificacao": dict(verif)}
    ps = {}
    for h in ("H1a", "H1b"):
        bloco = CONTRASTES_DA_H1[h]["bloco"]
        pessoas, frases, D = _tabela_do_contraste(por_part, bloco)
        r = ganho_dois_fatores(D, alfa=alfa)
        k3 = veredito_da_h1(por_part, chaves, bloco)
        acima = k3.get("acima_de_zero") or 0
        abaixo = k3.get("abaixo_de_zero") or 0
        n_com_ic = k3.get("n_participantes") or 0
        bloco_h = {
            "contraste": CONTRASTES_DA_H1[h]["nome"],
            "frases_por_participante": {
                p: sorted(f for f in por_part[p][bloco]["delta_por_frase"]) for p in pessoas},
            "individual_descritivo": {
                "ganhos": acima, "pioras": abaixo,
                "inconclusivos": max(0, n_com_ic - acima - abaixo),
                "fonte": "intervalo de cada pessoa pelo metodo_do_intervalo ({}); a "
                         "regra k=3 não decide aqui".format(
                             chaves.get("metodo_do_intervalo"))},
        }
        confirmatoria = h == "H1a" or verif["h1b_confirmatoria"]
        if not confirmatoria:
            bloco_h.update(status="descritiva", ganho_observado=r["ganho"],
                           diferencas_por_participante={
                               p: float(np.nanmean(D[i])) for i, p in enumerate(pessoas)},
                           limitacao=LIMITACAO_DA_H1B)
        elif r["n_participantes"] < verif["n_minimo_dois_fatores"]:
            bloco_h.update(status="abaixo_do_n_minimo", ganho_observado=r["ganho"],
                           motivo="{} participante(s); o candidato só decide com {} ou "
                                  "mais".format(r["n_participantes"],
                                                         verif["n_minimo_dois_fatores"]))
        else:
            bloco_h.update(teste=r, status=r["status"])
            if r["status"] in ("passou", "negativo"):
                ps[h] = r["p"]
        saida[h] = bloco_h
    if verif["h1b_confirmatoria"] and verif["holm_h1a_h1b"]:
        rej, limiares = holm({h: ps.get(h) for h in ("H1a", "H1b")}, alfa=alfa)
        saida["multiplicidade"] = {"metodo": "Holm, erro familiar {}".format(alfa),
                                   "limiares": limiares, "rejeitadas": rej}
    else:
        rej = {"H1a": "H1a" in ps and ps["H1a"] <= alfa, "H1b": None}
        saida["multiplicidade"] = {
            "metodo": ("nenhuma: a H1b não é confirmatória, então a família tem uma "
                       "hipótese só e a H1a usa alfa {}".format(alfa)),
            "rejeitadas": rej}
    for h in ("H1a", "H1b"):
        b = saida[h]
        if b["status"] in ("passou", "negativo"):
            b["sustentada"] = bool(rej.get(h))
    return saida


BLOCOS_COM_P_EXPLORATORIO = ("entre_participantes", "h2")


def _e_campo_de_p(chave, valor):
    """Campo que é um valor p."""
    if chave in ("p", "t_pareado", "wilcoxon"):
        return True
    return chave.startswith("p_") and (
        valor is None or isinstance(valor, (int, float)))


def _retirar_p(no):
    """Tira do `no` (in place) todo campo de p e devolve o que tirou, com o mesmo aninhamento."""
    if isinstance(no, dict):
        tirado = {}
        for chave in list(no):
            if _e_campo_de_p(chave, no[chave]):
                tirado[chave] = no.pop(chave)
            else:
                sub = _retirar_p(no[chave])
                if sub is not None:
                    tirado[chave] = sub
        return tirado or None
    if isinstance(no, list):
        subs = [_retirar_p(x) for x in no]
        return subs if any(x is not None for x in subs) else None
    return None


def mover_p_para_auditoria(resultado, n_confirmatorio):
    """Com N < 8, move os p exploratórios para `resultado["auditoria"]`."""
    if n_confirmatorio >= MINIMO_DE_PARTICIPANTES:
        return resultado
    auditoria = {
        "papel": ("AUDITORIA -- {} participante(s) no grupo confirmatorio, "
                  "abaixo do minimo de {}: a analise entre "
                  "participantes e a H2 sao so' descritivas, e os p abaixo "
                  "NAO sao relatados nem lidos como teste; ficam aqui so' "
                  "para conferencia".format(
                      n_confirmatorio, MINIMO_DE_PARTICIPANTES)),
    }
    for bloco in BLOCOS_COM_P_EXPLORATORIO:
        tirado = _retirar_p(resultado.get(bloco))
        if tirado is not None:
            auditoria[bloco] = tirado
            resultado[bloco]["onde_estao_os_p"] = (
                "N < {}: os p deste bloco estao em `auditoria.{}`"
                .format(MINIMO_DE_PARTICIPANTES, bloco))
    resultado["auditoria"] = auditoria
    return resultado


def mover_p_exploratorio_da_h1(resultado, n_confirmatorio):
    """Com N < 8, move para `resultado["auditoria"]` os p exploratórios da H1."""
    if n_confirmatorio >= MINIMO_DE_PARTICIPANTES:
        return resultado
    blocos = ["entre_pessoas_completo_exploratorio"]
    if (resultado.get("h1b") or {}).get("papel") == "exploratoria":
        blocos.append("h1b")
    auditoria = resultado.setdefault("auditoria", {
        "papel": ("AUDITORIA -- {} participante(s) no grupo confirmatorio, abaixo do "
                  "minimo de {}: os p abaixo NAO sao relatados nem lidos como teste; "
                  "ficam aqui so' para conferencia".format(
                      n_confirmatorio, MINIMO_DE_PARTICIPANTES))})
    for bloco in blocos:
        tirado = _retirar_p(resultado.get(bloco))
        if tirado is not None:
            auditoria[bloco] = tirado
            resultado[bloco]["onde_estao_os_p"] = (
                "N < {}: os p exploratorios deste bloco estao em `auditoria.{}`"
                .format(MINIMO_DE_PARTICIPANTES, bloco))
    return resultado


CAMPOS_DE_TESTE_ENTRE_PARTICIPANTES_DA_H2 = ("p", "p_metodo",
                                            "p_assintotico_do_scipy")
CAMPOS_DE_TESTE_DA_PARCIAL_DA_H2 = ("p", "p_do_scipy_gl_n_menos_2")


def so_descricao_entre_participantes_da_h2(resultado):
    """Tira do corpo do `h2` os p da análise entre participantes, com qualquer N."""
    h2_ = resultado.get("h2")
    if not isinstance(h2_, dict):
        return resultado
    tirado = {c: h2_.pop(c) for c in CAMPOS_DE_TESTE_ENTRE_PARTICIPANTES_DA_H2
              if c in h2_}
    parcial = h2_.get("parcial_controlando_gravidade")
    if isinstance(parcial, dict):
        da_parcial = {c: parcial.pop(c) for c in CAMPOS_DE_TESTE_DA_PARCIAL_DA_H2
                      if c in parcial}
        if da_parcial:
            tirado["parcial_controlando_gravidade"] = da_parcial
    h2_["teste_entre_participantes"] = (
        "nenhum: a analise entre participantes e' so' descricao, com qualquer "
        "numero de participantes; os p ficam em `auditoria.h2_entre_participantes`"
        ", so' para conferencia")
    if tirado:
        auditoria = resultado.setdefault("auditoria", {
            "papel": ("AUDITORIA -- valores p que o resultado NAO relata nem "
                      "le como teste; ficam aqui so' para conferencia")})
        auditoria["h2_entre_participantes"] = tirado
    return resultado


def _p_por_permutacao(x, y, chaves, semente=None):
    """p bilateral do Spearman por permutação."""
    import itertools
    from math import factorial

    n = len(x)
    rho_obs = stats.spearmanr(x, y).statistic
    if not np.isfinite(rho_obs):
        return float("nan"), {"rota": "indefinida", "motivo": "rho nao finito"}

    TETO_EXATO = 40320
    if factorial(n) <= TETO_EXATO:
        extremos = 0
        total = 0
        for perm in itertools.permutations(range(n)):
            r = stats.spearmanr(x, y[list(perm)]).statistic
            if np.isfinite(r):
                total += 1
                extremos += abs(r) >= abs(rho_obs) - 1e-12
        return (extremos / total if total else float("nan"),
                {"rota": "exata", "ordenacoes": total,
                 "nota": "todas as {} ordenacoes foram enumeradas".format(total)})

    sorteios = int(chaves.get("replicas_permutacao_h2", 20000))
    rng = np.random.default_rng(semente if semente is not None else SEMENTE)
    extremos = 0
    for _ in range(sorteios):
        r = stats.spearmanr(x, rng.permutation(y)).statistic
        if np.isfinite(r) and abs(r) >= abs(rho_obs) - 1e-12:
            extremos += 1
    p = (1 + extremos) / (1 + sorteios)
    ep = float(np.sqrt(max(p * (1 - p), 1e-12) / sorteios))
    return p, {"rota": "monte carlo", "sorteios": sorteios,
               "erro_padrao_do_p": ep,
               "meia_largura_95": 1.96 * ep,
               "ordenacoes_possiveis": "{:.3g}".format(float(factorial(n))),
               "nota": "p = (1 + acertos) / (1 + sorteios); nunca devolve zero"}


def _residuo_efeitos_aditivos(valores, linha_de, coluna_de, n_linhas, n_colunas,
                              extra=None):
    """Resíduo de `valores` após o ajuste de mínimos quadrados de linha + coluna."""
    n = len(valores)
    X = np.zeros((n, n_linhas + n_colunas))
    X[np.arange(n), linha_de] = 1.0
    X[np.arange(n), n_linhas + coluna_de] = 1.0
    if extra is not None:
        X = np.column_stack([X, np.asarray(extra, dtype=float)])
    beta, *_ = np.linalg.lstsq(X, valores, rcond=None)
    ajustado = X @ beta
    return valores - ajustado, ajustado


def _correlacao_de_residuos(a, b):
    den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else float("nan")


def _permuta_dentro(valores, grupos, rng):
    """Embaralha `valores` só dentro de cada grupo (participante)."""
    saida = valores.copy()
    for g in grupos:
        if g.size > 1:
            saida[g] = valores[g[rng.permutation(g.size)]]
    return saida


def h2_dentro_da_pessoa(celulas, chaves, semente=None, com_covariavel=False):
    """H2 decisora: permutação de Freedman e Lane dentro da pessoa, unilateral."""
    metodo = chaves.get("metodo_da_h2")
    if metodo is None:
        return {"resultado_do_teste": None, "p": None,
                "motivo": ("metodo_da_h2 nao escolhido: o metodo exato da H2 e "
                           "a escala do acerto na celula dependem da escolha do "
                           "pesquisador responsavel (plano de analise). Opcoes: {}.".format(
                               ", ".join(METODOS_DA_H2)))}
    if metodo not in METODOS_DA_H2:
        raise ValueError("metodo_da_h2 = {!r} nao existe. Opcoes: {}".format(
            metodo, ", ".join(METODOS_DA_H2)))

    celulas = [c for c in celulas if c["gravacoes"] > 0
               and c["sti"] is not None and np.isfinite(c["sti"])]
    if com_covariavel and any(c.get("covariavel") is None
                              or not np.isfinite(c["covariavel"])
                              for c in celulas):
        raise ValueError(
            "celula sem covariavel chegou ao modelo da sensibilidade; ela devia "
            "ter sido excluida, com motivo, em `h2_sensibilidades` -- nunca "
            "imputada")
    pessoas = sorted({c["pessoa"] for c in celulas})
    frases = sorted({c["frase"] for c in celulas})
    por_pessoa = Counter(c["pessoa"] for c in celulas)
    uteis = [p for p in pessoas if por_pessoa[p] >= 2]
    base = {"metodo": metodo, "unidade": "celula participante x frase",
            "direcao": "negativa (mais inconsistencia, menor acerto)",
            "celulas": len(celulas), "participantes": len(pessoas),
            "participantes_com_2_ou_mais_celulas": len(uteis),
            "frases": len(frases)}
    if com_covariavel:
        base["covariavel_no_modelo_reduzido"] = True
    gl = len(celulas) - (len(pessoas) + len(frases) - 1) - int(com_covariavel)
    if len(uteis) < 2 or gl < 2:
        return dict(base, resultado_do_teste=None, p=None, gl_do_residuo=gl,
                    motivo=("celulas insuficientes: {} participante(s) com 2+ "
                            "celulas e {} grau(s) de liberdade no residuo; o "
                            "teste nao e' calculavel".format(len(uteis), gl)))

    li = {p: i for i, p in enumerate(pessoas)}
    co = {f: j for j, f in enumerate(frases)}
    linha_de = np.array([li[c["pessoa"]] for c in celulas])
    coluna_de = np.array([co[c["frase"]] for c in celulas])
    k = np.array([c["acertos"] for c in celulas], dtype=float)
    n = np.array([c["gravacoes"] for c in celulas], dtype=float)
    if metodo == "freedman_lane_proporcao":
        y = k / n
    else:
        y = np.log((k + 0.5) / (n - k + 0.5))
    sti = np.array([c["sti"] for c in celulas], dtype=float)
    extra = (np.array([c["covariavel"] for c in celulas], dtype=float)
             if com_covariavel else None)

    ajuste = lambda v: _residuo_efeitos_aditivos(
        v, linha_de, coluna_de, len(pessoas), len(frases), extra)
    r_sti, _ = ajuste(sti)
    r_y, ajustado_y = ajuste(y)
    obs = _correlacao_de_residuos(r_sti, r_y)
    if not np.isfinite(obs):
        return dict(base, resultado_do_teste=None, p=None, gl_do_residuo=gl,
                    motivo=("o STI ou o acerto nao varia depois de remover os "
                            "efeitos de participante e de frase; a correlacao "
                            "nao existe"))
    if len(pessoas) < MINIMO_DE_PARTICIPANTES:
        return dict(base, resultado_do_teste=None, p=None, gl_do_residuo=gl,
                    correlacao_parcial=obs, papel="DESCRITIVO",
                    motivo=("{} participante(s) com celula, abaixo do minimo "
                            "de {}: a H2 fica so' descritiva; a "
                            "correlacao parcial sai sem teste e sem "
                            "resultado do teste".format(len(pessoas),
                                              MINIMO_DE_PARTICIPANTES)))

    B = int(chaves.get("replicas_permutacao_h2", 20000))
    rng = np.random.default_rng(semente if semente is not None else SEMENTE)
    grupos = [np.flatnonzero(linha_de == i) for i in range(len(pessoas))]
    extremos = 0
    for _ in range(B):
        r_perm, _ = ajuste(ajustado_y + _permuta_dentro(r_y, grupos, rng))
        est = _correlacao_de_residuos(r_sti, r_perm)
        if np.isfinite(est) and est <= obs + 1e-12:
            extremos += 1
    p = (1 + extremos) / (1 + B)
    ep = float(np.sqrt(max(p * (1 - p), 1e-12) / B))
    return dict(base, gl_do_residuo=gl, estatistica="correlacao parcial",
                correlacao_parcial=obs, p=float(p), unicaudal=True,
                alfa=0.05, sorteios=B,
                semente=semente if semente is not None else SEMENTE,
                erro_padrao_do_p=ep,
                resultado_do_teste=("p < 0,05 unilateral" if p < 0.05
                                    else "p >= 0,05 unilateral"),
                nota=("p = (1 + extremos) / (1 + sorteios), extremos = "
                      "estatistica permutada <= observada; a interpretacao e' "
                      "exploratoria (plano de analise)"))


SENSIBILIDADES_DA_H2 = {
    "qualidade_tecnica": {
        "chave": "sensibilidade_h2_qualidade_tecnica",
        "pergunta": ("a associacao permanece depois de ajustar pela qualidade "
                     "tecnica da gravacao na celula?"),
        "aviso": ("a DEFINICAO OPERACIONAL da qualidade tecnica (medida, "
                  "confiabilidade, resumo por pessoa x frase, cobrindo as "
                  "gravacoes de consistencia E as de teste) NAO esta' definida"
                  "; este script so' le o valor pronto por celula, "
                  "nao calcula nem escolhe a metrica, e nao aplica limiar"),
    },
    "duracao": {
        "chave": "sensibilidade_h2_duracao",
        "pergunta": ("a associacao permanece entre producoes de duracao "
                     "semelhante?"),
        "aviso": ("ajustar por variavel do mecanismo pode retirar parte da "
                  "relacao real (sobreajuste; Schisterman, Cole e Platt 2009"
                  "): uma associacao menor aqui NAO "
                  "demonstra que a principal era espuria"),
    },
}

LEITURA_DAS_SENSIBILIDADES_DA_H2 = (
    "leitura declarada: 'A persistencia da "
    "associacao indicara robustez a esses ajustes especificos, nao ausencia de "
    "confundimento nem prova de causalidade.' As sensibilidades NAO decidem a "
    "H2; a decisao e' a da analise principal (`decisora`).")


def ler_covariavel_por_celula(diretorio, nome_do_arquivo):
    """{(pessoa, frase): valor ou None}, de `<diretorio>/<nome_do_arquivo>`."""
    caminho = os.path.join(diretorio, nome_do_arquivo)
    if not os.path.exists(caminho):
        raise SystemExit(
            "A chave da sensibilidade da H2 nomeia {!r}, e o arquivo nao "
            "existe em {}. A sensibilidade nao roda em silencio sem ele.".format(
                nome_do_arquivo, diretorio))
    with open(caminho, encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        faltam = {"pessoa", "frase", "valor"} - set(leitor.fieldnames or [])
        if faltam:
            raise SystemExit(
                "{}: faltam as colunas {}; o formato e' pessoa,frase,valor, "
                "uma linha por celula.".format(caminho, sorted(faltam)))
        valores = {}
        for i, l in enumerate(leitor, 2):
            celula = (l["pessoa"], l["frase"])
            if celula in valores:
                raise SystemExit(
                    "{} linha {}: a celula {} aparece de novo. O resumo por "
                    "celula e' de quem produz o arquivo; o script nao escolhe "
                    "entre dois valores.".format(caminho, i, celula))
            bruto = (l["valor"] or "").strip()
            if bruto == "":
                valores[celula] = None
                continue
            try:
                v = float(bruto)
            except ValueError:
                v = float("nan")
            if not np.isfinite(v):
                raise SystemExit(
                    "{} linha {}: valor {!r} da celula {} nao e' numero "
                    "finito. Vazio quer dizer 'sem covariavel'; um valor que "
                    "existe e nao e' numero e' arquivo corrompido.".format(
                        caminho, i, l["valor"], celula))
            valores[celula] = v
    return valores


def _correlacao_parcial_nas_celulas(celulas, metodo):
    """Só a estatística observada da principal (sem covariável), sem permutação."""
    pessoas = sorted({c["pessoa"] for c in celulas})
    frases = sorted({c["frase"] for c in celulas})
    if not celulas:
        return None
    li = {p: i for i, p in enumerate(pessoas)}
    co = {f: j for j, f in enumerate(frases)}
    linha_de = np.array([li[c["pessoa"]] for c in celulas])
    coluna_de = np.array([co[c["frase"]] for c in celulas])
    k = np.array([c["acertos"] for c in celulas], dtype=float)
    n = np.array([c["gravacoes"] for c in celulas], dtype=float)
    y = k / n if metodo == "freedman_lane_proporcao" else \
        np.log((k + 0.5) / (n - k + 0.5))
    sti = np.array([c["sti"] for c in celulas], dtype=float)
    r_sti, _ = _residuo_efeitos_aditivos(sti, linha_de, coluna_de,
                                         len(pessoas), len(frases))
    r_y, _ = _residuo_efeitos_aditivos(y, linha_de, coluna_de,
                                       len(pessoas), len(frases))
    obs = _correlacao_de_residuos(r_sti, r_y)
    return obs if np.isfinite(obs) else None


PAPEL_DAS_SENSIBILIDADES_DA_H2 = "SENSIBILIDADE — nao decide a H2"


def h2_sensibilidades(celulas, chaves, covariaveis=None, semente=None):
    """As sensibilidades pré-declaradas da H2 decisora, cada uma separada."""
    covariaveis = covariaveis or {}
    metodo = chaves.get("metodo_da_h2")
    uteis = [c for c in celulas if c["gravacoes"] > 0
             and c["sti"] is not None and np.isfinite(c["sti"])]
    saida = {}
    for nome, definicao in SENSIBILIDADES_DA_H2.items():
        chave = definicao["chave"]
        bloco = {"executada": False, "chave": chave,
                 "papel": PAPEL_DAS_SENSIBILIDADES_DA_H2,
                 "arquivo": chaves.get(chave),
                 "pergunta": definicao["pergunta"], "aviso": definicao["aviso"],
                 "leitura": LEITURA_DAS_SENSIBILIDADES_DA_H2,
                 "separada": ("so' esta covariavel entra no modelo reduzido; a "
                              "outra sensibilidade e' outra analise")}
        saida[nome] = bloco
        if not chaves.get(chave):
            bloco["motivo"] = (
                "nao executada: a chave `{}` esta' nula. O arquivo com a "
                "covariavel por celula nao foi nomeado{}".format(
                    chave, " -- a definicao operacional da qualidade tecnica "
                    "ainda nao esta' definida"
                    if nome == "qualidade_tecnica" else ""))
            continue
        if metodo is None:
            bloco["motivo"] = (
                "nao executada: `metodo_da_h2` nao escolhido, e a sensibilidade "
                "usa o mesmo metodo e a mesma escala da analise principal, que "
                "tambem nao roda sem ele")
            continue
        valores = covariaveis.get(nome)
        if valores is None:
            bloco["motivo"] = (
                "nao executada: a chave `{}` nomeia {!r}, mas nenhuma "
                "covariavel foi carregada para esta analise".format(
                    chave, chaves.get(chave)))
            continue
        com, sem = [], []
        for c in uteis:
            v = valores.get((c["pessoa"], c["frase"]))
            if v is None:
                sem.append({"pessoa": c["pessoa"], "frase": c["frase"],
                            "motivo": ("celula ausente do arquivo"
                                       if (c["pessoa"], c["frase"]) not in valores
                                       else "valor vazio no arquivo")
                            + "; sai desta sensibilidade, sem imputacao"})
            else:
                com.append(dict(c, covariavel=float(v)))
        analisadas = {(c["pessoa"], c["frase"]) for c in uteis}
        resultado = h2_dentro_da_pessoa(com, chaves, semente=semente,
                                        com_covariavel=True)
        if "veredito" in resultado:
            resultado["resultado_do_teste"] = resultado.pop("veredito")
        bloco.update({
            "executada": True,
            "papel": PAPEL_DAS_SENSIBILIDADES_DA_H2,
            "celulas_com_covariavel": len(com),
            "celulas_sem_covariavel": sem,
            "celulas_do_arquivo_fora_da_analise": len(set(valores) - analisadas),
            "resultado": resultado,
            "correlacao_parcial_principal_nas_mesmas_celulas":
                _correlacao_parcial_nas_celulas(com, metodo),
        })
    return saida


def _celulas_da_h2(linhas, pessoas, sti_da_celula, chaves):
    """As células da H2: acerto do personalizado nas gravações de teste."""
    avaliaveis = frases_avaliaveis(linhas, chaves)
    ausentes = gravacoes_ausentes(linhas) if causa_disponivel(linhas) else set()
    conta = defaultdict(lambda: [0.0, 0])
    for l in linhas:
        if (l["pessoa"] in pessoas and l["condicao"] == "personalizado"
                and l["frase"] in avaliaveis.get(l["pessoa"], set())
                and _e_gravacao(l)
                and (l["pessoa"], l.get("gravacao")) not in ausentes):
            c = conta[(l["pessoa"], l["frase"])]
            c[0] += float(l["acerto"])
            c[1] += 1
    return [{"pessoa": p, "frase": f, "sti": sti_da_celula[(p, f)],
             "acertos": kk, "gravacoes": nn}
            for (p, f), (kk, nn) in sorted(conta.items())
            if (p, f) in sti_da_celula]


def _sti_por_celula(linhas):
    """{(pessoa, frase): mediana do STI nas linhas da célula} (só linhas com índice)."""
    por = defaultdict(list)
    for l in linhas:
        if l.get("sti") is not None and np.isfinite(l["sti"]):
            por[(l["pessoa"], l["frase"])].append(float(l["sti"]))
    return {c: float(np.median(v)) for c, v in por.items()}


def _celulas_com_minimo_de_frases(celulas, minimo):
    """(células de quem tem pelo menos `minimo` células, {pessoa: motivo} de quem sai)."""
    por_pessoa = Counter(c["pessoa"] for c in celulas)
    fora = {p: ("{} frase(s) com indice e gravacoes de teste suficientes; "
                "minimo {} (plano de analise)".format(n, minimo))
            for p, n in sorted(por_pessoa.items()) if n < minimo}
    return [c for c in celulas if c["pessoa"] not in fora], fora


PAPEL_DAS_SENSIBILIDADES_DO_DESENHO_DA_H2 = (
    "SENSIBILIDADE pre-declarada (plano de analise) -- nao decide "
    "a H2; a decisao e' a da analise principal (`decisora`)")


def h2_sensibilidades_de_escala_e_minimo(celulas, chaves, semente=None):
    """A H2 decisora refeita com outro mínimo de frases e com outra escala."""
    if chaves.get("metodo_da_h2") is None:
        return {"executada": False,
                "motivo": ("nao executada: `metodo_da_h2` nao escolhido, e as "
                           "sensibilidades usam o mesmo teste da analise principal")}
    principal = chaves["h2_minimo_de_frases_com_indice"]
    saida = {"executada": True, "papel": PAPEL_DAS_SENSIBILIDADES_DO_DESENHO_DA_H2}
    for minimo in range(principal + 1,
                        chaves["h2_minimo_de_frases_maximo_da_sensibilidade"] + 1):
        cel, fora = _celulas_com_minimo_de_frases(celulas, minimo)
        saida["minimo_de_{}_frases".format(minimo)] = {
            "minimo_de_frases_com_indice": minimo,
            "fora": fora,
            "resultado": h2_dentro_da_pessoa(cel, chaves, semente=semente)}
    piso = chaves["h2_gravacoes_minimas_da_sensibilidade_de_escala"]
    com_piso = [c for c in celulas if c["gravacoes"] >= piso]
    cel, fora = _celulas_com_minimo_de_frases(com_piso, principal)
    saida["escala"] = {
        "metodo": chaves["h2_escala_de_sensibilidade"],
        "gravacoes_minimas_por_celula": piso,
        "celulas_abaixo_do_piso": len(celulas) - len(com_piso),
        "minimo_de_frases_com_indice": principal,
        "fora": fora,
        "resultado": h2_dentro_da_pessoa(
            cel, dict(chaves, metodo_da_h2=chaves["h2_escala_de_sensibilidade"]),
            semente=semente)}
    return saida


def h2_em_verificacao(linhas, por_part, chaves, verif, sti_por_celula=None):
    """A H2 em verificação, exploratória, por Freedman–Lane dentro da pessoa."""
    sti_c = sti_por_celula if sti_por_celula is not None else _sti_por_celula(linhas)
    celulas = _celulas_da_h2(linhas, set(por_part), sti_c, chaves)
    minimo = verif["h2_minimo_de_frases_com_indice"]
    por_pessoa = Counter(c["pessoa"] for c in celulas)
    excluidos = {p: "{} frase(s) com índice, mínimo {}".format(n, minimo)
                 for p, n in sorted(por_pessoa.items()) if n < minimo}
    sem_celula = sorted(set(por_part) - set(por_pessoa))
    for p in sem_celula:
        excluidos[p] = "nenhuma frase com índice"
    celulas = [c for c in celulas if c["pessoa"] not in excluidos]
    base = dict(chaves)
    return {
        "papel": ("EXPLORATÓRIA (em verificação). Associação dentro "
                  "da pessoa entre a inconsistência e o acerto do Personalizado; nada "
                  "diagnóstico, nenhum limite clínico."),
        "minimo_de_frases_com_indice": minimo,
        "piso_de_gravacoes_por_frase": chaves["n_min_por_frase"],
        "excluidos": excluidos,
        "celulas": len(celulas),
        "principal_proporcao": h2_dentro_da_pessoa(
            celulas, dict(base, metodo_da_h2="freedman_lane_proporcao")),
        "sensibilidade_logito": h2_dentro_da_pessoa(
            celulas, dict(base, metodo_da_h2="freedman_lane_logito")),
    }


ARQUIVOS_DAS_SENSIBILIDADES = {
    "barulho": "barulho_por_gravacao.csv",
    "sti_6_primeiras": "sti_6_primeiras_repeticoes.csv",
    "generico_regua_alternativa": "generico_regua_alternativa.csv",
}
CHAVE_DE_CADA_ARQUIVO_DAS_SENSIBILIDADES = {
    "barulho": "sensibilidade_barulho",
    "sti_6_primeiras": "sensibilidade_6_primeiras",
    "generico_regua_alternativa": "sensibilidade_letras_palavras",
}


ARQUIVO_DA_ESCOLHA_FORCADA = "escolha_forcada.csv"
COLUNAS_DA_ESCOLHA_FORCADA = ("pessoa", "frase", "sessao", "gravacao", "versao",
                              "acerto", "empate")
REGRAS_DO_DEGRAU_2 = ("vencer_todos",)


def comparadores_do_degrau_2(chaves):
    """As versões da escolha forçada que o degrau 2 exige, na ordem das chaves."""
    versoes = [v.strip() for v in str(chaves["degrau_2_comparadores"]).split(",")
               if v.strip()]
    if not versoes or len(set(versoes)) != len(versoes):
        raise ValueError("degrau_2_comparadores = {!r}: precisa de pelo menos uma "
                         "versão, sem repetição".format(chaves["degrau_2_comparadores"]))
    if chaves["degrau_2_regra"] not in REGRAS_DO_DEGRAU_2:
        raise ValueError("degrau_2_regra = {!r}; vale uma de {}".format(
            chaves["degrau_2_regra"], ", ".join(REGRAS_DO_DEGRAU_2)))
    return versoes


def linhas_do_personalizado_do_degrau_1(linhas, pessoas, chaves, verif=None):
    avaliaveis = frases_avaliaveis(linhas, chaves, verif)
    ausentes = gravacoes_ausentes(linhas) if causa_disponivel(linhas) else set()
    return [l for l in linhas
            if l["pessoa"] in pessoas and l["condicao"] == "personalizado"
            and l["frase"] in avaliaveis.get(l["pessoa"], ())
            and _e_gravacao(l)
            and (l["pessoa"], l.get("gravacao")) not in ausentes]


def conferir_escolha_forcada(linhas_forcado, versoes):
    """Recusa, com o motivo, o arquivo fora do contrato, fora da regra de empate ou sem alguma das versões exigidas."""
    if not linhas_forcado:
        raise ValueError("{} está vazio".format(ARQUIVO_DA_ESCOLHA_FORCADA))
    faltam = [c for c in COLUNAS_DA_ESCOLHA_FORCADA if c not in linhas_forcado[0]]
    if faltam:
        raise ValueError("{} sem as colunas {}".format(ARQUIVO_DA_ESCOLHA_FORCADA, faltam))
    for i, l in enumerate(linhas_forcado, 2):
        empate = str(l["empate"]).strip()
        try:
            acerto = ler_acerto(l["acerto"])
        except ValueError as e:
            raise ValueError("linha {} de {}: {}".format(
                i, ARQUIVO_DA_ESCOLHA_FORCADA, e))
        if empate not in ("0", "1"):
            raise ValueError("linha {} de {}: empate {!r} tem de ser 0 ou 1".format(
                i, ARQUIVO_DA_ESCOLHA_FORCADA, l["empate"]))
        if empate == "1" and acerto == 1.0:
            raise ValueError(
                "linha {} de {}: gravação {} de {} com empate de notas e acerto 1; "
                "no empate o acerto é 1/t ou 0 (plano de análise)".format(
                    i, ARQUIVO_DA_ESCOLHA_FORCADA, l["gravacao"], l["pessoa"]))
        if empate == "0" and acerto not in (0.0, 1.0):
            raise ValueError(
                "linha {} de {}: gravação {} de {} com acerto {} sem empate de "
                "notas; acerto fracionário só existe no empate".format(
                    i, ARQUIVO_DA_ESCOLHA_FORCADA, l["gravacao"], l["pessoa"],
                    l["acerto"]))
    presentes = {str(l["versao"]).strip() for l in linhas_forcado}
    ausentes = [v for v in versoes if v not in presentes]
    if ausentes:
        raise ValueError(
            "{} sem a(s) versão(ões) {} da escolha forçada; o degrau 2 exige vencer "
            "{} (chave degrau_2_comparadores)".format(
                ARQUIVO_DA_ESCOLHA_FORCADA, ", ".join(ausentes), ", ".join(versoes)))


def tabela_do_degrau_2(linhas_personalizado, linhas_forcado, pessoas, versao):
    do_forcado = {}
    for l in linhas_forcado:
        if str(l["versao"]).strip() != versao:
            continue
        chave = (l["pessoa"], l["gravacao"])
        if chave in do_forcado:
            raise ValueError("gravação {} de {} repetida na versão {} de {}".format(
                l["gravacao"], l["pessoa"], versao, ARQUIVO_DA_ESCOLHA_FORCADA))
        do_forcado[chave] = l
    por_frase = defaultdict(lambda: ([], []))
    faltando = []
    for l in linhas_personalizado:
        par = do_forcado.get((l["pessoa"], l["gravacao"]))
        if par is None:
            faltando.append((l["pessoa"], l["gravacao"]))
            continue
        if par["frase"] != l["frase"]:
            raise ValueError("a gravação {} de {} é {!r} no Personalizado e {!r} na "
                             "escolha forçada ({})".format(
                                 l["gravacao"], l["pessoa"], l["frase"], par["frase"],
                                 versao))
        a, b = por_frase[(l["pessoa"], l["frase"])]
        a.append(ler_acerto(l["acerto"]))
        b.append(ler_acerto(par["acerto"]))
    if faltando:
        raise ValueError("{} gravação(ões) do Personalizado do degrau 1 sem linha da "
                         "versão {} em {} (a primeira: {} {}); o degrau 2 usa as "
                         "mesmas gravações do degrau 1".format(
                             len(faltando), versao, ARQUIVO_DA_ESCOLHA_FORCADA,
                             *faltando[0]))
    com_dado = sorted(p for p in pessoas if any(k[0] == p for k in por_frase))
    frases = sorted({f for (_p, f) in por_frase})
    D = np.full((len(com_dado), len(frases)), np.nan)
    for i, p in enumerate(com_dado):
        for j, f in enumerate(frases):
            if (p, f) in por_frase:
                a, b = por_frase[(p, f)]
                D[i, j] = float(np.mean(a) - np.mean(b))
    return com_dado, frases, D


def _resumo_do_teste(r, decisao):
    """Ganho médio, t, gl, p e o intervalo do teste (bilateral de 1 - 2·alfa, 92%) de um teste do degrau 2."""
    teste = decisao.get("teste") or {}
    intervalo = teste.get("intervalo_do_teste") or {}
    ic95 = teste.get("intervalo_95") or {}
    return {"ganho_medio": r.get("ganho"), "t": teste.get("t"), "gl": teste.get("gl"),
            "p": teste.get("p"),
            "ic_do_teste": ([float(x) for x in intervalo["limites"]] if intervalo
                            else None),
            "nivel_do_ic": intervalo.get("nivel") if intervalo else None,
            "ic_95": [float(x) for x in ic95["limites"]] if ic95 else None,
            "rotulo_do_ic_95": ic95.get("rotulo") if ic95 else None}


def decidir_vencer_todos(por_versao, versoes):
    """A regra do degrau 2: passa só se o teste passar contra todas as versões."""
    faltam = [v for v in versoes if v not in por_versao]
    if faltam:
        raise ValueError("falta(m) o(s) teste(s) contra {}".format(", ".join(faltam)))
    falharam = [v for v in versoes if por_versao[v]["veredito"] is False]
    sem_veredito = [v for v in versoes if por_versao[v]["veredito"] is None]
    if falharam:
        return dict(veredito=False, resultado="nao sustentada",
                    falharam=falharam, sem_veredito=sem_veredito,
                    motivo="não venceu: {}".format(", ".join(falharam)))
    if sem_veredito:
        return dict(veredito=None, resultado="sem veredito", falharam=[],
                    sem_veredito=sem_veredito,
                    motivo="sem veredito contra {}: {}".format(
                        ", ".join(sem_veredito),
                        "; ".join(por_versao[v]["motivo"] for v in sem_veredito)))
    return dict(veredito=True, resultado="sustentada", falharam=[], sem_veredito=[],
                motivo="")


def _tabelas_do_degrau_2(pessoas, linhas, chaves, verif, diretorio, versoes):
    """{versão: (usadas, frases, D)} do degrau 2, ou ValueError com o motivo."""
    linhas_forcado = _ler_csv_opcional(diretorio, ARQUIVO_DA_ESCOLHA_FORCADA)
    if linhas_forcado is None:
        raise ValueError("sem o arquivo {}: a escolha forçada não rodou".format(
            ARQUIVO_DA_ESCOLHA_FORCADA))
    conferir_escolha_forcada(linhas_forcado, versoes)
    do_degrau_1 = linhas_do_personalizado_do_degrau_1(linhas, pessoas, chaves, verif)
    return {v: tabela_do_degrau_2(do_degrau_1, linhas_forcado, pessoas, v)
            for v in versoes}


PAPEL_DO_DEGRAU_2_DESCRITIVO = (
    "DESCRITIVO: o degrau 1 nao passou (ou ficou sem veredito); o degrau 2 nao e' "
    "testado e aparece so' como descricao: ganho medio e ganho de cada participante "
    "contra cada versao da escolha forcada, sem teste, sem valor p, sem intervalo e "
    "sem veredito (decisao de 06/10/2026, 18h14 a 18h21)")


def degrau_2_descritivo(pessoas, linhas, chaves, verif, diretorio):
    """O degrau 2 só como descrição, para quando o degrau 1 não passa."""
    versoes = comparadores_do_degrau_2(chaves)
    try:
        tabelas = _tabelas_do_degrau_2(list(pessoas), linhas, chaves, verif, diretorio,
                                       versoes)
    except ValueError as erro:
        return {"papel": PAPEL_DO_DEGRAU_2_DESCRITIVO, "disponivel": False,
                "motivo": str(erro)}
    por_versao = {}
    for v in versoes:
        usadas, frases, D = tabelas[v]
        ganhos = ganhos_da_tabela(usadas, D)
        por_versao[v] = dict(_descricao_dos_ganhos(ganhos), n_participantes=len(ganhos),
                             frases_no_grupo=frases, ganho_por_participante=ganhos)
    return {"papel": PAPEL_DO_DEGRAU_2_DESCRITIVO, "disponivel": True,
            "por_versao": por_versao}


def degrau_2_da_h1a_na_analise(veredito_do_degrau_1, pessoas, linhas, chaves, verif,
                               diretorio):
    from src.sequencia_fixa import sequencia_fixa
    pessoas = list(pessoas)
    versoes = comparadores_do_degrau_2(chaves)

    def _degrau_2():
        tabelas = _tabelas_do_degrau_2(pessoas, linhas, chaves, verif, diretorio, versoes)
        por_versao, testes = {}, {}
        for v in versoes:
            usadas, frases, D = tabelas[v]
            r, decisao = _decidir_pelas_frases_fixas(D)
            por_versao[v] = dict(decisao, participantes_usados=usadas,
                                 frases_no_grupo=frases,
                                 ganho_por_participante=dict(
                                     zip(usadas, r["ganhos_por_participante"])))
            testes[v] = dict(_resumo_do_teste(r, decisao), status=decisao["status"],
                             veredito=decisao["veredito"])
        regra = decidir_vencer_todos(por_versao, versoes)
        primeira = por_versao[versoes[0]]
        return dict(regra, regra=chaves["degrau_2_regra"], comparadores=versoes,
                    referencia_de_silencio=chaves["degrau_2_referencia_de_silencio"],
                    testes=testes, por_versao=por_versao,
                    participantes_usados=primeira["participantes_usados"],
                    frases_no_grupo=primeira["frases_no_grupo"])

    saida = sequencia_fixa(veredito_do_degrau_1, _degrau_2)
    saida["degrau_1"].pop("veredito", None)
    saida["degrau_1"]["resultado_completo_em"] = "veredito_h1"
    if saida["degrau_1"]["estado"] != "passou":
        saida["degrau_2"]["descricao"] = degrau_2_descritivo(pessoas, linhas, chaves,
                                                             verif, diretorio)
    saida["procedimento"] = ("sequencia fixa: o degrau 2 so' e' testado se o degrau 1 "
                             "passar; o mesmo teste das frases fixas nos dois; o "
                             "degrau 1 a {:.0%}; no degrau 2, contra cada versao da "
                             "escolha forcada ({}), e passa so' se vencer todas, isto "
                             "e', se o maior dos tres p for menor ou igual ao limiar "
                             "que o Holm da familia {{degrau 2, H1b}} lhe der ({:.0%} "
                             "ou {:.0%}; `aplicar_holm_da_familia`). O degrau 1 e' a "
                             "afirmacao principal: a falha do degrau 2 qualifica o "
                             "alcance e nao apaga o degrau 1; sem o degrau 1, o degrau "
                             "2 e' so' descricao. O estado devolvido aqui e' o de cada "
                             "versao a {:.0%}, sem a familia; a analise o troca pelo "
                             "do Holm".format(ALFA_DA_H1A, ", ".join(versoes),
                                              ALFA_DA_H1A / 2, ALFA_DA_H1A, ALFA_DA_H1A))
    saida["empate_na_escolha_forcada"] = ("credito 1/t: com t frases empatadas na "
                                          "maior nota, vale 1/t se a verdadeira "
                                          "esta' entre elas e 0 se nao esta', como "
                                          "nas tres condicoes")
    saida["fonte"] = "plano de analise"
    return saida


ARQUIVO_DO_SERVICO = "servico_por_participante.csv"
FORMATO_DO_CODIGO_DE_SERVICO = re.compile(r"^S[0-9]{1,3}$")
MINIMO_DE_SERVICOS = 3
MINIMO_DE_PARTICIPANTES_NUM_SERVICO = 2
PAPEL_DA_SENSIBILIDADE_POR_SERVICO = (
    "SENSIBILIDADE -- nao decide: o degrau 1 refeito sobre as medias por servico "
    "(cada servico vale um ganho), para ver se a conclusao depende de participantes "
    "do mesmo servico; a conclusao principal e' condicional aos servicos da amostra")
NAO_VERIFICAVEL_SEM_SERVICO = (
    "nao verificavel: sem a categoria pseudonimizada do servico de cada participante "
    "({} ausente), a dependencia entre participantes do mesmo servico fica declarada "
    "e nao pode ser verificada".format(ARQUIVO_DO_SERVICO))


def ler_servico_por_participante(diretorio):
    """({pessoa: código}, motivo)."""
    linhas = _ler_csv_opcional(diretorio, ARQUIVO_DO_SERVICO)
    if linhas is None:
        return None, NAO_VERIFICAVEL_SEM_SERVICO
    mapa = {}
    for i, l in enumerate(linhas, 2):
        pessoa = (l.get("pessoa") or "").strip()
        codigo = (l.get("servico") or "").strip()
        if not pessoa or "servico" not in l:
            raise ValueError("{}: linha {} sem as colunas 'pessoa' e 'servico'".format(
                ARQUIVO_DO_SERVICO, i))
        if codigo and not FORMATO_DO_CODIGO_DE_SERVICO.match(codigo):
            raise ValueError("{}: linha {} com servico fora do codigo pseudonimizado "
                             "(S seguido de 1 a 3 digitos); o valor nao e' mostrado"
                             .format(ARQUIVO_DO_SERVICO, i))
        if pessoa in mapa:
            raise ValueError("{}: {} aparece mais de uma vez".format(
                ARQUIVO_DO_SERVICO, pessoa))
        mapa[pessoa] = codigo or None
    return mapa, None


def sensibilidade_por_servico(diretorio, ganho_por_participante, alfa=ALFA_DA_H1):
    """O bloco `sensibilidade_por_servico` do JSON."""
    base = {"papel": PAPEL_DA_SENSIBILIDADE_POR_SERVICO,
            "alcance_da_conclusao": ("condicional aos adolescentes, as familias e aos "
                                     "servicos da amostra"),
            "condicao": ("{} ou mais servicos entre os participantes do degrau 1 e algum "
                         "servico com {} ou mais deles".format(
                             MINIMO_DE_SERVICOS, MINIMO_DE_PARTICIPANTES_NUM_SERVICO))}
    mapa, motivo = ler_servico_por_participante(diretorio)
    if mapa is None:
        return dict(base, verificavel=False, motivo=motivo)
    ganhos = {p: g for p, g in ganho_por_participante.items()
              if g is not None and np.isfinite(g)}
    sem_codigo = sorted(p for p in ganhos if not mapa.get(p))
    por_servico = defaultdict(list)
    for p in sorted(ganhos):
        if mapa.get(p):
            por_servico[mapa[p]].append(p)
    tamanhos = {c: len(v) for c, v in sorted(por_servico.items())}
    base.update(participantes_por_servico=tamanhos, sem_codigo_de_servico=sem_codigo)
    if (len(por_servico) < MINIMO_DE_SERVICOS
            or max(tamanhos.values(), default=0) < MINIMO_DE_PARTICIPANTES_NUM_SERVICO):
        return dict(base, verificavel=False, motivo=(
            "nao verificavel: {} servico(s), o maior com {} participante(s); a "
            "sensibilidade exige {}".format(len(por_servico),
                                            max(tamanhos.values(), default=0),
                                            base["condicao"])))
    medias = {c: float(np.mean([ganhos[p] for p in v])) for c, v in sorted(por_servico.items())}
    r = ganho_frases_fixas(np.array([[m] for m in medias.values()]), alfa=alfa)
    return dict(base, verificavel=True, medias_por_servico=medias,
                n_servicos=len(medias), ganho_medio_das_medias=r["ganho"], t=r["t"],
                gl=r["gl"], p=r["p"], intervalo_do_teste=r["intervalo_do_teste"],
                intervalo_95=r["intervalo_95"],
                acusaria_ganho_a_este_nivel=bool(r["p"] is not None and r["p"] <= alfa),
                nao_altera_o_resultado_principal=True, motivo=r.get("motivo", ""))


def arquivos_de_entrada(chaves, verif):
    return (["acertos.csv"]
            + sorted(chaves[d["chave"]] for d in SENSIBILIDADES_DA_H2.values()
                     if chaves.get(d["chave"]))
            + sorted(ARQUIVOS_DAS_SENSIBILIDADES[nome] for nome, chave
                     in CHAVE_DE_CADA_ARQUIVO_DAS_SENSIBILIDADES.items()
                     if verif.get(chave)
                     or (nome == "barulho" and chaves.get("sensibilidade_barulho")))
            + [ARQUIVO_DO_SISTEMA_DO_APARELHO, ARQUIVO_DAS_DATAS,
               ARQUIVO_DA_ESCOLHA_FORCADA, ARQUIVO_DO_ENTRE_PESSOAS_IGUALADO,
               ARQUIVO_DOS_SORTEIOS_DO_IGUALADO, ARQUIVO_DO_SERVICO,
               ARQUIVO_DA_CONFIGURACAO_PADRAO])


def _ler_csv_opcional(diretorio, nome):
    caminho = os.path.join(diretorio, nome)
    if not os.path.exists(caminho):
        return None
    with open(caminho, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sessoes_completas(linhas):
    conta = Counter()
    for l in linhas:
        if l["condicao"] == "personalizado" and _e_gravacao(l):
            conta[(l["pessoa"], l["frase"], l["sessao"])] += 1
    if not conta:
        return set()
    cheio = Counter(conta.values()).most_common(1)[0][0]
    frases = {f for (_p, f, _s) in conta}
    sessoes = {s for (_p, _f, s) in conta}
    completas = set()
    for p in {pp for (pp, _f, _s) in conta}:
        if all(conta.get((p, f, s), 0) >= cheio for f in frases for s in sessoes):
            completas.add(p)
    return completas


def _resumo_da_h1(linhas_mod, pessoas, confirmatorio, chaves, verif):
    pp = por_participante(linhas_mod, pessoas, chaves, verif)
    conf = {p: pp[p] for p in confirmatorio if p in pp}
    dentro, _fora = participantes_no_grupo(conf, chaves["min_frases_no_grupo"])
    conf = {p: conf[p] for p in dentro}
    k3 = veredito_da_h1ab(conf, chaves)
    saida = {"n_confirmatorio": len(conf),
             "h1_regra_k3": {"H1a": k3["H1a"].get("veredito")},
             "h1b_descritiva_ganho_medio": k3["H1b"].get("ganho_medio"),
             "duar_pg_medio": (float(np.nanmean([r["duar_pg"] for r in conf.values()]))
                               if conf else None)}
    ff = h1a_frases_fixas(conf)
    saida["h1a_frases_fixas"] = {"resultado": ff["resultado"], "status": ff["status"],
                                 "ganho": (ff.get("teste") or {}).get(
                                     "ganho", ff.get("ganho_observado")),
                                 "p": (ff.get("teste") or {}).get("p"),
                                 "decide_a_h1a": chaves["alcance_da_h1a"]
                                 == "frases_fixas_peso_igual"}
    if saida["h1a_frases_fixas"]["decide_a_h1a"]:
        saida["papel_de_h1_regra_k3"] = ("H1a: DESCRITIVA (contagem, nao decide); "
                                         "H1b: descritiva")
    if verif["metodo_da_h1"] == "dois_fatores":
        c = h1_dois_fatores(conf, chaves, verif)
        saida["h1_dois_fatores"] = {h: {"status": c[h]["status"],
                                        "ganho": (c[h].get("teste") or {}).get("ganho",
                                                                              c[h].get("ganho_observado")),
                                        "p": (c[h].get("teste") or {}).get("p")}
                                    for h in ("H1a", "H1b")}
    return saida


def sensibilidades_do_pacote(linhas, diretorio, pessoas, confirmatorio, chaves, verif,
                             linhas_da_h2=None, fora_da_h2=None):
    """As sensibilidades em verificação, cada uma separada e só com a chave ligada."""
    saida = {}
    barulho_registrada = bool(chaves.get("sensibilidade_barulho"))
    if verif["sensibilidade_barulho"] or barulho_registrada:
        papel = ("registrada (chave congelada sensibilidade_barulho); não decide"
                 if barulho_registrada else "em verificação")
        marcas = _ler_csv_opcional(diretorio, ARQUIVOS_DAS_SENSIBILIDADES["barulho"])
        if marcas is None:
            saida["barulho"] = {"executada": False, "papel": papel,
                                "motivo": "arquivo de barulho ausente"}
        else:
            sim = {(m["pessoa"], m["gravacao"]) for m in marcas if m["teve_barulho"] == "sim"}
            mod = [l for l in linhas if (l["pessoa"], l.get("gravacao")) not in sim]
            saida["barulho"] = {"executada": True, "papel": papel,
                                "gravacoes_marcadas_com_barulho": len(sim),
                                "linhas_retiradas": len(linhas) - len(mod),
                                "regra": "sai só o que a família marcou 'sim'; vazio = "
                                         "desconhecido, fica",
                                "resultado": _resumo_da_h1(mod, pessoas, confirmatorio,
                                                           chaves, verif)}
    if verif["sensibilidade_sessoes_completas"]:
        completas = _sessoes_completas(linhas)
        so = [p for p in pessoas if p in completas]
        saida["sessoes_completas"] = {
            "executada": True, "participantes": so,
            "definicao": ("as duas sessões de teste com todas as frases e o número cheio "
                          "de gravações"),
            "resultado": _resumo_da_h1(linhas, so, [p for p in confirmatorio if p in completas],
                                       chaves, verif)}
    if verif["sensibilidade_letras_palavras"]:
        alt = _ler_csv_opcional(diretorio,
                                ARQUIVOS_DAS_SENSIBILIDADES["generico_regua_alternativa"])
        if alt is None:
            saida["letras_palavras"] = {"executada": False,
                                        "motivo": "arquivo da régua alternativa ausente"}
        else:
            nova = {(a["pessoa"], a["gravacao"]): ler_acerto(a["acerto"]) for a in alt}
            mod = []
            for l in linhas:
                chave = (l["pessoa"], l.get("gravacao"))
                if (l["condicao"] == "generico" and chave in nova
                        and l.get("causa") in (None, "acertou", "predicao_errada",
                                               "empate_no_topo")):
                    l = dict(l, acerto=nova[chave],
                             causa=None if l.get("causa") is None else
                             ("acertou" if nova[chave] == 1.0
                              else "empate_no_topo" if nova[chave] > 0
                              else "predicao_errada"))
                mod.append(l)
            saida["letras_palavras"] = {"executada": True,
                                        "regua": alt[0]["regua"] if alt else None,
                                        "resultado": _resumo_da_h1(mod, pessoas, confirmatorio,
                                                                   chaves, verif)}
    if verif["sensibilidade_6_primeiras"]:
        valores = _ler_csv_opcional(diretorio, ARQUIVOS_DAS_SENSIBILIDADES["sti_6_primeiras"])
        if valores is None:
            saida["6_primeiras"] = {"executada": False, "motivo": "arquivo ausente"}
        elif verif["metodo_da_h2_em_verificacao"] is None:
            saida["6_primeiras"] = {"executada": False,
                                    "motivo": "a H2 em verificação está desligada"}
        else:
            fora = set(fora_da_h2 or ())
            sti_c = {(v["pessoa"], v["frase"]): float(v["valor"]) for v in valores
                     if v["valor"] not in ("", None) and v["pessoa"] not in fora}
            pp_conf = {p: None for p in confirmatorio}
            saida["6_primeiras"] = {
                "executada": True,
                "fora_da_h2": sorted(fora),
                "regra": ("troca só o índice; quem saiu da H2 principal "
                          "fica fora também aqui"),
                "resultado": h2_em_verificacao(linhas_da_h2 or linhas, pp_conf, chaves,
                                               verif, sti_por_celula=sti_c)}
    return saida


def h2(linhas, por_part, chaves, covariaveis=None):
    """H2: a análise decisora dentro da pessoa, e o Spearman entre participantes."""
    por_frase = defaultdict(list)
    celulas_sem_indice = set()
    for l in linhas:
        if l.get("sti") is not None and np.isfinite(l["sti"]):
            por_frase[(l["pessoa"], l["frase"])].append(float(l["sti"]))
        else:
            celulas_sem_indice.add((l["pessoa"], l["frase"]))
    celulas_sem_indice -= set(por_frase)
    frases_com_indice = sorted({f for (_p, f) in por_frase})
    por_pessoa = defaultdict(list)
    for (pessoa_, _frase), vs in por_frase.items():
        por_pessoa[pessoa_].append(float(np.median(vs)))
    sti = {p: float(np.median(v)) for p, v in por_pessoa.items() if v}
    fora_por_uar = {
        p: "UAR do Personalizado indisponivel (nenhum par formado)"
        for p in por_part if p in sti
        and not np.isfinite(por_part[p]["uar"]["personalizado"])}
    pessoas = [p for p in por_part if p in sti and p not in fora_por_uar]
    cobertura = {
        "regra": "a H2 usa so' as frases do bloco de consistencia (as que tem "
                 "indice); as demais entram so' na H1",
        "fonte": "plano de analise (criterios de inferencia)",
        "frases_com_indice": len(frases_com_indice),
        "celulas_usadas": len(por_frase),
        "celulas_sem_indice": len(celulas_sem_indice),
        "celulas_por_pessoa": {p: len(v) for p, v in sorted(por_pessoa.items())},
    }
    sti_da_celula = {c: float(np.median(v)) for c, v in por_frase.items()}
    com_metodo = chaves.get("metodo_da_h2") is not None
    todas_as_celulas = (_celulas_da_h2(linhas, set(por_part), sti_da_celula, chaves)
                        if com_metodo else [])
    celulas_da_h2, fora_pelo_minimo = (
        _celulas_com_minimo_de_frases(todas_as_celulas,
                                      chaves["h2_minimo_de_frases_com_indice"])
        if com_metodo else ([], {}))
    decisora = h2_dentro_da_pessoa(celulas_da_h2, chaves)
    if com_metodo:
        decisora.update(piso_de_gravacoes_por_frase=chaves["n_min_por_frase"],
                        minimo_de_frases_com_indice=chaves["h2_minimo_de_frases_com_indice"],
                        fora_pelo_minimo_de_frases=fora_pelo_minimo)
    sensibilidades = h2_sensibilidades(celulas_da_h2, chaves, covariaveis)
    sensibilidades_do_desenho = h2_sensibilidades_de_escala_e_minimo(
        todas_as_celulas, chaves)
    papel = ("DESCRITIVO: os campos de topo sao a analise ENTRE PARTICIPANTES, que "
             "nao decide a H2; a decisao esta' em `decisora`")
    if len(pessoas) < 4:
        return {"erro": "menos de 4 participantes -- correlacao nao sai",
                "n": len(pessoas), "cobertura_da_h2": cobertura,
                "fora_da_correlacao": fora_por_uar,
                "papel": papel, "decisora": decisora,
                "sensibilidades_da_decisora": sensibilidades,
                "sensibilidades_de_escala_e_minimo": sensibilidades_do_desenho}
    x = np.array([sti[p] for p in pessoas])
    y = np.array([por_part[p]["uar"]["personalizado"] for p in pessoas])
    g = np.array([por_part[p]["uar"]["generico"] for p in pessoas])
    ganho = np.array([por_part[p]["duar_pg"] for p in pessoas])
    rho, _p_assintotico = stats.spearmanr(x, y)
    p, _detalhe_p = _p_por_permutacao(x, y, chaves)

    rng = np.random.default_rng(SEMENTE)
    reps = []
    for _ in range(chaves["replicas_bootstrap"]):
        i = rng.integers(0, len(pessoas), len(pessoas))
        if len(set(x[i])) < 3:
            continue
        reps.append(stats.spearmanr(x[i], y[i]).statistic)
    reps = np.array([r for r in reps if np.isfinite(r)])
    ic_perc = ([float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
               if len(reps) > 10 else None)
    n_p = len(pessoas)
    z = np.arctanh(np.clip(rho, -0.999, 0.999))
    ic_fisher = ic_fisher_corrigido = None
    if n_p > 3:
        se = 1.0 / np.sqrt(n_p - 3)
        ic_fisher = [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]
        se_c = np.sqrt((1.0 + rho ** 2 / 2.0) / (n_p - 3))
        ic_fisher_corrigido = [float(np.tanh(z - 1.96 * se_c)),
                               float(np.tanh(z + 1.96 * se_c))]

    com_g = np.isfinite(g) & np.isfinite(ganho)
    fora_da_parcial = {
        p: "UAR do Generico indisponivel: fica fora da parcial e do STI x ganho"
        for p, ok in zip(pessoas, com_g) if not ok}
    n_g = int(com_g.sum())
    parcial = None
    if n_g >= 5 and float(np.std(g[com_g])) > 0:
        px, py, pg = (stats.rankdata(v) for v in (x[com_g], y[com_g], g[com_g]))
        def _residuo(alvo, ctrl):
            b = np.polyfit(ctrl, alvo, 1)
            return alvo - np.polyval(b, ctrl)
        rho_p, _p_do_scipy = stats.spearmanr(_residuo(px, pg), _residuo(py, pg))
        if n_g > 3 and abs(rho_p) < 1.0:
            _t = rho_p * np.sqrt((n_g - 3) / (1.0 - rho_p ** 2))
            p_p = float(2 * stats.t.sf(abs(_t), n_g - 3))
        else:
            p_p = float("nan")
        parcial = {"rho": float(rho_p), "p": p_p,
                   "p_do_scipy_gl_n_menos_2": float(_p_do_scipy),
                   "gl": int(n_g - 3), "n": n_g,
                   "controle": "uar_generico (proxy de gravidade)",
                   "metodo": ("Spearman sobre residuos dos postos, com o p "
                              "recalculado em gl = n-3 (parcial de 1a ordem)")}

    rho_ganho = (stats.spearmanr(x[com_g], ganho[com_g]).statistic
                 if n_g >= 3 else float("nan"))

    return {"papel": papel, "decisora": decisora,
            "sensibilidades_da_decisora": sensibilidades,
            "sensibilidades_de_escala_e_minimo": sensibilidades_do_desenho,
            "n": n_p, "rho": float(rho), "p": float(p),
            "p_metodo": _detalhe_p,
            "p_assintotico_do_scipy": float(_p_assintotico),
            "desfecho": "uar_personalizado",
            "ic_bootstrap": ic_perc, "ic_fisher": ic_fisher,
            "ic_fisher_corrigido": ic_fisher_corrigido,
            "parcial_controlando_gravidade": parcial,
            "descritivo_sti_x_ganho": {
                "rho": float(rho_ganho), "n": n_g,
                "nota": ("DESCRITIVO, sem direcao prevista: a hipotese do STI "
                         "como preditor do GANHO nao faz parte do plano de analise"),
            },
            "cobertura_da_h2": cobertura,
            "fora_da_correlacao": fora_por_uar,
            "fora_da_parcial_e_do_ganho": fora_da_parcial,
            "replicas_uteis": int(len(reps))}


def _contagem_de_u(m, n):
    """Distribuição nula exata de U: quantos arranjos dão cada valor de U."""
    tab = {(0, j): [1] for j in range(n + 1)}
    for j in range(n + 1):
        tab[(0, j)] = [1]
    for i in range(1, m + 1):
        tab[(i, 0)] = [1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            a = tab[(i - 1, j)]
            b = tab[(i, j - 1)]
            tamanho = i * j + 1
            fora = [0] * tamanho
            for u, c in enumerate(a):
                if u + j < tamanho:
                    fora[u + j] += c
            for u, c in enumerate(b):
                if u < tamanho:
                    fora[u] += c
            tab[(i, j)] = fora
    return tab[(m, n)]


def _corte_exato_de_mann_whitney(m, n, alfa=0.05):
    """(corte, nivel_atingido) para o IC de Moses, pelo quantil exato."""
    from math import comb

    if m < 1 or n < 1:
        return None, 1.0
    cont = _contagem_de_u(m, n)
    total = comb(m + n, m)
    acum = 0
    corte = 0
    for u, c in enumerate(cont):
        prox = acum + c
        if prox / total > alfa / 2.0:
            break
        acum = prox
        corte = u + 1
    if corte == 0:
        return None, 1.0
    return corte, float(1.0 - 2.0 * acum / total)


def ic_de_moses(afi, tip):
    """(ic, nivel_atingido) do IC de Moses pelo quantil exato de Mann-Whitney."""
    difs = sorted(a - t for a in afi for t in tip)
    corte, atingido = _corte_exato_de_mann_whitney(len(afi), len(tip))
    if corte is None or len(difs) <= 2 * corte:
        return [difs[0], difs[-1]], 1.0
    return [difs[corte - 1], difs[-corte]], atingido


def objetivo_7(por_part, grupos):
    """Objetivo 7: o ganho P - G nos dois grupos, só descritivo."""
    fora = {p: "ganho P - G indisponivel: o Generico nao formou par"
            for p in por_part if grupos.get(p) in ("afi", "tipico")
            and not np.isfinite(por_part[p]["duar_pg"])}
    afi = [por_part[p]["duar_pg"] for p in por_part
           if grupos.get(p) == "afi" and p not in fora]
    tip = [por_part[p]["duar_pg"] for p in por_part
           if grupos.get(p) == "tipico" and p not in fora]
    if len(afi) < 2 or len(tip) < 2:
        return {"erro": "menos de 2 por grupo", "n_afi": len(afi),
                "n_tipico": len(tip), "fora": fora}
    difs = sorted(a - t for a in afi for t in tip)
    return {"hodges_lehmann": float(np.median(difs)),
            "mediana_afi": float(np.median(afi)),
            "mediana_tipico": float(np.median(tip)),
            "n_afi": len(afi), "n_tipico": len(tip), "fora": fora,
            "papel": ("DESCRITIVO -- sem teste entre grupos, sem p e sem "
                      "intervalo (plano de analise)"),
            "nota": ("ganho do Personalizado sobre o Generico (duar_pg), "
                     "mediana por grupo e diferenca de Hodges-Lehmann")}


def resumo_do_empate(linhas, chaves):
    """A regra de empate e, por condição, quantas gravações empataram no topo."""
    por_condicao = {}
    for cond in ("personalizado", "entre_pessoas", "generico"):
        dela = [l for l in linhas if l["condicao"] == cond and _e_gravacao(l)]
        empatadas = [l for l in dela if empatadas_do_predito(l.get("predito"))]
        por_condicao[cond] = {
            "gravacoes": len(dela),
            "com_empate_no_topo": len(empatadas),
            "taxa_de_empate": (len(empatadas) / len(dela)) if dela else None,
            "com_a_frase_alvo_no_empate": sum(
                1 for l in empatadas
                if l["frase"] in empatadas_do_predito(l.get("predito"))),
        }
    return {"regra": chaves["empate"],
            "descricao": ("com t frases empatadas na maior nota, a gravacao vale "
                          "1/t se a frase-alvo esta' entre elas e 0 se nao esta'; "
                          "nas tres condicoes e nos dois degraus da H1a"),
            "por_condicao": por_condicao}


def matriz_de_confusao(linhas):
    saida = {}
    for cond in ("personalizado", "entre_pessoas", "generico"):
        m = defaultdict(Counter)
        for l in linhas:
            if l["condicao"] != cond or not _e_gravacao(l):
                continue
            predito = l["predito"]
            if l.get("causa") in CAUSAS_SEM_RESPOSTA_DO_GENERICO:
                predito = "(sem saida)"
            empatadas = empatadas_do_predito(predito)
            if empatadas:
                for frase in empatadas:
                    m[l["frase"]][frase] += 1.0 / len(empatadas)
                continue
            m[l["frase"]][predito] += 1
        saida[cond] = {k: dict(v) for k, v in sorted(m.items())}
    return saida


CONDICOES_DESCRITIVAS = ("personalizado", "entre_pessoas", "generico")


def pratica_descritiva(linhas, pessoas, chaves):
    """A prática, só descritiva."""
    avaliaveis = frases_avaliaveis(linhas, chaves)
    ausentes = gravacoes_ausentes(linhas) if causa_disponivel(linhas) else set()
    saida = {}
    for pessoa in pessoas:
        dele = [l for l in linhas
                if l["pessoa"] == pessoa
                and l["frase"] in avaliaveis.get(pessoa, set())
                and _e_gravacao(l)
                and (l["pessoa"], l.get("gravacao")) not in ausentes]
        praticadas = sorted({l["frase"] for l in dele
                             if l.get("sti") is not None})
        nao_praticadas = sorted({l["frase"] for l in dele} - set(praticadas))
        sessoes = sorted({l["sessao"] for l in dele})
        res = {"frases_praticadas": praticadas,
               "frases_nao_praticadas": nao_praticadas,
               "sessoes": sessoes}
        for cond in CONDICOES_DESCRITIVAS:
            dela = [l for l in dele if l["condicao"] == cond]

            def _celula(ls):
                return {"uar": _uar(ls), "n": len(ls)}
            res[cond] = {
                "por_sessao": {str(s): _celula([l for l in dela
                                                if l["sessao"] == s])
                               for s in sessoes},
                "por_subconjunto": {
                    "praticadas": _celula([l for l in dela
                                           if l["frase"] in praticadas]),
                    "nao_praticadas": _celula([l for l in dela
                                               if l["frase"] in nao_praticadas]),
                },
            }
        saida[pessoa] = res
    return {
        "papel": ("DESCRITIVO: sem teste, sem veredito; nao "
                  "entra na H1 nem na H2"),
        "criterio_de_praticada": ("PROVISORIO: frase com indice (STI) no "
                                  "participante = frase do bloco de "
                                  "consistencia"),
        "por_participante": saida,
    }


def sinais_opostos_entre_sessoes(por_part):
    """Quem tem o ganho P - G com sinais opostos nas duas sessões."""
    marcados = sorted(p for p, r in por_part.items()
                      if r.get("sessoes_divergem_em_sinal"))
    return {"papel": ("MARCA DESCRITIVA: nao altera o veredito"),
            "contraste": "personalizado_x_generico (dUAR por sessao)",
            "limiar": "PROVISORIO: produto dos dois ganhos < 0 (zero nao conta)",
            "participantes": marcados,
            "n_marcados": len(marcados),
            "n_com_duas_sessoes": sum(
                1 for r in por_part.values()
                if len([v for v in (r.get("duar_pg_por_sessao") or {}).values()
                        if v is not None]) == 2)}


def curva_de_aprendizagem(linhas, por_part):
    """Marca de lugar da curva 24/48/96 (quanto treino compra quanto ganho)."""
    return {"nota": "requer reavaliacao com subconjuntos de treino; "
                    "nao derivavel do CSV de acertos",
            "volumes": [24, 48, 96]}


ARQUIVO_DA_INTELIGIBILIDADE = "inteligibilidade_no_treino.csv"
ARQUIVO_DA_ESCALA_DOS_PAIS = "ics_dos_pais.csv"
ARQUIVO_DA_AVALIACAO = "avaliacao.json"
GANHOS_DA_INTELIGIBILIDADE = (("personalizado_x_generico", "duar_pg"),
                              ("personalizado_x_entre_pessoas", "duar_pe"))
UAR_SOZINHAS_DA_INTELIGIBILIDADE = ("personalizado", "generico")
LEITURA_DO_GANHO_PG = (
    "relacao esperada em parte por construcao: o ganho contem a UAR do "
    "Generico, do mesmo transcritor da medida; diz que a distancia entre os "
    "sistemas aumenta, nao que o Personalizado melhora em quem fala menos "
    "inteligivel")


def _valor_por_pessoa(linhas, coluna):
    """{pessoa: float} das linhas com a coluna numérica; as demais ficam de fora."""
    saida = {}
    for l in linhas or []:
        try:
            v = float(l.get(coluna))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            saida[l["pessoa"]] = v
    return saida


def _postos_com_direcao(pares, alternativa):
    """Descrição dos pares e, com N suficiente, rho de Spearman unicaudal."""
    bloco = {"n": len(pares),
             "pares": [{"pessoa": p, "medida": x, "ganho": y}
                       for p, x, y in sorted(pares)]}
    if alternativa is None:
        bloco["papel"] = "descricao, sem estatistica"
        return bloco
    if len(pares) < MINIMO_DE_PARTICIPANTES:
        bloco["rho"] = None
        bloco["motivo"] = ("N = {} abaixo de {}: so' descricao, sem correlacao"
                           .format(len(pares), MINIMO_DE_PARTICIPANTES))
        return bloco
    x = [a for _, a, _ in pares]
    y = [b for _, _, b in pares]
    r = stats.spearmanr(x, y, alternative=alternativa)
    bloco["rho"] = float(r.statistic)
    bloco["p_unicaudal"] = float(r.pvalue)
    bloco["alternativa"] = alternativa
    return bloco


def procedencia_da_inteligibilidade(diretorio):
    """Confere o CSV da medida contra o sha256 declarado no `avaliacao.json`."""
    caminho_csv = os.path.join(diretorio, ARQUIVO_DA_INTELIGIBILIDADE)
    caminho_av = os.path.join(diretorio, ARQUIVO_DA_AVALIACAO)
    if not os.path.exists(caminho_av):
        return {"verificada": False,
                "motivo": "{} ausente: origem nao verificada".format(
                    ARQUIVO_DA_AVALIACAO)}
    try:
        with open(caminho_av, encoding="utf-8") as f:
            declarado = (json.load(f).get("metadados") or {}).get(
                "inteligibilidade_no_treino") or {}
    except (OSError, ValueError, AttributeError):
        return {"verificada": False,
                "motivo": "{} ilegivel: origem nao verificada".format(
                    ARQUIVO_DA_AVALIACAO)}
    esperado = declarado.get("sha256") if isinstance(declarado, dict) else None
    if not esperado:
        return {"verificada": False,
                "motivo": "{} sem sha256 da medida: origem nao verificada".format(
                    ARQUIVO_DA_AVALIACAO)}
    obtido = _sha256_do_arquivo(caminho_csv)
    if obtido != esperado:
        return {"verificada": False, "sha256_declarado": esperado,
                "sha256_lido": obtido,
                "motivo": "sha256 do CSV difere do declarado em {}: origem nao "
                          "verificada".format(ARQUIVO_DA_AVALIACAO)}
    return {"verificada": True, "sha256": obtido}


def _pares_da_uar_sozinha(do_grupo, por_part, valores, condicao):
    """Pares (medida, UAR da condição no teste), só descrição, sem estatística."""
    pares, fora = [], {}
    for p in do_grupo:
        u = ((por_part.get(p) or {}).get("uar") or {}).get(condicao)
        if p not in valores:
            fora[p] = "sem a medida"
        elif u is None or not np.isfinite(u):
            fora[p] = "sem a UAR do {}".format(condicao)
        else:
            pares.append((p, valores[p], float(u)))
    return {"n": len(pares), "papel": "descricao, sem estatistica",
            "pares": [{"pessoa": p, "medida": x, "uar": y}
                      for p, x, y in sorted(pares)],
            "fora": fora}


def inteligibilidade_e_ganho(diretorio, por_part, pessoas, grupos):
    """Exploratória: a taxa de erro do Genérico no treino e o ganho no teste."""
    medida_bruta = _ler_csv_opcional(diretorio, ARQUIVO_DA_INTELIGIBILIDADE)
    papel = ("EXPLORATORIA: nao entra na regra de nenhuma hipotese; medida no "
             "treino (sessoes 1 e 2), ganho no teste (sessoes 5 e 6)")
    if medida_bruta is None:
        return {"papel": papel, "disponivel": False,
                "motivo": "{} ausente: medida de inteligibilidade nao fornecida"
                          .format(ARQUIVO_DA_INTELIGIBILIDADE)}
    medida = _valor_por_pessoa(medida_bruta, "taxa_de_erro_media")
    sem_medida = {l["pessoa"]: (l.get("motivo") or "sem taxa de erro")
                  for l in medida_bruta if l["pessoa"] not in medida}
    escala_bruta = _ler_csv_opcional(diretorio, ARQUIVO_DA_ESCALA_DOS_PAIS)
    escala = _valor_por_pessoa(escala_bruta, "ics_media")
    fontes = {"taxa_de_erro_no_treino": ("maior taxa = menos inteligivel",
                                         medida, "greater")}
    if escala_bruta is not None:
        fontes["ics_dos_pais"] = ("menor nota = menos inteligivel", escala, None)
    saida = {"papel": papel, "disponivel": True,
             "direcao_declarada": "menos inteligivel -> mais ganho",
             "procedencia": procedencia_da_inteligibilidade(diretorio),
             "leitura_personalizado_x_generico": LEITURA_DO_GANHO_PG,
             "sem_medida": {p: sem_medida[p] for p in sorted(sem_medida)},
             "ics_dos_pais": ("fornecida" if escala_bruta is not None else
                              "{} ausente: conferencia nao executada".format(
                                  ARQUIVO_DA_ESCALA_DOS_PAIS)),
             "entradas": hashes_das_entradas(
                 diretorio, [n for n in (ARQUIVO_DA_INTELIGIBILIDADE,
                                         ARQUIVO_DA_ESCALA_DOS_PAIS)
                             if os.path.exists(os.path.join(diretorio, n))]),
             "por_grupo": {}}
    for grupo in sorted({grupos.get(p) for p in pessoas}, key=str):
        do_grupo = [p for p in pessoas if grupos.get(p) == grupo]
        bloco = {}
        for nome, (sentido, valores, alternativa) in fontes.items():
            for contraste, chave in GANHOS_DA_INTELIGIBILIDADE:
                pares, fora = [], {}
                for p in do_grupo:
                    g = (por_part.get(p) or {}).get(chave)
                    if p not in valores:
                        fora[p] = "sem a medida"
                    elif g is None or not np.isfinite(g):
                        fora[p] = "sem o ganho {}".format(contraste)
                    else:
                        pares.append((p, valores[p], float(g)))
                b = _postos_com_direcao(pares, alternativa)
                b["sentido_da_medida"] = sentido
                b["fora"] = fora
                if contraste == "personalizado_x_generico":
                    b["leitura"] = LEITURA_DO_GANHO_PG
                bloco["{}__{}".format(nome, contraste)] = b
            for condicao in UAR_SOZINHAS_DA_INTELIGIBILIDADE:
                b = _pares_da_uar_sozinha(do_grupo, por_part, valores, condicao)
                b["sentido_da_medida"] = sentido
                bloco["{}__uar_{}".format(nome, condicao)] = b
        saida["por_grupo"][str(grupo)] = bloco
    return saida


def figura_inteligibilidade_e_ganho(bloco, caminho, sistemas=None):
    if not bloco.get("disponivel"):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eixos = plt.subplots(1, len(GANHOS_DA_INTELIGIBILIDADE) + 1,
                              figsize=(13, 4))
    for eixo, (contraste, _) in zip(eixos, GANHOS_DA_INTELIGIBILIDADE):
        for grupo, marca in zip(sorted(bloco["por_grupo"]), "os^D"):
            b = bloco["por_grupo"][grupo]["taxa_de_erro_no_treino__" + contraste]
            if sistemas is None:
                eixo.scatter([q["medida"] for q in b["pares"]],
                             [q["ganho"] for q in b["pares"]],
                             marker=marca, label="{} (n = {})".format(grupo, b["n"]))
                continue
            _pontos_por_sistema(eixo, b["pares"], "ganho", marca, grupo, sistemas)
        eixo.axhline(0, linewidth=0.8, color="0.5")
        eixo.set_xlabel("taxa de erro do Generico no treino (sessoes 1 e 2)")
        eixo.set_ylabel("dUAR no teste ({})".format(contraste))
        eixo.legend(fontsize=8)
    eixo = eixos[-1]
    for condicao, cor in zip(UAR_SOZINHAS_DA_INTELIGIBILIDADE, ("C0", "C3")):
        for grupo, marca in zip(sorted(bloco["por_grupo"]), "os^D"):
            b = bloco["por_grupo"][grupo][
                "taxa_de_erro_no_treino__uar_" + condicao]
            eixo.scatter([q["medida"] for q in b["pares"]],
                         [q["uar"] for q in b["pares"]], marker=marca, color=cor,
                         label="{} {} (n = {})".format(condicao, grupo, b["n"]))
            for q in (b["pares"] if sistemas is not None else ()):
                eixo.annotate(_inicial_do_sistema(sistemas.get(q["pessoa"])),
                              (q["medida"], q["uar"]), fontsize=6,
                              xytext=(3, 3), textcoords="offset points")
    eixo.set_xlabel("taxa de erro do Generico no treino (sessoes 1 e 2)")
    eixo.set_ylabel("UAR no teste (descricao)")
    eixo.legend(fontsize=7)
    fig.suptitle("Exploratoria: inteligibilidade para a maquina x ganho")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dados")
    ap.add_argument("saida")
    for chave, padrao in CHAVES.items():
        tipo = str if (padrao is None or isinstance(padrao, bool)) else type(padrao)
        ap.add_argument("--" + chave, default=padrao, type=tipo)
    ap.add_argument("--modo", choices=MODOS, default="estudo",
                    help="estudo (padrao) recusa chave diferente do "
                         "arquivo congelado e arvore Git suja; em_verificacao usa as "
                         "mesmas chaves sem exigir commit e rotula o JSON; "
                         "exploratorio aceita chave diferente e rotula o JSON")
    a = ap.parse_args()
    try:
        congeladas, hash_das_chaves = chaves_congeladas()
    except (OSError, ValueError) as e:
        print("Erro: " + str(e), file=sys.stderr)
        return 3
    chaves = {}
    for k in CHAVES:
        padrao, valor = CHAVES[k], getattr(a, k)
        if padrao is None:
            chaves[k] = valor if isinstance(valor, str) and valor else None
        elif isinstance(padrao, bool):
            chaves[k] = str(valor).lower() in ("true", "1", "sim")
        else:
            chaves[k] = type(padrao)(valor)
    if chaves["alcance_da_h1a"] not in ALCANCES_DA_H1A:
        ap.error("alcance_da_h1a = {!r}; vale um de {}".format(
            chaves["alcance_da_h1a"], ", ".join(ALCANCES_DA_H1A)))
    if chaves["empate"] not in POLITICAS_DE_EMPATE:
        ap.error("empate = {!r}; vale uma de {}".format(
            chaves["empate"], ", ".join(POLITICAS_DE_EMPATE)))
    fora_do_arquivo = sorted(k for k in chaves if chaves[k] != congeladas[k])
    if fora_do_arquivo and a.modo in ("estudo", "em_verificacao"):
        ap.error(
            "modo {}: {} diferente(s) do arquivo congelado {}. "
            "No modo do estudo a analise so' roda com as chaves "
            "registradas. Uma escolha nova (ex.: o metodo do intervalo) entra "
            "NO ARQUIVO e em CHAVES, com a fonte da decisao; para explorar, "
            "use --modo exploratorio.".format(
                a.modo.replace("_", "-"),
                ", ".join("{}={!r} (registrado: {!r})".format(
                    k, chaves[k], congeladas[k]) for k in fora_do_arquivo),
                os.path.basename(ARQUIVO_DAS_CHAVES)))
    if fora_do_arquivo:
        print("Atenção: MODO EXPLORATORIO -- chaves fora do registro: {}. Este "
              "resultado NAO e' a analise do estudo.".format(
                  ", ".join(fora_do_arquivo)))

    _git_agora = estado_do_git()
    if a.modo == "estudo" and _git_agora.get("arvore_limpa") is not True:
        ap.error(
            "a análise do estudo exige um clone com a árvore Git limpa, e não é "
            "o caso ({}). O resultado precisa apontar para um commit reproduzível: "
            "numa cópia sem Git (por exemplo, um arquivo zip) este modo não roda, e "
            "num clone com arquivos alterados eles precisam estar num commit. Para "
            "medir assim mesmo, use --modo em_verificacao (o JSON sai rotulado "
            "\"análise em verificação\").".format(
                _git_agora.get("motivo") or "{} arquivo(s) alterado(s)".format(
                    _git_agora.get("arquivos_alterados"))))
    if _git_agora.get("arvore_limpa") is not True:
        print("Atenção: ARVORE GIT NAO LIMPA ou estado desconhecido ({}): este JSON "
              "nao aponta para um commit reproduzivel.".format(
                  _git_agora.get("motivo") or "{} arquivo(s) alterado(s)".format(
                      _git_agora.get("arquivos_alterados"))))

    try:
        verif = ler_em_verificacao()
    except (OSError, ValueError) as e:
        print("Erro: " + str(e), file=sys.stderr)
        return 3
    fonte_do_verif = (caminho_do_config() if os.path.exists(caminho_do_config())
                      else "config.yaml ausente ao lado do script: PADRÕES (a análise "
                           "congelada)")
    if not os.path.exists(caminho_do_config()):
        print("Atenção: " + fonte_do_verif)
    verif_fora_do_padrao = chaves_em_verificacao_fora_do_padrao(verif)
    try:
        conferir_constantes_da_h1(chaves)
        _config_lido = None
        if os.path.exists(caminho_do_config()):
            import yaml
            with open(caminho_do_config(), encoding="utf-8") as f:
                _config_lido = yaml.safe_load(f) or {}
        whisper_forcado_no_config = conferir_whisper_forcado_no_config(
            chaves, _config_lido, a.modo)
        conferir_regras_de_06_10(chaves)
        conferir_regras_de_07_10(chaves)
        grafias_no_config = conferir_grafias_no_config(chaves, _config_lido, a.modo)
    except (OSError, ValueError) as e:
        print("Erro: " + str(e), file=sys.stderr)
        return 3
    if marcas_a_preencher(chaves):
        print("Atenção: chaves com valor a preencher ({}). Sem esses valores o lote "
              "oficial e a carga dos pesos reais recusam rodar; estes dados nao vieram "
              "do caminho oficial completo.".format(", ".join(marcas_a_preencher(chaves))))
    if whisper_forcado_no_config.get("concordam") is False:
        print("Atenção: MODO EXPLORATORIO -- condicoes.whisper_forcado.ligado do "
              "config.yaml discorda da chave congelada.")
    if grafias_no_config.get("concordam") is False:
        print("Atenção: MODO EXPLORATORIO -- as sequencias literais da escolha forcada "
              "no config.yaml nao sao as congeladas.")
    if verif_fora_do_padrao:
        print("Atenção: CHAVES EM VERIFICAÇÃO fora do padrão: {}. O que elas "
              "mudam é candidato em verificação, não a análise congelada.".format(
                  ", ".join("{}={!r}".format(k, verif[k]) for k in verif_fora_do_padrao)))

    linhas = ler(a.dados)
    fora_da_h2 = _exigir_indice_nas_frases_de_consistencia(linhas, chaves)
    grupos = grupos_por_pessoa(linhas)
    tem_causa = causa_disponivel(linhas)
    primario, sens, motivos = elegiveis(linhas, chaves, verif)

    rotulo_da_execucao = rotulo_do_modo(fora_do_arquivo, verif_fora_do_padrao, a.modo)
    print({ROTULOS_DO_MODO["estudo"]: "ANALISE DO ESTUDO",
           ROTULOS_DO_MODO["em_verificacao"]: "ANALISE EM VERIFICACAO (nao e' a "
                                              "analise do estudo)",
           ROTULOS_DO_MODO["exploratoria"]: "ANALISE EXPLORATORIA (nao e' a "
                                            "analise do estudo)"}[rotulo_da_execucao])
    print("  {} linha(s) · {} participante(s)".format(
        len(linhas), len({l["pessoa"] for l in linhas})))
    if tem_causa:
        print("  coluna `causa` PRESENTE — as quatro clausulas se aplicam")
        for c, n in sorted(Counter(l["causa"] for l in linhas).items()):
            print("    {:>18}: {}".format(c, n))
    else:
        print("  coluna `causa` AUSENTE — so' a clausula 1 (gravacao "
              "ausente) se aplica; as outras tres ficam INAPLICAVEIS.")
        print("    Ausencia NAO vira 'predicao_errada': o motivo de cada "
              "erro fica como NAO SABIDO.")
    print("  primario: {} · sensibilidade: {} · fora: {}".format(
        len(primario), len(sens), len(motivos) - len(sens)))
    for p, m in sorted(motivos.items()):
        print("    {}: {}".format(p, m))
    for p, m in sorted(fora_da_h2.items()):
        print("    {}: {}".format(p, m))

    covariaveis_da_h2 = {
        nome: ler_covariavel_por_celula(a.dados, chaves[d["chave"]])
        for nome, d in SENSIBILIDADES_DA_H2.items() if chaves.get(d["chave"])}
    pp = por_participante(linhas, primario + sens, chaves, verif)
    confirmatorio = so_o_grupo_confirmatorio(primario, grupos)
    pp_conf = {p: pp[p] for p in confirmatorio}
    print("  grupo confirmatorio ({}): {} de {} no primario; os demais sao "
          "so' descritivos".format(GRUPO_CONFIRMATORIO, len(confirmatorio),
                                   len(primario)))
    print("  aparelho: {}".format(REGRA_DO_APARELHO))
    ep = entre_participantes(pp_conf, chaves)
    resultado = {
        "chaves": chaves,
        "modo": {
            "modo": a.modo,
            "chaves_fora_do_arquivo": fora_do_arquivo,
            "chaves_em_verificacao_fora_do_padrao": verif_fora_do_padrao,
            "rotulo": rotulo_da_execucao,
            "e_a_analise_do_estudo": (
                rotulo_da_execucao == ROTULOS_DO_MODO["estudo"]),
        },
        "acaso": ACASO,
        "causa": {
            "coluna_presente": tem_causa,
            "vocabulario": sorted(CAUSAS),
            "contagem": (dict(Counter(l["causa"] for l in linhas))
                         if tem_causa else None),
            "clausulas_aplicaveis": (sorted(REGRA_DE_FALTANTES) if tem_causa
                                     else ["gravacao_ausente"]),
            "nota": ("ausencia da coluna quer dizer NAO SE SABE o motivo de "
                     "cada erro -- nao 'predicao_errada'. As clausulas 2, 3 e 4 "
                     "da REGRA_DE_FALTANTES ficam inaplicaveis, e isso e' "
                     "declarado em vez de suprido por um padrao."),
        },
        "elegibilidade": {"primario": primario, "sensibilidade": sens,
                          "motivos": motivos},
        "empate": resumo_do_empate(linhas, chaves),
        "grupo_confirmatorio": {
            "grupo": GRUPO_CONFIRMATORIO,
            "participantes": confirmatorio,
            "aplica_a": ["entre_participantes", "veredito_h1", "h2"],
            "fonte": ("plano de analise: o grupo tipico "
                      "serve a' caracterizacao descritiva e nao entra na "
                      "hipotese confirmatoria"),
        },
        "treino_adequado": {
            "minimo_por_sessao_de_treino": chaves["treino_minimo_por_sessao_de_treino"],
            "sessoes_de_treino": 2,
            "gravacoes_de_treino_por_frase_em_cada_sessao": 4,
            "regra": ("a frase sem o minimo em alguma das duas sessoes de "
                      "treino sai das TRES condicoes daquele "
                      "participante; o participante continua com as frases "
                      "que sobraram, e o numero de frases avaliaveis de cada "
                      "um e' relatado"),
            "fonte": "plano de analise (exclusao de dados)",
            "aplicado_por": ("quem monta o acertos.csv, pela causa "
                             "`falta_de_treino`; este script LE a causa"),
        },
        "por_participante": pp,
        "entre_participantes": ep,
        "h1": None,
        "veredito_h1": None,
        "h2": dict(h2(_linhas_da_h2(linhas, fora_da_h2), pp_conf, chaves,
                      covariaveis_da_h2),
                   fora_da_h2=fora_da_h2),
        "objetivo_7": objetivo_7({p: pp[p] for p in primario}, grupos),
        "matriz_de_confusao": matriz_de_confusao(linhas),
        "descritivos_exploratorios": {
            "pratica": pratica_descritiva(linhas, primario + sens, chaves),
            "sinais_opostos_entre_sessoes": sinais_opostos_entre_sessoes(pp),
        },
        "curva_de_aprendizagem": (curva_de_aprendizagem(linhas, pp)
                                  if chaves["curva_aprendizagem"] else None),
        "procedencia": {
            "semente": SEMENTE,
            "hash_do_script": hashlib.sha256(
                open(__file__, "rb").read()).hexdigest()[:16],
            "hash_de_src_analise": _hash_de_src_analise(),
            "arquivo_das_chaves": os.path.basename(ARQUIVO_DAS_CHAVES),
            "hash_das_chaves": hash_das_chaves,
            "marcas_a_preencher": marcas_a_preencher(chaves),
            "whisper_forcado_no_config": whisper_forcado_no_config,
            "grafias_da_escolha_forcada_no_config": grafias_no_config,
            "hash_do_config": (_sha256_do_arquivo(caminho_do_config())
                               if os.path.exists(caminho_do_config())
                               else "indisponivel: " + fonte_do_verif),
            "hash_das_chaves_em_verificacao": hash_das_chaves_em_verificacao(verif),
            "entradas": hashes_das_entradas(a.dados, arquivos_de_entrada(chaves, verif)),
            "git": _git_agora,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
    }
    mover_p_para_auditoria(resultado, len(confirmatorio))
    so_descricao_entre_participantes_da_h2(resultado)
    resultado["aparelho"] = {
        "regra": REGRA_DO_APARELHO, "chave": chaves["aparelho"],
        "fonte_do_sistema": ("resposta da familia a' pergunta sobre o aparelho; "
                             "identificacao do navegador como conferencia"),
        "fonte": "decisao de 06/10/2026; plano de analise"}
    h1, resultado["resultado_do_grupo"] = resultado_do_grupo_da_h1(pp_conf, chaves)
    resultado["h1"] = h1
    _rg = resultado["resultado_do_grupo"]
    print("  resultado do grupo: {} de {} com {} ou mais frases avaliaveis".format(
        len(_rg["participantes"]), len(pp_conf), _rg["minimo_de_frases"]))
    for _p, _f in sorted(_rg["fora"].items()):
        print("    {}: {}".format(_p, _f["motivo"]))
    for _p in sorted(p for p in pp if pp[p].get("resultado_parcial")):
        print("    {}: resultado PARCIAL (mais de {} frases sem treino "
              "minimo)".format(_p, MAXIMO_DE_FRASES_SEM_TREINO_SEM_MARCA))
    resultado["em_verificacao"] = {
        "valores": verif, "fora_do_padrao": verif_fora_do_padrao, "fonte": fonte_do_verif,
        "aviso": ("chaves de `analise_em_verificacao` (config.yaml), fora do arquivo congelado; o que elas mudam "
                  "é candidato em verificação, não a análise do estudo")}
    if verif["metodo_da_h2_em_verificacao"] is not None:
        resultado["h2_em_verificacao"] = h2_em_verificacao(
            _linhas_da_h2(linhas, fora_da_h2), pp_conf, chaves, verif)
    resultado["sensibilidades_do_pacote"] = sensibilidades_do_pacote(
        linhas, a.dados, primario + sens, confirmatorio, chaves, verif,
        linhas_da_h2=_linhas_da_h2(linhas, fora_da_h2), fora_da_h2=fora_da_h2)
    resultado["sessoes_no_mesmo_dia"] = sessoes_no_mesmo_dia(a.dados)
    _smd = resultado["sessoes_no_mesmo_dia"]
    print("  sessoes abertas no mesmo dia da anterior: {}".format(
        _smd["total"] if _smd["disponivel"] else "NAO CONTADAS -- " + _smd["motivo"]))
    if verif["metodo_da_h1"] == "dois_fatores":
        resultado["h1_dois_fatores_em_verificacao"] = h1_dois_fatores(pp_conf, chaves, verif)
        h1["papel_nesta_execucao"] = ("a regra k=3 continua calculada; com "
                                      "metodo_da_h1=dois_fatores ela é lida como "
                                      "DESCRIÇÃO do candidato em verificação")
        _c = resultado["h1_dois_fatores_em_verificacao"]
        for _h in ("H1a", "H1b"):
            print("  {} candidato dois fatores (EM VERIFICAÇÃO): {}".format(
                _h, _c[_h]["status"]))
    resultado["veredito_h1"] = dict(
        h1["H1a"], aviso=("este bloco e' a H1a (Personalizado x Generico); a "
                          "H1b (contra o Entre-pessoas igualado) esta' em h1b e a "
                          "leitura da H1 inteira em h1_sequencia_confirmatoria."
                          "leitura; h1.H1b e' o Entre-pessoas completo (descritivo)"))
    ff_decide = chaves["alcance_da_h1a"] == "frases_fixas_peso_igual"
    fixas = _rg.pop("h1a_frases_fixas")
    for _p, _f in _rg["fora"].items():
        fixas["fora_e_por_que"].setdefault(_p, _f["motivo"])
    fixas["papel"] = ("DECIDE A H1a: a conclusao vale so' para as frases do banco"
                      if ff_decide else
                      "SECUNDARIA: nao decide a H1a (alcance_da_h1a = "
                      "frases_como_amostra)")
    resultado["h1a_frases_fixas"] = (
        {"papel": fixas["papel"], "resultado_completo_em": "veredito_h1"}
        if ff_decide else fixas)
    if ff_decide:
        h1["H1a"] = contagem_k3_so_descritiva(h1["H1a"])
        h1["H1a"]["papel"] = ("DESCRITIVA: contagem de participantes com ganho "
                              "individual demonstrado (regra k=3); nao decide a "
                              "H1a, que e' decidida pelo teste das frases fixas "
                              "(veredito_h1)")
        h1["fonte"] = ("H1a decidida pelo teste das frases fixas (veredito_h1); "
                       "contagem k=3 + trava so' descritiva; h1.H1b e' o bloco "
                       "legado do Entre-pessoas completo (descritivo), sem p, sem "
                       "intervalo e sem veredito; a H1b (contra o Entre-pessoas "
                       "igualado) esta' no bloco h1b")
        _s6h = _rg["sensibilidade_seis_frases_ou_mais"]["h1"]
        _s6h["H1a"] = contagem_k3_so_descritiva(_s6h["H1a"])
        _s6h["fonte"] = h1["fonte"]
        h1["papel_nesta_execucao"] = (
            "a H1a e' decidida pelo teste das frases fixas (veredito_h1, "
            "h1a_frases_fixas); a contagem k=3 da H1a e' so' descritiva; o "
            "Entre-pessoas completo (h1.H1b) e' descritivo; a H1b tem veredito "
            "proprio no bloco h1b")
        h1["leitura_combinada"]["veredito_da_parte_confirmatoria"] = fixas.get("veredito")
        h1["leitura_combinada"]["fonte_da_parte_confirmatoria"] = (
            "veredito_h1 (teste das frases fixas)")
        resultado["veredito_h1"] = dict(
            fixas, aviso=("este bloco e' a H1a (Personalizado x Generico) pelo "
                          "teste das frases fixas; a contagem k=3 da H1a esta' em "
                          "h1.H1a (descritiva); a H1b (contra o Entre-pessoas "
                          "igualado) esta' em h1b; h1.H1b e' o Entre-pessoas "
                          "completo (descritivo)"))
    _rot1 = (_rotulo_de_n(fixas["n_participantes"])
             if fixas.get("veredito") is not None else None)
    if ff_decide and resultado.get("veredito_h1") is not None:
        resultado["veredito_h1"]["rotulo_do_n"] = _rot1
    _t = fixas.get("teste") or {}
    print("  H1a (frases fixas, peso igual por frase, t unilateral, gl = N - 1, "
          "alfa {}) [{}]: {}{}{}".format(
              ALFA_DA_H1A, "DECIDE" if ff_decide else "SECUNDARIA", fixas["resultado"],
              " [{}]".format(_rot1.upper()) if _rot1 else "",
              " -- ganho={:+.4f} t={:.3f} gl={} p={:.4g} N={} · {:.0f}% [{:+.4f}; {:+.4f}], "
              "{} · 95% [{:+.4f}; {:+.4f}] (relatado ao lado; nao decide)".format(
                  _t["ganho"], _t["t"], _t["gl"], _t["p"], fixas["n_participantes"],
                  100 * _t["intervalo_do_teste"]["nivel"],
                  _t["intervalo_do_teste"]["limites"][0],
                  _t["intervalo_do_teste"]["limites"][1],
                  _t["intervalo_do_teste"]["rotulo"],
                  _t["intervalo_95"]["limites"][0], _t["intervalo_95"]["limites"][1])
              if _t.get("p") is not None else " -- " + str(fixas["motivo"])))
    if fixas["status"] == "sem_veredito":
        for _p, _g in sorted(fixas["ganho_por_participante"].items()):
            print("    descricao (sem veredito): {}: ganho={:+.4f}".format(_p, _g))
    for _p, _m in sorted(fixas["fora_e_por_que"].items()):
        print("    fora das frases fixas: {}: {}".format(_p, _m))
    resultado["h1a_degrau_2"] = degrau_2_da_h1a_na_analise(
        resultado["veredito_h1"], fixas["participantes_usados"], linhas, chaves,
        verif, a.dados)
    resultado["h1a_testes_de_apoio"] = apoio_do_degrau_1(
        fixas.get("ganho_por_participante") or {}, fixas.get("teste"))
    _ap = resultado["h1a_testes_de_apoio"]
    print("  H1a testes de apoio (NAO decidem): {}".format(
        "permutacao p={} ({}); Wilcoxon p={} ({})".format(
            _ap["permutacao_de_sinais"].get("p"),
            _ap["permutacao_de_sinais"].get("leitura") or _ap["permutacao_de_sinais"].get("motivo"),
            _ap["wilcoxon"].get("p"),
            _ap["wilcoxon"].get("leitura") or _ap["wilcoxon"].get("motivo"))
        if _ap.get("calculado") else _ap["motivo"]))
    def _estado_da_h1a():
        return dict(estado_da_h1a(resultado["h1a_degrau_2"]["degrau_1"],
                                  resultado["h1a_degrau_2"]["degrau_2"]),
                    hipotese="H1a (Personalizado x Generico)",
                    alfa_unilateral=ALFA_DA_H1A,
                    regra_dos_degraus=chaves["h1a_degraus"],
                    degrau_1=resultado["h1a_degrau_2"]["degrau_1"]["estado"],
                    degrau_2=resultado["h1a_degrau_2"]["degrau_2"]["estado"],
                    resultado_completo_em=["veredito_h1", "h1a_degrau_2"])
    _h1a = _estado_da_h1a()
    resultado["h1b"] = h1b_na_analise(_h1a, fixas["participantes_usados"], linhas,
                                      chaves, verif, a.dados, modo=a.modo)
    resultado["familia_depois_do_degrau_1"] = aplicar_holm_da_familia(
        resultado["h1a_degrau_2"]["degrau_2"], resultado["h1b"], chaves)
    _h1a = _estado_da_h1a()
    _h1a["n_elegivel"] = fixas["n_participantes"]
    _h1a["rotulo_do_n"] = (_rotulo_de_n(fixas["n_participantes"])
                           if _h1a["veredito"] is not None else None)
    resultado["h1a"] = _h1a
    _f = resultado["familia_depois_do_degrau_1"]
    _d2 = resultado["h1a_degrau_2"]["degrau_2"]
    _v2 = _d2.get("veredito") or {}
    _n2 = len(_v2.get("participantes_usados") or []) or None
    _d2["rotulo_do_n"] = (_rotulo_de_n(_n2) if _d2.get("estado") in ("passou", "nao_passou")
                          else None)
    _mot2 = _d2.get("motivo") or _v2.get("motivo")
    print("  H1a degrau 2 (escolha forcada, so' se o degrau 1 passar; vencer as tres "
          "versoes; na familia de Holm com a H1b): {}{}{}".format(
              _d2["estado"],
              " [{}]".format(_d2["rotulo_do_n"].upper()) if _d2.get("rotulo_do_n") else "",
              " -- " + str(_mot2) if _mot2 else ""))
    _desc2 = _d2.get("descricao")
    if _desc2 is not None:
        print("    degrau 2 so' como DESCRICAO (sem teste, sem p, sem veredito): {}".format(
            "; ".join("{} ganho medio={:+.4f} (n={})".format(
                _v, _r["ganho_medio"], _r["n_participantes"])
                for _v, _r in _desc2["por_versao"].items()
                if _r.get("ganho_medio") is not None)
            if _desc2.get("disponivel") else _desc2.get("motivo")))
    for _versao, _t in (_v2.get("testes") or {}).items():
        _ih = _t.get("intervalo_no_nivel_de_holm")
        print("    contra {}: ganho={} t={} gl={} p={} IC{:.0f}={} IC95={}{} ({})".format(
            _versao, _t["ganho_medio"], _t["t"], _t["gl"], _t["p"],
            100 * (1 - 2 * ALFA_DA_H1A), _t["ic_do_teste"], _t.get("ic_95"),
            " IC{:.0f} (nivel de Holm usado)={}".format(100 * _ih["nivel"], _ih["limites"])
            if _ih else "", _t["status"] + " a 4% sem a familia"))
    if _f.get("aberta"):
        print("  familia depois do degrau 1 (Holm a {:.0%}): {}".format(
            _f["alfa_da_familia"], "; ".join(
                "{} p={:.4g} limiar={:.0%} {}".format(
                    _m, _r["p"], _r["limiar"],
                    "REJEITADA" if _r["rejeitada"] else
                    ("nao rejeitada" if _r["alcancado"] else "nao alcancada"))
                if _r["p"] is not None else "{} sem p (nao rejeitavel)".format(_m)
                for _m, _r in sorted(_f["holm"].items(),
                                     key=lambda kv: kv[1]["posicao"]))))
        print("    p do degrau 2 = maior dos tres p ({}); p da H1b = maior entre Nadeau e "
              "Bengio e jackknife ({})".format(
                  _f["p_do_degrau_2"].get("versao_do_maior_p"),
                  _f["p_da_h1b"].get("metodo_do_maior_p")))
    else:
        print("  familia depois do degrau 1: nao aberta -- {}".format(_f.get("motivo")))
    print("  H1a (degrau 1 decide a {:.0%}; degrau 2 qualifica; familia de Holm depois): "
          "{}{} -- {}{}".format(
              ALFA_DA_H1A, _h1a["resultado"],
              " [{}: N = {}]".format(_h1a["rotulo_do_n"].upper(), _h1a["n_elegivel"])
              if _h1a.get("rotulo_do_n") else "",
              _h1a["leitura"],
              " [QUALIFICADA: {}]".format(_h1a["qualificacao"])
              if _h1a.get("qualificacao") else ""))
    _b = resultado["h1b"]
    _b["rotulo_do_n"] = (_rotulo_de_n(len(_b.get("participantes_usados") or []))
                         if _b.get("veredito") is not None else None)
    _tb = _b.get("teste") or {}
    _ihb = _tb.get("intervalo_no_nivel_de_holm")
    print("  H1b [{}] (Entre-pessoas igualado, K = {}; confirmatoria so' com a H1a "
          "sustentada; os dois metodos (Nadeau e Bengio E jackknife) no limiar de Holm "
          "da familia): {}{}{}".format(
              _b["papel"].upper(), chaves["h1b_igualado_sorteios"],
              _b["estado"],
              " [{}]".format(_b["rotulo_do_n"].upper()) if _b.get("rotulo_do_n") else "",
              " -- ganho={:+.4f} p(NB)={:.4g} p(jackknife)={:.4g} p(maior EP)={:.4g}{} "
              "IC{:.0f}={} IC{:.0f}={}{}".format(
                  _tb["ganho_medio"], _tb["nadeau_bengio"]["p"], _tb["jackknife"]["p"],
                  _tb["p"], " (exploratorio, nao decide)" if _b["papel"] == "exploratoria"
                  else "", 100 * _tb["intervalo_do_teste"]["nivel"],
                  _tb["intervalo_do_teste"]["limites"],
                  100 * _tb["intervalo_95"]["nivel"], _tb["intervalo_95"]["limites"],
                  " IC{:.0f} (nivel de Holm usado)={}".format(100 * _ihb["nivel"],
                                                              _ihb["limites"])
                  if _ihb else "")
              if _tb.get("p") is not None else " -- " + str(_b.get("motivo"))))
    _ds = (_b.get("distribuicao_dos_sorteios") or {}).get("resumo")
    if _ds:
        print("    os {} sorteios do igualado (ganho medio de cada um; descricao): minimo "
              "{:+.4f}, mediana {:+.4f}, maximo {:+.4f}".format(
                  chaves["h1b_igualado_sorteios"], _ds["minimo"], _ds["mediana"],
                  _ds["maximo"]))
    _pds = _b.get("procedencia_dos_sorteios")
    if _pds is not None:
        print("    procedencia dos sorteios do igualado (semente, K, doadoras, contagem): "
              "{}".format("conferida" if _pds.get("conferida") else _pds.get("motivo")))
    resultado["entre_pessoas_completo_exploratorio"] = entre_pessoas_completo_exploratorio(
        fixas["participantes_usados"], linhas, chaves, verif, a.dados)
    try:
        resultado["sensibilidade_por_servico"] = sensibilidade_por_servico(
            a.dados, fixas.get("ganho_por_participante") or {})
    except ValueError as e:
        resultado["sensibilidade_por_servico"] = {
            "papel": PAPEL_DA_SENSIBILIDADE_POR_SERVICO, "verificavel": False,
            "motivo": "arquivo recusado: " + str(e)}
    _sv = resultado["sensibilidade_por_servico"]
    print("  sensibilidade por servico (NAO decide): {}".format(
        "{} servicos; media das medias={:+.4f}; p={}".format(
            _sv["n_servicos"], _sv["ganho_medio_das_medias"], _sv["p"])
        if _sv.get("verificavel") and _sv.get("ganho_medio_das_medias") is not None
        else _sv.get("motivo")))
    resultado["sensibilidade_seis_frases"] = sensibilidade_seis_frases_nos_contrastes(
        pp_conf, fixas, _h1a, resultado["h1a_degrau_2"]["degrau_2"], resultado["h1b"],
        linhas, chaves, verif, a.dados)
    _s6c = resultado["sensibilidade_seis_frases"]
    for _nome in ("degrau_1", "h1b", "entre_pessoas_completo"):
        _x = _s6c.get(_nome) or {}
        print("  sensibilidade 6+ frases ({}, NAO decide): {}".format(
            _nome, "n restante={} de {}; ganho medio={}{}".format(
                _x.get("n_restante"), _x.get("n_da_analise_principal"),
                _x.get("ganho_medio"),
                "; p={}".format(_x["p"]) if _x.get("p") is not None else
                " (so' estimativa)") if "n_restante" in _x else _x.get("motivo")))
    resultado["sensibilidade_configuracao_padrao"] = \
        sensibilidade_configuracao_padrao_na_analise(fixas, linhas, chaves, verif, a.dados)
    _sp = resultado["sensibilidade_configuracao_padrao"]
    print("  sensibilidade configuracao padrao (NAO decide): {}".format(
        "n={}; ganho medio={}{}".format(_sp["n"], _sp.get("ganho_medio"),
                                        "; p={}".format(_sp["p"]) if _sp.get("p") is not None
                                        else " (so' estimativa)")
        if _sp.get("disponivel") else _sp.get("motivo")))
    mover_p_exploratorio_da_h1(resultado, len(confirmatorio))
    _c = resultado["entre_pessoas_completo_exploratorio"]
    print("  Entre-pessoas completo (EXPLORATORIO, sem veredito): ganho medio={} "
          "IC92={} IC95={} p exploratorio={}{}".format(
              _c.get("ganho_medio"),
              _c["intervalo"]["limites"] if _c.get("intervalo") else None,
              _c["intervalo_95"]["limites"] if _c.get("intervalo_95") else None,
              _c.get("p_exploratorio"),
              " -- " + _c["motivo"] if _c.get("motivo") else ""))
    resultado["h1_sequencia_confirmatoria"] = {
        "ordem": ["H1a: generico_livre (decide a H1a)",
                  "H1a: whisper_escolha_forcada (so' com o degrau 1; qualifica)",
                  "H1b: entre_pessoas_igualado (so' com a H1a sustentada)"],
        "regra": ("na H1a, o degrau 1 decide, unilateral a {:.0%}; so' se ele passar, o "
                  "degrau 2 e a H1b formam uma familia testada por Holm a {:.0%} (menor "
                  "p <= {:.0%}; se rejeitado, o outro com p <= {:.0%}; senao, nenhum "
                  "passa); o p do degrau 2 e' o maior dos tres p unilaterais e o da H1b, "
                  "o maior entre Nadeau e Bengio e jackknife; a falha do degrau 2 "
                  "qualifica a H1a e nao a apaga; a H1b e' sempre calculada e so' recebe "
                  "veredito (confirmatoria) se a H1a for sustentada, senao sai "
                  "exploratoria; a H1a e a H1b tem vereditos proprios, e a H1b nao muda o "
                  "da H1a".format(ALFA_DA_H1A, ALFA_DA_H1A, ALFA_DA_H1A / 2, ALFA_DA_H1A)),
        "erro_da_familia": ERRO_DA_FAMILIA_DA_H1,
        "familia_depois_do_degrau_1": resultado["familia_depois_do_degrau_1"],
        "rotulos_do_n": {"H1a": _h1a.get("rotulo_do_n"),
                         "degrau_2": resultado["h1a_degrau_2"]["degrau_2"].get("rotulo_do_n"),
                         "H1b": _b.get("rotulo_do_n"),
                         "regra": ("de {} a {} participantes elegiveis: '{}'; com {} ou "
                                   "mais: '{}'; abaixo de {}: sem veredito".format(
                                       N_MINIMO_DA_H1, N_PLENO_DA_H1 - 1,
                                       ROTULO_PRELIMINAR, N_PLENO_DA_H1, ROTULO_PLENO,
                                       N_MINIMO_DA_H1))},
        "papel_da_h1b": _b["papel"],
        "degrau_1": resultado["h1a_degrau_2"]["degrau_1"],
        "degrau_2": resultado["h1a_degrau_2"]["degrau_2"],
        "vereditos": {"H1a": _h1a["veredito"], "H1b": _b["veredito"]},
        "resultados": {"H1a": _h1a["resultado"], "H1b": _b["resultado"]},
        "leitura": leitura_da_h1(_h1a, _b),
        "fonte": ("decisoes de 05/10/2026 e de 06/10/2026; "
                  "chaves h1b_ordem, h1a_degraus, h1_multiplicidade e "
                  "h1_rotulo_de_n_pequeno"),
    }
    for rotulo, _campo in ROTULOS_EXPLORATORIOS:
        r = ep[rotulo]["permutacao_de_sinais"]
        print("  {}: n={} media={:+.4f} p={} ({})".format(
            rotulo, r["n"], r.get("media", float("nan")),
            r.get("p", "nao relatado com N < 8; esta no bloco `auditoria` do JSON"),
            "exato" if r.get("exato") else "amostrado"))
    print("  permutacao de sinais do contraste PxG: EXPLORATORIO, p "
          "bruto, nao decide (H1a: {})".format(
              "teste das frases fixas" if ff_decide else "regra k=3"))
    _b = h1["H1b"]
    print("  [{}] sem p, sem intervalo e sem veredito: n={} "
          "ganho medio={} mediana={} melhoras={} pioras={} empates={}".format(
              CONTRASTES_DA_H1["H1b"]["nome"], _b["n_participantes"],
              "{:+.4f}".format(_b["ganho_medio"]) if _b["ganho_medio"] is not None
              else "sem dado", "{:+.4f}".format(_b["ganho_mediano"])
              if _b["ganho_mediano"] is not None else "sem dado",
              _b["melhoras"], _b["pioras"], _b["empates"]))
    for _p, _motivo in sorted(_b["fora_da_descricao"].items()):
        print("    {}: {}".format(_p, _motivo))
    for _h in ("H1a",):
        _v1 = h1[_h]
        _c1 = _v1.get("contagem_k3_descritiva")
        print("  {} (regra k=3, intervalo {}) [{}]: {}".format(
            "H1" if _h == "H1a" else "H1b", _v1.get("metodo_do_intervalo"),
            CONTRASTES_DA_H1[_h]["nome"],
            _c1["leitura"] if _c1 is not None
            else _v1.get("veredito") if _v1.get("veredito") is not None
            else "NAO SAI -- " + str(_v1.get("motivo"))) +
            (" [DESCRITIVA: contagem, nao decide a H1a]" if ff_decide and _h == "H1a"
             else ""))
        _s1 = _v1.get("sensibilidade_do_intervalo") or {}
        if _s1.get("executada"):
            print("  {} sensibilidade ({}, NAO decide): resultado da regra={} · {}".format(
                "H1" if _h == "H1a" else "H1b",
                _s1["metodo"], _s1["resultado"]["resultado_da_regra"],
                "concorda" if _s1["concorda"] else
                "DISCORDA -- " + "; ".join(_s1["discordancia"])))
        else:
            print("  {} sensibilidade: nao executada -- {}".format(
                _h, _s1.get("motivo")))
    _lc = resultado["h1_sequencia_confirmatoria"]["leitura"]
    print("  H1 inteira (H1a: {}; H1b: {}): {} -- {}".format(
        resultado["h1a"]["resultado"], resultado["h1b"]["resultado"],
        _lc["leitura"], _lc["texto"]))
    _s6 = _rg["sensibilidade_seis_frases_ou_mais"]
    _f6 = _s6["h1a_frases_fixas"]
    _t6 = _f6.get("teste") or {}
    print("  H1a sensibilidade ({} ou mais frases, {} participante(s), NAO "
          "decide): teste das frases fixas {}{}".format(
              _s6["minimo_de_frases"], len(_s6["participantes"]), _f6["resultado"],
              " -- ganho={:+.4f} p={:.4g}".format(_t6["ganho"], _t6["p"])
              if _t6.get("p") is not None else " -- " + str(_f6["motivo"])))
    _dec = (resultado["h2"] or {}).get("decisora") or {}
    print("  H2 decisora (dentro da pessoa): {}".format(
        "p={} ({})".format(_dec["p"], _dec["resultado_do_teste"]) if _dec.get("p")
        is not None else "NAO SAI -- " + str(_dec.get("motivo"))))
    for _nome, _s in sorted(((resultado["h2"] or {}).get(
            "sensibilidades_da_decisora") or {}).items()):
        _r = _s.get("resultado") or {}
        print("  H2 sensibilidade ({}) [{}]: {}".format(
            _nome, _s.get("papel"), ("correlacao parcial={} p={} (resultado do "
                                     "teste: {})".format(
                _r.get("correlacao_parcial"), _r.get("p"),
                _r.get("resultado_do_teste"))
                if _r.get("p") is not None else "correlacao parcial={}, SEM p "
                "-- {}".format(_r.get("correlacao_parcial"), _r.get("motivo")))
            if _s.get("executada") else _s.get("motivo")))
    _sd = (resultado["h2"] or {}).get("sensibilidades_de_escala_e_minimo") or {}
    for _nome, _s in sorted((k, v) for k, v in _sd.items() if isinstance(v, dict)):
        _r = _s.get("resultado") or {}
        print("  H2 sensibilidade ({}, NAO decide): {}".format(
            _nome, "p={} ({})".format(_r["p"], _r["resultado_do_teste"])
            if _r.get("p") is not None else "SEM p -- " + str(_r.get("motivo"))))
    print("  H2 descritiva (entre participantes): rho={}".format(
        resultado["h2"].get("rho")))
    for _chave in ("fora_da_correlacao", "fora_da_parcial_e_do_ganho"):
        for _p, _motivo in sorted(((resultado["h2"] or {}).get(_chave)
                                   or {}).items()):
            print("    {}: {}".format(_p, _motivo))
    _cob = (resultado["h2"] or {}).get("cobertura_da_h2")
    if _cob:
        print("    {} frase(s) com indice · {} celula(s) usada(s) · "
              "{} sem indice (entram so' na H1)".format(
                  _cob["frases_com_indice"], _cob["celulas_usadas"],
                  _cob["celulas_sem_indice"]))
    print("  Objetivo 7 (descritivo): diferenca de Hodges-Lehmann AFI - tipico "
          "no ganho Personalizado - Generico = {}".format(
              resultado["objetivo_7"].get("hodges_lehmann")))
    for _p, _motivo in sorted((resultado["objetivo_7"].get("fora") or {}).items()):
        print("    {}: {}".format(_p, _motivo))

    resultado["fluxo_de_participantes"] = fluxo_de_participantes(
        a.dados, linhas, grupos, motivos, {
            "h1a": h1["H1a"].get("motivos_sem_intervalo"),
            "h1b_descritiva": h1["H1b"].get("fora_da_descricao"),
            "h2": fora_da_h2,
            "h2_correlacao_entre_participantes":
                (resultado["h2"] or {}).get("fora_da_correlacao"),
            "h2_parcial_e_ganho":
                (resultado["h2"] or {}).get("fora_da_parcial_e_do_ganho"),
            "objetivo_7": resultado["objetivo_7"].get("fora"),
        })
    _fl = resultado["fluxo_de_participantes"]
    print("  fluxo de participantes: {} participante(s); {}".format(
        len(_fl["participantes"]),
        "sessoes conferidas" if _fl["sessoes_disponiveis"] else _fl["motivo"]))

    resultado["sistema_do_aparelho"] = sistema_do_aparelho(
        a.dados, linhas, pp, confirmatorio, grupos, chaves, fora_da_h2,
        covariaveis_da_h2)
    _sa = resultado["sistema_do_aparelho"]
    if _sa["disponivel"]:
        _sa["figura"] = figura_ganho_por_sistema(
            _sa, os.path.splitext(a.saida)[0] + "_ganho_por_sistema.png")
    _spp = _sa["sensibilidade_por_plataforma"]
    print("  sistema do aparelho (descritivo): {} · sensibilidade por plataforma "
          "(so' descritiva): {}".format(
              "{} participante(s)".format(len(_sa["por_participante"]))
              if _sa["disponivel"] else _sa["motivo"],
              "executada, n por plataforma={}".format(
                  {k: v["n"] for k, v in _spp["por_plataforma"].items()})
              if _spp.get("executada") else "nao executada -- " + _spp["motivo"]))

    resultado["inteligibilidade_e_ganho"] = inteligibilidade_e_ganho(
        a.dados, pp, primario, grupos)
    _ig = resultado["inteligibilidade_e_ganho"]
    if _ig["disponivel"]:
        _ig["figura"] = figura_inteligibilidade_e_ganho(
            _ig, os.path.splitext(a.saida)[0] + "_inteligibilidade.png",
            sistemas=sistemas_para_figura(_sa))
    print("  inteligibilidade x ganho (exploratoria): {}".format(
        "{} grupo(s)".format(len(_ig["por_grupo"])) if _ig["disponivel"]
        else _ig["motivo"]))

    json.dump(resultado, open(a.saida, "w"), ensure_ascii=False, indent=1,
              default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o))
    print("\nJSON: {}  (hash do script: {} · hash das chaves: {} · modo: {})".format(
        a.saida, resultado["procedencia"]["hash_do_script"],
        hash_das_chaves[:16], a.modo))
    return 0


ARQUIVO_DO_SISTEMA_DO_APARELHO = "sistema_do_aparelho_por_gravacao.csv"
SISTEMAS_POR_GRAVACAO = ("ios", "android", "indeterminado")
CORES_DO_SISTEMA = {"ios": "C0", "android": "C2", "misto": "C1",
                    "indeterminado": "0.55"}
ORDEM_DOS_SISTEMAS = ("ios", "android", "misto", "indeterminado")
PLATAFORMAS_DOS_ESTRATOS = ("ios", "android")
N_MINIMO_POR_PLATAFORMA = 3
PAPEL_DO_SISTEMA = ("DESCRITIVO: o sistema do aparelho por participante e o "
                    "ganho de cada um; nao entra na regra de nenhuma hipotese")
LIMITACOES_DO_SISTEMA = (
    "a classe vem da resposta livre da familia a' pergunta sobre o aparelho; "
    "quando ela nao nomeia sistema nem marca reconhecida, vem da identificacao "
    "do navegador, que distingue iOS de Android, mas nao o modelo nem a versao "
    "do sistema (o Chrome do Android manda uma versao fixa)",
    "o Safari do iPad se apresenta por padrao como computador Mac: um iPad "
    "assim fica 'indeterminado', nao 'ios' (comportamento do navegador, nao "
    "medido neste projeto)",
    "sistema e aparelho variam junto com a familia, o comodo e a pessoa: "
    "diferenca de ganho entre sistemas nao isola o efeito do aparelho",
    "a medicao de engenharia com um aparelho emprestado descreve aquele modelo, "
    "nao a categoria inteira de aparelhos do mesmo sistema")
REGRA_DO_APARELHO = (
    "todos os participantes entram na H2 e na H1, qualquer que seja o aparelho; "
    "nenhuma plataforma e' excluida e nao ha ramo do aparelho; com {} ou mais "
    "participantes em cada plataforma (iOS e Android), a H2 e os ganhos sao "
    "repetidos por plataforma so' como descricao, sem valor p; a qualidade tecnica "
    "e' avaliada por gravacao (plano de analise), e nao pela marca do "
    "aparelho".format(N_MINIMO_POR_PLATAFORMA))
PAPEL_DA_SENSIBILIDADE_POR_PLATAFORMA = (
    "SENSIBILIDADE SO' DESCRITIVA: a H2 e os ganhos em cada plataforma, sem valor p "
    "e sem veredito; nao substitui nem muda a analise principal, que tem todos")


DISCORDANCIAS_ACEITAS = ("", "sim", "nao")


def ler_sistema_do_aparelho(diretorio):
    """(linhas, motivo): linhas do CSV conferidas, ou None com o motivo."""
    linhas = _ler_csv_opcional(diretorio, ARQUIVO_DO_SISTEMA_DO_APARELHO)
    if linhas is None:
        return None, ("{} ausente: o sistema do aparelho nao chegou a' "
                      "analise".format(ARQUIVO_DO_SISTEMA_DO_APARELHO))
    for i, l in enumerate(linhas, 2):
        if l.get("pessoa") in (None, "") or l.get("sistema") is None:
            return None, ("{} sem as colunas 'pessoa' e 'sistema' (linha {})"
                          .format(ARQUIVO_DO_SISTEMA_DO_APARELHO, i))
        if l["sistema"] not in SISTEMAS_POR_GRAVACAO:
            return None, ("{}, linha {}: sistema {!r} fora de {}".format(
                ARQUIVO_DO_SISTEMA_DO_APARELHO, i, l["sistema"],
                "/".join(SISTEMAS_POR_GRAVACAO)))
        if (l.get("discordancia") or "") not in DISCORDANCIAS_ACEITAS:
            return None, ("{}, linha {}: discordancia {!r} fora de {}".format(
                ARQUIVO_DO_SISTEMA_DO_APARELHO, i, l.get("discordancia"),
                "/".join(d for d in DISCORDANCIAS_ACEITAS if d)))
    return linhas, None


def sistema_por_participante(linhas_do_sistema, pessoas):
    """{pessoa: {"sistema", "contagem", "sessoes", "motivos"}}."""
    por = defaultdict(list)
    for l in linhas_do_sistema:
        por[l["pessoa"]].append(l)
    saida = {}
    for p in pessoas:
        conta = Counter()
        sessoes = defaultdict(set)
        motivos = Counter()
        fontes = Counter()
        discordantes = []
        for l in por.get(p, []):
            if not (l.get("gravacao") or "").strip():
                motivos[l.get("motivo") or "sem motivo declarado"] += 1
                continue
            conta[l["sistema"]] += 1
            fontes[l.get("fonte") or "nao declarada"] += 1
            if l.get("discordancia") == "sim":
                discordantes.append({
                    "gravacao": l["gravacao"], "sessao": l.get("sessao"),
                    "pela_familia": l.get("sistema_pela_familia"),
                    "pelo_navegador": l.get("sistema_pelo_navegador")})
            if str(l.get("sessao") or "").strip():
                sessoes[l["sistema"]].add(str(l["sessao"]).strip())
            if l["sistema"] == "indeterminado":
                motivos[l.get("motivo") or "sem motivo declarado"] += 1
        conhecidos = [s for s in ("ios", "android") if conta[s]]
        if len(conhecidos) == 2:
            sistema = "misto"
        elif conhecidos:
            sistema = conhecidos[0]
        else:
            sistema = "indeterminado"
        registro = {"sistema": sistema,
                    "contagem": {s: conta[s] for s in SISTEMAS_POR_GRAVACAO},
                    "sessoes": {s: sorted(v, key=lambda x: (len(x), x))
                                for s, v in sorted(sessoes.items())},
                    "motivos_das_indeterminadas": dict(sorted(motivos.items())),
                    "fontes_da_classe": dict(sorted(fontes.items())),
                    "discordancias_familia_x_navegador": discordantes}
        if sistema == "indeterminado":
            registro["motivo"] = ("; ".join(sorted(motivos)) if motivos else
                                  "nenhuma gravacao desta pessoa em {}".format(
                                      ARQUIVO_DO_SISTEMA_DO_APARELHO))
        saida[p] = registro
    return saida


def _mediana_ou_nada(valores):
    v = [float(x) for x in valores if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _sensibilidade_por_plataforma(linhas, pp, confirmatorio, marcas, chaves,
                                  fora_da_h2, covariaveis):
    """A H2 e o ganho repetidos em cada plataforma, só como descrição."""
    base = {"papel": PAPEL_DA_SENSIBILIDADE_POR_PLATAFORMA,
            "regra": REGRA_DO_APARELHO,
            "minimo_por_plataforma": N_MINIMO_POR_PLATAFORMA}
    estratos = {s: sorted(p for p in confirmatorio if marcas[p]["sistema"] == s)
                for s in PLATAFORMAS_DOS_ESTRATOS}
    fora_dos_estratos = {p: marcas[p]["sistema"] for p in sorted(confirmatorio)
                         if marcas[p]["sistema"] not in PLATAFORMAS_DOS_ESTRATOS}
    base.update(n_por_plataforma={s: len(v) for s, v in estratos.items()},
                fora_dos_estratos=fora_dos_estratos,
                n_confirmatorio=len(confirmatorio))
    curtas = [s for s, v in estratos.items() if len(v) < N_MINIMO_POR_PLATAFORMA]
    if curtas:
        return dict(base, executada=False, motivo=(
            "nao executada: a sensibilidade por plataforma exige {} ou mais "
            "participantes em cada plataforma; ha {}".format(
                N_MINIMO_POR_PLATAFORMA,
                ", ".join("{} em {}".format(len(estratos[s]), s)
                          for s in PLATAFORMAS_DOS_ESTRATOS))))
    por_plataforma, auditoria = {}, {}
    for s, quem in estratos.items():
        pp_sub = {p: pp[p] for p in quem}
        so_eles = [l for l in linhas if l["pessoa"] in pp_sub]
        fora = {p: m for p, m in (fora_da_h2 or {}).items() if p in pp_sub}
        resultado_h2 = dict(h2(_linhas_da_h2(so_eles, fora), pp_sub, chaves,
                               covariaveis), fora_da_h2=fora)
        dec = resultado_h2.get("decisora")
        rt = dec.get("resultado_do_teste") if isinstance(dec, dict) else None
        if isinstance(dec, dict):
            dec["resultado_do_teste"] = None
        tirado = _retirar_p(resultado_h2) or {}
        if rt is not None:
            tirado.setdefault("decisora", {})["resultado_do_teste"] = rt
        if tirado:
            auditoria[s] = tirado
        por_plataforma[s] = {
            "participantes": quem, "n": len(quem), "h2": resultado_h2,
            "duar_pg_mediana": _mediana_ou_nada([pp_sub[p].get("duar_pg") for p in quem]),
            "duar_pe_mediana": _mediana_ou_nada([pp_sub[p].get("duar_pe") for p in quem])}
    saida = dict(base, executada=True, por_plataforma=por_plataforma)
    if auditoria:
        saida["auditoria"] = auditoria
    return saida


def sistema_do_aparelho(diretorio, linhas, pp, confirmatorio, grupos, chaves,
                        fora_da_h2=None, covariaveis=None):
    linhas_do_sistema, motivo = ler_sistema_do_aparelho(diretorio)
    if linhas_do_sistema is None:
        return {"papel": PAPEL_DO_SISTEMA, "disponivel": False, "motivo": motivo,
                "limitacoes": list(LIMITACOES_DO_SISTEMA),
                "sensibilidade_por_plataforma": {
                    "papel": PAPEL_DA_SENSIBILIDADE_POR_PLATAFORMA,
                    "executada": False,
                    "motivo": "sem o sistema do aparelho: " + motivo}}
    pessoas = sorted(pp)
    marcas = sistema_por_participante(linhas_do_sistema, pessoas)
    por_participante = {}
    for p in pessoas:
        por_participante[p] = dict(
            marcas[p], grupo=grupos.get(p),
            duar_pg=pp[p].get("duar_pg"), duar_pe=pp[p].get("duar_pe"))
    resumo = {}
    for grupo in sorted({grupos.get(p) for p in pessoas}, key=str):
        do_grupo = [p for p in pessoas if grupos.get(p) == grupo]
        resumo[str(grupo)] = {}
        for s in ORDEM_DOS_SISTEMAS:
            quem = [p for p in do_grupo if marcas[p]["sistema"] == s]
            if quem:
                resumo[str(grupo)][s] = {
                    "n": len(quem), "participantes": quem,
                    "duar_pg_mediana": _mediana_ou_nada(
                        [pp[p].get("duar_pg") for p in quem]),
                    "duar_pe_mediana": _mediana_ou_nada(
                        [pp[p].get("duar_pe") for p in quem])}
    return {"papel": PAPEL_DO_SISTEMA, "disponivel": True,
            "limitacoes": list(LIMITACOES_DO_SISTEMA),
            "entradas": hashes_das_entradas(diretorio,
                                            [ARQUIVO_DO_SISTEMA_DO_APARELHO]),
            "por_participante": por_participante,
            "resumo_por_grupo_e_sistema": resumo,
            "sensibilidade_por_plataforma": _sensibilidade_por_plataforma(
                linhas, pp, [p for p in confirmatorio if p in pp], marcas,
                chaves, fora_da_h2, covariaveis)}


def sistemas_para_figura(bloco):
    """{pessoa: sistema} para as figuras, ou None sem o bloco disponível."""
    if not (bloco or {}).get("disponivel"):
        return None
    return {p: r["sistema"] for p, r in bloco["por_participante"].items()}


def _inicial_do_sistema(sistema):
    return {"ios": "i", "android": "a", "misto": "m"}.get(sistema, "?")


def _pontos_por_sistema(eixo, pares, campo_y, marca, grupo, sistemas):
    """Os pares de um grupo, um scatter por sistema, com a cor fixa do sistema; pessoa sem marca entra como 'indeterminado'."""
    for s in ORDEM_DOS_SISTEMAS:
        qs = [q for q in pares if (sistemas.get(q["pessoa"]) or "indeterminado") == s]
        if qs:
            eixo.scatter([q["medida"] for q in qs], [q[campo_y] for q in qs],
                         marker=marca, color=CORES_DO_SISTEMA[s],
                         label="{} {} (n = {})".format(grupo, s, len(qs)))

def figura_ganho_por_sistema(bloco, caminho):
    """Ganho por participante (P - G e P - E), um ponto por pessoa, separado por sistema no eixo e na cor; marcador pelo grupo."""
    if not (bloco or {}).get("disponivel"):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pp = bloco["por_participante"]
    grupos = sorted({str(r["grupo"]) for r in pp.values()})
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
    for eixo, (chave, nome) in zip(eixos, (("duar_pg", "personalizado_x_generico"),
                                           ("duar_pe", "personalizado_x_entre_pessoas"))):
        for gi, (grupo, marca) in enumerate(zip(grupos, "os^D")):
            for xi, s in enumerate(ORDEM_DOS_SISTEMAS):
                quem = sorted(p for p, r in pp.items() if str(r["grupo"]) == grupo
                              and r["sistema"] == s and r.get(chave) is not None
                              and np.isfinite(r[chave]))
                if not quem:
                    continue
                xs = [xi + 0.15 * (gi - (len(grupos) - 1) / 2)
                      + 0.04 * (k - (len(quem) - 1) / 2) for k in range(len(quem))]
                eixo.scatter(xs, [pp[p][chave] for p in quem], marker=marca,
                             color=CORES_DO_SISTEMA[s],
                             label="{} {} (n = {})".format(grupo, s, len(quem)))
        eixo.axhline(0, linewidth=0.8, color="0.5")
        eixo.set_xticks(range(len(ORDEM_DOS_SISTEMAS)))
        eixo.set_xticklabels(ORDEM_DOS_SISTEMAS)
        eixo.set_ylabel("dUAR no teste ({})".format(nome))
        eixo.legend(fontsize=7)
    fig.suptitle("Descritivo: ganho por participante, por sistema do aparelho")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


if __name__ == "__main__":
    sys.exit(main())

