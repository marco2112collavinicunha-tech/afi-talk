"""Análise estatística do estudo prevista no protocolo."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


class AnaliseIndisponivel(ValueError):
    """A análise não pôde ser feita; a mensagem diz por quê."""


def graus_de_liberdade_between_within(
    n_grupos, n_observacoes, n_parametros_fixos, nivel="entre"
):
    """Graus de liberdade do denominador pelo método Between-Within."""
    if nivel not in ("entre", "dentro"):
        raise AnaliseIndisponivel(
            "Nível desconhecido: {!r}. Use \"entre\" (preditor que varia entre "
            "participantes) ou \"dentro\" (preditor que varia dentro de cada "
            "participante). O nível é do PREDITOR, não do modelo.".format(nivel)
        )
    if nivel == "entre":
        if n_grupos <= n_parametros_fixos:
            raise AnaliseIndisponivel(
                "Between-Within precisa de mais grupos ({}) que parâmetros fixos "
                "({}). Com menos, não há graus de liberdade para testar nada — e "
                "devolver um número aqui seria inventar precisão.".format(
                    n_grupos, n_parametros_fixos
                )
            )
        return int(n_grupos - n_parametros_fixos)

    gl = n_observacoes - n_grupos - n_parametros_fixos
    if gl <= 0:
        raise AnaliseIndisponivel(
            "O nível 'dentro' não sobra grau de liberdade nenhum: {} observações "
            "− {} grupos − {} parâmetros = {}. Ou há poucas observações por "
            "participante, ou parâmetros demais no nível de dentro.".format(
                n_observacoes, n_grupos, n_parametros_fixos, gl
            )
        )
    return int(gl)


def h2_agregada(sti_por_frase, acuracia_por_participante, agregacao="mediana"):
    """Correlação entre participantes: os mais inconsistentes são reconhecidos pior?"""
    if agregacao not in ("mediana", "media"):
        raise AnaliseIndisponivel(
            "Agregação desconhecida: {!r} — use 'mediana' ou 'media' "
            "(sensibilidade declarada).".format(agregacao)
        )
    pessoas = sorted(set(sti_por_frase) & set(acuracia_por_participante))
    faltando = sorted(set(sti_por_frase) ^ set(acuracia_por_participante))
    if faltando:
        raise AnaliseIndisponivel(
            "Estes participantes têm STI ou acurácia, mas não os dois: {}. "
            "Correlacionar sobre a interseção em silêncio mudaria o n sem "
            "ninguém ver.".format(faltando)
        )
    if len(pessoas) < 3:
        raise AnaliseIndisponivel(
            "A H2 precisa de ao menos 3 participantes com os dois valores; "
            "vieram {}.".format(len(pessoas))
        )

    resumir = np.median if agregacao == "mediana" else np.mean
    sti = np.array([resumir(sti_por_frase[p]) for p in pessoas], dtype=float)
    uar = np.array([float(acuracia_por_participante[p]) for p in pessoas])

    if np.std(sti) == 0 or np.std(uar) == 0:
        raise AnaliseIndisponivel(
            "Um dos lados não varia entre participantes (STI: {:.4g}, UAR: {:.4g}). "
            "Sem variação não há correlação a estimar.".format(np.std(sti), np.std(uar))
        )

    r, p_bruto = stats.pearsonr(sti, uar)
    rho, p_rho = stats.spearmanr(sti, uar)
    ddf = graus_de_liberdade_between_within(len(pessoas), len(pessoas), 2)
    t = r * np.sqrt(ddf / max(1e-12, 1 - r**2))
    p = 2 * stats.t.sf(abs(t), ddf)
    return {
        "hipotese": "H2",
        "unidade": "participante",
        "agregacao": agregacao,
        "n": len(pessoas),
        "participantes": pessoas,
        "r_pearson": float(r),
        "rho_spearman": float(rho),
        "p_spearman": float(p_rho),
        "graus_de_liberdade": ddf,
        "metodo_ddf": "Between-Within (Li & Redden, 2015)",
        "t": float(t),
        "p": float(p),
        "convergiu": True,
        "aviso": (
            "Correlação não demonstra que a inconsistência CAUSA o desempenho. "
            "A gravidade da AFI pode mover as duas juntas."
        ),
    }


def h2b_por_frase(observacoes):
    import statsmodels.formula.api as smf

    quadro = pd.DataFrame(observacoes)
    for coluna in ("participante", "frase", "sti"):
        if coluna not in quadro.columns:
            raise AnaliseIndisponivel(
                "Falta a coluna {!r} nas observações. Esperado: participante, "
                "frase, sti e (acerto ou acuracia).".format(coluna)
            )
    alvo = "acerto" if "acerto" in quadro.columns else "acuracia"
    if alvo not in quadro.columns:
        raise AnaliseIndisponivel(
            "As observações precisam de 'acerto' (0/1) ou 'acuracia' (contínua)."
        )
    if quadro["participante"].nunique() < 3:
        raise AnaliseIndisponivel(
            "A H2b precisa de ao menos 3 participantes; vieram {}.".format(
                quadro["participante"].nunique()
            )
        )

    binaria = alvo == "acerto"
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        try:
            if binaria:
                modelo = smf.mixedlm("acerto ~ sti", quadro, groups=quadro["participante"])
            else:
                modelo = smf.mixedlm("acuracia ~ sti", quadro, groups=quadro["participante"])
            ajuste = modelo.fit(reml=True)
            convergiu = bool(getattr(ajuste, "converged", False))
            erro = None
        except Exception as falha:
            return {
                "hipotese": "H2b", "unidade": "frase", "convergiu": False,
                "erro": "{}: {}".format(type(falha).__name__, falha),
                "aviso": _AVISO_CONVERGENCIA,
            }
        mensagens = [str(a.message) for a in capturados]

    ddf = graus_de_liberdade_between_within(
        quadro["participante"].nunique(), len(quadro), 2
    )
    return {
        "hipotese": "H2b",
        "unidade": "frase",
        "n_observacoes": int(len(quadro)),
        "n_participantes": int(quadro["participante"].nunique()),
        "n_frases": int(quadro["frase"].nunique()),
        "desfecho": alvo,
        "binaria": binaria,
        "coeficiente_sti": float(ajuste.params.get("sti", np.nan)),
        "erro_padrao": float(ajuste.bse.get("sti", np.nan)),
        "p": float(ajuste.pvalues.get("sti", np.nan)),
        "variancia_participante": float(ajuste.cov_re.iloc[0, 0]),
        "graus_de_liberdade": ddf,
        "metodo_ddf": "Between-Within (Li & Redden, 2015)",
        "convergiu": convergiu,
        "avisos_do_ajuste": mensagens,
        "aviso": None if convergiu else _AVISO_CONVERGENCIA,
    }


_AVISO_CONVERGENCIA = (
    "O MODELO NÃO CONVERGIU. Eager & Roy (2017) mostram que, com resposta "
    "binária e poucos grupos, isso acontece mesmo quando a estrutura de efeitos "
    "aleatórios é a correta. NÃO remova o termo aleatório para 'fazer rodar': "
    "o modo de falha documentado é exatamente esse, e leva a concluir que não há "
    "variância num nível onde ela existe. Reporte a não convergência."
)


def componentes_de_variancia(observacoes, reamostragens=400, semente=42):
    """Quanto da variação do STI é de pessoa e quanto é de frase (descritivo)."""
    import statsmodels.formula.api as smf

    quadro = pd.DataFrame(observacoes)
    for coluna in ("participante", "frase", "sti"):
        if coluna not in quadro.columns:
            raise AnaliseIndisponivel("Falta a coluna {!r}.".format(coluna))
    pessoas = sorted(quadro["participante"].unique())
    if len(pessoas) < 3:
        raise AnaliseIndisponivel(
            "Estimar componentes precisa de ao menos 3 participantes; vieram "
            "{}.".format(len(pessoas))
        )

    def _ajustar(dados):
        dados = dados.copy()
        dados["_todos"] = 1
        modelo = smf.mixedlm(
            "sti ~ 1", dados, groups=dados["_todos"],
            vc_formula={
                "participante": "0 + C(participante)",
                "frase": "0 + C(frase)",
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ajuste = modelo.fit(reml=True)
        nomes = list(ajuste.model.exog_vc.names)
        componentes = dict(zip(nomes, [float(x) for x in ajuste.vcomp]))
        v_pessoa = componentes.get("participante", 0.0)
        v_frase = componentes.get("frase", 0.0)
        v_res = float(ajuste.scale)
        total = v_pessoa + v_frase + v_res
        if total <= 0:
            return None
        return (v_pessoa / total, v_frase / total, bool(getattr(ajuste, "converged", False)))

    ponto = _ajustar(quadro)
    if ponto is None:
        return {"convergiu": False, "erro": "variância total não positiva",
                "aviso": _AVISO_CONVERGENCIA}
    p_pessoa, p_frase, convergiu = ponto

    rng = np.random.default_rng(semente)
    amostras = []
    for _ in range(reamostragens):
        sorteados = rng.choice(pessoas, size=len(pessoas), replace=True)
        pedacos = []
        for novo, antigo in enumerate(sorteados):
            bloco = quadro[quadro["participante"] == antigo].copy()
            bloco["participante"] = "r{}".format(novo)
            pedacos.append(bloco)
        try:
            r = _ajustar(pd.concat(pedacos, ignore_index=True))
        except Exception:
            r = None
        if r is not None:
            amostras.append(r[:2])

    def _ic(valores):
        if len(valores) < 20:
            return (float("nan"), float("nan"))
        return (float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5)))

    ic_pessoa = _ic([a[0] for a in amostras])
    ic_frase = _ic([a[1] for a in amostras])
    return {
        "estatuto": "DESCRITIVO — não afirma proporção de variância",
        "n_participantes": len(pessoas),
        "n_frases": int(quadro["frase"].nunique()),
        "proporcao_pessoa": p_pessoa,
        "ic95_pessoa": ic_pessoa,
        "proporcao_frase": p_frase,
        "ic95_frase": ic_frase,
        "reamostragens_validas": len(amostras),
        "convergiu": convergiu,
        "direcao": _direcao(ic_pessoa, ic_frase),
        "aviso": (
            "Com poucos participantes o IC é largo. Leia a DIREÇÃO, não o "
            "número. Se os dois intervalos se sobrepuserem, a leitura é que os "
            "dois níveis contribuem de forma comparável — e isso é resultado."
        ) if convergiu else _AVISO_CONVERGENCIA,
    }


def _direcao(ic_pessoa, ic_frase):
    """A leitura que o IC sustenta, ou "indeterminada" quando ele não sustenta."""
    if any(np.isnan(x) for x in (*ic_pessoa, *ic_frase)):
        return "indeterminada (reamostragens insuficientes)"
    if ic_pessoa[0] > ic_frase[1]:
        return "predominantemente DE PESSOA"
    if ic_frase[0] > ic_pessoa[1]:
        return "predominantemente DE FRASE"
    return "os dois níveis contribuem de forma comparável (intervalos se sobrepõem)"


def robustez_h2_logistica(observacoes):
    """Checagem da H2 no desfecho binário, sem agregar acerto em taxa."""
    import statsmodels.formula.api as smf

    quadro = pd.DataFrame(observacoes)
    if "acerto" not in quadro.columns:
        raise AnaliseIndisponivel(
            "Esta checagem da H2 é sobre desfecho BINÁRIO: as observações "
            "precisam da coluna 'acerto' (0/1). Não confundir com a robustez "
            "com o segundo classificador, que é `robustez_do_segundo_classificador`."
        )
    valores = set(pd.unique(quadro["acerto"]))
    if not valores <= {0, 1, True, False}:
        raise AnaliseIndisponivel(
            "'acerto' precisa ser 0/1; vieram {}.".format(sorted(valores)[:5])
        )
    if len(valores) < 2:
        raise AnaliseIndisponivel(
            "'acerto' não varia (só {}). Sem variação no desfecho não há "
            "regressão a ajustar.".format(valores)
        )
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        try:
            ajuste = smf.logit("acerto ~ sti", quadro).fit(disp=False)
            convergiu = bool(ajuste.mle_retvals.get("converged", False))
        except Exception as falha:
            return {"analise": "robustez_h2_logistica", "convergiu": False,
                    "erro": "{}: {}".format(type(falha).__name__, falha),
                    "aviso": _AVISO_CONVERGENCIA}
        mensagens = [str(a.message) for a in capturados]
    return {
        "analise": "robustez_h2_logistica (NÃO é a checagem com o segundo classificador)",
        "n_observacoes": int(len(quadro)),
        "coeficiente_sti": float(ajuste.params.get("sti", np.nan)),
        "erro_padrao": float(ajuste.bse.get("sti", np.nan)),
        "p": float(ajuste.pvalues.get("sti", np.nan)),
        "convergiu": convergiu,
        "avisos_do_ajuste": mensagens,
        "aviso": (
            "Isto é CHECAGEM da H2, não a análise principal e não a checagem com o segundo classificador. O "
            "erro-padrão é otimista (sem efeito aleatório de participante). Se o "
            "SINAL divergir da H2, reporte a divergência — não escolha a que agrada."
        ) if convergiu else _AVISO_CONVERGENCIA,
    }


CLASSIFICADOR_DE_ROBUSTEZ = "regressao_logistica"


def robustez_do_segundo_classificador(linhas, metrica="uar"):
    """Sobre as linhas da varredura: a conclusão sobrevive à troca de classificador?"""
    if metrica not in ("uar", "acuracia"):
        raise ValueError(
            "Métrica de robustez desconhecida: {!r}. Use \"uar\" (primária) "
            "ou \"acuracia\" (secundária).".format(metrica)
        )
    ok = [l for l in linhas if l.get("status") == "ok" and l.get(metrica) is not None]
    if not ok:
        raise AnaliseIndisponivel(
            "Nenhuma linha da varredura tem {} — sem número não há robustez a "
            "checar. Veja os motivos das linhas indisponíveis.".format(metrica)
        )

    por_condicao = {}
    for linha in ok:
        por_condicao.setdefault(linha["condicao"], []).append(linha)

    quadro = {}
    for condicao, suas in por_condicao.items():
        vencedora = max(suas, key=lambda l: l[metrica])
        pares = [
            l for l in suas
            if l.get("classificador") == CLASSIFICADOR_DE_ROBUSTEZ
            and l.get("camada") == vencedora.get("camada")
            and l.get("checkpoint") == vencedora.get("checkpoint")
            and l.get("esquema") == vencedora.get("esquema")
        ]
        if vencedora.get("classificador") == CLASSIFICADOR_DE_ROBUSTEZ:
            quadro[condicao] = {
                "indisponivel": (
                    "A configuração vencedora de {!r} JÁ usa o classificador de "
                    "robustez ({}). Repetir com ele mesmo não checa nada — o "
                    "segundo classificador aqui seria o kNN, e essa troca de papéis "
                    "precisa ser decidida, não presumida.".format(
                        condicao, CLASSIFICADOR_DE_ROBUSTEZ
                    )
                )
            }
            continue
        if not pares:
            quadro[condicao] = {
                "indisponivel": (
                    "Não há linha de {!r} com o classificador {} na MESMA "
                    "configuração da vencedora (camada {}, esquema {}). Sem o par "
                    "não há repetição — ative o classificador no config.yaml e "
                    "rode a varredura de novo.".format(
                        condicao, CLASSIFICADOR_DE_ROBUSTEZ,
                        vencedora.get("camada"), vencedora.get("esquema"),
                    )
                )
            }
            continue
        par = pares[0]
        quadro[condicao] = {
            "esquema": vencedora.get("esquema"),
            "principal": {"classificador": vencedora["classificador"],
                          metrica: float(vencedora[metrica])},
            "robustez": {"classificador": par["classificador"],
                         metrica: float(par[metrica])},
            "diferenca": float(par[metrica]) - float(vencedora[metrica]),
        }

    return {
        "analise": "robustez_do_segundo_classificador (sobre as linhas da varredura)",
        "metrica": metrica,
        "por_condicao": quadro,
        "contraste_central": _contraste_sobrevive(quadro, metrica),
        "aviso": (
            "Isto NÃO é a varredura. A varredura SELECIONA o melhor; esta função "
            "REPETE com o segundo e compara. Reportar só a vencedora seria omitir "
            "a checagem que o protocolo exige."
        ),
    }


def _contraste_sobrevive(quadro, metrica, alvo="personalizado", controle="entre_pessoas"):
    """O contraste central muda de sinal quando se troca o classificador?"""
    for nome in (alvo, controle):
        if nome not in quadro:
            return {"veredito": None,
                    "motivo": "a condição {!r} não está no quadro".format(nome)}
        if "indisponivel" in quadro[nome]:
            return {"veredito": None, "motivo": quadro[nome]["indisponivel"]}

    esquemas = {quadro[alvo]["esquema"], quadro[controle]["esquema"]}
    if len(esquemas) > 1:
        return {
            "veredito": None,
            "motivo": (
                "{!r} e {!r} foram avaliados por esquemas DIFERENTES ({}). A régua "
                "comum existe só no teste reservado (`avaliar_condicoes_no_teste`), "
                "e subtrair esquemas diferentes da varredura produziria um número "
                "sem significado.".format(
                    alvo, controle, sorted(e for e in esquemas if e)
                ),
            ),
        }

    principal = quadro[alvo]["principal"][metrica] - quadro[controle]["principal"][metrica]
    robusto = quadro[alvo]["robustez"][metrica] - quadro[controle]["robustez"][metrica]
    mesmo_sinal = (principal > 0) == (robusto > 0)
    return {
        "veredito": bool(mesmo_sinal),
        "contraste_principal": float(principal),
        "contraste_robustez": float(robusto),
        "razao_de_magnitude": (
            float(robusto / principal) if principal not in (0.0,) else None
        ),
        "aviso": (
            "O sinal se manteve. Isso NÃO diz que a magnitude se manteve — "
            "compare `razao_de_magnitude` antes de escrever que o resultado é robusto."
            if mesmo_sinal else
            "O SINAL INVERTEU ao trocar o classificador. A conclusão depende do "
            "classificador, e isso precisa ser reportado como resultado, não "
            "resolvido escolhendo a versão que agrada."
        ),
    }


DEGRAUS_DA_CASCATA = (
    "1_glmm_completo",
    "2_reduzido_sem_frase",
    "3_pareado_intra_participante",
)

_POR_QUE_A_FRASE_CAI_PRIMEIRO = (
    "A FRASE cai primeiro, e o termo aleatório NUNCA cai. Duas razões, e a "
    "segunda é a que decide: (a) `frase` custa 7 dos 9 parâmetros do modelo — é o "
    "maior alívio de graus de liberdade pelo menor custo de hipótese; (b) "
    "Eager e Roy (2017) mostram que não convergir com resposta "
    "binária e poucos grupos NÃO é evidência de variância zero, e que remover o "
    "termo aleatório é o modo de falha documentado. Derrubar `(1|participante)` "
    "para 'fazer rodar' inverteria o sentido do achado."
)


def cascata_de_nao_convergencia(observacoes, formula_completa=None):
    """Roda os degraus na ordem pré-especificada e diz em qual o resultado nasceu."""
    import statsmodels.formula.api as smf

    quadro = pd.DataFrame(observacoes)
    for coluna in ("acerto", "condicao", "participante"):
        if coluna not in quadro.columns:
            raise AnaliseIndisponivel(
                "A cascata precisa da coluna {!r}. Vieram: {}.".format(
                    coluna, sorted(quadro.columns)
                )
            )
    tem_frase = "frase" in quadro.columns
    completa = formula_completa or (
        "acerto ~ condicao + frase" if tem_frase else "acerto ~ condicao"
    )

    tentativas = []
    for degrau, formula in (
        (DEGRAUS_DA_CASCATA[0], completa),
        (DEGRAUS_DA_CASCATA[1], "acerto ~ condicao"),
    ):
        if degrau == DEGRAUS_DA_CASCATA[1] and formula == completa:
            tentativas.append({"degrau": degrau, "convergiu": None,
                               "motivo": "sem coluna `frase`: o degrau 2 seria "
                                         "idêntico ao 1, e foi pulado"})
            continue
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter("always")
            try:
                ajuste = smf.mixedlm(
                    formula, quadro, groups=quadro["participante"]
                ).fit()
                convergiu = bool(getattr(ajuste, "converged", False))
            except Exception as falha:
                tentativas.append({"degrau": degrau, "formula": formula,
                                   "convergiu": False,
                                   "erro": "{}: {}".format(type(falha).__name__, falha)})
                continue
            avisos = [str(a.message) for a in capturados]
        tentativas.append({"degrau": degrau, "formula": formula,
                           "convergiu": convergiu, "avisos_do_ajuste": avisos})
        if convergiu:
            return {
                "degrau": degrau,
                "formula": formula,
                "convergiu": True,
                "coeficientes": {k: float(v) for k, v in ajuste.params.items()},
                "tentativas": tentativas,
                "aviso": (
                    "O resultado nasceu no degrau {!r}. Ele TEM de ser relatado "
                    "com o degrau — um número do degrau 2 não é um número do "
                    "degrau 1.".format(degrau)
                ),
                "por_que_a_ordem": _POR_QUE_A_FRASE_CAI_PRIMEIRO,
            }

    pareado = _contraste_pareado(quadro)
    return {
        "degrau": DEGRAUS_DA_CASCATA[2],
        "convergiu": False,
        "pareado": pareado,
        "tentativas": tentativas,
        "aviso": (
            "NENHUM modelo misto convergiu. O resultado vem do contraste PAREADO "
            "intra-participante, que é evidência mais fraca e não sustenta "
            "afirmação sobre a população. " + _AVISO_CONVERGENCIA
        ),
        "por_que_a_ordem": _POR_QUE_A_FRASE_CAI_PRIMEIRO,
    }


def _contraste_pareado(quadro, alvo="personalizado"):
    """Degrau 3: diferença por participante entre o alvo e cada outra condição."""
    taxas = quadro.groupby(["participante", "condicao"])["acerto"].mean().unstack()
    if alvo not in taxas.columns:
        raise AnaliseIndisponivel(
            "A condição {!r} não está nas observações (há: {}).".format(
                alvo, sorted(taxas.columns)
            )
        )
    saida = {}
    for outra in taxas.columns:
        if outra == alvo:
            continue
        pares = taxas[[alvo, outra]].dropna()
        diffs = (pares[alvo] - pares[outra]).tolist()
        saida[outra] = {
            "n_participantes": int(len(diffs)),
            "diferencas": [float(d) for d in diffs],
            "mediana": float(np.median(diffs)) if diffs else None,
            "quantos_a_favor": int(sum(1 for d in diffs if d > 0)),
        }
    return saida


NIVEL_PADRAO = 0.95


def _wilson(k, n, z):
    """Intervalo de Wilson para uma proporção, a peça do método de Newcombe."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    meia = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(centro - meia, 0.0), min(centro + meia, 1.0)


def _ic_escore_pareado(a, b, c, d, nivel=NIVEL_PADRAO):
    """IC de escore para a diferença de proporções pareadas (Newcombe, método 8)."""
    from scipy.stats import norm

    n = a + b + c + d
    if n <= 0:
        raise AnaliseIndisponivel(
            "IC pareado sem pares: n = 0. Verifique se as duas condições foram "
            "avaliadas nas mesmas gravações."
        )
    delta = (b - c) / n

    z = float(norm.ppf(0.5 + nivel / 2))
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = _wilson(a + b, n, z)
    l2, u2 = _wilson(a + c, n, z)

    denom = (a + b) * (c + d) * (a + c) * (b + d)
    phi = ((a * d - b * c) / np.sqrt(denom)) if denom > 0 else 0.0
    phi = float(min(max(phi, -1.0), 1.0))

    termo_lo = (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    termo_hi = (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    inferior = delta - np.sqrt(max(termo_lo, 0.0))
    superior = delta + np.sqrt(max(termo_hi, 0.0))
    return float(delta), float(max(inferior, -1.0)), float(min(superior, 1.0))


def _pares_por_classe(observacoes, alvo, controle):
    """{classe: (a, b, c, d)}: a tabela 2×2 dos pares, por intenção."""
    quadro = pd.DataFrame(observacoes)
    for coluna in ("acerto", "condicao", "gravacao", "classe"):
        if coluna not in quadro.columns:
            raise AnaliseIndisponivel(
                "O tamanho de efeito pareado exige as colunas 'acerto', "
                "'condicao', 'gravacao' e 'classe'; faltou {!r}. 'gravacao' é "
                "o que torna o par possível: sem ela não se sabe qual acerto do "
                "alvo corresponde a qual do controle.".format(coluna)
            )
    largo = quadro.pivot_table(
        index=["gravacao", "classe"], columns="condicao", values="acerto", aggfunc="first"
    )
    for condicao in (alvo, controle):
        if condicao not in largo.columns:
            raise AnaliseIndisponivel(
                "A condição {!r} não aparece nas observações; há {}.".format(
                    condicao, sorted(quadro["condicao"].unique())
                )
            )
    largo = largo.dropna(subset=[alvo, controle])
    if largo.empty:
        raise AnaliseIndisponivel(
            "Nenhuma gravação foi avaliada nas DUAS condições — o dado não é "
            "pareado, e a avaliação exige as mesmas gravações."
        )
    saida = {}
    for classe, bloco in largo.groupby(level="classe"):
        a = bloco[alvo].astype(int).to_numpy()
        c_ = bloco[controle].astype(int).to_numpy()
        saida[str(classe)] = (int(((a == 1) & (c_ == 1)).sum()),
                              int(((a == 1) & (c_ == 0)).sum()),
                              int(((a == 0) & (c_ == 1)).sum()),
                              int(((a == 0) & (c_ == 0)).sum()))
    return saida


def tamanho_de_efeito_por_participante(observacoes, alvo="personalizado",
                                       controle="generico", nivel=NIVEL_PADRAO):
    """ΔUAR pareado por participante, com IC de escore."""
    pares = _pares_por_classe(observacoes, alvo, controle)
    tamanhos = {sum(x) for x in pares.values()}
    a_t = sum(x[0] for x in pares.values())
    b_t = sum(x[1] for x in pares.values())
    c_t = sum(x[2] for x in pares.values())
    d_t = sum(x[3] for x in pares.values())

    delta, lo, hi = _ic_escore_pareado(a_t, b_t, c_t, d_t, nivel)
    n_total = a_t + b_t + c_t + d_t
    b_total, c_total = b_t, c_t
    recalls = {}
    for classe, x in pares.items():
        n_classe = sum(x)
        recalls[classe] = ((x[1] - x[2]) / n_classe) if n_classe else float("nan")
    delta_uar = float(np.mean(list(recalls.values())))
    if not (np.isnan(delta_uar) or -1.0 <= delta_uar <= 1.0):
        raise AnaliseIndisponivel(
            "delta_uar saiu {:.4f}, fora da faixa possivel [-1, 1]. Diferenca de "
            "duas taxas nao pode sair dai: isto e' defeito de calculo, nao "
            "resultado extremo. Confira o denominador por classe em "
            "`_pares_por_classe`.".format(delta_uar)
        )

    return {
        "analise": "tamanho_de_efeito_por_participante (pareado, IC de escore)",
        "alvo": alvo,
        "controle": controle,
        "nivel": nivel,
        "delta_uar": delta_uar,
        "delta_acuracia": float(delta),
        "ic": (lo, hi),
        "n_pares": n_total,
        "n_discordante": int(b_total + c_total),
        "por_classe": {k: {"delta": recalls[k], "a": pares[k][0], "b": pares[k][1],
                           "c": pares[k][2], "d": pares[k][3],
                           "n": sum(pares[k])} for k in pares},
        "classes_balanceadas": len(tamanhos) == 1,
        "aviso": (
            "RECOMENDAÇÃO, NÃO DECISÃO. O IC agrega as "
            "discordâncias de todas as classes e supõe classes de tamanho igual; "
            "confira `classes_balanceadas`. `n_discordante` = 0 significa que as "
            "duas condições acertaram as MESMAS gravações — ausência de "
            "informação, não efeito zero preciso."
        ),
    }

