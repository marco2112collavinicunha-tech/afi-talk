"""Carregamento do config.yaml — usado por todos os módulos do pipeline."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

VARIAVEL_DOS_DADOS = "AFI_DADOS"


def dentro_do_repositorio(absoluta):
    """`absoluta` (já resolvida) é o repositório ou está dentro dele."""
    raiz = str(RAIZ_PROJETO).casefold()
    alvo = str(Path(absoluta)).casefold()
    return alvo == raiz or alvo.startswith(raiz.rstrip(os.sep) + os.sep)


def checkout_do_projeto_acima(absoluta):
    """A primeira pasta, de `absoluta` para cima, que é outra cópia do projeto: tem `src/receptor.py` e `pagina_coleta/`."""
    for pasta in [Path(absoluta), *Path(absoluta).parents]:
        if (pasta / "src" / "receptor.py").is_file() and (pasta / "pagina_coleta").is_dir():
            return pasta
    return None


class RaizDosDadosNaoDefinida(RuntimeError):
    """`AFI_DADOS` ausente."""


def raiz_dos_dados():
    """Pasta onde ficam os dados de participante, sempre fora do repositório."""
    bruto = os.environ.get(VARIAVEL_DOS_DADOS)
    if not bruto or not bruto.strip():
        raise RaizDosDadosNaoDefinida(
            "A variável de ambiente {0} não está definida, e sem ela o sistema não "
            "sabe onde guardar os dados de participante.\n\n"
            "Defina-a apontando para uma pasta FORA do repositório e FORA de "
            "qualquer pasta sincronizada por nuvem:\n"
            "    export {0}=~/dados-afi          # macOS/Linux\n"
            "    $env:{0} = \"$HOME\\dados-afi\"    # Windows (PowerShell)\n\n"
            "Atenção: Nao use ~/Documents, ~/Desktop nem ~/Área de Trabalho: no macOS elas "
            "costumam ser sincronizadas pelo iCloud, e a regra existe porque ignorar no "
            "git nao protege contra sincronizacao.\n"
            "Atenção: Nao ha padrao de fallback de proposito: cair para dentro do repositorio "
            "seria violar em silencio a unica fronteira irreversivel do projeto.".format(
                VARIAVEL_DOS_DADOS
            )
        )
    raiz = Path(bruto).expanduser()
    absoluta = (raiz if raiz.is_absolute() else Path.cwd() / raiz).resolve()
    outro = checkout_do_projeto_acima(absoluta)
    if outro is not None and not dentro_do_repositorio(absoluta):
        raise RaizDosDadosNaoDefinida(
            "{} aponta para dentro de um checkout do projeto ({}), e a regra de guarda proíbe "
            "dado de participante no caminho de qualquer cópia do repositório. "
            "Aponte para fora do repositório e fora de pasta sincronizada.".format(
                VARIAVEL_DOS_DADOS, outro))
    if dentro_do_repositorio(absoluta):
        raise RaizDosDadosNaoDefinida(
            "{} aponta para DENTRO do repositório ({}), e a regra de guarda proíbe dado de "
            "participante no caminho do repositório mesmo em pasta ignorada — "
            "ignorar protege contra o git, não contra backup e sincronização de "
            "nuvem. Aponte para fora do repositório e fora de pasta sincronizada.".format(
                VARIAVEL_DOS_DADOS, absoluta
            )
        )
    return raiz


def carregar_config(caminho=None):
    """Lê o config.yaml e devolve um dicionário com todas as escolhas do projeto."""
    caminho = Path(caminho) if caminho is not None else RAIZ_PROJETO / "config.yaml"
    if not caminho.exists():
        raise FileNotFoundError(
            "Arquivo de configuração não encontrado: {}".format(caminho)
        )
    with open(caminho, encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)


class CaminhoDoDadoNaoDecidido(RuntimeError):
    """`coleta.caminho_do_dado` sem valor ou com trajeto não implementado."""


CAMINHOS_DO_DADO = {
    "servidor_local": "a familia grava e o arquivo vai direto para a maquina do estudo",
    "drive": "a familia grava e o arquivo passa pelo Drive institucional antes de ser ingerido",
}


def caminho_do_dado(config):
    """Qual trajeto o áudio percorre entre o aparelho da família e o estudo."""
    escolhido = (config.get("coleta") or {}).get("caminho_do_dado")
    if not escolhido:
        raise CaminhoDoDadoNaoDecidido(
            "`coleta.caminho_do_dado` esta sem valor, e nao ha padrao de proposito.\n\n"
            "O trajeto do audio muda guarda, retirada e rastreabilidade. Ele esta "
            "em aberto: depende da escolha da pesquisadora responsavel pela etica, "
            "e o valor escolhido e' registrado nesta chave do config.\n"
            "Valores implementados: {}.".format(", ".join(sorted(CAMINHOS_DO_DADO)))
        )
    if escolhido not in CAMINHOS_DO_DADO:
        raise CaminhoDoDadoNaoDecidido(
            "`coleta.caminho_do_dado` vale {!r}, que nao e um trajeto implementado.\n"
            "Valores implementados: {}.".format(escolhido, ", ".join(sorted(CAMINHOS_DO_DADO)))
        )
    return escolhido

