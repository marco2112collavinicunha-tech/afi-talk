# AFI-Talk

Reconhecimento personalizado de frases funcionais produzidas por adolescentes com apraxia de fala na infância, em português brasileiro.

Este repositório reúne o programa de análise, a página de gravação e as simulações de um estudo que será pré-registrado no OSF antes do início da coleta. Ele não contém dados de participantes. [English version](README.md).

## O estudo

Reconhecedores de fala treinados sobretudo com fala típica erram mais em fala alterada, e na apraxia de fala na infância (AFI) a mesma palavra pode ser produzida de forma diferente a cada tentativa. O estudo mede quanto um reconhecedor ganha com poucas gravações da fala do próprio usuário, numa tarefa de conjunto fechado: identificar qual de doze frases do dia a dia foi pedida ao adolescente com AFI.

Doze adolescentes com AFI e doze com desenvolvimento típico, de 12 a 18 anos, gravam seis sessões no celular da família, em dias diferentes, sempre no mesmo lugar: em casa ou no consultório da fonoaudióloga do adolescente. As hipóteses tratam dos adolescentes com AFI; o grupo com desenvolvimento típico é uma referência descritiva. As sessões 1 e 2 dão as gravações de treino (quatro por frase em cada), as sessões 3 e 4 trazem o bloco de consistência, usado só na hipótese 2 (doze repetições de cada uma de oito frases, quatro frases por sessão), e as sessões 5 e 6 são o teste reservado (três gravações por frase em cada).

Três sistemas classificam as mesmas gravações de teste:

| Sistema | Treinado com | Modelo |
|---|---|---|
| Personalizado | as gravações de treino do próprio adolescente, até oito por frase | classificador de vizinho mais próximo (k = 1) ou de regressão logística sobre a média das representações congeladas do XLSR-53 (wav2vec 2.0) na camada 12 ou 18; camada e classificador são escolhidos por participante, só com as sessões de treino |
| Entre-pessoas igualado | a mesma quantidade de gravações por frase e sessão, de outros adolescentes com AFI | o mesmo procedimento |
| Genérico | nenhum dado do estudo | Whisper medium (via WhisperX), em transcrição livre em português; a transcrição é associada à mais parecida das doze frases por semelhança de caracteres |

A hipótese 1 compara o Personalizado com os outros pelo ganho de cada participante na taxa média de acerto por frase, em testes unilaterais a 4 %, numa sequência fixa: 1a contra o Genérico, 1b contra o Entre-pessoas igualado e 1c contra o Whisper restrito às doze frases por escolha forçada, em que o Whisper pontua cada frase como transcrição da gravação e a mais provável é a escolhida. A comparação igualada pergunta se, com a mesma quantidade de fala, as gravações do próprio usuário valem mais que as de outros usuários com AFI. A hipótese 2, exploratória, pergunta se as frases que o adolescente produz com mais variação acústica entre repetições são as que o Personalizado mais erra. As regras de exclusão, de dados faltantes e de parada, e cada limiar de decisão, serão fixados no pré-registro, registrado antes da primeira gravação de participante.

## Situação

Em 16 de novembro de 2026.

| | |
|---|---|
| Ética | aprovado pelo Comitê de Ética em Pesquisa do Centro de Ciências da Saúde da Universidade Federal da Paraíba (CAAE 00394526.8.0000.5188; parecer 8.739.574, de 17 de setembro de 2026) |
| Recrutamento | encerrado; 12 adolescentes com AFI e 12 com desenvolvimento típico, todos incluídos antes de qualquer sessão |
| Pré-registro | em preparação no OSF; o link será incluído aqui antes da primeira sessão |
| Coleta | não começou; nenhuma gravação de participante existe |
| Análise | única, depois do fim da coleta, por uma versão congelada deste programa, identificada por tag e hash |

Este commit é anterior ao pré-registro. O programa de análise e as simulações da hipótese 1b ainda seguem um plano anterior, em que a escolha forçada e a comparação igualada eram testadas juntas pelo procedimento de Holm e o treino igualado era sorteado, frase a frase, entre todas as gravações dos doadores. No código, a escolha forçada é o degrau 2 (`degrau_2`) da H1a, e o rótulo H1b de `CONTRASTES_DA_H1` designa uma comparação descritiva com o Entre-pessoas completo. Onde o código e o pré-registro divergirem, vale o pré-registro, e a versão congelada vai implementá-lo.

## Pastas

| Pasta | Conteúdo |
|---|---|
| `src/` | pré-processamento do áudio, extração das representações do XLSR-53, classificadores, divisão por sessão, os três sistemas, transcrição pelo Whisper e escolha forçada |
| `inconsistencia/` | o índice espaço-temporal (STI), calculado sobre a envoltória de amplitude, usado na hipótese 2 |
| `analise/` | o programa de análise, rodado uma única vez depois da coleta, e `chaves.json`, com as definições congeladas dele |
| `simulacoes/` | simulações com dados sintéticos que estimam poder e taxa de falso positivo |
| `pagina_coleta/` | cópia de trabalho da página de gravação |

O `config.yaml` guarda as definições do estudo (frases, sessões, revisões dos modelos e seus SHA-256), e o `requirements.txt` lista as dependências do sistema completo. Alguns trechos de código e comentários citam módulos do repositório de trabalho do projeto que ainda não estão aqui, como `src/agenda.py`, `src/io_store.py`, `src/receptor.py`, o servidor local e os testes; eles entram neste repositório quando o programa de análise for congelado.

## Simulações

As simulações usam só `numpy` e `scipy` (versões em `simulacoes/requirements.txt`) e nenhuma gravação. Cada programa grava em `resultados/` um JSON com a semente, as versões dos pacotes e o SHA-256 do programa de análise, e o `tabela.py` de cada pasta resume esses arquivos em `resultados/tabelas.md`. Duas saídas são mais simples: o arquivo de calibração de `sequencia_completa.py` e o `sorteio_doadores.json`, que registra só a semente e o tamanho da rodada. São as tabelas citadas no rascunho do pré-registro e no relatório do projeto. Os comandos abaixo rodam um cenário cada; os outros arquivos de `resultados/` mudam só o argumento do cenário.

**Hipótese 1a (Personalizado contra Genérico).**

```
cd simulacoes/h1
python a_nivel.py S1 20000 resultados/nivel_S1.json
python a_h1.py S2 5000 resultados/alternativas_S2.json 3,4,5,6,8,10,12 0.04
python frases_como_amostra.py resultados/frases_como_amostra.json 4000
python frases_faltantes.py resultados/frases_faltantes.json 2000
python frases_seis_ou_mais.py resultados/frases_seis_ou_mais.json 4000
python misturas_media_nula.py resultados/misturas_media_nula.json 20000
python tabela.py
```

| Programa | O que mede |
|---|---|
| `a_nivel.py` | poder e taxa de falso positivo do teste t de uma amostra em níveis de 2,5 % a 5 %, com 3 a 12 participantes e ganho verdadeiro de 0, 10 ou 15 pontos percentuais, nos cenários S1 a S9; é a base do nível de 4 % |
| `a_h1.py` | compara o teste t com o t de Johnson, o bootstrap-t, a permutação de sinais, o teste de Wilcoxon e dois testes de média aparada (20 %), todos no mesmo nível, nos cenários S1 a S4 e S9 |
| `frases_como_amostra.py` | a taxa de falso positivo com as doze frases fixas e quando o resultado é lido como válido para outras frases |
| `frases_faltantes.py` | a regra registrada para frases faltantes, com perda ao acaso ou maior quando o ganho é baixo |
| `frases_seis_ou_mais.py` | o mesmo cenário, com o teste restrito a quem mantém seis ou mais frases |
| `misturas_media_nula.py` | o teste t, a permutação de sinais e o Wilcoxon quando poucos participantes perdem muito e os demais ganham pouco, com média nula |

| Cenário | Como o ganho varia entre os participantes |
|---|---|
| S1 | de forma simétrica, sem cauda |
| S2 | o Personalizado falha em 20 % dos participantes |
| S3 | o Personalizado falha muito em 10 % dos participantes |
| S4 | o Genérico falha em 20 % dos participantes |
| S5 | participantes agrupados de três em três por serviço de atendimento, com a conclusão lida como válida para outros serviços |
| S6 | metade dos participantes num só serviço e os demais em serviços separados, com a conclusão lida como válida para outros serviços |
| S7 | como S5, com a conclusão lida só para os serviços da amostra |
| S8 | S2 e S5 juntos |
| S9 | ganhos com assimetria forte à esquerda (cauda exponencial de perdas) |

**Hipótese 1b (Personalizado contra Entre-pessoas igualado).** Cada sistema igualado é treinado com gravações dos outros participantes, e por isso os ganhos não são independentes. Seis cenários sintéticos (`mundos` no código), cada um com 3, 5, 8 e 12 participantes; os comandos abaixo rodam um deles:

```
cd simulacoes/entre_pessoas
python h1b_igualado_dois_metodos.py entre_media 8 10000 resultados/entre_media_N8.json 2000
python t_comum_igualado.py entre_media 8 10000 resultados/t_comum/entre_media_N8.json 2000
python tabela.py
```

`h1b_igualado_dois_metodos.py` mede o falso positivo e o poder da regra registrada, que exige que o t corrigido de Nadeau e Bengio (2003), adaptado à saída de um participante por vez, e o jackknife por participante indiquem ganho. `t_comum_igualado.py` aplica o teste t comum aos mesmos estudos; a taxa de falso positivo dele passa de 4 % à medida que os falantes diferem mais e chega a cerca de 13 % em `entre_alta`. Em `nulo_exato` e `nulo_heterogeneo`, a fala do próprio adolescente não traz vantagem. Em `entre_baixa`, `entre_media` e `entre_alta`, a variação própria de cada falante em cada frase cresce, e com ela a vantagem das gravações do próprio falante; `pior_caso` mantém a variação de `entre_media` e aumenta o deslocamento comum a todas as frases do falante (voz e aparelho), os efeitos de sessão e a heterogeneidade entre falantes.

`entre_pessoas_sorteio/sorteio_doadores.py` compara, no mesmo gerador, o sorteio anterior do treino igualado (em cada frase e sessão, quatro gravações sorteadas entre todas as gravações dos doadores) com o sorteio registrado (os mesmos quatro doadores em todas as frases de uma sessão, uma gravação de cada), e relata também uma variante com os mesmos doadores nas duas sessões. As simulações de `entre_pessoas` e de `sequencia` ainda usam o sorteio anterior. O programa roda na própria pasta e grava `sorteio_doadores.json` nela:

```
cd simulacoes/entre_pessoas_sorteio
python sorteio_doadores.py ../entre_pessoas
```

**Sequência anterior.** `sequencia/sequencia_completa.py` mede a taxa de erro por família e o poder de cada passo do plano anterior, com 3, 5, 8 e 12 participantes. O programa também relata a hipótese 1b testada a 4 % logo depois da 1a, a ordem prevista para o pré-registro.

```
cd simulacoes/sequencia
python sequencia_completa.py cal resultados/calibracao.json
python sequencia_completa.py A 12 10000 resultados/a12.json resultados/calibracao.json
python tabela.py
```

## Página de gravação

As famílias usam a página publicada a partir do repositório [coleta-afi](https://github.com/marco2112collavinicunha-tech/coleta-afi), aberta por um link pessoal no próprio celular. A versão usada na coleta tem três botões em cada gravação (Ouvir, Gravar e Parar), seguidos de uma confirmação antes do envio, e será publicada lá antes da primeira sessão, com o commit e o SHA-256 de cada arquivo informados no pré-registro. A pasta `pagina_coleta/` é a cópia de trabalho.

O adolescente vê só os pictogramas da frase, sem texto escrito, para que a tarefa não vire leitura em voz alta; a família pode ver a frase escrita antes de gravar, e ela some enquanto o microfone está ligado. A página toca a frase-modelo com o microfone desligado, grava a imitação do adolescente em WAV de 16 bits, sem compressão, e envia ao endereço que vem no link.

| Arquivo | Função |
|---|---|
| `index.html` | a sessão de gravação: pictogramas, frase-modelo, gravação e envio |
| `captura.js` | gravação pelo microfone (Web Audio API), conversão para WAV e envio |
| `agenda.js` e `rng_python.js` | a ordem das gravações de cada sessão e a divisão das frases de consistência entre as sessões 3 e 4, sorteadas com sementes formadas pela semente do estudo, pelo código do participante e pela sessão, iguais às do módulo `random` do Python |
| `trava_de_participante.js` | bloqueia a sessão de um participante enquanto algum item do protocolo estiver pendente |
| `protocolo.json` | o protocolo de coleta: sessões, frases e pictogramas |
| `diagnostico.html` e `diagnostico_microfone.js` | teste do microfone, só com códigos de engenharia (`ENG###`), que não grava nem envia nada |

O endereço de envio e a chave de cada família vêm no link pessoal e não estão no código. As gravações da frase-modelo, feitas por uma fonoaudióloga da equipe, não estão neste repositório.

Os pictogramas de `pagina_coleta/pictogramas/` são do ARASAAC (arasaac.org), de autoria de Sergio Palao, propriedade do Governo de Aragão, sob a licença CC BY-NC-SA. Um deles foi adaptado pelo projeto (`3345_sem_texto.png`, sem o texto do balão) e segue a mesma licença.

## Dados e privacidade

Nenhuma gravação ou dado de participante está neste repositório, e nenhuma gravação será publicada: a voz pode identificar quem fala, e o estudo não divulga as gravações. Sistemas treinados com gravações de participantes também não serão publicados, porque o classificador de vizinho mais próximo guarda as representações das gravações de treino. Resultados por participante aparecem só pelo código de participante.

## Autor

Marco Antonio Collavini Cunha

## Como citar

Veja o arquivo `CITATION.cff`. Cada versão usada no estudo, no pré-registro e no congelamento da análise, receberá uma tag.

## Licença

Código sob a licença MIT (arquivo `LICENSE`). Os textos deste repositório estão sob a licença CC BY 4.0. Os pictogramas seguem a própria licença, CC BY-NC-SA.
