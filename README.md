# AFI-Talk

Código do projeto **AFI-Talk: sistema de reconhecimento personalizado de intenções comunicativas em adolescentes com apraxia de fala na infância**.

Na apraxia de fala na infância (AFI), a mesma palavra pode sair diferente a cada tentativa, e os reconhecedores de voz comuns erram muito com essa fala. O projeto compara três sistemas que ouvem a gravação de um adolescente e dizem qual de doze frases do dia a dia ele quis dizer:

- **Personalizado**: representações do XLSR-53 (wav2vec 2.0, congelado) e um classificador treinado com poucas gravações do próprio adolescente;
- **Entre-pessoas igualado**: a mesma receita, treinada com a mesma quantidade de gravações de outros adolescentes com AFI;
- **Genérico**: o Whisper, que transcreve a fala e escolhe a frase mais parecida, em transcrição livre e em escolha forçada.

No código, a comparação do Personalizado com o Genérico aparece como H1a, e a comparação com o Entre-pessoas igualado, como H1b.

## Pastas

| Pasta | O que tem |
|---|---|
| `src/` | pré-processamento do áudio, extração das representações, classificadores, divisão entre treino e teste e os três sistemas |
| `inconsistencia/` | índice espaço-temporal (STI), que mede o quanto as repetições de uma frase variam, calculado sobre a envoltória de amplitude |
| `analise/` | programa que aplica as regras de decisão do estudo às gravações, uma única vez, depois da coleta |
| `simulacoes/` | simulações com dados sintéticos que estimam o poder e a taxa de falso positivo das comparações |
| `pagina_coleta/` | página de gravação usada pelas famílias no celular |

## Simulações

As simulações usam só `numpy` e `scipy` (versões em `simulacoes/requirements.txt`) e não precisam de nenhuma gravação. Cada programa grava em `resultados/` um JSON com a semente, as versões usadas e o sha256 do programa de análise, e o `tabela.py` de cada pasta resume os resultados em `resultados/tabelas.md`. São os números da seção 6 do relatório, inclusive os das Figuras 6 e 7.

**Primeira comparação (Personalizado contra Genérico).**

```
cd simulacoes/h1
python a_nivel.py S1 20000 resultados/nivel_S1.json
python a_h1.py S2 5000 resultados/alternativas_S2.json 3,4,5,6,8,10,12 0.04
python frases_como_amostra.py resultados/frases_como_amostra.json 4000
python frases_faltantes.py resultados/frases_faltantes.json 2000
python tabela.py
```

| Programa | O que mede |
|---|---|
| `a_nivel.py` | poder e taxa de falso positivo do teste t em vários níveis, de 3 a 12 participantes, com ganho verdadeiro de 0, 10 ou 15 pontos percentuais (Figura 6 e escolha do nível de 4%) |
| `a_h1.py` | o teste t contra as alternativas simuladas (t de Johnson, bootstrap-t, permutação de sinais, Wilcoxon e média aparada em duas formas), todas no mesmo nível, nos cenários S1 a S4 e S9 |
| `frases_como_amostra.py` | a taxa de falso positivo com as doze frases fixas e quando o resultado é lido como válido para outras frases |
| `frases_faltantes.py` | a regra das frases presentes quando faltam frases a alguns participantes, inclusive quando faltam as de menor ganho |

Cenários de `a_nivel.py` e `a_h1.py`:

| Cenário | Como o ganho varia entre os participantes |
|---|---|
| S1 | de forma simétrica |
| S2 | o Personalizado falha em 20% dos participantes |
| S3 | o Personalizado falha muito em 10% dos participantes |
| S4 | o Genérico falha em 20% dos participantes |
| S5 | participantes agrupados de três em três por serviço de atendimento |
| S6 | metade dos participantes no mesmo serviço |
| S7 | como S5, com a conclusão valendo só para os serviços da amostra |
| S8 | S2 e S5 juntos |
| S9 | ganho com assimetria forte |

**Comparação com o Entre-pessoas igualado.** Seis cenários, com 3, 5, 8 e 12 participantes:

```
cd simulacoes/entre_pessoas
python h1b_igualado_dois_metodos.py entre_media 8 10000 resultados/entre_media_N8.json 2000
python t_comum_igualado.py entre_media 8 10000 resultados/t_comum/entre_media_N8.json 2000
python tabela.py
```

`h1b_igualado_dois_metodos.py` mede a regra dos dois métodos (Figura 7). `t_comum_igualado.py` refaz os mesmos estudos e aplica a eles o teste t comum, que trata os ganhos como independentes; é ele que mostra por que essa comparação precisa de outra regra. Nos cenários `nulo_exato` e `nulo_heterogeneo` a fala do próprio adolescente não traz vantagem; em `entre_baixa`, `entre_media` e `entre_alta` a diferença entre falantes cresce; `pior_caso` tem mais variação de voz, aparelho e sessão.

**Sequência das três comparações.** A primeira comparação e, se ela for sustentada, a escolha forçada e o Entre-pessoas igualado, testados pelo procedimento de Holm:

```
cd simulacoes/sequencia
python sequencia_completa.py cal resultados/calibracao.json
python sequencia_completa.py A 12 10000 resultados/a12.json resultados/calibracao.json
python tabela.py
```

`sequencia_completa.py` mede a chance de pelo menos uma conclusão falsa na sequência inteira, quando nenhuma das comparações tem efeito e quando só algumas têm, e o poder de cada parte da sequência, com 3, 5, 8 e 12 participantes.

## Página de coleta

A pasta `pagina_coleta/` tem a página que a família abre no celular, pelo link pessoal de cada participante. Ela diz qual é a próxima gravação, mostra o pictograma da frase, toca a frase-modelo, grava a imitação do adolescente e envia cada gravação ao endereço de envio que vem no link.

| Arquivo | O que faz |
|---|---|
| `index.html` | a sessão de gravação: pictograma, frase-modelo, gravação e envio |
| `captura.js` | gravação pelo microfone do celular (Web Audio API), conversão para WAV e envio |
| `agenda.js` e `rng_python.js` | ordem das gravações de cada sessão, sorteada com uma semente fixa e reproduzível em Python |
| `trava_de_participante.js` | não deixa abrir a sessão de um participante enquanto algo do protocolo estiver marcado como pendente |
| `protocolo.json` | protocolo de coleta: sessões, frases e pictogramas |
| `diagnostico.html` e `diagnostico_microfone.js` | teste do microfone, só com os códigos de teste, que não grava nem envia nada |

O endereço de envio e a chave de cada família vêm no link pessoal e não estão no código. As gravações da frase-modelo, feitas por uma fonoaudióloga da equipe, não estão neste repositório. Os comentários do código citam programas e documentos do repositório de trabalho do projeto (como `src/agenda.py`, o servidor local e os testes), que não fazem parte desta cópia.

Os pictogramas de `pagina_coleta/pictogramas/` são do ARASAAC (arasaac.org), de autoria de Sergio Palao e propriedade do Governo de Aragão, sob a licença CC BY-NC-SA. Um deles foi adaptado pelo projeto (`3345_sem_texto.png`, sem o texto do balão) e segue a mesma licença.

## Dados

Nenhuma gravação nem dado de participante está neste repositório. O estudo foi aprovado pelo Comitê de Ética em Pesquisa.

## Autor

Marco Antonio Collavini Cunha

## Licença

Código sob a licença MIT (arquivo `LICENSE`). Os pictogramas seguem a licença própria, CC BY-NC-SA.
