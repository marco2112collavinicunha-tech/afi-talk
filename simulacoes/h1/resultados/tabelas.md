# Primeira comparação (dados sintéticos)

Gerado por `tabela.py` a partir dos JSON desta pasta. SINTÉTICO: não cobre fala apráxica real.

## Nível do teste (`a_nivel.py`)

Falso positivo (ganho verdadeiro 0) e poder (ganho de 10 e 15 pontos) do t do programa. Pior falso positivo entre 3 e 12 participantes, com o N entre parênteses, e poder com 12 participantes. Em porcentagem.

| cenário | pior FP a 5% | pior FP a 4% | poder +10 a 5% | poder +10 a 4% | poder +15 a 5% | poder +15 a 4% |
|---|---|---|---|---|---|---|
| S1 simétrico | 5,11 (6) | 4,04 (6) | 70,19 | 65,69 | 94,48 | 92,96 |
| S2 o Personalizado falha em 20% | 6,34 (4) | 5,17 (4) | 51,94 | 47,61 | 77,87 | 74,19 |
| S3 o Personalizado falha muito em 10% | 6,97 (6) | 5,73 (6) | 53,80 | 49,90 | 76,13 | 72,73 |
| S4 o Genérico falha em 20% | 4,28 (10) | 3,42 (10) | 58,46 | 53,55 | 88,03 | 85,08 |
| S5 trios por serviço, outros serviços | 9,69 (10) | 8,21 (6) | 57,34 | 53,73 | 81,99 | 79,03 |
| S6 metade num serviço, outros serviços | 10,42 (12) | 8,80 (12) | 57,67 | 54,18 | 81,25 | 78,67 |
| S7 trios por serviço, estes serviços | 5,05 (3) | 3,98 (3) | 61,57 | 56,17 | 90,35 | 87,76 |
| S8 S2 e S5 juntos | 8,87 (3) | 7,32 (3) | 46,39 | 42,38 | 70,65 | 66,72 |
| S9 assimetria forte | 7,20 (5) | 5,91 (5) | 80,51 | 77,39 | 96,51 | 95,47 |

Poder do t do programa a 4% pelo número de participantes, no cenário simétrico (Figura 6):

| N | +10 | +15 |
|---|---|---|
| 3 | 15,96 | 26,25 |
| 4 | 22,91 | 40,08 |
| 5 | 29,03 | 52,38 |
| 6 | 36,10 | 63,34 |
| 8 | 47,08 | 78,03 |
| 10 | 58,35 | 87,02 |
| 12 | 65,69 | 92,96 |

## Alternativas ao t (`a_h1.py`)

Todos os testes no mesmo nível. Pior falso positivo entre 3 e 12 participantes (N entre parênteses) e poder com ganho de 10 pontos com 3, 8 e 12 participantes. Em porcentagem.

### S1 simétrico (nível de 4%, 5000 réplicas)

| teste | pior FP | poder +10, N = 3 | poder +10, N = 8 | poder +10, N = 12 |
|---|---|---|---|---|
| t do programa | 4,42 (12) | 15,38 | 46,78 | 66,36 |
| t de Johnson | 4,58 (12) | 13,90 | 44,86 | 64,76 |
| bootstrap-t | 4,60 (12) | 16,78 | 43,28 | 64,14 |
| permutação de sinais | 4,34 (12) | 0,00 | 45,78 | 66,28 |
| Wilcoxon | 4,16 (12) | 0,00 | 45,34 | 63,80 |
| média aparada (a) | 5,54 (5) | 6,56 | 29,38 | 51,72 |
| média aparada (b) | 4,78 (5) | 15,38 | 40,84 | 58,22 |

### S2 o Personalizado falha em 20% (nível de 4%, 5000 réplicas)

| teste | pior FP | poder +10, N = 3 | poder +10, N = 8 | poder +10, N = 12 |
|---|---|---|---|---|
| t do programa | 5,46 (5) | 14,64 | 34,92 | 45,86 |
| t de Johnson | 5,44 (5) | 12,68 | 30,68 | 41,32 |
| bootstrap-t | 5,66 (3) | 12,76 | 29,08 | 39,86 |
| permutação de sinais | 4,94 (8) | 0,00 | 33,70 | 45,58 |
| Wilcoxon | 4,98 (8) | 0,00 | 33,16 | 45,34 |
| média aparada (a) | 7,28 (5) | 6,20 | 24,78 | 41,94 |
| média aparada (b) | 6,60 (10) | 14,64 | 35,04 | 46,72 |

### S3 o Personalizado falha muito em 10% (nível de 4%, 5000 réplicas)

| teste | pior FP | poder +10, N = 3 | poder +10, N = 8 | poder +10, N = 12 |
|---|---|---|---|---|
| t do programa | 5,90 (5) | 17,54 | 38,94 | 50,76 |
| t de Johnson | 5,70 (6) | 15,50 | 34,80 | 44,66 |
| bootstrap-t | 6,08 (3) | 15,72 | 33,52 | 43,50 |
| permutação de sinais | 5,34 (10) | 0,00 | 38,38 | 50,46 |
| Wilcoxon | 5,76 (12) | 0,00 | 37,22 | 52,20 |
| média aparada (a) | 7,42 (10) | 6,90 | 32,06 | 52,22 |
| média aparada (b) | 6,82 (10) | 17,54 | 43,22 | 57,16 |

### S4 o Genérico falha em 20% (nível de 4%, 5000 réplicas)

| teste | pior FP | poder +10, N = 3 | poder +10, N = 8 | poder +10, N = 12 |
|---|---|---|---|---|
| t do programa | 3,58 (12) | 11,60 | 35,62 | 53,16 |
| t de Johnson | 3,98 (12) | 10,92 | 36,98 | 55,32 |
| bootstrap-t | 4,70 (3) | 15,16 | 37,32 | 55,38 |
| permutação de sinais | 3,58 (12) | 0,00 | 35,58 | 53,00 |
| Wilcoxon | 3,34 (12) | 0,00 | 35,24 | 50,04 |
| média aparada (a) | 3,94 (5) | 4,66 | 19,52 | 37,32 |
| média aparada (b) | 3,58 (6) | 11,60 | 30,44 | 43,06 |

### S9 assimetria forte (nível de 4%, 5000 réplicas)

| teste | pior FP | poder +10, N = 3 | poder +10, N = 8 | poder +10, N = 12 |
|---|---|---|---|---|
| t do programa | 5,84 (5) | 26,92 | 61,52 | 78,26 |
| t de Johnson | 5,62 (5) | 23,12 | 52,56 | 69,80 |
| bootstrap-t | 5,84 (3) | 19,62 | 49,84 | 68,70 |
| permutação de sinais | 5,46 (8) | 0,00 | 60,38 | 77,80 |
| Wilcoxon | 5,76 (12) | 0,00 | 59,34 | 78,14 |
| média aparada (a) | 8,18 (10) | 12,32 | 50,88 | 75,74 |
| média aparada (b) | 7,40 (10) | 26,92 | 62,86 | 79,94 |

## Frases fixas e frases lidas como amostra (`frases_como_amostra.py`)

Porcentagem de estudos em que a primeira comparação sai sustentada, a 4%, sem ganho médio. tau é o tamanho dos efeitos de frase. 4000 réplicas por célula.

| tau | N | estas doze frases (falso positivo) | lido para outras frases |
|---|---|---|---|
| 0,05 | 3 | 4,20 | 4,15 |
| 0,05 | 4 | 3,95 | 4,85 |
| 0,05 | 5 | 3,62 | 4,23 |
| 0,05 | 6 | 4,00 | 5,10 |
| 0,05 | 8 | 3,80 | 5,42 |
| 0,05 | 10 | 4,15 | 4,95 |
| 0,05 | 12 | 4,25 | 6,15 |
| 0,10 | 3 | 4,20 | 4,95 |
| 0,10 | 4 | 4,52 | 5,78 |
| 0,10 | 5 | 4,05 | 7,65 |
| 0,10 | 6 | 3,20 | 7,67 |
| 0,10 | 8 | 4,00 | 8,77 |
| 0,10 | 10 | 3,80 | 11,22 |
| 0,10 | 12 | 3,48 | 11,95 |
| 0,20 | 3 | 3,67 | 8,77 |
| 0,20 | 4 | 4,67 | 12,15 |
| 0,20 | 5 | 3,90 | 15,53 |
| 0,20 | 6 | 4,50 | 16,83 |
| 0,20 | 8 | 3,82 | 18,62 |
| 0,20 | 10 | 4,30 | 21,38 |
| 0,20 | 12 | 3,65 | 24,30 |

Poder com ganho de 10 pontos igual em todos os participantes e sem efeito de frase:

| N | poder |
|---|---|
| 3 | 33,05 |
| 4 | 52,25 |
| 5 | 66,33 |
| 6 | 76,02 |
| 8 | 90,18 |
| 10 | 95,67 |
| 12 | 98,38 |

## Frases faltantes (`frases_faltantes.py`)

Regra das frases presentes, a 4%. FP = falso positivo (ganho 0); poder com ganho de 10 pontos; cobertura do intervalo de 92% sem ganho. 2000 réplicas por célula.

| N | perda | estrutura | células faltando | FP | poder | cobertura |
|---|---|---|---|---|---|---|
| 3 | aleatória | aditiva | 24,75 | 4,05 | 20,80 | 92,15 |
| 3 | aleatória | com interação | 24,86 | 4,35 | 19,40 | 90,85 |
| 3 | ligada ao ganho | aditiva | 32,00 | 6,40 | 26,45 | 90,80 |
| 3 | ligada ao ganho | com interação | 31,94 | 6,90 | 27,15 | 90,55 |
| 5 | aleatória | aditiva | 25,19 | 3,85 | 38,80 | 91,90 |
| 5 | aleatória | com interação | 24,95 | 3,45 | 39,30 | 92,80 |
| 5 | ligada ao ganho | aditiva | 32,02 | 7,35 | 48,70 | 90,10 |
| 5 | ligada ao ganho | com interação | 32,00 | 9,05 | 53,40 | 89,45 |
| 8 | aleatória | aditiva | 25,16 | 3,60 | 61,70 | 92,05 |
| 8 | aleatória | com interação | 25,13 | 3,60 | 61,35 | 92,25 |
| 8 | ligada ao ganho | aditiva | 32,04 | 7,50 | 73,00 | 90,45 |
| 8 | ligada ao ganho | com interação | 32,08 | 10,80 | 74,85 | 88,25 |
| 12 | aleatória | aditiva | 25,11 | 3,15 | 81,45 | 93,15 |
| 12 | aleatória | com interação | 25,12 | 3,10 | 79,95 | 93,25 |
| 12 | ligada ao ganho | aditiva | 31,96 | 9,05 | 89,05 | 89,30 |
| 12 | ligada ao ganho | com interação | 32,11 | 12,50 | 92,05 | 86,65 |
