# AFI-Talk

Personalized recognition of functional phrases produced by adolescents with childhood apraxia of speech, in Brazilian Portuguese.

This repository holds the analysis program, the recording page and the simulations of a study that will be preregistered on OSF before data collection begins. It holds no participant data. [Versão em português](README.pt-BR.md).

## The study

Speech recognizers trained mostly on typical speech make more errors on disordered speech, and in childhood apraxia of speech (CAS; in Portuguese, *apraxia de fala na infância*, AFI) the same word can be produced differently from one attempt to the next. The study measures how much a recognizer gains from a few recordings of the user's own speech in a closed-set task: identifying which of twelve everyday phrases an adolescent with CAS was asked to repeat.

Twelve adolescents with CAS and twelve typically developing adolescents, aged 12 to 18, record six sessions on the family's phone, on different days, always in the same place: at home or at the adolescent's speech-language pathologist's office. The hypotheses concern the adolescents with CAS; the typically developing group is a descriptive reference. Sessions 1 and 2 provide the training recordings (four per phrase in each), sessions 3 and 4 contain a consistency block used only in Hypothesis 2 (twelve repetitions of each of eight phrases, four phrases per session), and sessions 5 and 6 are the held-out test (three recordings per phrase in each).

Three systems classify the same test recordings:

| System | Trained on | Model |
|---|---|---|
| Personalized | the adolescent's own training recordings, up to eight per phrase | 1-nearest-neighbor or logistic-regression classifier on mean-pooled frozen XLSR-53 (wav2vec 2.0) representations from layer 12 or 18; layer and classifier are chosen per participant on the training sessions only |
| Matched cross-speaker | the same number of recordings per phrase and session, from other adolescents with CAS | the same procedure |
| Generic | no study data | Whisper medium (through WhisperX), unconstrained transcription in Portuguese; the transcript is mapped to the most similar of the twelve phrases by character similarity |

Hypothesis 1 compares the Personalized system with the others through per-participant gains in unweighted average recall, in a fixed sequence of one-sided tests at the 4 % level: 1a against the Generic system, 1b against the Matched cross-speaker system, and 1c against Whisper restricted to the twelve phrases by forced choice, in which Whisper scores each phrase as a transcript of the recording and the most likely phrase is chosen. The matched comparison asks whether, for the same amount of speech, a user's own recordings are worth more than those of other users with CAS. Hypothesis 2, exploratory, asks whether the phrases an adolescent produces with more acoustic variability across repetitions are the ones the Personalized system recognizes less often. Exclusion, missing-data and stopping rules, and every decision threshold, will be fixed in the preregistration, which is registered before the first participant recording.

## Status

As of 16 November 2026.

| | |
|---|---|
| Ethics | approved by the Research Ethics Committee of the Health Sciences Center, Federal University of Paraíba (CAAE 00394526.8.0000.5188; approval 8.739.574, 17 September 2026) |
| Recruitment | closed; 12 adolescents with CAS and 12 typically developing adolescents enrolled before any session |
| Preregistration | in preparation on OSF; the link will be added here before the first session |
| Data collection | not started; no participant recording exists |
| Analysis | a single run after data collection, by a frozen version of this program identified by tag and hash |

This commit predates the preregistration. The analysis program and the simulations of Hypothesis 1b still follow an earlier plan, in which the forced-choice comparison and the matched comparison were tested together by the Holm procedure and the matched training set was drawn per phrase from all donor recordings. In the code, the forced-choice comparison is step 2 (`degrau_2`) of H1a, and the label H1b in `CONTRASTES_DA_H1` names a descriptive comparison with the full cross-speaker system. Where the code and the preregistration differ, the preregistration governs, and the frozen version will implement it.

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | audio preprocessing, XLSR-53 representation extraction, classifiers, the split by session, the three systems, Whisper transcription and forced choice |
| `inconsistencia/` | the spatiotemporal index (STI) computed on the amplitude envelope, used in Hypothesis 2 |
| `analise/` | the analysis program, run once after data collection, and `chaves.json`, its frozen settings |
| `simulacoes/` | simulations on synthetic data that estimate power and false-positive rates |
| `pagina_coleta/` | working copy of the recording page |

`config.yaml` holds the study settings (phrases, sessions, model revisions and their SHA-256), and `requirements.txt` lists the dependencies of the full pipeline. Identifiers and comments are in Portuguese. Some code and comments refer to modules of the project's working repository that are not yet here, such as `src/agenda.py`, `src/io_store.py`, `src/receptor.py`, the local server and the tests; they will be added when the analysis program is frozen.

## Simulations

The simulations need only `numpy` and `scipy` (versions in `simulacoes/requirements.txt`) and use no recordings. Each program writes a JSON file to `resultados/` with its seed, the package versions and the SHA-256 of the analysis program, and `tabela.py` in each folder summarizes these files in `resultados/tabelas.md`. Two outputs are simpler: the calibration file of `sequencia_completa.py`, and `sorteio_doadores.json`, which records only the seed and the run size. These tables are cited in the draft preregistration and in the project report. The commands below run one scenario each; the other files in `resultados/` change only the scenario argument.

**Hypothesis 1a (Personalized vs. Generic).**

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

| Program | What it measures |
|---|---|
| `a_nivel.py` | power and false-positive rate of the one-sample t test at levels from 2.5 % to 5 %, with 3 to 12 participants and a true gain of 0, 10 or 15 percentage points, in scenarios S1 to S9; the basis for the 4 % level |
| `a_h1.py` | compares the t test with Johnson's t, the bootstrap-t, the sign-flip permutation test, the Wilcoxon signed-rank test and two trimmed-mean tests (20 % trimming), all at the same level, in scenarios S1 to S4 and S9 |
| `frases_como_amostra.py` | the false-positive rate with the twelve phrases fixed, and when the result is read as valid for other phrases |
| `frases_faltantes.py` | the registered rule for missing phrases, with phrases missing at random or more often when the gain is low |
| `frases_seis_ou_mais.py` | the same scenario, with the test restricted to participants who keep six or more phrases |
| `misturas_media_nula.py` | the t test, the sign-flip test and the Wilcoxon test when a few participants lose a lot and the rest gain a little, with zero mean |

| Scenario | How the gain varies across participants |
|---|---|
| S1 | symmetrically, with no tail |
| S2 | the Personalized system fails in 20 % of participants |
| S3 | the Personalized system fails badly in 10 % of participants |
| S4 | the Generic system fails in 20 % of participants |
| S5 | participants clustered in threes by clinical service, with the conclusion read as valid for other services |
| S6 | half of the participants in one service and the rest in separate services, with the conclusion read as valid for other services |
| S7 | as S5, with the conclusion read only for the services in the sample |
| S8 | S2 and S5 combined |
| S9 | strongly left-skewed gains (an exponential tail of losses) |

**Hypothesis 1b (Personalized vs. Matched cross-speaker).** Each matched system is trained on the other participants' recordings, so the per-participant gains are not independent. Six synthetic scenarios (`mundos` in the code), each run with 3, 5, 8 and 12 participants; the commands below run one of them:

```
cd simulacoes/entre_pessoas
python h1b_igualado_dois_metodos.py entre_media 8 10000 resultados/entre_media_N8.json 2000
python t_comum_igualado.py entre_media 8 10000 resultados/t_comum/entre_media_N8.json 2000
python tabela.py
```

`h1b_igualado_dois_metodos.py` measures the false-positive rate and power of the preregistered rule, which requires both the corrected resampled t test of Nadeau and Bengio (2003), adapted to leaving one participant out, and a participant-level jackknife to indicate a gain. `t_comum_igualado.py` applies the ordinary t test to the same studies; its false-positive rate rises above 4 % as speakers differ more, to about 13 % in `entre_alta`. In `nulo_exato` and `nulo_heterogeneo` the adolescent's own speech brings no advantage. In `entre_baixa`, `entre_media` and `entre_alta`, each speaker's own variation in every phrase grows, and with it the advantage of the speaker's own recordings; `pior_caso` keeps the speaker variation of `entre_media` and raises the shift common to all of a speaker's phrases (voice and phone), the session effects and the heterogeneity between speakers.

`entre_pessoas_sorteio/sorteio_doadores.py` compares, on the same generator, the earlier draw of the matched training set (for each phrase and session, four recordings drawn from all donor recordings) with the preregistered draw (the same four donors for every phrase within a session, one recording each), and reports a variant with the same donors in both sessions. The simulations in `entre_pessoas` and `sequencia` still use the earlier draw. Run it from its own folder; it writes `sorteio_doadores.json` there:

```
cd simulacoes/entre_pessoas_sorteio
python sorteio_doadores.py ../entre_pessoas
```

**Earlier sequence.** `sequencia/sequencia_completa.py` measures the family-wise error rate and the power of each step of the earlier plan, with 3, 5, 8 and 12 participants. It also reports Hypothesis 1b tested at 4 % right after 1a, the order planned for the preregistration.

```
cd simulacoes/sequencia
python sequencia_completa.py cal resultados/calibracao.json
python sequencia_completa.py A 12 10000 resultados/a12.json resultados/calibracao.json
python tabela.py
```

## Recording page

Families use the page published from the [coleta-afi](https://github.com/marco2112collavinicunha-tech/coleta-afi) repository, opened through a personal link on their own phone. The version used in data collection has three buttons for each recording (Listen, Record, Stop), followed by a confirmation before upload, and will be published there before the first session, with the commit and the SHA-256 of each file stated in the preregistration. `pagina_coleta/` is the working copy.

The adolescent sees only the phrase's pictograms, with no written text, so that the task does not turn into reading aloud; the family can display the written phrase before recording, and it is hidden while the microphone is on. The page plays the model phrase with the microphone off, records the adolescent's imitation as uncompressed 16-bit WAV and sends it to the upload address given in the link.

| File | Role |
|---|---|
| `index.html` | the recording session: pictograms, model phrase, recording and upload |
| `captura.js` | microphone capture (Web Audio API), WAV encoding and upload |
| `agenda.js` and `rng_python.js` | the order of recordings in each session and the split of the consistency phrases between sessions 3 and 4, drawn with seeds built from the study seed, the participant code and the session, identical to Python's `random` module |
| `trava_de_participante.js` | blocks a participant's session while any protocol item is still marked as pending |
| `protocolo.json` | the collection protocol: sessions, phrases and pictograms |
| `diagnostico.html` and `diagnostico_microfone.js` | microphone check, available only with engineering test codes (`ENG###`); it neither records nor uploads |

The upload address and each family's key are carried in the personal link and are not in the code. The model-phrase recordings, made by a speech-language pathologist on the team, are not in this repository.

The pictograms in `pagina_coleta/pictogramas/` are from ARASAAC (arasaac.org), by Sergio Palao, owned by the Government of Aragon, under CC BY-NC-SA. One of them was adapted by the project (`3345_sem_texto.png`, with the speech-bubble text removed) and keeps the same license.

## Data and privacy

No participant recordings or data are in this repository, and no recordings will be published: a voice can identify its speaker, and the study does not release recordings. Systems trained on participant recordings will not be published either, because a nearest-neighbor classifier stores the representations of its training recordings. Per-participant results are reported only by participant code.

## Author

Marco Antonio Collavini Cunha

## Citation

See `CITATION.cff`. Each version used in the study, at preregistration and at the analysis freeze, will receive a tag.

## License

Code under the MIT license (`LICENSE`). Text in this repository is under CC BY 4.0. The pictograms keep their own license, CC BY-NC-SA.
