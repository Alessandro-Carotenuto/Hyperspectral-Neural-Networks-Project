# Piano incrementale: classificazione HSI con CNN e Transformer

## TL;DR
python -c "import torch, numpy, scipy, sklearn, matplotlib, seaborn, pandas, yaml; print('Setup completato'); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
Classificare immagini iperspettrali ad alta dimensionalità combinando:

- CNN per apprendere feature locali spaziali e spettrali;
- Transformer per modellare contesto globale e relazioni a lungo raggio.

Il progetto cresce per livelli. Ogni livello produce un risultato eseguibile, valutabile e presentabile. Se il tempo finisce, si consegna l'ultimo livello completato. Se resta tempo, si continua verso la versione CTA-Net prevista per gruppi di tre.

```text
Pipeline dati
    |
    v
CNN baseline
    |
    v
CNN + Transformer                 <- obiettivo minimo coerente col TL;DR
    |
    v
Channel-spatial attention
    |
    v
Sample amplification + ablation  <- versione estesa per gruppi di tre
    |
    v
Adaptive band pruning            <- Honors, solo alla fine
```

## Vincoli dell'esame

- lavoro individuale ammesso;
- difficoltà proporzionata alla dimensione del gruppo;
- paper recente, preferibilmente successivo al 2022;
- paper e obiettivo devono essere approvati dal docente;
- consegna entro le 23:59 della data comunicata dal docente;
- notebook Jupyter obbligatorio o formato equivalente;
- presentazione di 5–7 slide;
- esposizione di circa 5 minuti, seguita da domande;
- vincoli computazionali e limiti devono essere dichiarati;
- Honors richiede comunicazione preventiva e lavoro eccezionale.

La regola testuale parla di due giorni prima dell'esame, mentre l'esempio del 26 giugno indica il 23 giugno. Usare la data ufficiale pubblicata per il proprio appello oppure chiedere conferma al docente.

## Interpretazione dello scope individuale

Il nucleo individuale è il modello CNN + Transformer. Questo risponde direttamente all'obiettivo di combinare apprendimento locale e contesto globale.

La progressione consigliata è:

- baseline CNN come riferimento necessario;
- CNN + Transformer come progetto minimo completo;
- channel-spatial attention come prima estensione;
- sample amplification e ablation completa come avvicinamento allo scope per gruppi di tre;
- adaptive pruning solo come possibile Honors.

L'approvazione del docente resta necessaria. Non serve proporre subito tutta CTA-Net come requisito individuale.

## Regola di lavoro

Completare, valutare e documentare un livello prima di iniziare il successivo.

Un livello è completo solo quando:

1. il codice parte da un comando documentato;
2. produce un checkpoint;
3. calcola metriche sul test set;
4. salva configurazione e seed;
5. genera almeno una tabella o figura utilizzabile nel notebook finale.

Non lasciare tre modelli incompleti. Meglio un livello inferiore completo e difendibile.

Ogni livello deve aggiornare anche notebook e materiale del report. Codice completo senza analisi non soddisfa i criteri dell'esame.

## Decisioni comuni a tutti i livelli

### Ambiente di sviluppo e training

Usare due ambienti con ruoli distinti:

- VS Code locale per scrivere, organizzare e testare il codice su piccoli batch;
- Kaggle Notebook per training completi e generazione dei risultati finali.

Creare comunque `.venv` locale. Serve a controllare import, testare preprocessing e usare lo stesso codice fuori da Kaggle. Non serve replicare localmente potenza GPU.

Su Kaggle:

- abilitare acceleratore solo quando il modello è pronto;
- mantenere dataset come input separato dal codice;
- salvare checkpoint, configurazione, log, metriche e figure negli output;
- registrare versioni di Python, PyTorch e librerie principali;
- fissare seed;
- verificare che notebook funzioni dall'inizio alla fine dopo un riavvio della sessione;
- scaricare risultati importanti, perché una sessione non è archivio permanente del progetto.

Tenere logica riutilizzabile in moduli Python. Usare notebook per orchestrazione, esperimenti, grafici e spiegazione. Evitare di duplicare tutto il modello in celle scollegate.

### Dataset

Usare solo Pavia University:

- 9 classi;
- 103 bande utilizzabili;
- dimensione gestibile;
- classi meno fragili rispetto a Indian Pines;
- un solo dataset riduce codice e tempo sperimentale.

### Protocollo few-sample

### Decisione sul task

Il modello classifica pixel, ma riceve patch come input:

- unità da classificare: pixel centrale;
- input: patch iperspettrale centrata sul pixel;
- target: classe ground truth del solo pixel centrale;
- output: 9 score, uno per classe;
- una patch non riceve una classe di maggioranza e non viene trattata come immagine con una sola etichetta globale.

Configurazione iniziale della patch del modello:

- dimensione spaziale: `11 x 11`;
- profondità spettrale: 103 bande originali oppure 30 componenti dopo PCA;
- target della patch centrata in `(r, c)`: `ground_truth[r, c]`.

### Decisione sullo split spaziale

Dividere l'intera scena in blocchi spaziali uguali e non sovrapposti. Randomizzare i blocchi, non le singole patch, quindi assegnare ogni blocco interamente a training, validation oppure test.

```text
1 1 | 2 2
1 1 | 2 2
----+----
3 3 | 4 4
3 3 | 4 4
```

Esempio: blocchi 1, 2 e 4 al training; blocco 3 al test.

Procedura:

1. creare griglia di blocchi non sovrapposti;
2. contare pixel di ogni classe presenti in ciascun blocco;
3. assegnare blocchi con seed fisso;
4. verificare che ogni classe compaia nel training e, se possibile, anche in validation e test;
5. usare come candidati solo pixel etichettati appartenenti al rispettivo gruppo di blocchi;
6. escludere centri troppo vicini ai confini tra gruppi diversi;
7. salvare assegnazione dei blocchi e coordinate selezionate.

Con patch `11 x 11`, raggio spaziale è 5 pixel. La fascia di sicurezza deve quindi impedire che una patch di training contenga pixel appartenenti a un blocco di validation o test.

Configurazione few-sample iniziale, applicata dopo split dei blocchi:

- training: 30 campioni per classe;
- validation: 10 campioni per classe;
- test: pixel etichettati idonei nei blocchi di test;
- sviluppo: 1 seed;
- risultati finali: 3 seed, se il tempo lo permette.

Se split a blocchi non offre abbastanza campioni per una classe, ridurre quota in modo documentato oppure modificare assegnazione dei blocchi. Ogni modello deve usare esattamente stessi blocchi, coordinate e seed.

### Preprocessing

- normalizzazione per banda;
- PCA a 30 componenti;
- patch del modello `11 x 11`, centrate sui pixel da classificare;
- flip e rotazioni di 90° solo nel training;
- batch size 32 o 64;
- early stopping sulla validation loss.

PCA, dimensione patch e quote dello split sono configurabili. Non fare ricerca estesa: cambiare un valore solo per risolvere un problema osservato.

### Metriche

Calcolare fin dal primo modello:

- overall accuracy (OA);
- average accuracy per classe (AA);
- macro-F1;
- Cohen's kappa;
- confusion matrix;
- numero di parametri;
- tempo medio di inferenza.

Questo evita di ricostruire la valutazione alla fine.

## Livello preliminare: approvazione

**Tempo attivo:** circa 30 minuti.

Inviare al docente:

1. paper CTA-Net selezionato;
2. obiettivo sintetico;
3. scope individuale incrementale;
4. dataset scelto;
5. nota che training verrà eseguito su Kaggle.

Messaggio proposto:

> Vorrei sviluppare individualmente un classificatore per Pavia University che combini una CNN per feature locali e un Transformer leggero per contesto globale, partendo da una baseline CNN. Se il tempo lo consente, aggiungerei channel-spatial attention e successivamente sample amplification, avvicinandomi alla versione CTA-Net indicata per gruppi di tre. Il training verrà eseguito su Kaggle. Propongo di lasciare adaptive band pruning fuori dallo scope base. Il progetto e il paper sono adeguati per il lavoro individuale?

**Artefatto:** approvazione scritta dello scope. Durante l'attesa si può completare il Livello 0.

## Livello 0: Fondamenta dati

**Tempo:** 1–2 giorni.

### Obiettivo

Ottenere pipeline dati corretta e ispezionabile.

### Attività

1. creare ambiente Python e fissare dipendenze;
2. scaricare Pavia University e ground truth;
3. visualizzare una banda, ground truth e distribuzione delle classi;
4. dividere scena in blocchi non sovrapposti;
5. assegnare interi blocchi a training, validation e test con seed fisso;
6. verificare copertura delle classi e applicare fascia di sicurezza;
7. salvare blocchi e coordinate dello split;
8. normalizzare dati senza usare informazioni del test;
9. applicare PCA senza adattarla sul test;
10. estrarre patch centrate sui pixel etichettati idonei;
11. associare ogni patch alla classe del solo pixel centrale;
12. creare Dataset e DataLoader;
13. verificare forme, etichette e assenza di augmentation nel test.

### Perché

- normalizzazione stabilizza addestramento;
- PCA riduce rumore, memoria e costo;
- patch fornisce contesto locale per classificare pixel centrale;
- split a blocchi limita contaminazione spaziale tra training e test;
- fascia di sicurezza impedisce alle patch di attraversare confini tra gruppi;
- split salvato rende confronti validi;
- controlli automatici evitano errori silenziosi.

### Artefatto consegnabile

Pipeline riproducibile, analisi del dataset e figure descrittive. Non è ancora il progetto completo, ma costituisce una prima sezione solida del notebook.

### Limiti da dichiarare

Split a blocchi può produrre classi sbilanciate o assenti in un gruppo, perché classi sono concentrate in regioni specifiche. Assegnazione dei blocchi deve quindi essere controllata usando distribuzione delle classi, non uno shuffle cieco. Fascia di sicurezza riduce numero di campioni disponibili ma protegge indipendenza spaziale.

## Livello 1: Baseline CNN 3D

**Tempo aggiuntivo:** 1 giorno.

### Obiettivo

Produrre primo classificatore HSI completo.

```text
patch HSI
Conv3D + BatchNorm + ReLU
Conv3D + BatchNorm + ReLU
global pooling
linear classifier
```

### Perché

CNN 3D apprende congiuntamente pattern locali spaziali e spettrali. Baseline verifica preprocessing, training, checkpoint e metriche prima di introdurre Transformer.

### Prove minime

- modello riesce a sovra-adattarsi a un batch piccolo;
- training loss diminuisce;
- miglior checkpoint si ricarica;
- valutazione produce tutte le metriche;
- curve training/validation vengono salvate.

### Artefatto consegnabile

Sistema end-to-end per classificazione HSI, con baseline quantitativa e confusion matrix.

### Stato rispetto al TL;DR

Parziale: classifica HSI e apprende feature locali, ma non include ancora contesto globale del Transformer.

## Livello 2: CNN + Transformer leggero

**Tempo aggiuntivo:** 1–2 giorni.

### Obiettivo

Raggiungere nucleo del progetto espresso dal TL;DR.

```text
patch HSI
CNN spectral-spatial encoder
feature map convertita in token
positional encoding
1 Transformer encoder block
global pooling
linear classifier
```

Configurazione iniziale:

- embedding dimension: 64;
- attention heads: 4;
- Transformer blocks: 1;
- MLP ratio: 2;
- dropout: 0.1.

### Perché

- CNN estrae strutture locali;
- token rappresentano regioni o feature della patch;
- positional encoding conserva informazione sulla posizione;
- self-attention collega token distanti;
- Transformer piccolo limita overfitting e costo.

Il CT block deve terminare con pooling e classificatore. Un blocco isolato non produce una classificazione valutabile.

### Esperimento

Confrontare, sullo stesso split:

1. CNN 3D;
2. CNN + Transformer.

Domanda sperimentale: aggiungere contesto globale migliora classificazione rispetto alle sole feature locali?

### Artefatto consegnabile

Progetto minimo completo e coerente col TL;DR: modello ibrido, confronto con baseline, metriche e analisi.

### Prima stop condition

Se scadenza è vicina, fermarsi qui. Consolidare risultati, README e notebook invece di aggiungere moduli fragili.

## Livello 3: Channel-spatial attention

**Tempo aggiuntivo:** 0,5–1 giorno.

### Obiettivo

Avvicinare modello a CTA-Net senza cambiare pipeline.

Aggiungere modulo leggero ispirato a CBAM:

1. channel attention pesa feature channel;
2. spatial attention pesa posizioni della patch;
3. feature ricalibrate passano a pooling e classificatore.

### Perché

- non tutti i feature channel sono ugualmente informativi;
- non tutte le posizioni della patch aiutano il pixel centrale;
- attention apprende una selezione morbida end-to-end.

### Esperimento incrementale

Confrontare:

1. CNN 3D;
2. CNN + Transformer;
3. CNN + Transformer + channel-spatial attention.

Domanda sperimentale: ricalibrare canali e spazio aggiunge valore oltre al CT block?

### Artefatto consegnabile

Versione CTA compatta, con ablation chiara dei due contributi principali.

### Seconda stop condition

Se modello completo funziona ma tempo è poco, fermarsi. Eseguire 3 seed, produrre tabella finale e completare notebook.

## Livello 4: Versione estesa per gruppi di tre

**Tempo aggiuntivo:** 2–4 giorni.

Iniziare solo quando livelli 0–3 sono completi e documentati.

### 4A. Sample amplification

Implementare una sola strategia controllata. Prima scelta: interpolazione tra due patch della stessa classe oppure mixup intra-classe.

Confrontare modello completo:

- senza sample amplification;
- con sample amplification.

Perché: CTA-Net nasce per pochi esempi etichettati. Espandere training set mira a ridurre overfitting.

Non applicare trasformazione a validation o test. Non generare campioni prima dello split.

### 4B. Ablation completa

| Modello | CNN | Transformer | Attention | Amplification |
|---|---:|---:|---:|---:|
| Baseline | sì | no | no | no |
| CT | sì | sì | no | no |
| CTA | sì | sì | sì | no |
| CTA estesa | sì | sì | sì | sì |

Riportare media e deviazione standard su 3 seed.

### 4C. Analisi finale

- metriche globali e per classe;
- confusion matrix;
- curve di apprendimento;
- costo in parametri e latenza;
- eventuale mappa di classificazione;
- discussione delle classi con spettri simili.

### Artefatto consegnabile

Versione vicina allo scope assegnato a gruppi di tre: data augmentation per few-sample, CT block, channel-spatial attention e ablation completa.

## Livello 5: Honors: adaptive band pruning

**Tempo aggiuntivo:** almeno 2–3 giorni.

Questa fase non appartiene al piano base. Iniziarla solo quando codice, esperimenti principali, notebook e slide sono quasi completi.

### Problema con PCA

PCA miscela bande originali in componenti. Eliminare una componente PCA non equivale a eliminare una banda spettrale. Per vero spectral band pruning serve una variante che riceve bande originali.

### Versione minima

1. addestrare modello senza PCA;
2. ottenere importanza media delle bande tramite attention o gate apprendibili;
3. ordinare bande;
4. mantenere 100%, 75%, 50% e 25% delle bande;
5. fare breve fine-tuning;
6. misurare OA, macro-F1, latenza e FLOPs;
7. tracciare accuracy rispetto a bande rimosse.

Azzerare bande senza ridurre forma del tensore non produce speedup reale. Per dichiarare accelerazione, input e modello devono elaborare meno bande.

### Artefatto consegnabile

Grafico accuracy-costo e discussione del compromesso tra informazione spettrale e velocità.

## Calendario tecnico a checkpoint

### Dopo 2 giorni

- Livello 0 completo;
- figure del dataset;
- split salvato;
- DataLoader verificato.

### Dopo 3 giorni

- Livello 1 completo;
- primo classificatore;
- metriche e confusion matrix.

### Dopo 5 giorni

- Livello 2 completo;
- TL;DR soddisfatto;
- confronto CNN contro CNN-Transformer.

### Dopo 6 giorni

- Livello 3 completo;
- confronto dei tre modelli.

### Dopo 7 giorni

- 3 seed se possibili;
- tabella finale;
- README, notebook e slide consolidati.

### Tempo ulteriore

- prima Livello 4;
- poi Honors;
- mai iniziare Honors con notebook o slide principali incompleti.

## Ordine di priorità se manca tempo

1. correttezza di dati e split;
2. baseline end-to-end;
3. CNN + Transformer;
4. metriche e notebook;
5. channel-spatial attention;
6. tre seed;
7. sample amplification;
8. confronto opzionale con split casuale per pixel usato da parte della letteratura;
9. Honors pruning.

## Non-obiettivi iniziali

- secondo dataset;
- replica identica del paper;
- grid search estesa;
- più varianti di Transformer;
- più varianti di attention;
- confronto con molti modelli esterni;
- GUI o deployment;
- pruning prima della conclusione di notebook e slide principali.

## Struttura progressiva del notebook

Scrivere durante sviluppo:

- Livello 0 completa sezioni dataset, protocollo e preprocessing;
- Livello 1 completa baseline e setup sperimentale;
- Livello 2 completa metodo principale e primo confronto;
- Livello 3 completa CTA compatta e ablation;
- Livello 4 amplia few-sample study;
- Livello 5 diventa estensione Honors separata.

Così ogni checkpoint lascia anche notebook parzialmente pronto, non solo codice.

## Struttura del notebook finale

Usare un notebook principale, pensato per essere letto e rieseguito:

1. titolo, nome, matricola e corso;
2. obiettivo del progetto;
3. paper selezionato e contributi rilevanti;
4. background su HSI, CNN, Transformer e attention usata;
5. dataset e distribuzione delle classi;
6. preprocessing e protocollo few-sample;
7. architetture, con schema e motivazione;
8. configurazione degli esperimenti;
9. risultati con tabelle, curve e confusion matrix;
10. interpretazione dei risultati;
11. limiti e vincoli computazionali di Kaggle;
12. conclusioni e sviluppi futuri;
13. riferimenti a paper, dataset e repository;
14. istruzioni di riproducibilità.

Il notebook può importare moduli dal repository. Deve però mostrare chiaramente flusso, configurazione e risultati. Per la consegna, salvare anche output delle celle essenziali, così il docente può leggere risultati senza rilanciare training.

Separare due modalità:

- `FAST_DEV`: pochi campioni o una sola epoca per verificare notebook;
- `FULL_TRAIN`: configurazione usata per risultati dichiarati.

La modalità veloce serve a provare esecuzione completa. Non presentare sue metriche come risultati finali.

## Struttura delle slide

Preparare 6 slide principali:

1. titolo, autore, corso e data;
2. motivazione, HSI e obiettivo CNN locale + Transformer globale;
3. metodo e architettura incrementale;
4. dataset, split e configurazione Kaggle;
5. risultati e ablation con una tabella e una figura;
6. conclusioni, limiti e sviluppo successivo.

Settima slide opzionale: adaptive pruning o altri sviluppi futuri. Non inserirla come risultato se non è stato implementato e valutato.

Per una presentazione di circa 5 minuti, dedicare circa 40–50 secondi per slide. Evitare dettagli di codice; mostrare scelte, prove e interpretazione.

## Piano a ritroso dalla consegna

Indicare con `T` la scadenza ufficiale di consegna.

- `T-7` o prima: approvazione docente e Livello 0;
- `T-6`: baseline completa;
- `T-5` e `T-4`: CNN + Transformer completa;
- `T-3`: ultima estensione ammessa e avvio run finali;
- `T-2`: congelare architettura, raccogliere risultati e completare notebook;
- `T-1`: riesecuzione rapida, controllo riproducibilità, slide e prova orale;
- `T`: solo controllo e consegna, senza nuovi esperimenti.

Se un run fallisce dopo `T-2`, usare ultimo checkpoint valido. Non cambiare architettura all'ultimo giorno.

## Definizione di completamento per possibile consegna

Prima di fermarsi a qualsiasi livello, verificare:

- repository parte da ambiente pulito seguendo README;
- notebook Kaggle conserva output finali essenziali;
- notebook indica come abilitare acceleratore e dove trovare dati;
- dati non sono inclusi in Git;
- split e seed sono riproducibili;
- checkpoint migliore viene salvato;
- valutazione usa test set una sola volta per risultato finale;
- tabella indica chiaramente moduli presenti;
- limiti metodologici sono dichiarati;
- slide rispettano limite di 5–7 pagine;
- presentazione è stata provata entro circa 5 minuti;
- lavoro incompleto compare come sviluppo futuro, non come risultato ottenuto.
