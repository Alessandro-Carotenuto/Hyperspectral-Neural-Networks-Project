# Roadmap incrementale: replica CTA-Net per classificazione HSI

## Obiettivo

Classificare i pixel di immagini iperspettrali combinando:

- CNN per feature locali;
- Transformer per feature non locali;
- channel-spatial attention per raffinare le feature;
- sample amplification per lavorare con pochi campioni etichettati.

Paper di riferimento:

> Chuan Fu et al., *CNN-Transformer and Channel-Spatial Attention based network for hyperspectral image classification with few samples*, Neural Networks 186, 2025, 107283. DOI: 10.1016/j.neunet.2025.107283.

La priorità è ottenere presto un sistema completo. Ogni livello produce un risultato eseguibile, valutabile e presentabile. I livelli successivi avvicinano prima alla metodologia completa del paper, poi allo scope per tre studenti e infine all'Honors.

```text
Dataset e patch
    |
    v
CNN-only baseline
    |
    v
CT block: CNN + Transformer          progetto base completo
    |
    v
Channel-spatial attention
    |
    v
Sample amplification                 replica CTA-Net su Pavia
    |
    v
Ablation e replica estesa            estensione per tre studenti
    |
    v
Adaptive spectral band pruning       Honors
```

## Che cosa significa CT

CT significa **CNN-Transformer**.

Nel paper CNN e Transformer non sono disposti in sequenza. Lavorano in parallelo sulle stesse feature:

```text
feature iniziali
      |
  +---+---+
  |       |
 CNN  Transformer
  |       |
  +--concat--+
       |
    Conv 1x1
       |
   residual connection
```

- ramo CNN: estrae informazioni locali e multiscala;
- ramo Transformer: estrae relazioni non locali;
- concatenazione: unisce i due tipi di informazione;
- convoluzione `1x1`: riporta il numero di canali alla dimensione originale;
- residual connection: somma risultato e input del CT block.

## Task esatto

Il modello classifica un pixel usando il suo spettro e il contesto spaziale circostante.

Per ogni pixel etichettato in posizione `(r, c)`:

```text
input: patch iperspettrale 15 x 15 x 103
target: classe del solo pixel centrale ground_truth[r, c]
output: 9 logits, uno per classe
```

La patch non riceve una classe di maggioranza. I pixel vicini forniscono contesto, ma il target resta il pixel centrale.

Al termine, il modello viene applicato ai pixel della scena per produrre una mappa di classificazione.

## Protocollo principale del paper

### Dataset

Usare Pavia University:

- dimensione: `610 x 340` pixel;
- bande: `103` dopo rimozione delle 12 bande rumorose già assenti nel file distribuito;
- classi: `9`;
- pixel etichettati: `42.776`;
- etichetta `0`: pixel non etichettato, escluso da loss e metriche.

### Input

- patch principale: `15 x 15`;
- tutte le `103` bande;
- nessuna PCA nella replica principale;
- valore `13 x 13` conservato solo per eventuale confronto, perché sul dataset UP il paper trova buoni risultati con `13 x 13` e `15 x 15`.

### Split few-shot

Per ogni classe:

- training: `10` pixel scelti casualmente;
- validation: `5` pixel scelti casualmente tra quelli rimanenti;
- test: tutti gli altri pixel etichettati.

Salvare coordinate e seed di ogni split. La prima esecuzione usa un solo seed. La replica estesa usa dieci split casuali, come il paper.

Questo protocollo sostituisce lo split spaziale a blocchi nella replica principale. Il codice dei blocchi può essere conservato per un esperimento futuro, ma non appartiene al percorso principale.

### Training riportato dal paper

- optimizer: Adam;
- learning rate: `0.00008`;
- batch size: `32`;
- epoche: `150`;
- feature channels: `128`;
- loss: cross-entropy multiclasse;
- ambiente originale: PyTorch 1.10 e RTX 3090;
- ambiente del progetto: PyTorch corrente su Kaggle, con differenza documentata.

Il paper non descrive chiaramente ogni dettaglio di normalizzazione e padding. Tali scelte devono essere esplicite nella nostra implementazione e riportate come differenze riproduttive.

### Metriche

- accuracy per classe;
- Overall Accuracy, OA;
- Average Accuracy, AA;
- Cohen's Kappa;
- confusion matrix;
- mappa di classificazione.

Macro-F1, parametri e tempo di inferenza possono essere aggiunti, ma non sostituiscono le metriche del paper.

## Ambiente di lavoro

- VS Code locale: codice, notebook, test rapidi e ispezione dati;
- `.venv` locale: dipendenze e controlli senza GPU;
- Kaggle: training completi con GPU;
- notebook finale: spiegazione, esperimenti, figure e risultati;
- moduli Python: dataset, modelli, training e metriche riutilizzabili.

Ogni run deve salvare:

- configurazione;
- seed;
- coordinate dello split;
- checkpoint migliore;
- metriche;
- curve di training;
- mappa di classificazione.

## Livello 0: dataset ed esplorazione

### Obiettivo

Verificare contenuto e struttura di Pavia University.

### Attività

1. caricare `PaviaU.mat` e `PaviaU_gt.mat`;
2. verificare shape, dtype e intervalli;
3. visualizzare alcune bande;
4. visualizzare ground truth;
5. contare pixel per classe;
6. mostrare firme spettrali di alcuni pixel;
7. documentare classi e sbilanciamento.

### Artefatto disponibile

Prima parte del notebook con dataset, figure e descrizione del problema.

### Stato: completato

Completato nel notebook:

- caricamento del cubo e della ground truth;
- verifica di shape, dtype, intervallo e label;
- conteggio dei pixel per classe;
- visualizzazione delle bande 10, 50 e 90;
- visualizzazione della ground truth;
- grafico dello sbilanciamento delle classi etichettate;
- firme spettrali medie delle nove classi;
- configurazione della patch principale a `15 x 15`.

Le figure ottenute sono già riutilizzabili nel notebook finale.

## Livello 1: pipeline patch e split del paper

### Stato: completato

Completato nel notebook:

- split few-shot riproducibile con seed `42`;
- `10` coordinate train e `5` validation per ciascuna classe;
- tutte le restanti `42.641` coordinate etichettate assegnate al test;
- split salvato in tre CSV sotto `data/splits/seed_42/`;
- verifica di conteggi, copertura, label e assenza di sovrapposizioni;
- normalizzazione min-max per banda sull'intera scena, documentata come preprocessing transduttivo;
- reflect padding con cubo raw invariato;
- estrazione on-demand di patch `15 x 15 x 103`;
- `Dataset` e `DataLoader` PyTorch con batch size `32`;
- conversione dei target originali `1-9` negli indici PyTorch `0-8`.

### Obiettivo

Creare campioni corretti e riproducibili.

### Attività

1. raccogliere coordinate di tutti i pixel con etichetta da `1` a `9`;
2. per ogni classe, mescolare coordinate con seed fisso;
3. selezionare `10` coordinate train e `5` validation;
4. assegnare tutte le coordinate rimanenti al test;
5. estrarre patch `15 x 15 x 103` centrate sulle coordinate;
6. gestire bordi con una regola di padding documentata;
7. creare `Dataset` e `DataLoader` PyTorch;
8. verificare shape, etichette e assenza di sovrapposizione tra coordinate centrali;
9. salvare split e configurazione.

### Test minimi

- ogni classe ha esattamente `10` campioni train;
- ogni classe ha esattamente `5` campioni validation;
- coordinate train, validation e test sono disgiunte;
- patch centrale corrisponde alla coordinata selezionata;
- target corrisponde alla ground truth del centro;
- classe `0` non compare tra i target.

### Artefatto disponibile

Pipeline few-shot completa e ispezionabile.

## Livello 2: CNN-only baseline

### Stato: completato

Completato nel notebook e nei moduli Python:

- convoluzione iniziale da `103` bande a `128` feature channel;
- ramo CNN multiscala del CT block con quattro branch paralleli;
- concatenazione, convoluzione `1x1` e residual connection;
- global average pooling e classificatore a `9` logits;
- forward/backward sanity check e overfitting test su un piccolo batch;
- training con Adam, checkpoint migliore e tre strategie di learning rate;
- valutazione con OA, AA, Kappa, accuracy per classe e confusion matrix;
- confusion matrix normalizzata e classification map completa;
- progressi periodici durante l'inferenza sul test;
- salvataggio di configurazione e metriche in un report JSON;
- split esplorativi `RANDOM_NO_CENTER_OVERLAP` e
  `RANDOM_NO_OVERLAP` per misurare il bias dovuto all'overlap spaziale;
- notebook usato come main leggibile per configurazione, chiamate,
  controlli e interpretazione dei risultati;
- codice riutilizzabile separato in `data.py`, `models.py`,
  `training.py`, `evaluation.py` e `plotting.py`.

I checkpoint creati prima del refactor restano compatibili perché i nomi e
la struttura dei layer non sono cambiati.

### Obiettivo

Ottenere rapidamente un classificatore end-to-end e validare tutta la pipeline.

La baseline deve derivare dal ramo CNN del CT block, non da una rete 3D estranea al paper.

Ramo CNN del paper:

- quattro branch paralleli;
- convoluzione `1x1`;
- convoluzione `3x3`;
- due convoluzioni `3x3` impilate;
- tre convoluzioni `3x3` impilate;
- BatchNorm e GELU tra convoluzioni impilate;
- concatenazione dei branch;
- convoluzione `1x1`;
- residual connection;
- global average pooling;
- fully connected classifier.

### Verifiche

- il modello sovra-adatta un batch piccolo;
- training loss diminuisce;
- checkpoint migliore viene ricaricato;
- metriche test vengono calcolate;
- mappa di classificazione viene generata.

### Artefatto disponibile

Primo progetto completo e presentabile, anche se non soddisfa ancora il TLDR CNN-Transformer.

## Livello 3: CT block

### Stato: completato

Completato nel notebook e nei moduli Python:

- ramo Transformer in stile Conformer con due Feed Forward Module,
  Multi-Head Self-Attention, CNN module interno e residual connection;
- tokenizzazione spaziale delle feature `15 x 15` in `225` token da
  `128` feature;
- self-attention con `4` head e positional bias relativa 2D;
- input e output del ramo Transformer con shape `(B, 128, 15, 15)`;
- CT block con CNN e Transformer paralleli, concatenazione, convoluzione
  `1x1` e residual connection esterna;
- enum dell'architettura e factory per selezionare CNN-only oppure CT;
- checkpoint, report e titoli dei grafici distinti per architettura;
- forward/backward sanity check, smoke training, evaluation e
  classification map completi;
- run principale CT da `150` epoche sullo split `RANDOM`, split seed
  `42` e training seed `100`;
- best checkpoint all'epoca `133`, con validation loss `0,2517`;
- risultati test CT: OA `84,39%`, AA `90,46%` e Kappa `80,16%`;
- confronto sullo stesso protocollo con la CNN-only: il CT riduce l'OA
  di circa `1,91` punti ma aumenta l'AA di circa `1,36` punti nel singolo
  seed analizzato.

Il primo MVP individuale CNN-Transformer è quindi completo. Il confronto
su un solo seed mostra un miglior bilanciamento medio tra classi, ma non
un miglioramento dell'Overall Accuracy. La verifica statistica su più seed
resta parte del Livello 6.

### Obiettivo

Soddisfare il nucleo del progetto: CNN locale e Transformer non locale.

### Architettura

1. convoluzione iniziale per portare feature a `128` canali;
2. ramo CNN multiscala;
3. ramo Transformer in parallelo;
4. concatenazione delle feature;
5. convoluzione `1x1` per ridurre i canali;
6. residual connection con input del CT block;
7. global average pooling;
8. fully connected classifier.

Il Transformer del paper deriva da Conformer e contiene:

- Feed Forward Module;
- Multi-Head Self-Attention;
- CNN module interno;
- secondo Feed Forward Module;
- residual connection per ogni modulo;
- relative positional encoding nella self-attention.

### Esperimento

Confrontare sullo stesso split:

1. CNN-only;
2. CT block.

### Domanda

Il ramo Transformer migliora la classificazione rispetto alle sole feature locali?

### Artefatto disponibile

Progetto individuale minimo completo e coerente con il TLDR.

### Stop condition

Se il tempo è limitato, fermarsi qui. Consolidare codice, notebook, risultati e slide.

## Livello 4: channel-spatial attention

### Stato: completato

Sono disponibili il blocco `ChannelSpatialAttentionBlock` e le quattro
architetture selezionabili `CNN_ONLY`, `CNN_CSA`, `CNN_TRANSFORMER` e
`CNN_TRANSFORMER_CSA`. I controlli sintetici di shape, forward e backward
e gli smoke training delle due nuove architetture sono superati. La variante
`outer_residual`, che rimuove il residual interno dal ramo CNN e conserva il
residual esterno mostrato dal paper, e' stata validata con training completi.
CT + CSA senza sample amplification, con weight decay nullo, ha ottenuto su
dieci training seed OA `84,06 +/- 2,28%`, AA `88,88 +/- 1,03%` e Kappa
`79,76 +/- 2,59%`, mantenendo fisso lo split seed `42`.

Il confronto sistematico comprendente anche CNN + CSA resta utile, ma viene
trattato come parte dell'ablation completa del Livello 6 e non blocca la
replica metodologica CTA-Net del Livello 5.

### Obiettivo

Replicare Att block del paper dopo CT block.

### Channel attention

- apprende un peso per ogni feature channel;
- usa convoluzione 1D con kernel `3`;
- moltiplica pesi e feature;
- applica residual connection.

### Spatial attention

- calcola massimo, minimo, media e deviazione standard lungo i canali;
- elabora statistiche spaziali;
- usa convoluzione `5x5` e PReLU;
- concatena feature;
- usa convoluzione `1x1`;
- moltiplica mappa di pesi e input;
- applica residual connection.

### Esperimento

Confrontare:

1. CNN-only;
2. CNN + CSA;
3. CT;
4. CT + CSA.

### Artefatto disponibile

Architettura CNN-Transformer-Attention quasi completa.

## Livello 5: sample amplification

### Stato: completato

La Sample Amplification e' implementata come espansione offline del solo
training set, materializzata separatamente per ogni training seed. La
configurazione `CTA_NET` produce quattro pool paralleli: campioni originali,
rumore gaussiano fuori dalla regione centrale `3 x 3`, rotazione casuale in
`[-180, 180]` e somma diretta intra-classe con un anchor casuale per classe.
Con 10 campioni originali per classe risultano 39 campioni per classe e 351
campioni totali. Validation e test restano invariati.

I test verificano cardinalita', regione centrale protetta, coerenza spaziale
della rotazione tra tutte le bande, riproducibilita' per seed e isolamento di
validation e test. Checkpoint e report registrano modalita', seed e parametri
della SA.

Su dieci training seed, con split seed `42` e weight decay nullo, CTA-Net
completa ha ottenuto:

- OA `85,47 +/- 3,55%`;
- AA `88,88 +/- 2,23%`;
- Kappa `81,37 +/- 4,19%`.

Rispetto a CT + CSA senza SA, l'incremento e' `+1,41` punti OA e `+1,61`
punti Kappa. La magnitudo OA e' coerente con i `+1,23` punti attribuiti alla
SA dall'ablation del paper. La SA viene quindi congelata in questa forma;
eventuali confronti paired o ulteriori ablation appartengono al Livello 6.

### Obiettivo

Completare replica metodologica di CTA-Net su Pavia University.

Applicare solo ai campioni training:

1. rumore gaussiano fuori dalla regione centrale della patch;
2. rotazione casuale della patch;
3. somma lineare tra patch appartenenti alla stessa classe;
4. mantenimento dei campioni originali.

Validation e test non vengono aumentati.

Quantità di campioni generati, intensità del rumore e dettagli non completamente specificati dal paper devono diventare parametri documentati.

### Esperimento

Confrontare:

1. CT;
2. CT + CSA;
3. SA + CT + CSA, cioè CTA-Net.

### Artefatto disponibile

Replica della metodologia CTA-Net su Pavia University, con differenze implementative dichiarate.

### Seconda stop condition

Fermarsi qui prima di aggiungere nuove idee. Consolidare replica, metriche, notebook e slide.

## Livello 6: estensione per tre studenti

Iniziare solo dopo replica funzionante su Pavia.

### Replica sperimentale completa

- dieci split casuali;
- media e deviazione standard;
- ablation dei componenti SA, CT e CSA secondo la Tabella 7 del paper:
  `CT + CSA + SA`, `CT + SA`, `CT + CSA` e `CSA + SA` senza CT;
- implementare la variante `CSA + SA` senza CT: l'attuale `CNN_CSA` non e'
  equivalente perche' mantiene il blocco CNN multiscala mentre rimuove il
  Transformer. La nuova variante deve saltare il modulo CT, mantenendo CSA e
  sample amplification;
- `CT + CSA` senza SA e `CT + CSA + SA` sono state eseguite su split seed
  `42` con la stessa lista di dieci training seed. La pipeline completa ha
  ottenuto OA `85,55 +/- 4,36%`, AA `89,00 +/- 2,17%` e Kappa
  `81,48 +/- 5,23%`, contro rispettivamente `84,10 +/- 4,06%`,
  `88,03 +/- 2,13%` e `79,68 +/- 4,77%` senza SA. Le differenze tra le
  medie sono `+1,45` punti OA, `+0,97` AA e `+1,80` Kappa;
- anche `CT + SA` e' stata eseguita con lo stesso split e gli stessi seed:
  OA `87,57 +/- 2,55%`, AA `90,58 +/- 1,44%` e Kappa `83,95 +/- 3,09%`.
  Rispetto a `CT + CSA + SA`, le medie sono superiori di `2,02` punti OA,
  `1,58` AA e `2,47` Kappa; in questo split l'effetto osservato della CSA
  con SA attiva e' quindi negativo, diversamente dal piccolo incremento OA
  riportato dal paper;
- resta da completare `CSA + SA` senza CT; richiede una nuova variante che
  salti il CT module, perche' `CNN_CSA` mantiene il blocco CNN multiscala;
- analisi con `5, 10, 15, 20, 25, 30, 40, 50, 100` campioni per classe;
- analisi numero feature channel;
- analisi patch size da `7 x 7` a `23 x 23`;
- confronto con almeno alcuni metodi pubblicati riproducibili.

### Estensione dataset

Aggiungere uno tra:

- WHU-Hi-HongHu;
- WHU-Hi-HanChuan.

Un secondo dataset ha priorità maggiore di molte piccole varianti architetturali, perché verifica generalizzazione.

### Esperimento robusto opzionale

Confrontare split casuale del paper con split spaziale a blocchi. Presentare lo split spaziale come test aggiuntivo, non come replica diretta.

## Livello 7: Honors

### Obiettivo

Adaptive spectral band pruning guidato da attention.

Il channel attention del paper pesa feature interne, non direttamente le 103 bande originali. Honors richiede quindi un meccanismo aggiuntivo prima della convoluzione iniziale.

Possibile soluzione:

1. aggiungere gate apprendibile per ogni banda;
2. combinare gate e informazioni del Transformer;
3. stimare importanza media delle 103 bande;
4. mantenere diverse percentuali di bande;
5. ricostruire input e primo layer con meno bande;
6. fare fine-tuning;
7. misurare OA, AA, Kappa, FLOPs e latenza;
8. tracciare accuracy rispetto alla percentuale di bande eliminate.

Azzerare bande mantenendo input da 103 canali non riduce realmente FLOPs. Il modello finale deve elaborare un numero minore di bande.

Questa estensione richiede approvazione preventiva per Honors.

## Ordine di priorità

1. pipeline patch corretta;
2. split few-shot riproducibile;
3. CNN-only funzionante;
4. CT block;
5. metriche, mappa e notebook;
6. channel-spatial attention;
7. sample amplification;
8. più seed e ablation;
9. secondo dataset;
10. Honors pruning.

## Checkpoint temporali

### Checkpoint 1

- esplorazione dataset completa;
- split salvato;
- patch e DataLoader verificati.

### Checkpoint 2

- CNN-only allenata;
- metriche e mappa disponibili.

### Checkpoint 3

- CT block allenato;
- confronto CNN contro CT;
- obiettivo individuale raggiunto.

### Checkpoint 4

- channel-spatial attention;
- confronto CT contro CT + CSA.

### Checkpoint 5

- sample amplification;
- CTA-Net completa su Pavia;
- ablation minima.

### Checkpoint 6

- esperimenti estesi oppure Honors;
- nessuna estensione prima di aver consolidato notebook e slide.

## Notebook finale

1. titolo, autore, matricola e corso;
2. obiettivo e paper di riferimento;
3. teoria HSI e classificazione pixel-wise;
4. CNN, Transformer, CT block e attention;
5. Pavia University e distribuzione classi;
6. patch e protocollo few-shot;
7. implementazione incrementale;
8. configurazione sperimentale;
9. risultati CNN, CT e CTA;
10. ablation;
11. mappe e confusion matrix;
12. limiti e differenze dal paper;
13. conclusioni;
14. riproducibilità e riferimenti.

## Slide

Preparare 6 slide principali:

1. titolo e obiettivo;
2. HSI e motivazione few-shot;
3. architettura CTA-Net;
4. dataset e protocollo;
5. risultati incrementali;
6. conclusioni e limiti.

Settima slide opzionale: estensione per tre studenti oppure Honors.

## Non obiettivi iniziali

- split spaziale come protocollo principale;
- classificazione densa di tile `10 x 10`;
- PCA obbligatoria;
- CNN 3D non collegata al paper;
- replica di tutti i modelli concorrenti;
- grid search estesa;
- secondo dataset prima del completamento su Pavia;
- pruning prima della replica CTA-Net.

## Criterio di completamento di ogni livello

Un livello è completo quando:

1. codice eseguibile da configurazione salvata;
2. test minimi superati;
3. checkpoint ricaricabile;
4. metriche prodotte;
5. almeno una figura o tabella pronta per notebook;
6. differenze rispetto al paper documentate.
