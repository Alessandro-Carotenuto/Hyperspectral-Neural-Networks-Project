# Quanto è studiato il pruning delle bande spettrali?

Ricerca bibliografica aggiornata al **22 ottobre 2026**. La domanda è stata confrontata con il meccanismo implementato in `models.py` e `training.py` nel repository.

## Risposta breve

- **La selezione delle bande iperspettrali è un tema maturo e ampio.** Esistono rassegne dedicate e famiglie classiche basate su ranking, ricerca di sottoinsiemi, clustering, sparsità, embedding e metodi ibridi. Anche la selezione deep-learning end-to-end è studiata da anni.
- **“Pruning” è un’etichetta ambigua.** Una parte dei lavori chiama pruning l’eliminazione di pesi o canali interni della CNN per comprimere la rete; un’altra affronta la selezione delle bande originali in ingresso. Sono obiettivi collegati, ma non equivalenti.
- **Il selettore appreso con gate, sparsità e STE ha precedenti diretti**, tra cui BHCNN (2021), metodi embedded supervisionati con gate stocastici/Concrete (pubblicati online nel 2026) e un lavoro IGARSS 2020 che chiama esplicitamente la procedura “neural network pruning” per selezionare bande.
- **Il gating dipendente dal campione seguito da aggregazione in un ranking globale è già presente in forma molto vicina in un articolo del 2026** (ABG). Perciò non è prudente presentare come nuova l’idea generale di imparare score condizionati all’input e poi farne la media per scegliere un sottoinsieme fisso.
- La parte specifica del repository — descrittore CTA/Transformer più statistiche spaziali delle bande grezze, warm-up, maschera STE globale, narrowing della prima proiezione e fine-tuning — appare una combinazione architetturale specifica. Questa ricerca non dimostra però una priorità bibliografica esclusiva né un vantaggio scientifico: servirebbe un confronto sperimentale e bibliografico più sistematico.

## Cosa fa il progetto

Il modello produce un gate per ogni banda originale. Il gate è condizionato dal campione: concatena il contesto medio dell’output di attenzione Transformer e la media spaziale separata di ogni banda grezza, quindi applica una testa MLP e una sigmoid con pendenza configurabile. Nella fase iniziale i gate sono morbidi e la loss riceve una penalità proporzionale alla loro somma.

Dopo il warm-up, il forward usa una maschera binaria fissa, mentre la formula straight-through passa ai gate morbidi il gradiente approssimato. Per decidere quali bande tenere, il training loader viene passato in inferenza e gli score di ciascuna banda sono mediati su tutti i campioni. Si eliminano le k bande con media più bassa. Infine `build_compact_pruned_model` costruisce un classificatore CTA standard con la proiezione iniziale `1×1` ridotta alle colonne delle bande scelte, copia i pesi compatibili e il modello compatto viene fine-tuned.

Quindi, durante l’inferenza finale il selettore **non** sceglie bande diverse per ogni patch: le risposte sono dipendenti dal campione durante l’apprendimento/valutazione degli score, ma la selezione esportata è un sottoinsieme globale fisso.

## Quanto è studiato il tema generale?

La rassegna IEEE di Sun e Du (2019) organizza il campo in sei famiglie: ranking, ricerca, clustering, sparsità, embedding-learning e ibridi. La survey di Sawant e Prabukumar (2020) è focalizzata sulla selezione per classificazione e descrive categorie, tecniche, difficoltà e protocolli. Queste rassegne, insieme alla letteratura successiva, indicano un campo consolidato, non un’idea marginale.

Anche la selezione integrata nelle reti è consolidata. BS-Nets (2019/2020) ha introdotto un modulo di attenzione sulle bande accoppiato a una rete di ricostruzione; il lavoro sulla spectral attention di Mou e Zhu (2020) impara gate per ripesare in modo adattivo le bande in un classificatore; BHCNN (2021) combina convoluzioni indipendenti per banda, hard thresholding e STE in una rete end-to-end di selezione/classificazione. Esistono inoltre metodi supervisionati, non supervisionati, basati su reinforcement learning, autoencoder e Transformer.

La distinzione utile è dunque:

| Domanda | Valutazione |
|---|---|
| Selezione delle bande iperspettrali in generale | **Molto studiata**; campo ampio con survey dedicate e numerosi paradigmi. |
| Selettori deep-learning, attention/gating e apprendimento con task downstream | **Ben studiati e ancora attivi**; almeno dal 2019–2021 sono presenti metodi end-to-end. |
| Selezione con hard mask/STE o rilassamenti differenziabili | **Già studiata**; BHCNN è un precedente diretto, e i metodi embedded recenti continuano su gate e rilassamenti discreti. |
| Specifica architettura CTA + descrittori raw per banda + score mediati + compattazione della proiezione | **Più specifica e meno standardizzata**; trovati antecedenti molto vicini per componenti e pipeline, non una corrispondenza dimostrata di tutti i dettagli. |

## Quanto è studiato questo tipo specifico?

I precedenti più rilevanti sono:

1. **Wang et al., “Neural Network Pruning for Hyperspectral Image Band Selection” (IGARSS 2020).** Il metodo addestra una rete con tutte le bande, aggrega i parametri associati a ogni banda per stimarne l’importanza, impone un vincolo di significatività, elimina componenti e riaddestra una sub-rete compatta. È un precedente diretto per la formulazione “pruning per selezionare bande”, lo score globale, la compattazione e il riaddestramento. Differisce dal repository perché l’importanza deriva dai parametri della rete, non da gate per-campione costruiti da feature CTA e statistiche raw.

2. **Feng et al., BHCNN (IEEE Transactions on Cybernetics, 2021).** Impiega convoluzioni `1×1` indipendenti per banda, hard thresholding e STE per apprendere la selezione insieme alla classificazione. È il precedente più chiaro per il gate binario differenziabile/STE applicato alla selezione spettrale. L’architettura e la loss coarse-to-fine sono diverse; non usa il particolare passaggio del repository dal gate contestuale al ranking medio e alla prima proiezione CTA ristretta.

3. **Zhu et al., ABG (Remote Sensing, 23 agosto 2026).** Il selettore combina un prior globale con risposte di gating specifiche per campione; aggrega gli score dei campioni sorgente e seleziona un top-K globale di bande originali. Questo è molto vicino al punto concettuale che distingue il repository dal semplice vettore di gate statico. ABG differisce per il gating globale+sample-specific, l’espansione del dominio, gli obiettivi di classificazione/contrasto/sparsità e la valutazione cross-scene con SVM; non esporta il classificatore CTA ristretto come nel progetto.

4. **Zimmer, Lindenbaum e Glickman, “Supervised Embedded Methods for Hyperspectral Band Selection” (online 25 agosto 2026).** EHBS usa gate stocastici e approssimazione `L0`; CHBS usa un Concrete Selector/Gumbel-Softmax. Entrambi integrano il selettore nella rete per ottimizzare il task downstream. Confermano che selezione embedded e rilassamenti differenziabili sono una linea attiva; non replicano l’aggregazione di score CTA e la conversione della proiezione iniziale usata qui.

Questi lavori rendono il sottoinsieme “pruning/gating appreso di bande” più piccolo del campo complessivo, ma non nuovo. Inoltre, parole chiave come *band selection*, *band attention*, *gating*, *hard thresholding*, *sparse gates* e *network pruning* indicizzano filoni sovrapposti: cercare solo “spectral band pruning” sottostima la letteratura pertinente.

## Valutazione della novità

La formulazione più accurata è: **“selettore di bande spettrali appreso e specifico per CTA-Net, seguito da esportazione di un modello compatto”**. È ragionevole descrivere come contributo d’implementazione l’uso congiunto del contesto di attenzione CTA e delle statistiche per-banda, il warm-up con sparsità, il passaggio STE verso una maschera globale e il trasferimento dei pesi nella proiezione ristretta.

Non è supportato dai precedenti dire che siano nuovi in sé: la selezione spettrale, l’attenzione per banda, le maschere apprese, lo STE, la valutazione delle importanze su più campioni, il top-K globale, il pruning di canali di input o il fine-tuning del modello compatto. In particolare ABG (2026) rende troppo ampia una rivendicazione di primato sul gating campione-condizionato seguito da aggregazione globale.

Per sostenere una novità più forte servirebbero una ricerca sistematica ulteriore e ablation che isolino il contributo di ciascun elemento, oltre a un confronto equo con selettori noti. Per la valutazione pratica del codice è anche utile chiamarlo **input-band selection / structured spectral-channel pruning**: non vengono potati i pesi arbitrari o i canali intermedi dell’intera rete; vengono rimosse bande in ingresso e ristretto il primo layer che le proietta.

## Fonti

- Sun, W. e Du, Q. (2019), [“Hyperspectral Band Selection: A Review”](https://doi.org/10.1109/MGRS.2019.2911100), *IEEE Geoscience and Remote Sensing Magazine*, 7(2), 118–139.
- Sawant, S. S. e Prabukumar, M. (2020), [“A survey on band selection techniques for hyperspectral image classification”](https://doi.org/10.1255/jsi.2020.a5), *Journal of Spectral Imaging*, 9, a5.
- Sun et al. (2019/2020), [“BS-Nets: An End-to-End Framework for Band Selection of Hyperspectral Image”](https://doi.org/10.1109/TGRS.2019.2951433), *IEEE TGRS*, 58(3), 1969–1984.
- Mou, L. e Zhu, X. X. (2020), [“Learning to Pay Attention on Spectral Domain”](https://doi.org/10.1109/TGRS.2019.2933609), *IEEE TGRS*, 58(1), 110–122.
- Wang, Q. et al. (2020), [“Neural Network Pruning for Hyperspectral Image Band Selection”](https://doi.org/10.1109/IGARSS39084.2020.9323467), *IGARSS 2020*, 2041–2044.
- Feng, J. et al. (2021), [“Convolutional Neural Network Based on Bandwise-Independent Convolution and Hard Thresholding for Hyperspectral Band Selection”](https://doi.org/10.1109/TCYB.2020.3000725), *IEEE Transactions on Cybernetics*, 51(9), 4414–4428.
- Zhu, H. et al. (2026), [“Reliability-Aware Adaptive Band Gating with Domain Expansion for Cross-Scene Hyperspectral Band Selection”](https://doi.org/10.3390/rs18172855), *Remote Sensing*, 18(17), 2855.
- Zimmer, Y., Lindenbaum, O. e Glickman, O. (2026), [“Supervised Embedded Methods for Hyperspectral Band Selection”](https://doi.org/10.3233/FAIA251180), ECAI 2025 proceedings, first published online 25 agosto 2026.
- Li et al. (2026), [“Hyperspectral band selection based on deep learning: a review”](https://doi.org/10.1007/s44443-026-00786-4), rassegna recente del sottofilone deep-learning.
