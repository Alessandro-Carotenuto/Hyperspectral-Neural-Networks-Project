# Normalizzazione del cubo Pavia University

## Risposta breve

Nei codici originali consultati non emerge un'unica regola, ma la prassi dominante e' normalizzare **separatamente ogni banda** sull'intera scena, prima del padding e dell'estrazione delle patch. Le due scelte ricorrenti sono:

- min-max per banda in `[0, 1]`;
- z-score per banda, cioe' media 0 e deviazione standard 1.

Per la replica principale senza PCA, la scelta piu' vicina a un Transformer HSI ufficiale e piu' semplice da spiegare e' il **min-max per banda sull'intero cubo**. Va documentato che le statistiche sono calcolate sull'intera scena: non usa label di validation/test, ma e' un preprocessing transduttivo che vede la distribuzione spettrale dei loro pixel.

## Evidenze da implementazioni originali

| Implementazione | Trasformazione | Dati usati per le statistiche | Evidenza |
|---|---|---|---|
| SpectralFormer, codice ufficiale | Min-max `[0,1]` per banda | Intera scena, prima di individuare train e test | In `demo.py` scorre le bande, calcola minimo e massimo di `input[:, :, i]`, poi normalizza; split e patch vengono dopo. [Codice, righe 304-317](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer/blob/main/demo.py#L304-L317) |
| DeepHyperX, toolbox degli autori della review comparativa | Min-max `[0,1]` per banda | Intera scena, nel loader, prima dello split | Rimodella il cubo come `(pixel, bande)` e applica `preprocessing.minmax_scale` con l'asse feature predefinito. [Codice, righe 185-192](https://github.com/eecn/Hyperspectral-Classification/blob/master/datasets.py#L185-L192) |
| DBDA, codice ufficiale | Z-score per banda | Intera scena, prima del campionamento | Rimodella in `(pixel, bande)`, usa `preprocessing.scale(data)`, ricostruisce il cubo e solo dopo chiama `sampling`. [Codice, righe 30-34 e 65-77](https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/blob/master/DBDA/main.py#L30-L77) |
| SSTN, codice ufficiale per PaviaU | Centratura per banda, poi divisione per un massimo globale | Intera scena, prima dello split | Sottrae la media spaziale di ogni banda e divide per `MAX = data_IN.max()`. [Codice PaviaU](https://github.com/zilongzhong/SSTN/blob/master/train_UP.py) |
| HybridSN e SSFTT | PCA con whitening | Intera scena, prima dello split | [HybridSN originale](https://github.com/gokriznastic/HybridSN/blob/master/Hybrid-Spectral-Net.ipynb) e [SSFTT ufficiale](https://github.com/zgr6010/HSI_SSFTT/blob/main/cls_SSFTT_IP/IP_train.py#L18-L29) applicano `PCA(..., whiten=True)`. Non sono un modello adatto alla replica principale, che esclude PCA. |

## Cosa non risulta dal paper di Fu et al.

Nel PDF locale di Fu et al. (2025), la ricerca nel testo per termini relativi a normalizzazione, standardizzazione e preprocessing non individua una regola esplicita. Una ricerca mirata per titolo, DOI e autori non ha inoltre individuato un repository ufficiale pubblico di CTA-Net. Non e' quindi corretto attribuire agli autori min-max o z-score senza ulteriore evidenza.

## Distinzioni importanti

- **Per banda** significa calcolare 103 coppie di statistiche, una per ciascuna banda. Mantiene confrontabili bande con intervalli numerici diversi.
- **Globale** significa usare un solo minimo/massimo o una sola media/deviazione per tutti i valori del cubo. Nei codici consultati non e' la scelta principale attiva per PaviaU.
- **Per pixel** significa normalizzare separatamente ogni firma spettrale. Cambia l'ampiezza relativa tra pixel e non e' equivalente alla normalizzazione per banda.
- **Intera scena** e' comune nei benchmark HSI consultati. E' riproducibile e non usa le classi, ma incorpora statistiche delle feature di test.
- **Solo train** e' il protocollo induttivo piu' rigoroso, ma con appena 90 pixel centrali produce stime fragili. Calcolare le statistiche sulle patch train offre piu' valori, ma nel random pixel split le patch possono contenere pixel che sono centri di validation/test.

## Decisione consigliata per questo progetto

Usare, nella replica principale:

```python
cube_float = cube.astype(np.float32)
band_min = cube_float.min(axis=(0, 1), keepdims=True)
band_max = cube_float.max(axis=(0, 1), keepdims=True)
normalized_cube = (cube_float - band_min) / (band_max - band_min)
```

Poi applicare reflect padding a `normalized_cube` ed estrarre le patch. Il cubo raw resta invariato. Prima di adottarlo nel notebook vanno controllate eventuali bande costanti con `band_max == band_min` e salvata chiaramente la scelta metodologica.

Per una successiva ablation, confrontare questo metodo con z-score per banda. Non cambiare normalizzazione durante il confronto tra CNN baseline e CTA-Net, altrimenti il confronto architetturale non resta controllato.
