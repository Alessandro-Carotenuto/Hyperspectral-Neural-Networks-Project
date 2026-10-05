# Sample Amplification di CTA-Net: ricostruzione piu' fedele possibile

## Risposta breve

La replica piu' vicina al paper e' una **espansione offline del solo training set**, costruita una volta per run da quattro insiemi paralleli:

1. patch originali;
2. una copia con rumore gaussiano per ogni patch, lasciando intatta una regione centrale `3x3`;
3. una copia ruotata per ogni patch, con angolo continuo uniforme in `[-180, 180]`;
4. per ogni classe, una patch anchor scelta casualmente e sommata a ciascuna delle altre patch della stessa classe.

Con 10 campioni originali per classe, questa lettura produce `10 + 10 + 10 + 9 = 39` patch per classe, cioe' 351 patch di training su Pavia University. Validation e test restano invariati.

La struttura sopra segue letteralmente il testo di Fu et al.; **dimensione `3x3`, intensita' del rumore, interpolazione/fill della rotazione e trattamento numerico della somma non sono pubblicati dagli autori**. Questi parametri devono quindi essere salvati nel report come scelte di replica, non presentati come dettagli del metodo originale.

## Cosa dice esplicitamente CTA-Net

La Sezione 2.2 definisce tre trasformazioni e il dataset finale come:

```text
X_noise = Addnoise(X)
X_rotation = Rotation(X)
X_linear = Linear(X)
X_final = {X_noise, X_rotation, X_linear, X}
```

Il testo precisa inoltre che:

- il rumore e' gaussiano e viene aggiunto ai pixel **fuori dalla regione centrale**;
- la rotazione e' casuale entro `+/-180` gradi;
- la trasformazione lineare sceglie un campione casuale di una classe e lo somma agli altri campioni della stessa classe;
- l'espansione lineare procede classe per classe;
- gli originali restano nel training set;
- tutti i campioni partecipano ai rami rumore e rotazione.

Queste affermazioni sono nella Sezione 2.2 e nell'Eq. (1) del [paper CTA-Net](../../CNN-Transformer%20and%20Channel-Spatial%20Attention%20based%20network%20for.pdf), disponibile anche nella [pagina ufficiale ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608025001625). Il paper usa patch `15x15`, 10 campioni train e 5 validation per classe su Pavia University; addestra per 150 epoche con Adam, learning rate fisso `8e-5` e batch size 32. Ogni esperimento viene ripetuto dieci volte.

La Tabella 7 attribuisce a SA, sul modello CT+CSA di Pavia University, il passaggio da `93.19 +/- 2.48` a `94.42 +/- 1.90` OA, da `91.40 +/- 2.42` a `92.84 +/- 1.69` AA e da `91.13 +/- 3.15` a `92.71 +/- 2.43` Kappa. Quindi SA vale circa `+1.23` punti OA nel paper: e' utile per completare la replica, ma non spiega da sola un divario molto piu' grande.

## Cosa il paper non specifica

CTA-Net non pubblica:

- forma o lato della regione centrale protetta;
- media, deviazione standard o dipendenza per banda del rumore;
- distribuzione esatta degli angoli;
- interpolazione, espansione della tela e valore di riempimento della rotazione;
- se la "somma lineare" sia `x + a`, una media `(x + a) / 2` o una combinazione pesata;
- clipping o rinormalizzazione dopo le trasformazioni;
- cardinalita' esatta del ramo lineare;
- generazione offline o rigenerazione a ogni epoca;
- un seed separato per l'augmentation.

La sezione SA non cita un lavoro precedente da cui recuperare questi dettagli. La dichiarazione di disponibilita' dice inoltre che non sono stati usati dati per la ricerca; il PDF e la pagina dell'editore non indicano codice o supplementi. Ricerche mirate per titolo, DOI, autori e nome `CTA-net` non hanno individuato un repository ufficiale pubblico degli autori alla data di questa nota (6 ottobre 2026). L'assenza di un risultato pubblico non dimostra che il codice non esista, ma impedisce di attribuire agli autori parametri ulteriori.

## Precedenti primari utili, senza confonderli con CTA-Net

### DeepHyperX: scala del rumore e mixing same-class

Il codice originale di DeepHyperX normalizza Pavia per banda in `[0,1]` e implementa rumore additivo con coefficiente `beta = 1/25 = 0.04`. Il suo `mixture_noise` seleziona dati della stessa classe, usa due pesi casuali positivi normalizzati e aggiunge ancora rumore con `beta=0.04`. E' un precedente HSI concreto e riproducibile, non la fonte dichiarata di CTA-Net: [normalizzazione e augmentation nel codice DeepHyperX](https://github.com/nshaud/DeepHyperX/blob/master/datasets.py#L220-L349).

Questo precedente sostiene `sigma=0.04` come primo valore ragionevole quando gli input sono min-max, ma non giustifica l'uso del mixing pesato al posto della somma letterale descritta da Fu et al.

### Linear mixture HSI

Ball e Wei scelgono coppie casuali di pixel della stessa classe e le mescolano linearmente, mantenendo una maggioranza della classe vera. Il lavoro dimostra che combinazioni intra-classe sono un'augmentation HSI plausibile, ma opera su firme spettrali/pixel e non prescrive la specifica somma di patch di CTA-Net: [paper primario su arXiv](https://arxiv.org/abs/1807.10574).

### Rotazione coerente con PyTorch 1.10

CTA-Net dichiara PyTorch 1.10. Nella documentazione coeva di torchvision 0.11, `RandomRotation(180)` campiona nel range `[-180,180]`; i default sono interpolazione nearest, `expand=False`, centro geometrico e fill 0. I tensori possono avere forma `[..., H, W]`, quindi la stessa trasformazione geometrica puo' essere applicata insieme a tutte le bande: [documentazione torchvision 0.11](https://docs.pytorch.org/vision/0.11/transforms.html#torchvision.transforms.RandomRotation), [sorgente della trasformazione](https://docs.pytorch.org/vision/0.11/_modules/torchvision/transforms/transforms.html#RandomRotation).

Questo rende i default torchvision la scelta conservativa piu' vicina allo stack dichiarato, ma il paper non dice che gli autori abbiano usato torchvision.

## Specifica raccomandata

| Aspetto | Implementazione raccomandata | Stato dell'evidenza |
|---|---|---|
| Dati ammessi | Solo patch i cui centri appartengono al train; mai validation/test | **Esplicito**: SA usa gli existing training samples |
| Composizione | Concatenare `X`, `noise(X)`, `rotate(X)`, `linear(X)`; non concatenare le trasformazioni tra loro | **Esplicito** dall'Eq. (1) |
| Frequenza | Una variante noise e una rotation per ogni originale | **Esplicito**: tutti i campioni partecipano |
| Momento | Materializzare l'insieme una volta per run, prima del DataLoader | **Inferito** dalla definizione di un augmented sample set finito |
| Regione protetta | Quadrato centrale `3x3` della patch `15x15` | **Nostra scelta**: minimo intorno completo del pixel centrale |
| Rumore | `epsilon ~ N(0, 0.04^2)` i.i.d. per valore spaziale-spettrale, applicato solo fuori dal `3x3` | **Nostra scelta**, con `0.04` mutuato dal precedente DeepHyperX |
| Post-rumore | Nessun clipping | **Nostra scelta**, coerente con DeepHyperX; CTA non lo specifica |
| Angolo | `theta ~ Uniform(-180, 180)` continuo, un angolo per patch e comune a tutte le bande | **Inferito** da "random rotation +/-180" |
| Rotazione | Centro geometrico, output `15x15`, nearest, fill 0, senza espansione | **Nostra scelta**, corrispondente ai default torchvision coevi |
| Anchor lineare | Per ogni classe scegliere uniformemente un solo anchor tra i 10 originali | **Esplicito** nel testo, distribuzione uniforme **inferita** |
| Destinatari lineari | Sommare l'anchor alle altre 9 patch, escludendo l'anchor stesso | **Inferito letterale** da "sum it to other samples" |
| Formula lineare principale | `x_linear = x + anchor` | **Lettura letterale**; non e' fornita una formula numerica |
| Post-somma | Nessun clipping, averaging o rinormalizzazione | **Nostra scelta conservativa**: non aggiunge operazioni non dichiarate |
| Label | Conservare la label della classe comune | **Esplicito/necessario** |
| Seed | Usare un `augmentation_seed` deterministico derivato dal training seed e salvarlo nel report | **Nostra scelta** per riproducibilita' |

### Cardinalita' risultante

Per una classe con `n=10`:

```text
originali:  n     = 10
noise:      n     = 10
rotation:   n     = 10
linear:     n - 1 =  9
totale:           = 39
```

La sottrazione di uno nel ramo lineare deriva dalla parola "other": l'anchor non viene sommato a se stesso. Se si volesse forzare esattamente `4n`, bisognerebbe consentire anche `anchor + anchor` o scegliere un partner distinto per ciascun campione; nessuna delle due regole compare nel paper.

## Pseudocodice della replica principale

```python
augmented = list(original_train_patches)

for x, y in original_train_patches:
    eps = normal(mean=0.0, std=0.04, shape=x.shape)
    x_noise = x.copy()
    x_noise[:, outside_central_3x3] += eps[:, outside_central_3x3]
    augmented.append((x_noise, y))

    theta = uniform(-180.0, 180.0)
    x_rot = rotate_all_bands_together(
        x, theta, interpolation="nearest", expand=False, fill=0
    )
    augmented.append((x_rot, y))

for class_id in classes:
    class_patches = originals_with_label(class_id)
    anchor_index = uniform_choice(range(len(class_patches)))
    anchor = class_patches[anchor_index]
    for i, x in enumerate(class_patches):
        if i != anchor_index:
            augmented.append((x + anchor, class_id))
```

`x` e' qui una patch gia' normalizzata, con layout adattato all'implementazione. La maschera e la rotazione agiscono solo sugli assi spaziali; l'angolo e' identico per tutte le 103 bande.

## Unica ablation necessaria per l'ambiguita' maggiore

La somma diretta puo' portare input min-max da `[0,1]` a `[0,2]`. E' comunque la replica principale piu' letterale. Se il training diventa instabile o le attivazioni risultano chiaramente fuori distribuzione, eseguire una sola ablation controllata:

```python
x_linear = 0.5 * (x + anchor)
```

La media e' sostenuta concettualmente da SamplePairing, che combina due immagini pixel per pixel tramite media, ma non e' specifica di HSI e non va presentata come CTA-Net: [paper primario SamplePairing](https://arxiv.org/abs/1801.02929). Non introdurre coefficienti Beta/MixUp casuali nella replica principale: sarebbero una nuova tecnica.

## Controlli prima del training

- Il numero finale deve essere 39 campioni per classe e 351 totali su Pavia University.
- Noise e rotation devono produrre esattamente una patch per originale.
- Nei campioni noise, il `3x3` centrale deve essere bit-identico all'originale.
- Nella rotazione, tutte le bande devono condividere la stessa griglia e lo stesso angolo.
- Ogni campione lineare deve usare due patch originali della stessa classe.
- Validation e test devono mantenere indici, valori e cardinalita' originali.
- Il report deve registrare seed, anchor per classe, angoli, `sigma`, maschera centrale, interpolazione, fill, clipping e cardinalita' dei quattro insiemi.
- Prima di lanciare 10 run, una visualizzazione diagnostica di alcune bande e gli intervalli min/max per ramo devono verificare che la pipeline faccia esattamente quanto dichiarato.

## Decisione

Implementare prima la variante letterale sopra. E' la combinazione che aggiunge il minor numero di assunzioni rispetto a Fu et al. e rende ogni assunzione mancante auditabile. Se non porta un incremento vicino all'ordine di grandezza riportato (`~1.2` punti OA), provare soltanto la media nel ramo lineare; ulteriori variazioni di sigma, regione centrale, interpolazione o clipping diventerebbero tuning di una nuova augmentation, non una replica fedele del paper.
