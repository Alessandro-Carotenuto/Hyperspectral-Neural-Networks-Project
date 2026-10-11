# Ablation study — Pavia University

Risultati dell'ablation della Tabella 7 di CTA-Net e registro delle prove architetturali.

## Protocollo dei risultati raccolti

- Dataset: Pavia University.
- Split fisso: `SPLIT_SEED = 42`.
- Training: 10 run per configurazione, riusando la stessa lista di seed riportata sotto.
- Patch: `15 x 15`; 10 campioni di training e 5 di validation per classe.
- Metriche: media ± deviazione standard tra i 10 training run.
- Il paper riporta 10 esperimenti randomizzati; i nostri risultati qui sotto sono condizionati allo split 42.
- Heads usati nei risultati multi-seed: **4** (confermato dall'utente; coincide con il valore corrente del notebook).

```python
TRAINING_SEEDS_FIXED = [
    1496815513,
    1940408339,
    194592138,
    628378136,
    109688099,
    2039541324,
    1831162321,
    1559800566,
    469802734,
    1660496806,
]
```

## Baseline completa

Configurazione: **CT + CSA + SA** (CTA-Net completa).

| Provenienza | OA | AA | Kappa |
|---|---:|---:|---:|
| Nostro risultato, split 42 | **85,55 ± 4,36%** | **89,00 ± 2,17%** | **81,48 ± 5,23%** |
| Paper, Pavia University | 94,42 ± 1,90% | 92,84 ± 1,69% | 92,71 ± 2,43% |

## Ablation dei componenti già eseguite

| Configurazione | SA | CT | CSA | Nostro OA | Nostro AA | Nostro Kappa | Paper OA | Paper AA | Paper Kappa |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| CT + CSA, senza SA | — | ✓ | ✓ | 84,10 ± 4,06% | 88,03 ± 2,13% | 79,68 ± 4,77% | 93,19 ± 2,48% | 91,40 ± 2,42% | 91,13 ± 3,15% |
| CT + SA, senza CSA | ✓ | ✓ | — | 87,57 ± 2,55% | 90,58 ± 1,44% | 83,95 ± 3,09% | 94,07 ± 2,22% | 92,13 ± 2,03% | 92,27 ± 2,83% |
| **CT + CSA + SA (baseline)** | ✓ | ✓ | ✓ | **85,55 ± 4,36%** | **89,00 ± 2,17%** | **81,48 ± 5,23%** | **94,42 ± 1,90%** | **92,84 ± 1,69%** | **92,71 ± 2,43%** |
| CSA + SA, senza CT | ✓ | — | ✓ | Da eseguire | Da eseguire | Da eseguire | 86,16 ± 4,60% | 82,90 ± 2,57% | 82,30 ± 5,42% |

`CT + CSA` senza SA è l'ablazione della SA; `CT + SA` senza CSA è l'ablazione della CSA; `CSA + SA` senza CT è l'ablazione del CT. La variante esistente `CNN_CSA` non implementa quest'ultima riga, perché mantiene il CNN multiscala del blocco CT.

### Differenze tra le medie osservate

- Aggiungere SA a CT + CSA: **+1,45 punti OA**, **+0,97 AA**, **+1,80 Kappa** (paper: +1,23, +1,44, +1,58).
- Aggiungere CSA a CT + SA: **−2,02 punti OA**, **−1,58 AA**, **−2,47 Kappa** (paper: +0,35, +0,71, +0,44).

## Ablation architetturali da rifare sulla pipeline completa

Le prove seguenti vanno eseguite con pipeline completa **CT + CSA + SA**, split 42 e gli stessi `TRAINING_SEEDS_FIXED` per ogni configurazione. Cambiare un solo parametro alla volta.

| Fattore | Valore attuale | Valori da provare |
|---|---:|---|
| Transformer heads | 4 | 2, 8 |
| Transformer dropout | 0,1 | 0,3 e 0,5 (eseguiti) |
| FFN expansion factor | 4 | 2, 8 (entrambi eseguiti) |
| FFN residual scale | 0,5 | 1,0 e 2 (eseguiti) |
| Feature channels | 128 | 64 e 68 (run singolo precedente identificato dal nome file come `f68`) |
| Positional encoding | Learned 2D | Conformer 1D (eseguito) |
| Normalizzazione CNN module | BatchNorm | LayerNorm (eseguita) |
| Input normalization | Min-max | Z-score (eseguita) |
| Patch size | 15 × 15 | 7 × 7, 11 × 11, 19 × 19 e 23 × 23 (eseguite) |
| LR scheduler | Fixed | Cosine annealing (eseguito) |

Registrare per ogni run la configurazione, i seed, le metriche OA/AA/Kappa e media ± deviazione standard. Sono stati raccolti i risultati per heads, FFN residual scale, expansion 2/8, dropout 0,3/0,5, feature channels 64/256, positional encoding Conformer 1D, normalizzazione LayerNorm, input Z-score, patch 7/11/19/23 × 23 e scheduler cosine annealing.

### Transformer heads — risultati

Pipeline completa CT + CSA + SA, split 42 e lista fissa dei training seed. Il riferimento `heads = 4` è la baseline completa riportata sopra.

| Heads | OA | AA | Kappa |
|---:|---:|---:|---:|
| 2 | 86,32 ± 3,22% | 89,80 ± 1,60% | 82,40 ± 3,78% |
| 4 (baseline) | 85,55 ± 4,36% | 89,00 ± 2,17% | 81,48 ± 5,23% |
| 8 | **88,42 ± 2,67%** | **90,07 ± 1,28%** | **84,96 ± 3,21%** |
| 16 | 86,96 ± 2,82% | 88,95 ± 1,74% | 83,09 ± 3,43% |

#### Accuracy per classe

| Classe | Heads 2 | Heads 8 | Heads 16 |
|---|---:|---:|---:|
| Asphalt | 87,14 ± 5,08% | 83,81 ± 3,24% | 82,11 ± 4,08% |
| Meadows | 83,18 ± 9,17% | 88,81 ± 7,65% | 87,43 ± 7,23% |
| Gravel | 81,77 ± 5,18% | 78,33 ± 6,29% | 78,06 ± 11,72% |
| Trees | 90,65 ± 2,33% | 89,12 ± 2,18% | 90,18 ± 2,48% |
| Painted metal sheets | 96,98 ± 3,34% | 98,17 ± 1,33% | 97,90 ± 2,00% |
| Bare Soil | 83,30 ± 10,71% | 86,00 ± 8,55% | 82,13 ± 9,82% |
| Bitumen | 93,95 ± 6,43% | 95,82 ± 2,32% | 94,50 ± 3,62% |
| Self-Blocking Bricks | 94,67 ± 2,47% | 95,24 ± 3,38% | 93,77 ± 5,78% |
| Shadows | 96,57 ± 0,81% | 95,34 ± 1,74% | 94,44 ± 2,08% |

### Transformer dropout — risultati

| Transformer dropout | OA | AA | Kappa |
|---:|---:|---:|---:|
| 0,1 (baseline) | 85,55 ± 4,36% | 89,00 ± 2,17% | 81,48 ± 5,23% |
| 0,3 | 85,96 ± 4,94% | 88,66 ± 1,41% | 82,02 ± 5,74% |
| 0,5 | 80,86 ± 8,29% | 84,44 ± 4,87% | 75,77 ± 9,81% |

#### Accuracy per classe

| Classe | Dropout 0,3 | Dropout 0,5 |
|---|---:|---:|
| Asphalt | 83,90 ± 3,52% | 82,89 ± 4,70% |
| Meadows | 83,43 ± 12,32% | 76,67 ± 17,33% |
| Gravel | 78,62 ± 10,60% | 63,85 ± 16,54% |
| Trees | 87,85 ± 2,23% | 86,96 ± 3,86% |
| Painted metal sheets | 95,17 ± 3,46% | 95,16 ± 4,03% |
| Bare Soil | 88,23 ± 6,21% | 81,93 ± 13,39% |
| Bitumen | 89,17 ± 4,67% | 85,00 ± 10,00% |
| Self-Blocking Bricks | 94,77 ± 5,18% | 91,03 ± 7,05% |
| Shadows | 96,82 ± 2,15% | 96,50 ± 2,90% |

### FFN residual scale — risultati

| FFN residual scale | OA | AA | Kappa |
|---:|---:|---:|---:|
| 0,5 (baseline) | 85,55 ± 4,36% | 89,00 ± 2,17% | 81,48 ± 5,23% |
| 1,0 | 84,76 ± 4,44% | 88,27 ± 1,90% | 80,48 ± 5,11% |
| 2,0 | **87,00 ± 3,72%** | **89,37 ± 1,72%** | **83,21 ± 4,42%** |

#### Accuracy per classe

| Classe | Scale 1,0 | Scale 2,0 |
|---|---:|---:|
| Asphalt | 80,93 ± 1,34% | 83,53 ± 2,82% |
| Meadows | 82,76 ± 10,90% | 85,92 ± 9,11% |
| Gravel | 77,98 ± 11,20% | 81,38 ± 5,69% |
| Trees | 90,22 ± 2,64% | 88,06 ± 2,66% |
| Painted metal sheets | 96,37 ± 3,40% | 96,43 ± 2,77% |
| Bare Soil | 82,32 ± 6,92% | 85,64 ± 7,36% |
| Bitumen | 93,61 ± 5,78% | 91,68 ± 4,23% |
| Self-Blocking Bricks | 94,27 ± 3,18% | 95,48 ± 1,25% |
| Shadows | 95,95 ± 2,61% | 96,17 ± 2,03% |

### FFN expansion factor — risultati

| FFN expansion factor | OA | AA | Kappa |
|---:|---:|---:|---:|
| 2 | **87,02 ± 1,88%** | **89,54 ± 1,77%** | **83,22 ± 2,43%** |
| 4 (baseline) | 85,55 ± 4,36% | 89,00 ± 2,17% | 81,48 ± 5,23% |
| 8 | 86,07 ± 3,74% | 88,93 ± 2,36% | 81,98 ± 4,62% |

#### Accuracy per classe

| Classe | Expansion 2 |
|---|---:|
| Asphalt | 83,06 ± 4,03% |
| Meadows | 85,60 ± 3,73% |
| Gravel | 77,17 ± 6,33% |
| Trees | 89,96 ± 1,52% |
| Painted metal sheets | 97,48 ± 1,31% |
| Bare Soil | 87,53 ± 8,65% |
| Bitumen | 92,36 ± 5,38% |
| Self-Blocking Bricks | 95,57 ± 1,48% |
| Shadows | 97,12 ± 0,68% |

#### Accuracy per classe

| Classe | Expansion 8 |
|---|---:|
| Asphalt | 82,96 ± 6,30% |
| Meadows | 85,23 ± 8,64% |
| Gravel | 78,18 ± 9,49% |
| Trees | 87,38 ± 2,53% |
| Painted metal sheets | 98,13 ± 1,30% |
| Bare Soil | 81,07 ± 13,60% |
| Bitumen | 95,38 ± 3,87% |
| Self-Blocking Bricks | 95,81 ± 2,07% |
| Shadows | 96,27 ± 1,81% |

### Feature channels — risultati

| Feature channels | OA | AA | Kappa |
|---:|---:|---:|---:|
| 128 (baseline) | 85,55 ± 4,36% | 89,00 ± 2,17% | 81,48 ± 5,23% |
| 68 (dal nome report del run singolo) | **88,20 ± 2,79%** | **89,70 ± 1,41%** | **84,65 ± 3,54%** |
| 256 | 85,37 ± 2,64% | 89,04 ± 1,77% | 81,21 ± 3,26% |

Il messaggio iniziale del run singolo indicava `FEATURE CHANNELS = 64`, ma il report salvato termina in `f68` e contiene `FEATURE CHANNELS = 68`; qui le metriche sono quindi attribuite a 68 in base all'identificativo del report. I run successivi con 8 heads riportano esplicitamente `f64`.

#### Accuracy per classe

| Classe | Run singolo, 68 channels | 256 channels |
|---|---:|---:|
| Asphalt | 85,07 ± 5,64% | 83,56 ± 5,21% |
| Meadows | 88,30 ± 4,56% | 82,41 ± 4,74% |
| Gravel | 82,49 ± 5,35% | 77,46 ± 6,98% |
| Trees | 88,11 ± 3,00% | 90,92 ± 1,28% |
| Painted metal sheets | 95,83 ± 2,95% | 96,71 ± 2,01% |
| Bare Soil | 85,15 ± 8,65% | 84,30 ± 7,58% |
| Bitumen | 92,64 ± 5,36% | 93,87 ± 4,38% |
| Self-Blocking Bricks | 94,79 ± 2,67% | 94,90 ± 2,65% |
| Shadows | 94,95 ± 2,09% | 97,22 ± 0,99% |

### Positional encoding — risultati

| Positional encoding | OA | AA | Kappa |
|---|---:|---:|---:|
| Learned 2D (baseline) | 85,55 ± 4,36% | 89,00 ± 2,17% | 81,48 ± 5,23% |
| Conformer 1D | 85,60 ± 5,01% | 89,78 ± 1,72% | 81,63 ± 5,90% |

#### Accuracy per classe

| Classe | Conformer 1D |
|---|---:|
| Asphalt | 84,84 ± 4,17% |
| Meadows | 81,19 ± 11,11% |
| Gravel | 82,55 ± 6,73% |
| Trees | 90,27 ± 3,30% |
| Painted metal sheets | 96,84 ± 1,90% |
| Bare Soil | 87,58 ± 5,06% |
| Bitumen | 93,26 ± 3,48% |
| Self-Blocking Bricks | 94,83 ± 1,70% |
| Shadows | 96,63 ± 1,26% |

### Normalizzazione nel CNN module — risultati

| Normalizzazione | OA | AA | Kappa |
|---|---:|---:|---:|
| BatchNorm (baseline) | 85,55 ± 4,36% | 89,00 ± 2,17% | 81,48 ± 5,23% |
| LayerNorm | **88,69 ± 1,83%** | 89,38 ± 0,98% | **85,18 ± 2,18%** |

#### Accuracy per classe

| Classe | LayerNorm |
|---|---:|
| Asphalt | 82,72 ± 3,72% |
| Meadows | 91,28 ± 5,19% |
| Gravel | 81,79 ± 7,83% |
| Trees | 86,75 ± 2,58% |
| Painted metal sheets | 97,32 ± 2,29% |
| Bare Soil | 83,95 ± 7,60% |
| Bitumen | 91,24 ± 5,92% |
| Self-Blocking Bricks | 92,11 ± 3,75% |
| Shadows | 97,30 ± 1,21% |

### Input normalization — risultati

| Input normalization | OA | AA | Kappa |
|---|---:|---:|---:|
| Min-max (baseline) | 85,55 ± 4,36% | 89,00 ± 2,17% | 81,48 ± 5,23% |
| Z-score | 87,58 ± 2,46% | **90,77 ± 0,87%** | 83,99 ± 3,03% |

#### Accuracy per classe

| Classe | Z-score |
|---|---:|
| Asphalt | 81,52 ± 4,48% |
| Meadows | 85,85 ± 5,68% |
| Gravel | 89,26 ± 3,34% |
| Trees | 90,08 ± 2,24% |
| Painted metal sheets | 96,17 ± 1,32% |
| Bare Soil | 91,03 ± 2,71% |
| Bitumen | 94,87 ± 2,80% |
| Self-Blocking Bricks | 91,55 ± 2,18% |
| Shadows | 96,57 ± 1,78% |

### Patch size — risultati

| Patch size | OA | AA | Kappa |
|---:|---:|---:|---:|
| 15 × 15 (baseline) | 85,55 ± 4,36% | 89,00 ± 2,17% | 81,48 ± 5,23% |
| 7 × 7 | 81,58 ± 2,99% | 84,86 ± 1,35% | 76,50 ± 3,49% |
| 11 × 11 (small) | **87,64 ± 1,63%** | **89,90 ± 2,20%** | **83,96 ± 2,06%** |
| 19 × 19 (large) | 86,66 ± 1,84% | 86,97 ± 1,21% | 82,73 ± 2,26% |
| 23 × 23 (largest) | 82,28 ± 4,01% | 80,71 ± 1,95% | 77,06 ± 5,00% |

#### Accuracy per classe

| Classe | Patch 7 × 7 | Patch 11 × 11 | Patch 19 × 19 | Patch 23 × 23 |
|---|---:|---:|---:|---:|
| Asphalt | 91,66 ± 3,76% | 93,62 ± 2,16% | 80,71 ± 4,22% | 78,16 ± 4,71% |
| Meadows | 75,82 ± 7,72% | 85,34 ± 5,06% | 87,41 ± 4,86% | 85,79 ± 7,67% |
| Gravel | 78,42 ± 9,40% | 81,55 ± 6,76% | 79,22 ± 9,43% | 76,34 ± 6,53% |
| Trees | 91,87 ± 6,03% | 91,68 ± 0,99% | 81,84 ± 2,70% | 71,35 ± 5,29% |
| Painted metal sheets | 99,56 ± 0,58% | 98,24 ± 1,48% | 86,99 ± 4,21% | 74,92 ± 6,59% |
| Bare Soil | 80,95 ± 6,34% | 82,58 ± 11,22% | 90,42 ± 4,92% | 81,68 ± 13,71% |
| Bitumen | 63,37 ± 3,87% | 88,68 ± 6,99% | 93,02 ± 4,08% | 90,47 ± 8,12% |
| Self-Blocking Bricks | 82,11 ± 8,56% | 88,39 ± 3,48% | 93,61 ± 3,62% | 84,81 ± 6,88% |
| Shadows | 99,96 ± 0,09% | 99,02 ± 0,93% | 89,51 ± 2,95% | 82,85 ± 3,68% |

### LR scheduler — risultati

| Scheduler | OA | AA | Kappa |
|---|---:|---:|---:|
| Fixed (baseline) | 85,55 ± 4,36% | 89,00 ± 2,17% | 81,48 ± 5,23% |
| Cosine annealing | 85,98 ± 3,38% | 89,16 ± 1,59% | 81,96 ± 4,04% |

#### Accuracy per classe

| Classe | Cosine annealing |
|---|---:|
| Asphalt | 83,61 ± 4,55% |
| Meadows | 83,73 ± 7,23% |
| Gravel | 80,35 ± 6,23% |
| Trees | 90,31 ± 1,84% |
| Painted metal sheets | 96,68 ± 2,24% |
| Bare Soil | 84,03 ± 8,25% |
| Bitumen | 91,81 ± 5,31% |
| Self-Blocking Bricks | 95,30 ± 1,82% |
| Shadows | 96,60 ± 1,62% |

## Combinazione di tutte le migliori ablation

Configurazione combinata indicata dal report Kaggle: **heads 8**, FFN expansion **2**, dropout **0,3**, FFN residual scale **2**, positional encoding **Conformer 1D**, normalizzazione del CNN module **LayerNorm**, input **Z-score**, patch **11 × 11**, feature channels **64**, LR scheduler **cosine annealing**, Sample Amplification **CTA-Net**. Training seed fissi: la stessa lista riportata sopra. Sono stati eseguiti due split distinti.

| Split seed | OA | AA | Kappa |
|---:|---:|---:|---:|
| 42 | 85,09 ± 1,57% | 87,71 ± 1,23% | 80,73 ± 1,94% |
| 1 | 78,51 ± 5,17% | 86,61 ± 2,12% | 73,12 ± 5,90% |

#### Accuracy per classe

| Classe | Split 42 | Split 1 |
|---|---:|---:|
| Asphalt | 88,39 ± 6,39% | 80,83 ± 4,88% |
| Meadows | 83,01 ± 3,51% | 67,53 ± 10,43% |
| Gravel | 82,68 ± 4,70% | 65,79 ± 6,58% |
| Trees | 92,64 ± 1,11% | 92,94 ± 1,35% |
| Painted metal sheets | 95,13 ± 2,08% | 98,66 ± 0,51% |
| Bare Soil | 81,28 ± 6,52% | 88,43 ± 4,53% |
| Bitumen | 84,48 ± 3,13% | 92,08 ± 3,80% |
| Self-Blocking Bricks | 83,12 ± 4,12% | 94,49 ± 4,25% |
| Shadows | 98,65 ± 0,61% | 98,78 ± 0,30% |

## Ablation aggiuntive sulla baseline + 8 heads

Pipeline completa, split 42, stessi 10 training seed. Un solo fattore aggiuntivo varia per run; tutti gli altri parametri restano quelli del riferimento baseline + 8 heads.

| Configurazione | OA | AA | Kappa |
|---|---:|---:|---:|
| Baseline + 8 heads | 88,42 ± 2,67% | 90,07 ± 1,28% | 84,96 ± 3,21% |
| 8 heads + LayerNorm | 86,54 ± 2,48% | 89,60 ± 1,40% | 82,66 ± 2,96% |
| 8 heads + Z-score | 87,56 ± 2,79% | **90,68 ± 1,24%** | 83,97 ± 3,46% |
| 8 heads + cosine annealing | 86,99 ± 3,18% | 89,28 ± 2,36% | 83,12 ± 4,10% |
| 8 heads + patch 11 × 11 | 87,36 ± 1,99% | 89,22 ± 1,65% | 83,61 ± 2,44% |
| 8 heads + 64 channels | 86,93 ± 2,86% | 90,10 ± 1,12% | 83,21 ± 3,47% |
| 8 heads + 64 channels + cosine annealing | 86,33 ± 1,75% | 89,41 ± 0,79% | 82,34 ± 2,16% |
| 8 heads + patch 11 × 11 + LayerNorm | 86,49 ± 5,37% | 88,77 ± 1,56% | 82,67 ± 6,34% |
| 8 heads + patch 11 × 11 + Z-score | 86,33 ± 1,47% | 89,34 ± 0,95% | 82,35 ± 1,78% |
| 8 heads + 256 channels | 86,07 ± 3,46% | 88,47 ± 1,59% | 82,02 ± 4,23% |
| 8 heads + 256 channels + FFN expansion 2 | 86,67 ± 2,73% | 89,04 ± 1,48% | 82,73 ± 3,39% |
| 8 heads + Conformer 1D | 85,94 ± 3,62% | 90,00 ± 1,18% | 82,03 ± 4,29% |

#### Accuracy per classe

| Classe | 8h + LayerNorm | 8h + Z-score | 8h + cosine | 8h + patch11 | 8h + 64ch | 8h + 64ch + cosine | 8h + patch11 + LN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Asphalt | 83,26 ± 2,85% | 78,53 ± 3,85% | 84,65 ± 3,70% | 94,35 ± 1,84% | 84,31 ± 3,65% | 83,95 ± 3,32% | 90,44 ± 3,12% |
| Meadows | 84,43 ± 6,16% | 86,35 ± 5,97% | 86,12 ± 5,23% | 84,86 ± 5,20% | 84,22 ± 6,40% | 84,65 ± 3,94% | 83,91 ± 12,86% |
| Gravel | 79,59 ± 7,95% | 88,89 ± 5,56% | 77,75 ± 6,83% | 76,95 ± 10,36% | 81,12 ± 8,73% | 84,36 ± 6,46% | 81,78 ± 5,29% |
| Trees | 89,59 ± 3,54% | 89,85 ± 1,29% | 89,59 ± 3,24% | 92,06 ± 1,96% | 89,70 ± 2,37% | 88,02 ± 2,73% | 92,11 ± 3,03% |
| Painted metal sheets | 97,78 ± 1,27% | 95,11 ± 2,97% | 97,69 ± 1,76% | 98,94 ± 1,04% | 97,85 ± 1,54% | 97,69 ± 1,88% | 97,84 ± 2,05% |
| Bare Soil | 87,51 ± 7,20% | 91,80 ± 1,91% | 82,75 ± 17,10% | 83,91 ± 8,06% | 90,79 ± 4,72% | 85,38 ± 6,73% | 85,42 ± 5,27% |
| Bitumen | 94,18 ± 2,65% | 95,17 ± 2,45% | 94,05 ± 3,72% | 85,87 ± 7,50% | 95,55 ± 2,41% | 94,87 ± 3,66% | 82,23 ± 6,64% |
| Self-Blocking Bricks | 94,02 ± 2,82% | 93,78 ± 1,13% | 96,03 ± 1,81% | 87,66 ± 5,16% | 92,01 ± 3,92% | 90,85 ± 4,61% | 86,19 ± 3,63% |
| Shadows | 96,01 ± 2,01% | 96,68 ± 1,22% | 94,87 ± 2,29% | 98,37 ± 1,66% | 95,32 ± 1,77% | 94,94 ± 2,38% | 99,07 ± 0,51% |

#### Accuracy per classe — configurazioni a 256 channels

| Classe | 8 heads + 256 channels | 8 heads + 256 channels + expansion 2 |
|---|---:|---:|
| Asphalt | 81,40 ± 3,93% | 81,41 ± 3,20% |
| Meadows | 85,41 ± 7,50% | 86,40 ± 4,75% |
| Gravel | 73,37 ± 8,48% | 74,43 ± 9,61% |
| Trees | 89,12 ± 3,63% | 90,22 ± 3,42% |
| Painted metal sheets | 96,71 ± 3,10% | 98,38 ± 1,43% |
| Bare Soil | 85,18 ± 8,41% | 83,97 ± 8,11% |
| Bitumen | 93,93 ± 4,17% | 94,44 ± 3,72% |
| Self-Blocking Bricks | 94,30 ± 3,17% | 95,62 ± 1,62% |
| Shadows | 96,85 ± 1,38% | 96,53 ± 1,22% |

## Altre combinazioni — 4 heads

Pipeline completa, split 42 e lista fissa dei 10 training seed. Configurazioni indicate dai report Kaggle.

| Configurazione | OA | AA | Kappa |
|---|---:|---:|---:|
| 4 heads + LayerNorm + FFN expansion 2 | 87,19 ± 2,03% | 89,18 ± 1,61% | 83,33 ± 2,58% |
| 4 heads + LayerNorm + FFN expansion 2 + cosine annealing | 86,84 ± 3,12% | 89,55 ± 1,41% | 82,98 ± 3,71% |

#### Accuracy per classe

| Classe | 4h + LN + x2 | 4h + LN + x2 + cosine |
|---|---:|---:|
| Asphalt | 84,05 ± 2,32% | 83,70 ± 3,37% |
| Meadows | 87,31 ± 3,44% | 85,85 ± 8,12% |
| Gravel | 81,10 ± 7,54% | 82,14 ± 7,69% |
| Trees | 89,64 ± 3,02% | 88,84 ± 2,05% |
| Painted metal sheets | 97,32 ± 1,24% | 97,56 ± 1,07% |
| Bare Soil | 82,07 ± 9,00% | 83,80 ± 7,22% |
| Bitumen | 90,68 ± 2,82% | 92,21 ± 3,24% |
| Self-Blocking Bricks | 93,14 ± 3,53% | 94,17 ± 1,46% |
| Shadows | 97,32 ± 1,34% | 97,66 ± 0,96% |

#### Accuracy per classe — nuove combinazioni

| Classe | 8h + patch11 + Z-score | 8h + Conformer 1D |
|---|---:|---:|
| Asphalt | 86,40 ± 2,97% | 85,05 ± 4,29% |
| Meadows | 84,14 ± 3,34% | 81,59 ± 8,48% |
| Gravel | 88,91 ± 1,76% | 80,23 ± 7,11% |
| Trees | 92,51 ± 1,08% | 91,07 ± 2,54% |
| Painted metal sheets | 96,69 ± 1,62% | 97,72 ± 1,17% |
| Bare Soil | 84,82 ± 3,14% | 88,35 ± 5,18% |
| Bitumen | 85,21 ± 4,58% | 94,07 ± 4,79% |
| Self-Blocking Bricks | 86,20 ± 3,32% | 95,54 ± 2,20% |
| Shadows | 99,20 ± 0,24% | 96,35 ± 2,13% |

## Combinazioni a coppie: heads 8

Entrambi i run usano la pipeline completa, split 42, gli stessi 10 training seed e mantengono gli altri parametri della baseline.

| Configurazione | Heads | FFN expansion | FFN residual scale | OA | AA | Kappa |
|---|---:|---:|---:|---:|---:|---:|
| Riferimento heads 8 | 8 | 4 | 0,5 | 88,42 ± 2,67% | 90,07 ± 1,28% | 84,96 ± 3,21% |
| A: heads 8 + expansion 2 | 8 | 2 | 0,5 | 86,84 ± 2,34% | 89,20 ± 1,54% | 82,92 ± 2,96% |
| B: heads 8 + residual scale 2 | 8 | 4 | 2 | 87,26 ± 2,22% | 89,59 ± 1,61% | 83,45 ± 2,72% |
| C: heads 4 + expansion 2 + residual scale 2 | 4 | 2 | 2 | **87,56 ± 1,62%** | **90,30 ± 0,85%** | 83,88 ± 1,98% |
| D: heads 8 + expansion 2 + residual scale 2 | 8 | 2 | 2 | 85,46 ± 6,24% | 89,27 ± 2,18% | 81,44 ± 7,09% |

#### Accuracy per classe

| Classe | A: h8 + x2 | B: h8 + s2 | C: h4 + x2 + s2 | D: h8 + x2 + s2 |
|---|---:|---:|---:|---:|
| Asphalt | 82,66 ± 4,45% | 83,46 ± 2,36% | 83,79 ± 3,48% | 83,80 ± 4,95% |
| Meadows | 86,62 ± 4,57% | 87,26 ± 5,62% | 86,52 ± 3,75% | 82,55 ± 15,35% |
| Gravel | 77,38 ± 5,89% | 85,98 ± 9,70% | 84,78 ± 6,56% | 82,38 ± 5,08% |
| Trees | 89,77 ± 2,08% | 88,06 ± 3,12% | 89,08 ± 2,35% | 91,44 ± 1,31% |
| Painted metal sheets | 97,48 ± 1,64% | 95,96 ± 3,56% | 97,14 ± 2,18% | 98,01 ± 2,39% |
| Bare Soil | 82,17 ± 6,93% | 82,88 ± 8,43% | 85,58 ± 4,72% | 82,39 ± 8,73% |
| Bitumen | 94,62 ± 4,58% | 95,29 ± 3,63% | 95,29 ± 2,18% | 92,54 ± 4,98% |
| Self-Blocking Bricks | 95,79 ± 1,12% | 91,96 ± 6,28% | 94,15 ± 2,89% | 94,44 ± 2,02% |
| Shadows | 96,28 ± 1,87% | 95,49 ± 1,70% | 96,37 ± 2,33% | 95,86 ± 2,48% |

