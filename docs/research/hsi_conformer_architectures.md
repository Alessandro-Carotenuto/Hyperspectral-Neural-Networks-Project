# Conformer per classificazione hyperspectral: cosa specificano davvero le fonti

## Risposta breve

Il blocco Transformer disegnato da CTA-Net non corrisponde al **Visual Conformer di Peng et al. (2021)** che il paper cita. Peng usa due rami concorrenti, CNN e ViT, collegati da Feature Coupling Units; non usa il blocco seriale `FFN -> MHSA -> CNN -> FFN`.

Il disegno di CTA coincide invece quasi esattamente con lo **Speech Conformer di Gulati et al. (2020)** e con **FusionNet**, una sua trasposizione diretta alla classificazione HSI. Queste fonti giustificano due FFN con residui `0.5`, attenzione con posizione relativa e convoluzione depthwise. CTA, pero', dichiara esplicitamente soltanto posizione relativa e convoluzioni 2D `1x1`/`3x3`: non dice che il `3x3` sia depthwise, non scrive il fattore `0.5` e chiama la normalizzazione interna soltanto `norm` nella figura.

## Confronto delle scelte concrete

| Fonte | Tokenizzazione | Posizione e attenzione | FFN e residui | Modulo convoluzionale e norm | Codice ufficiale |
|---|---|---|---|---|---|
| **CTA-Net, Fu et al. (2025)** | Patch HSI `15x15`; una conv allinea i canali. Il testo non specifica flatten, ordine dei token o proiezione token. | MHSA con encoding relativo, senza definirne formula o implementazione. | Due FFM con residui; la figura mostra `LayerNorm -> Linear -> Swish -> Dropout -> Linear -> Dropout`, ma testo/equazioni non specificano scala `0.5`. | `LayerNorm` prima del modulo; `1x1 + GLU`, poi conv 2D `3x3`, `norm`, Swish, `1x1`, dropout. Non dichiara `groups`/depthwise ne' il tipo di `norm`. | Nessun repository degli autori indicato nel paper o trovato con ricerca mirata. [Paper/DOI](https://doi.org/10.1016/j.neunet.2025.107283), [PDF locale](../../CNN-Transformer%20and%20Channel-Spatial%20Attention%20based%20network%20for.pdf). |
| **FusionNet, Yang et al. (2022), HSI/PaviaU** | Cubo 3D -> conv kernel 7 stride 2 -> pooling kernel 3 stride 2; aggiunge un class token al ramo globale. | Dice sia che la conv iniziale rende superfluo un encoding assoluto aggiunto, sia che l'MHSA del Convolution-Transformer block usa embedding relativo. | Macaron esplicito: `x + 0.5 MLP`, MHSA, conv, `LayerNorm(x + 0.5 MLP)`. Swish. | Pointwise + depthwise convolution, GLU e Swish; il ramo locale usa bottleneck ordinari `1x1-3x3-1x1`. LayerNorm nel ramo Transformer; BatchNorm nelle proiezioni verso mappe CNN. | Nessun repository ufficiale collegato o trovato. [Paper ufficiale](https://www.mdpi.com/2072-4292/14/16/4066), [DOI](https://doi.org/10.3390/rs14164066). |
| **MCE-ST, Khotimah et al. (2023), HSI spettrale** | Firma spettrale 1D -> `Conv1d` (config pubblicata: kernel 21) -> ReLU -> `AvgPool1d` (5) -> Linear. Nessun class token; concatena/flattena tutti i token nel classificatore. | Posizione assoluta learned sommata ai token; QKV multi-head. Nota: il codice divide il softmax per `sqrt(embed_dim)` invece di scalare i logits per `sqrt(head_dim)`, quindi non va copiato alla cieca. | Due FFN, GELU, espansione 4; configurazione produttiva `scaler=1`, non `0.5`. Ordine MHSA -> FFN -> conv -> FFN, con LayerNorm dopo ogni residuo. | `1x1 Conv1d` espande x2; due rami depthwise dilatati con kernel 7 e 9, BatchNorm1d + Swish; somma, proiezione `1x1`, dropout. | [Repository degli autori](https://github.com/Weejaa04/MCE-ST-GitHub), [tokenizzazione/posizione/attenzione](https://github.com/Weejaa04/MCE-ST-GitHub/blob/main/MCE_ST.py#L929-L1027), [conv e blocco](https://github.com/Weejaa04/MCE-ST-GitHub/blob/main/MCE_ST.py#L1105-L1340), [DOI](https://doi.org/10.1016/j.jag.2023.103286). |
| **Visual Conformer, Peng et al. (2021), visione RGB** | Stem `7x7/s2` + max-pool; proiezione conv non-overlap (Conformer-S: `4x4/s4`) in 14x14 token + class token. | MHSA QKV scaled-dot-product standard, pre-LayerNorm. **Nessun positional embedding**: posizione indotta da stem e conv. | Un solo MLP per blocco, GELU, ratio 4; residui pieni, non Macaron e non `0.5`. | Ramo CNN ResNet: `1x1-3x3-1x1`, conv ordinaria (`groups=1`), BatchNorm + ReLU. FCU: CNN->token usa `1x1`, avg-pool, LayerNorm+GELU; token->CNN usa reshape, `1x1`, BatchNorm+ReLU, interpolate. | [Paper CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Conformer_Local_Features_Coupling_Global_Representations_for_Visual_Recognition_ICCV_2021_paper.html), [repository ufficiale](https://github.com/pengzhiliang/Conformer), [implementazione](https://github.com/pengzhiliang/Conformer/blob/main/conformer.py#L26-L445). |
| **Speech Conformer, Gulati et al. (2020)** | Sequenza audio dopo convolutional subsampling; non definisce una tokenizzazione 2D per immagini. | MHSA pre-norm con encoding sinusoidale relativo di Transformer-XL. | Due FFN Macaron, espansione 4, Swish; ciascun output FFN pesa `0.5`; LayerNorm finale. | Pointwise Conv1d espande x2 + GLU -> singola depthwise Conv1d -> BatchNorm -> Swish -> pointwise. Kernel usato nei modelli: 32. | Implementato originariamente nel toolkit Lingvo; il paper e' la specifica primaria. [Paper ufficiale Interspeech](https://www.isca-archive.org/interspeech_2020/gulati20_interspeech.html), [arXiv](https://arxiv.org/abs/2005.08100). |

## Il problema della citazione di Peng in CTA

CTA scrive che il proprio Transformer deriva da Peng et al. e contemporaneamente descrive due FFM, MHSA relativo e un CNN module seriale. Questi elementi sono propri del Conformer di Gulati, non di quello di Peng:

- Peng mantiene per tutta la rete due rappresentazioni distinte, una mappa CNN e una sequenza ViT, e le scambia tramite FCU.
- Nel [codice ufficiale di Peng](https://github.com/pengzhiliang/Conformer/blob/main/conformer.py#L26-L445) non esistono ne' doppio FFN Macaron, ne' residui `0.5`, ne' encoding relativo, ne' un CNN module inserito dentro il Transformer.
- CTA non cita Gulati, ma la struttura della sua Fig. 3 e' una variante 2D di Gulati. FusionNet, pubblicato per HSI nel 2022, rende esplicita la stessa derivazione e fornisce anche le equazioni con `0.5`.

Quindi Peng non puo' risolvere le ambiguita' implementative di CTA. E' una fonte valida solo per un'architettura alternativa, concorrente a due rami.

## Implicazioni operative per la replica Pavia CT+CSA

Ordine consigliato delle prove, mantenendo identici split, seed e preprocessing:

1. **Conservare BatchNorm nel CNN module.** CTA lascia `norm` ambiguo, ma Gulati usa BatchNorm e FusionNet usa LayerNorm nel percorso token/Transformer e BatchNorm nelle operazioni su feature map. Anche Peng usa BatchNorm nel ramo CNN. Il risultato sperimentale locale con LayerNorm peggiore e' quindi coerente con le fonti; non c'e' evidenza primaria per sostituire il BatchNorm 2D di default.
2. **Provare `CONFORMER_1D` contro `LEARNED_2D` come ablation controllata.** CTA richiede una posizione relativa ma non ne specifica la forma. La variante Transformer-XL 1D e' la trasposizione piu' letterale di Gulati/CTA; il bias learned 2D preserva invece la geometria della patch ed e' metodologicamente piu' naturale per 15x15. Nessuna fonte primaria stabilisce quale dei due abbia usato CTA, quindi entrambi vanno etichettati come scelte di replica, non come fedelta' certa.
3. **Tenere il fattore FFN `0.5` come default, ma registrarlo come inferenza.** E' esplicito in Gulati e FusionNet e distingue il Macaron Conformer; CTA mostra due FFN ma omette l'equazione. L'ablation sensata e' `0.5` vs `1.0`, non la rimozione casuale di un FFN.
4. **Aggiungere un enum per convoluzione `STANDARD_2D` vs `DEPTHWISE_2D`.** CTA dice soltanto conv 2D `3x3`, quindi `STANDARD_2D` e' la lettura letterale conservativa. `DEPTHWISE_2D` e' sostenuto da Gulati e FusionNet, ma resta un'ipotesi per CTA. E' una prova piu' motivata del cambio BatchNorm->LayerNorm.
5. **Non migrare ora al Visual Conformer di Peng.** Richiederebbe class token, doppia testa/loss, ramo ResNet persistente e FCU ripetute: sarebbe un modello diverso, non una piccola correzione della replica CTA.
6. **Non importare le scelte MCE-ST senza adattamento.** E' utile come prova che un Conformer HSI ufficiale puo' usare token spettrali, posizione assoluta learned, conv depthwise e residui pieni, ma opera su firme 1D e non su patch spaziali Pavia; inoltre la sua formula di scaling dell'attenzione nel codice e' anomala.

## Stato dell'evidenza

- **Esplicito in CTA:** patch `15x15`, MHSA con posizione relativa, due FFM, CNN module con conv 2D `1x1` e `3x3`, residui.
- **Non specificato da CTA:** ordinamento/flatten dei token, formula dell'encoding relativo, `0.5` FFN, depthwise vs standard, tipo di `norm` dopo il `3x3`.
- **Migliore precedente HSI vicino a CTA:** FusionNet, perche' usa PaviaU e rende espliciti Macaron `0.5`, posizione relativa e depthwise convolution.
- **Codice HSI Conformer disponibile:** MCE-ST, ma per token spettrali 1D e un compito di crop-stress; non e' una replica Pavia patch-based.
