# Metodologia Statistica per la Tomografia Quantistica

## Panoramica

Tre analisi statistiche complementari per confrontare MLE e VAE nella ricostruzione di stati quantistici:

1. **Intervalli di confidenza su $F_c$** — quantifica l'incertezza sulla qualità della ricostruzione
2. **Test χ²** — valuta la compatibilità statistica tra distribuzione ricostruita e dati
3. **Curva $F_c$ vs $N_{sample}$** — studia l'efficienza dei due metodi al variare dei dati disponibili

---

## 1. Intervalli di Confidenza su $F_c$

### Idea

La fidelity classica di Bhattacharyya

$$F_c(p, q) = \sum_i \sqrt{p_i \cdot q_i}$$

è una statistica — dipende dai dati su cui è stata stimata. Ripetendo l'esperimento con dataset diversi si otterrebbe un valore diverso. L'obiettivo è quantificare questa variabilità.

### Procedura

Poiché i dati sono **sintetici**, si possono generare $B$ dataset indipendenti — questo è più corretto del bootstrap, che approssima la stessa cosa ma è necessario solo quando i dati reali sono fissi e non riproducibili.

Per ogni dataset $b = 1, \dots, B$:

1. Genera $N$ misure sintetiche da $p_{exact} = \text{Tr}(\rho_{true} \cdot M_i)$
2. Esegui MLE → $\hat{\rho}^{(b)}_{MLE}$ → $p^{(b)}_{MLE} = \text{Tr}(\hat{\rho}^{(b)}_{MLE} \cdot M_i)$ → $F_c^{MLE,(b)}$
3. Addestra VAE su questi dati → campiona molti outcome ($N_{gen} \gg N$) → $p^{(b)}_{VAE}$ → $F_c^{VAE,(b)}$

Si ottengono due distribuzioni empiriche $\{F_c^{MLE,(b)}\}$ e $\{F_c^{VAE,(b)}\}$.

L'intervallo di confidenza al 95% è:

$$\text{CI}_{95\%} = \left[ \text{percentile}_{2.5\%},\ \text{percentile}_{97.5\%} \right]$$

### Nota sul confronto

Usando gli stessi $B$ dataset per entrambi i metodi il confronto è **paired**: si controlla la stessa fonte di variabilità e si isola la differenza tra i metodi. Si possono confrontare sia le medie che le varianze — un metodo con media simile ma varianza minore è più stabile e preferibile in pratica.

---

## 2. Test χ² come Goodness-of-Fit

### Idea

Dopo la ricostruzione, si vuole rispondere alla domanda: *la distribuzione degli outcome predetta dal metodo è statisticamente compatibile con i dati osservati?*

### Derivazione della formula

I conteggi $n_i$ (numero di volte che si osserva l'outcome $i$ su $N$ misure totali) seguono una distribuzione **multinomiale**. Per $N$ sufficientemente grande, ogni $n_i$ è approssimabile con una Poissoniana:

$$n_i \sim \text{Poisson}(N p_i)$$

con media $\mathbb{E}[n_i] = N p_i$ e varianza $\text{Var}(n_i) = N p_i$.

La statistica di test è costruita normalizzando gli scarti quadratici per la varianza attesa:

$$\chi^2 = \sum_i \frac{(n_i - N p_i)^2}{\text{Var}(n_i)} = \sum_i \frac{(n_i - N p_i)^2}{N p_i}$$

Il denominatore $N p_i$ **è** la stima dell'errore statistico di Poisson — non va stimato separatamente, è già incorporato nella formula. Sotto $H_0$ (il modello è corretto), questa statistica segue asintoticamente una distribuzione $\chi^2$ con $d = k - 1$ gradi di libertà, dove $k$ è il numero di outcome POVM distinti.

### Procedura

- **Per MLE**: $p_i = \text{Tr}(\hat{\rho}_{MLE} \cdot M_i)$
- **Per VAE**: $p_i = p_{VAE,i}$ stimata con $N_{gen} \gg N$ campioni

Si calcola la statistica $\chi^2$, si ricava il p-value da `scipy.stats.chi2.sf(chi2_stat, df=k-1)`.

### Attenzione

L'approssimazione χ² richiede $N p_i \gtrsim 5$ per ogni bin. Se alcuni outcome POVM sono molto rari, si possono raggruppare i bin con pochi conteggi attesi prima di calcolare la statistica.

### Interpretazione

| p-value | Interpretazione |
|---|---|
| $> 0.05$ | Ricostruzione compatibile con i dati |
| $< 0.05$ | Discrepanza statistica significativa |
| $\ll 0.01$ | Ricostruzione chiaramente inadeguata |

Con pochi sample ci si aspetta che MLE e la ricostruzione diretta abbiano p-value bassi; VAE dovrebbe essere più robusto.

---

## 3. Curva $F_c$ vs $N_{sample}$

### Idea

Studia come la qualità della ricostruzione dipende dal numero di misure disponibili. Rivela il regime in cui ciascun metodo è vantaggioso.

### Comportamento atteso

- **$N$ piccolo**: VAE meglio di MLE — ha imparato una prior implicita sugli stati fisici, è più robusto alla scarsità di dati
- **$N$ grande**: MLE converge alla soluzione vera e raggiunge o supera il VAE — con abbastanza dati la prior del VAE diventa irrilevante
- **Crossover**: il punto in cui MLE diventa competitivo con il VAE è il risultato più interessante — quantifica il "vantaggio" del VAE in termini di sample efficiency

Per ogni valore di $N$ si calcolano media e banda di confidenza dai $B$ dataset, ottenendo curve con errori per entrambi i metodi.

---

## Parametri Consigliati per GHZ a 3 Qubit

### Ragionamento

Uno stato GHZ a 3 qubit ha matrice densità $8 \times 8$, con $4^3 - 1 = 63$ parametri reali liberi. La tomografia completa richiede misure in $3^3 = 27$ basi (per misure di Pauli), per un totale di $2^3 \times 27 = 216$ outcome distinti.

### Valori suggeriti

| Parametro | Valore suggerito | Motivazione |
|---|---|---|
| $N_{sample}$ (griglia) | 50, 100, 200, 500, 1000, 2000 | Copre il regime scarso e abbondante |
| $N_{gen}$ (campioni VAE) | 50000 – 200000 | Riduce lo shot noise su $p_{VAE}$ |
| $B$ (dataset indipendenti) | 50 – 100 | Sufficiente per stimare percentili stabili |
| Stati da confrontare | GHZ-3, W-3, stato prodotto | Entanglement decrescente |

### Note pratiche

- Con $B = 50$ e riaddestramento VAE: stima ~50 training run — valuta il tempo su un singolo run e moltiplica
- Se il tempo è proibitivo, riduci a $B = 20$ — gli intervalli saranno meno stabili ma il confronto rimane valido
- Per MLE a 4 qubit (63 → 255 parametri) i 5 minuti con Minuit sono già un segnale: la curva di scalabilità è essa stessa un risultato da mostrare

### Stati con diverso entanglement

Un confronto naturale su 3 qubit:

| Stato | Entanglement | Note |
|---|---|---|
| GHZ-3 | Massimo, multipartito | Già implementato |
| W-3 | Medio, più robusto | Già implementato |
| Stato prodotto $\|000\rangle$ | Nullo | Caso banale, baseline |

Aggiungere uno stato parzialmente entangled (es. rotazione di GHZ) renderebbe la curva più continua, ma i tre sopra sono già sufficienti per una narrativa chiara.

---

## Riepilogo Operativo

```
Per ogni stato (GHZ-3, W-3, prodotto):
  Per ogni N in [50, 100, 200, 500, 1000, 2000]:
    Per b in 1..B:
      1. Genera dataset sintetico di N misure
      2. MLE → p_mle → Fc_mle[b], chi2_mle[b]
      3. VAE train → campiona N_gen → p_vae → Fc_vae[b], chi2_vae[b]
    
    Fc_mle_mean[N], Fc_mle_ci[N] = mean, percentile(Fc_mle)
    Fc_vae_mean[N], Fc_vae_ci[N] = mean, percentile(Fc_vae)

Plot: Fc vs N con bande di confidenza, MLE vs VAE
Plot: chi2 p-value vs N, MLE vs VAE
```