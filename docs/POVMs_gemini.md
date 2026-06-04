# Fondamenti di Tomografia Quantistica basata su SIC-POVM e Modelli Generativi

## 1. Il Formalismo della Matrice di Densità e la Sfera di Bloch

Uno stato quantistico generale (puro o misto) di un sistema a due livelli (_qubit_) è descritto da un operatore di densità $\rho$ che agisce sullo spazio di Hilbert $\mathcal{H} \cong \mathbb{C}^2$. Per essere fisicamente ammissibile, $\rho$ deve soddisfare tre proprietà fondamentali [1]:

1. **Hermiticità:** $\rho^\dagger = \rho$
2. **Traccia unitaria:** $\text{Tr}(\rho) = 1$ (normalizzazione della probabilità)
3. **Semidefinizione positiva:** $\rho \ge 0$ (autovalori non negativi $\lambda_i \ge 0$)

Sfruttando l'algebra delle matrici di Pauli $\{\sigma_x, \sigma_y, \sigma_z\}$, unitamente alla matrice Identità $I$, possiamo esprimere qualsiasi matrice di densità $2 \times 2$ nella base computazionale standard $\{|0\rangle, |1\rangle\}$ attraverso la parametrizzazione geometrica della **Sfera di Bloch** [1]:

$$\rho = \frac{1}{2} \left( I + r_x\sigma_x + r_y\sigma_y + r_z\sigma_z \right) = \frac{1}{2} \left( I + \vec{r} \cdot \vec{\sigma} \right)$$

dove $\vec{r} = (r_x, r_y, r_z) \in \mathbb{R}^3$ è il vettore di Bloch. Per gli stati puri vale $\|\vec{r}\| = 1$ (punti sulla superficie), mentre per gli stati misti vale $\|\vec{r}\| < 1$ (punti interni). Lo stato di massima ignoranza (rumore bianco) si colloca al centro della sfera con $\vec{r} = (0,0,0)$, ovvero $\rho = I/2$.

Lo spazio complessivo delle matrici Hermitiane $2 \times 2$ costituisce uno spazio vettoriale reale a 4 dimensioni (Spazio di Liouville), in cui il prodotto scalare interno è definito dal prodotto traccia di Hilbert-Schmidt: $\langle A, B \rangle = \text{Tr}(A^\dagger B)$.

## 2. Teoria delle POVM e Geometria del Tetraedro (SIC-POVM)

Nella formulazione standard di Von Neumann, una misura quantistica è descritta da un insieme di proiettori ortogonali $\{P_m\}$ tali che $P_m P_{m'} = \delta_{m,m'} P_m$ e $\sum_m P_m = I$. In un sistema a $d$ dimensioni, una misura proiettiva può restituire al massimo $d$ esiti distinti.

Una **POVM (Positive Operator-Valued Measure)** estende questo concetto postulando un insieme di operatori Hermitiani $\{M_m\}_{m=0}^{k-1}$ che soddisfano unicamente due vincoli:

1. **Positività:** $M_m \ge 0 \quad \forall m$
2. **Completezza (Risoluzione dell'Identità):** $\sum_{m=0}^{k-1} M_m = I$

La probabilità di ottenere l'esito $m$ dallo stato $\rho$ è governata dalla **Regola di Born generalizzata**:

$$p(m) = \text{Tr}\left( M_m \rho \right)$$

Una POVM si dice **Informationally Complete (IC)** se i suoi operatori $\{M_m\}$ formano un insieme di generatori che scherma interamente lo spazio degli operatori quantistici, costituendo cioè una base dello spazio degli operatori. Per un qubit, lo spazio ha 4 dimensioni reali; a causa del vincolo di completezza ($\sum M_m = I$), una POVM deve possedere almeno $k = 4$ elementi indipendenti per essere IC.

La scelta ottimale ricade sulla **SIC-POVM (Symmetric Informationally Complete)**, strutturata secondo la geometria di un **tetraedro regolare** inscritto nella Sfera di Bloch. I suoi 4 elementi sono definiti come proiettori riscalati lungo i 4 vertici del tetraedro $\vec{s}_m$ [2]:

$$M_m = \frac{1}{2} |\phi_m\rangle\langle\phi_m| = \frac{1}{4} \left( I + \vec{s}_m \cdot \vec{\sigma} \right)$$

I 4 vettori tridimensionali $\vec{s}_m$ sono unitari ($|\vec{s}_m|=1$) e disposti a un angolo costante. Il loro prodotto scalare geometrico è costante:

$$\vec{s}_m \cdot \vec{s}_{m'} = \begin{cases} 1 & \text{se } m=m' \\ -\frac{1}{3} & \text{se } m \neq m' \end{cases}$$

Questo si riflette in una sovrapposizione algebrica (prodotto traccia, ovvero elementi della matrice di Gram o di Overlap locale $T_{\text{local}}$) perfettamente uniforme [2, 3]:

$$T_{m,m'} = \text{Tr}(M_m M_{m'}) = \begin{cases} \frac{1}{4} = \frac{1}{d^2} & \text{se } m=m' \\ \frac{1}{12} = \frac{1}{d^2(d+1)} & \text{se } m \neq m' \end{cases}$$

## 3. Vantaggi della Simmetria SIC: Ricostruzione Analitica e Ottimalità Statistica

### A. Soluzione Analitica Chiusa

Per una POVM IC generica, l'inversione delle equazioni di Born richiede di calcolare numericamente l'inversa della matrice di Gram, che spesso è priva di struttura. Nei SIC, invece, la struttura uniforme (della forma $aI + bJ$, con diagonale e fuori-diagonale costanti) possiede un'inversa analitica esplicita che elimina ogni calcolo numerico [3]. Ne discende direttamente la formula chiusa di ricostruzione (Frame Duale) per un singolo sistema di dimensione $d$ [2, 3]:

$$\rho = \sum_{m=1}^{d^2} \left[ (d + 1) p_m - \frac{1}{d} \right] |\phi_m\rangle\langle\phi_m|$$

Sostituendo il valore del singolo qubit ($d=2$), la scomposizione dello stato diventa una combinazione lineare immediata dei 4 proiettori del tetraedro:

$$\rho = \sum_{m=0}^{3} \left( 3p_m - \frac{1}{2} \right) |\phi_m\rangle\langle\phi_m|$$

### B. Ottimalità Statistica (Minimizzazione del Rumore)

Le probabilità $p_m$ misurate in pratica sono affette da fluttuazioni statistiche dovute a un numero finito di ripetizioni (_shot noise_), con un'incertezza dell'ordine di $1/\sqrt{N_s}$ [3]. Tale errore si propaga inevitabilmente nella matrice $\rho$ ricostruita, amplificandosi in misura proporzionale alla traccia dell'inversa della matrice di Gram ($\text{Tr}[T^{-1}]$).

È dimostrato matematicamente che il minimo di questa amplificazione del rumore viene raggiunto esattamente quando la matrice assume la struttura uniforme caratteristica dei SIC [3]. La simmetria del tetraedro è quindi **statisticamente ottimale** per minimizzare l'impatto del rumore tomografico, riducendo altresì la presenza di minimi locali artificiali durante l'ottimizzazione numerica.

## 4. Sistemi Multi-Qubit, la Matrice di Overlap e la Rottura della Simmetria Globale

Per un sistema composto da $N$ qubit, lo spazio degli operatori di densità scala a dimensione $2^N \times 2^N$. Sperimentalmente, la misura globale viene eseguita applicando le POVM locali in modo indipendente su ciascun qubit. Gli operatori POVM globali $M^{(\mathbf{a})}$ vengono costruiti tramite il prodotto tensoriale (prodotto di Kronecker $\otimes$) degli operatori locali ad 1 qubit:

$$M^{(\mathbf{a})} = M^{(a_1)} \otimes M^{(a_2)} \otimes \dots \otimes M^{(a_N)}$$

dove $\mathbf{a} = (a_1, a_2, \dots, a_N)$ rappresenta una stringa di esiti di lunghezza $N$, con $a_i \in \{0, 1, 2, 3\}$. Il numero totale di esiti globali possibili è $4^N$.

Poiché gli operatori SIC-POVM non sono mutuamente ortogonali, la **Matrice di Overlap** $T$ è definita come la matrice delle tracce incrociate degli operatori globali:

$$T_{\mathbf{a}, \mathbf{a}'} = \text{Tr}\left[ M^{(\mathbf{a})} M^{(\mathbf{a}')} \right]$$

Sfruttando le proprietà distributive della traccia rispetto al prodotto tensoriale, la matrice di overlap globale $T_{\text{global}}$ di dimensione $4^N \times 4^N$ si riduce al prodotto tensoriale iterato delle matrici di overlap locali $4 \times 4$:

$$T_{\text{global}} = \underbrace{T_{\text{local}} \otimes T_{\text{local}} \otimes \dots \otimes T_{\text{local}}}_{N \text{ volte}}$$

L'indipendenza lineare degli operatori della SIC-POVM garantisce che $T_{\text{global}}$ sia strettamente definita positiva e, pertanto, **invertibile**.

### Il Limite della Formula Analitica Chiusa

Si presenti una cruciale distinzione algebrica: sebbene l'insieme di operatori globali generato per prodotto tensoriale sia Informationally Complete (IC), **esso non costituisce una SIC-POVM globale nel senso stretto dell'equazione simmetrica** [3, 4]. Gli operatori globali non sono tutti mutuamente equiangolari nello spazio di Hilbert totale di dimensione $d=2^N$.

Per questo motivo, la formula analitica chiusa semplificata **non può essere applicata direttamente impostando $d=2^N$**, poiché la matrice di overlap globale perde la struttura a fuori-diagonale costante. Diventa invece necessario ricorrere alla formula di inversione generale basata sull'inversione esplicita (eseguita tramite decomposizione di Kronecker) della matrice $T_{\text{global}}$ (Frame Duale generalizzato) [4]:

$$\rho = \sum_{\mathbf{a}, \mathbf{a}'} P(\mathbf{a}) T^{-1}_{\mathbf{a}, \mathbf{a}'} M^{(\mathbf{a}')}$$

Questa equazione soffre della _maledizione della dimensionalità_ al crescere di $N$, agendo da collo di bottiglia computazionale e giustificando l'impiego di architetture neurali come i Variational Autoencoder (VAE) per campionare e processare lo spazio degli esiti senza ricostruire esplicitamente l'operatore [4].

### Generazione dei Campioni (Sampling) e Fluttuazioni Statistiche

All'atto pratico (sia in hardware reale che in una simulazione Monte Carlo basata su Qiskit), non si accede alle probabilità analitiche continue $P(\mathbf{a})$, ma a un istogramma di frequenze discrete calcolato su un numero finito di misurazioni ($N_s$, chiamati _shots_):

$$\hat{P}(\mathbf{a}) = \frac{\text{Counts}(\mathbf{a})}{N_s}$$

All'aumentare degli shots ($N_s \to \infty$), la distribuzione empirica converge a quella teorica per la legge dei grandi numeri. Tuttavia, per $N_s$ finiti, la presenza dello _shot noise_ (rumore poissoniano) introduce fluttuazioni statistiche. Se si inserisce direttamente la distribuzione campionaria $\hat{P}(\mathbf{a})$ nell'equazione di ricostruzione lineare, la matrice risultante $\hat{\rho}$ potrebbe violare il criterio di positività ($\hat{\rho} \not\ge 0$), presentando piccoli autovalori negativi. Ciò giustifica l'utilizzo di tecniche avanzate come la Maximum Likelihood Estimation (MLE).

## 5. Realizzabilità Fisica delle POVM: Il Teorema di Dilatazione di Naimark

(ADD: merge di due file md, mantenuto tutto anche se parzialmente sovrapposto)

Un'obiezione fondamentale sollevata dall'assiomatica standard della meccanica quantistica (formulata originariamente da Von Neumann e Dirac) riguarda l'impossibilità di realizzare una misura non ortogonale in modo diretto [1]. Secondo il postulato della misura proiettiva, un sistema quantistico a $d$ dimensioni può generare al massimo $d$ esiti di collasso mutuamente ortogonali. Per un singolo qubit ($d=2$), è fisicamente impossibile far collassare lo stato direttamente lungo i 4 vettori non ortogonali del tetraedro SIC-POVM.

Questo apparente conflitto viene risolto dal **Teorema di Dilatazione di Naimark** [3, 5], il quale stabilisce che _qualsiasi misura POVM non ortogonale operante in uno spazio di Hilbert ridotto può essere reinterpretata come una misura proiettiva standard (ortogonale) operante in uno spazio di Hilbert esteso (dilatato)_.

All'atto pratico, l'estensione dello spazio di Hilbert si realizza accoppiando i qubit di sistema a dei qubit ausiliari, denominati **ancelle**, inizializzati nello stato $|0\rangle$. Esistono due strategie implementative per realizzare questo accoppiamento: una modulare e locale (a 2 ancelle per qubit), e una globale e non-locale (a 1 sola ancella totale per l'intero sistema).

#### Il Meccanismo di Accoppiamento con l'Ancella (idea intuitiva)

Per realizzare sperimentalmente la SIC-POVM a 4 esiti su un qubit principale $|\psi\rangle \in \mathcal{H}_S$, si introduce un sistema ausiliario denominato **ancella** (diversi qubit non di computazione) inizializzato nello stato canonico $|0\rangle \in \mathcal{H}_A$. Lo spazio di Hilbert totale del sistema esteso diventa il prodotto tensoriale $\mathcal{H}_{\text{tot}} = \mathcal{H}_S \otimes \mathcal{H}_A$, la cui dimensione è $2 \times 2 = 4$.

Il processo si articola in tre fasi matematiche distinte [1, 5]:

1. **Inizializzazione dello Stato Globale:** Lo stato iniziale complessivo è $|\Psi_0\rangle = |\psi\rangle \otimes |0\rangle$.
2. **Evoluzione Unitaria (Entanglement):** Si applica un operatore unitario globale $U$ su entrambi i qubit. Questa evoluzione modella l'interazione coerente tra il sistema e l'ambiente controllato, "dilatando" le 3 coordinate informative del qubit principale nelle 4 dimensioni dello spazio esteso:
   $$|\Psi_f\rangle = U \left( |\psi\rangle \otimes |0\rangle \right)$$
3. **Misura Proiettiva standard:** Si esegue una misura proiettiva Von Neumann standard lungo l'asse $Z$ sia sul qubit principale che sull'ancella.

Nello spazio esteso, i 4 autostati di collasso ($|00\rangle, |01\rangle, |10\rangle, |11\rangle$) sono perfettamente **mutuamente ortogonali**, rispettando rigorosamente gli assiomi standard della meccanica quantistica. Tuttavia, la trasformazione unitaria $U$ è ingegnerizzata in modo tale che le probabilità di collasso sui 4 esiti proiettivi coincidano esattamente con le proiezioni geometriche della SIC-POVM sul qubit locale:

$$p(m) = \langle\Psi_f| \left( |m\rangle\langle m| \right) |\Psi_f\rangle = \text{Tr}\left( M_m |\psi\rangle\langle\psi| \right)$$

Questo implica che mentre la generazione algebrica dei sample nel codice può applicare direttamente la regola di Born $p_m = \text{Tr}(M_m \rho)$, la simulazione hardware su Qiskit richiede la costruzione esplicita del circuito di Naimark a due qubit per estrarre i conteggi statistici dai 4 collassi ortogonali della base computazionale.


### A. La Dilatazione Canonica Modulare: L'Approccio a 2 Qubit Ancella per Qubit
Nelle architetture sperimentali reali si preferisce spesso adottare una dilatazione locale che prevede l'accoppiamento di ogni singolo qubit di sistema a **due qubit ancella dedicati** ($\mathcal{H}_{A_1} \otimes \mathcal{H}_{A_2}$). Per un sistema a 3 qubit (come lo stato GHZ), questo setup richiede $3 \times 2 = 6$ ancelle, portando lo spazio di Hilbert totale a $2^9 = 512$ dimensioni (9 qubit fisici).

In questo schema, l'operazione è **locale**: l'unitaria di accoppiamento agisce separatamente qubit per qubit. Il qubit 1 interagisce solo con le sue due ancelle tramite porte logiche a 2 e 3 qubit, e lo stesso avviene in parallelo per gli altri. Al termine del circuito, la misura proiettiva lungo $Z$ viene eseguita esclusivamente sulle 6 ancelle, i cui $2^6 = 64$ esiti proiettivi standard ($|000000\rangle_A, \dots, |111111\rangle_A$) corrispondono direttamente alle frequenze dei 64 esiti della POVM globale del prodotto tensoriale.

* **Vantaggi:** Massima modularità e semplicità di compilazione del circuito.
* **Svantaggi:** Enorme costo in termini di risorse hardware (6 ancelle per soli 3 qubit di sistema).

### B. La Dilatazione Minimale Globale: L'Approccio a 1 Sola Ancella Totale
Dal punto di vista del puro conteggio dei qubit, lo spazio dei 3 qubit possiede $2^3 = 8$ dimensioni. Per estrarre i 64 esiti della POVM globale, lo spazio di Hilbert totale minimo richiesto deve avere almeno 64 dimensioni, obiettivo che si raggiunge aggiungendo **un'unica ancella globale al sistema** ($2^3 \times 2^3 = 64$ dimensioni, ovvero un sistema a $3+3=6$ qubit totali, oppure accoppiando il registro a un'unica ancella complessiva opportunamente scalata). 

Per comprendere come realizzare concettualmente la misura con **1 sola ancella globale per l'intero sistema**, occorre analizzare la struttura delle ampiezze di probabilità nello spazio esteso:
1. **Inizializzazione globale:** Lo stato di partenza è composto dai 3 qubit di sistema e dall'ancella collettiva: $|\Psi_0\rangle = |\psi\rangle_S \otimes |0\rangle_A$.
2. **Mappatura sulle ampiezze:** L'obiettivo della dilatazione di Naimark minimale è costringere un'unica trasformazione unitaria globale $U_{\text{tot}}$ a ridistribuire l'informazione geometrica delle 64 proiezioni del tetraedro direttamente sulle ampiezze delle 64 basi computazionali dello spazio a 6 qubit.
3. **Misura proiettiva:** Al termine del circuito, si esegue una misura proiettiva standard lungo $Z$ su tutti e 6 i qubit. Ognuna delle 64 stringhe di bit estratte ($|000000\rangle, \dots, |111111\rangle$) simulerà esattamente uno specifico esito della POVM combinata.



### C. Il Concetto di Non-Località e il Costo Ingegneristico
Il motivo per cui l'approccio a 1 sola ancella totale è raramente utilizzato nell'hardware reale risiede nella natura **non-locale** della trasformazione unitaria $U_{\text{tot}}$ richiesta.

In informazione quantistica, un'operazione si definisce **locale** quando può essere eseguita manipolando i qubit in modo isolato o a compartimenti stagni nel laboratorio. Al contrario, un'operazione è **non-locale** quando richiede di connettere e intrecciare qubit distanti tra loro attraverso il fenomeno dell'entanglement, sfruttando porte logiche a due corpi (come i `CNOT` o `CZ`).

Mentre l'approccio a 2 ancelle permette di confinare l'interazione all'interno di piccole "isole" locali (qubit-ancelle private), l'approccio a 1 sola ancella totale distrugge questa separazione:
* Per mappare le corrette probabilità POVM su un'unica struttura di ampiezze, l'operatore $U_{\text{tot}}$ deve essere un blocco massiccio di porte logiche che fa interagire **contemporaneamente e collettivamente** il qubit 1, il qubit 2, il qubit 3 e l'ancella.
* Questo richiede una decomposizione in una rete densa e profonda di porte `CNOT` incrociate tra tutti i qubit del registro. 



In sintesi, la realizzabilità fisica delle POVM impone un trade-off ingegneristico assoluto: l'approccio a 2 ancelle per qubit è *localmente semplice* ma *esponenziale nel numero di qubit*; l'approccio a 1 ancella globale è *economico nel numero di qubit* (minimale) ma *estremamente complesso e non-locale* nella profondità e connettività delle porte logiche richieste, aumentando drasticamente l'esposizione del circuito all'errore hardware prima ancora che la misura abbia luogo.


## 6. Tomografia Quantistica Statistica: Maximum Likelihood Estimation (MLE) con Minuit

L'inversione lineare diretta (Frame Duale) applicata a frequenze di campionamento empiriche $\hat{P}(\mathbf{a})$ affette da _shot noise_ produce spesso matrici di densità non fisiche che violano il postulato di positività, presentando autovalori negativi ($\rho \not\ge 0$). La **Stima di Massima Verosimiglianza (MLE)** capovolge il paradigma tomografico [3]: invece di calcolare direttamente lo stato dai dati, cerca lo stato fisico valido $\rho_{\text{MLE}}$ che massimizza la probabilità matematica di aver osservato lo specifico set di conteggi sperimentali raccolti.

### A. Vincoli Assiomatici e la Codifica tramite Decomposizione di Cholesky

Un algoritmo di ottimizzazione numerica standard applicato direttamente sugli elementi di una matrice $\rho$ complessa tenderebbe a violare i vincoli fisici quantistici durante le iterazioni del gradiente. Per imporre i vincoli di Hermiticità ($\rho^\dagger = \rho$), positività ($\rho \ge 0$) e traccia unitaria ($\text{Tr}(\rho) = 1$) senza ricorrere a complessi algoritmi di ottimizzazione vincolata, si adotta la **Decomposizione di Cholesky modificata** [3, 6].

La matrice di densità per un sistema di $N$ qubit ha dimensione $d \times d$ (dove $d = 2^N$) e contiene $d^2$ parametri reali indipendenti. La si codifica tramite una matrice ausiliaria $T$ triangolare inferiore:

$$T(\vec{t}) = \begin{pmatrix} 
t_1 & 0 & 0 & \dots & 0 \\
t_{d+1} + i t_{d+2} & t_2 & 0 & \dots & 0 \\
t_{d+3} + i t_{d+4} & t_{d+5} + i t_{d+6} & t_3 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\dots & \dots & \dots & \dots & t_d
\end{pmatrix}$$

Gli elementi sulla diagonale principale ($t_1, t_2, \dots, t_d$) sono numeri puramente reali; gli elementi al di sotto della diagonale sono numeri complessi, ognuno codificato tramite una coppia di parametri reali (parte reale e parte immaginaria). Il vettore $\vec{t} = (t_1, t_2, \dots, t_{d^2})$ contiene esattamente $4^N$ parametri reali (per 3 qubit, $d=8$, quindi $\vec{t}$ è lungo $8^2 = 64$).

L'operatore di densità candidato viene generato attraverso il prodotto operatore auto-aggiunto, normalizzato esplicitamente per la sua traccia:

$$\rho(\vec{t}) = \frac{T^\dagger(\vec{t}) T(\vec{t})}{\text{Tr}\left( T^\dagger(\vec{t}) T(\vec{t}) \right)}$$

Per le proprietà algebriche intrinseche del prodotto $T^\dagger T$, la matrice risultante $\rho(\vec{t})$ è strutturalmente **semidefinita positiva e a traccia unitaria** indipendentemente dai valori numerici di $\vec{t}$. Questo trasforma un problema di ottimizzazione vincolata in un problema di **ottimizzazione libera (non vincolata)** nello spazio dei parametri reali $\vec{t}$.

### B. La Funzione di Log-Likelihood Quantistica

Dato un insieme di conteggi sperimentali discreti $n_{\mathbf{a}}$ registrati per ciascuno dei $4^N$ esiti della POVM globale (con $\sum_{\mathbf{a}} n_{\mathbf{a}} = N_s$ shots totali), la probabilità di osservare tale specifica combinazione segue una distribuzione multinomiale governata dalle probabilità teoriche Born $p_{\mathbf{a}}(\vec{t}) = \text{Tr}\left( M^{(\mathbf{a})} \rho(\vec{t}) \right)$.

Escludendo i fattori combinatori costanti, si applica il logaritmo naturale per trasformare la produttoria in una sommatoria computazionalmente stabile e si inverte il segno. Il problema MLE si riduce formalmente alla **minimizzazione della Negative Log-Likelihood** ($\mathcal{L}$) [3]:

$$\min_{\vec{t}} \mathcal{L}(\vec{t}) = - \sum_{\mathbf{a}=1}^{4^N} n_{\mathbf{a}} \ln \left[ \text{Tr}\left( M^{(\mathbf{a})} \rho(\vec{t}) \right) \right]$$

### C. Implementazione e Ottimizzazione tramite Minuit (`iminuit`)

Per risolvere questa minimizzazione numerica in contesti di fisica sperimentale, lo strumento d'elezione è **Minuit** (accessibile in Python tramite la libreria `iminuit`), rinomato per la sua stabilità nel calcolo delle matrici di covarianza e nell'analisi degli errori statistici [8].

L'algoritmo si articola nei seguenti passaggi ingegneristici:

1. **La Funzione Costo:** Si scrive una funzione obiettivo in Python, `loss_function(t_vector)`, che accetta un array NumPy di 64 elementi reali (nel caso di 3 qubit). All'interno, la funzione ricostruisce la matrice $T$, calcola $\rho(\vec{t})$, esegue i 64 prodotti traccia $\text{Tr}(M^{(\mathbf{a})}\rho)$ e restituisce il valore scalare di $\mathcal{L}(\vec{t})$.
2. **Inizializzazione:** Si crea un'istanza del minimizzatore passando la funzione costo e un'ipotesi iniziale per il vettore (ad esempio, lo stato di massima ignoranza o valori casuali vicini a zero): `m = Minuit(loss_function, t_start)`.
3. **Esecuzione di MIGRAD:** Si lancia l'algoritmo principale tramite `m.migrad()`. Minuit utilizza l'algoritmo _Davidon-Fletcher-Powell_ (una variante quasi-Newtoniana) per calcolare numericamente il gradiente $\nabla_{\vec{t}} \mathcal{L}$ e la matrice delle derivate seconde (Hessiana).

Grazie all'ottimalità statistica della geometria del tetraedro SIC-POVM, che riduce al minimo la presenza di minimi locali artificiali [3], Minuit è in grado di convergere verso il minimo globale $\vec{t}_{\text{MLE}}$ in pochi secondi per sistemi a 3 qubit, restituendo la matrice di densità fisica ottimale e i relativi errori sui parametri.

### D. Il Collo di Bottiglia Computazionale (The Bottleneck)

La MLE eseguita tramite Minuit costituisce il _benchmark_ ideale per validare l'accuratezza predittiva del Variational Autoencoder (VAE), poiché estrae il massimo contenuto informativo possibile dai dati sperimentali senza assumere alcuna conoscenza pregressa dello stato. Tuttavia, la MLE soffre della _maledizione della dimensionalità_:

- **Esplosione dei parametri:** il numero di parametri reali nel vettore $\vec{t}$ da ottimizzare scala esponenzialmente come $4^N$.
- **Costo delle tracce:** ad ogni passo del gradiente, il computer deve calcolare $4^N$ prodotti traccia di matrici la cui dimensione scala come $2^N \times 2^N$.

A $N=3$ qubit lo spazio dei parametri (64) è perfettamente gestibile da Minuit. Tuttavia, a soli $N=8$ qubit, Minuit si troverebbe a dover ottimizzare simultaneamente $4^8 = 65.536$ parametri reali, calcolando a ogni ciclo $65.536$ tracce di matrici di dimensione $256 \times 256$. Questo collo di bottiglia computazionale arresta la MLE intorno ai 6-7 qubit. È in questo gap che si inserisce il valore del VAE [4]: esso aggira la necessità di calcolare esplicitamente matrici giganti o di ottimizzare decine di migliaia di parametri reali individuali, imparando a mappare la fisica dello stato direttamente dentro uno spazio latente compresso e scalando in regioni di qubit altrimenti inaccessibili per la tomografia classica.

## 7. Criteri di Validazione: Fedeltà Classica vs Fedeltà Quantistica

Per quantificare l'accuratezza di un modello generativo (come un Variational Autoencoder, VAE) addestrato a replicare gli stati quantistici, si introducono due distinte classi di metriche.

### A. Fedeltà Classica (Bhattacharyya Coefficient)

La fedeltà classica si limita a confrontare le due distribuzioni di probabilità (quella esatta della teoria $P$ e quella campionata o generata dal VAE $P_{\text{vae}}$) nello spazio degli esiti discreti della POVM [4]:

$$F_c(P_{\text{vae}}, P) = \sum_{\mathbf{x}} \sqrt{P_{\text{vae}}(\mathbf{x})P(\mathbf{x})}$$

Questa metrica quantifica la sovrapposizione degli istogrammi e gode della proprietà $0 \le F_c \le 1$, dove $1$ indica distribuzioni identiche. Il suo limite intrinseco risiede nell'incapacità di rilevare le violazioni della coerenza quantistica: trattando i dati come distribuzioni statistiche classiche, la metrica ignora la struttura dello spazio di Hilbert. Due matrici di densità radicalmente differenti dal punto di vista fisico (ad esempio uno stato puro entangled e una miscela statistica decoerente) possono produrre la medesima proiezione classica, ingannando la metrica $F_c$.

### B. Fedeltà Quantistica (Fidelity di Jozsa)

Una volta ricostruite le matrici di densità stimate attraverso il processo tomografico (inversione tramite $T^{-1}$ o MLE), la vicinanza fisica tra lo stato teorico di riferimento $\rho_{\text{exact}}$ e lo stato ricostruito dal modello $\rho_{\text{vae}}$ viene misurata tramite la Fedeltà Quantistica di Jozsa [1]:

$$F_q(\rho_{\text{exact}}, \rho_{\text{vae}}) = \left( \text{Tr} \sqrt{\sqrt{\rho_{\text{exact}}} \rho_{\text{vae}} \sqrt{\rho_{\text{exact}}}} \right)^2$$

Se lo stato esatto generato dal circuito è uno stato puro ($\rho_{\text{exact}} = |\psi_{\text{exact}}\rangle\langle\psi_{\text{exact}}|$), come nel caso dello stato GHZ, la formula si riduce al valore di aspettazione dello stato proiettato sulla matrice stimata, eliminando la necessità di estrarre radici quadrate di matrici [1]:

$$F_q = \langle\psi_{\text{exact}}| \rho_{\text{vae}} |\psi_{\text{exact}}\rangle$$

La fedeltà quantistica funge da certificato definitivo per i modelli generativi in informazione quantistica: essa è sensibile alle relazioni di fase globali, cattura la presenza di entanglement multifile e garantisce l'esatta corrispondenza delle coerenze nello spazio di Hilbert.

---

## Riferimenti Bibliografici

- **[1] Nielsen, M. A., & Chuang, I. L. (2010).** _Quantum Computation and Quantum Information_. Cambridge University Press.
  _(Riferimento per i postulati della matrice di densità, proprietà di positività/traccia, definizione della Sfera di Bloch e formalismo della fedeltà quantistica di Jozsa)._

- **[2] Renes, J. M., Blume-Kohout, R., Scott, A. J., & Caves, C. M. (2004).** _Symmetric informationally complete quantum measurements_. Journal of Mathematical Physics, 45(6), 2171–2180.
  _(Il lavoro fondamentale che definisce la struttura geometrica equiangolare dei SIC-POVM, la relazione d'angolo del tetraedro e la derivazione della formula di ricostruzione analitica chiusa)._

- **[3] Paris, M., & Řeháček, J. (Eds.). (2004).** _Quantum State Estimation_. Lecture Notes in Physics, Vol. 649. Springer.
  _(Monografia di riferimento per la teoria della stima degli stati. Contiene le dimostrazioni formali sull'ottimalità statistica dei SIC, la minimizzazione della traccia dell'inversa della matrice di Gram, i formalismi della Maximum Likelihood Estimation e la convergenza tramite algoritmi iterativi a punto fisso come il Diluted MLE)._

- **[4] Carrasquilla, J., Torlai, G., Melko, R. G., & Leandro, A. (2019).** _Reconstructing quantum states with generative models_. Nature Machine Intelligence, 1(3), 155–161.
  _(Paper cardine che unisce il campionamento da misure POVM locali basate su prodotti tensoriali con modelli accoppiati ad architetture neurali generative, introducendo il calcolo della matrice di overlap e il confronto tramite fedeltà classica e quantistica)._

- **[5] Peres, A. (2006).** _Quantum Theory: Concepts and Methods_. Springer Science & Business Media.
  _(Testo fondamentale per la formulazione geometrica delle POVM e la derivazione formale del Teorema di Dilatazione di Naimark tramite l'estensione dello spazio di Hilbert con qubit ancella)._

- **[8] James, F., & Roos, M. (1975).** _Minuit: A system for function minimization and analysis of the parameter errors and correlations_. Computer Physics Communications, 10(6), 343–367.
  _(Il lavoro fondamentale che descrive l'architettura di Minuit, l'algoritmo MIGRAD e la logica di calcolo delle matrici di errore per la minimizzazione di funzioni di verosimiglianza in fisica delle alte energie e quantistica)._