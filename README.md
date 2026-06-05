# statistical-qst-vae
Statistical methods for Quantum State Tomography: evaluating Maximum Likelihood Estimation and Variational Autoencoders

Stati che sarebbe interessante simulare (esempi su 3 qubit N=3):
- 
- stato prodotto $\ket{+}\ket{+}\ket{+}$
- GHZ $\frac{1}{\sqrt{2}}(\ket{000} + \ket{111})$
- W  $\frac{1}{\sqrt{3}}(\ket{100}+\ket{010}+\ket{001})$
- Stato misto (rumore simulato): uno dei precedenti a cui viene associato un modello di rumore --> matrice di densità stato misto

Gli esempi sono indicativi, potremmo partire da uno stato come GHZ e decidere poi se estendere

Fasi progetto:
- 
- MLE (baseline): ricostruire matrice densità tramite stimatore MLE su POVMs --> limitato a stati "piccoli" [4-5-6 qubit]
- VAE + algebra: ricostruire distiribuzione di probabilità di misure POVMs e usare algebra (matrice di overlap T) per ricostruire matrice densità + confronto con risultato MLE [4-5-6 qubit]
- VAE come stimatore della fidelity classica: utilizzare vae per valutare fidelity di stati in dimensione maggiore (notevole dimensione) [$\sim50$ qubit]

[...] rappresentano i limiti *teorici* sulla dimensione dello stato per la simulazione. Sono indicativi, e frutto di stime by Gemini. Nota: nel paper simulano 3-8 qubit. Magari possiamo partire con 3 o 4 e vedere se ampliare

Simulazione dati:
- 
i dati possono essere generati nei seguenti modi:
- Nota $\rho_{true}$, è possibile campionare direttamente le distribuzioni di probabilità per ciascun outcome possibile delle misure POVMs sullo  stato 
- è possibile *simulare* anche l'atto di misura stesso, da effettuare tramite Neimark expansion (sono necessari qubit ancilla, stato dim N --> 3N qubit totali). Ai fini del progetto, non è indispensabile e ha un costo computazionale non indifferente; interessante come spunto per repo o curiosità personale.

(NB: la matrice può essere calcolata sia analiticamente, sia ottenuta tramite *DensityMatrix* in Qiskit su un circuito + quantum_info ed evolve() )

TO DO:
-
- Nome progetto e repo &#10004;
- Definire stato di simulazione, alcune opzioni: 
     - GHZ $\longrightarrow$ abbastanza semplice
     - W $\longrightarrow$ come paper
     - Dicke (?) $\longrightarrow$ da capire
     - Werner (?) $\longrightarrow$ da capire
- Preparare simulazione dati e quindi dataset (programmazione modulare)
- Analisi classica con MLE
- Classi e funzioni per VAE (fare affidamento, almeno in principio, a proposte paper su dimensioni + DeGuio per struttura)
- Confronti statistici + fidelity (nb: con matrici è possibile stimare F_quantum, con distribuzioni VAE solo F_classica)
- Altro: idee su come rendere più "statistica" il progetto...