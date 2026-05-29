"""
Campionamento POVM Tetraedrico via Naimark — N qubit (misura CONGIUNTA)
=======================================================================

Estensione di povm_naimark.py al caso di N qubit di sistema.

Idea (vedi povm_naimark.py per il caso 1 qubit):
    ogni qubit di sistema viene "dilatato" con 2 ancilla, e la misura
    proiettiva sull'ancilla in base Z restituisce l'outcome POVM in {0,1,2,3}.

PUNTO CHIAVE — perché tutto su UN solo circuito:
    Per stati entangled (es. GHZ) la distribuzione congiunta
        P(a_1,...,a_N) = Tr[(M^{a_1} ⊗ ... ⊗ M^{a_N}) rho]
    NON fattorizza nel prodotto delle marginali. Quindi NON si può
    campionare un qubit alla volta e poi affiancare i risultati:
    si perderebbero le correlazioni. Si prepara lo stato entangled
    su tutti i qubit di sistema nello stesso circuito, si applica la
    dilatazione di Naimark su ogni blocco, e si misurano TUTTE le
    ancilla insieme leggendo la stringa congiunta (a_1,...,a_N).

Risorse: N qubit di sistema + 2N ancilla = 3N qubit fisici simulati.
(È l'overhead intrinseco di Naimark: l'algebra pura in povm_sampling.py
 ottiene gli stessi campioni lavorando solo in 2^N.)

Niente statevector: si usa solo AerSimulator con shot reali, esattamente
come accadrebbe su hardware.

Riferimenti:
    - Naimark dilation: arxiv.org/abs/1807.08449
    - SIC-POVM tetraedrico: Renes et al. 2004, quant-ph/0310075
"""

import numpy as np
from collections import Counter
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

# riusa l'unitario di Naimark a 1 qubit e il POVM tetraedrico
from povm_naimark import build_naimark_unitary, M1
# riusa utility già scritte e coerenti col paper
from povm_sampling import samples_to_onehot, ghz_state_algebra, povm_probability


# ── Stato di sistema: GHZ a N qubit ──────────────────────────────────────────
def ghz_circuit(N):
    """Circuito che prepara |GHZ> = (|0..0> + |1..1>)/sqrt(2) su N qubit."""
    qc = QuantumCircuit(N)
    qc.h(0)
    for i in range(N - 1):
        qc.cx(i, i + 1)
    return qc


# ── Costruzione del circuito dilatato a N qubit ──────────────────────────────
def build_naimark_circuit_Nqubit(state_circuit, N, U_naimark):
    """
    Assembla il circuito 3N-qubit con la dilatazione di Naimark su ogni qubit.

    Layout (blocco k = 0..N-1, 3 qubit fisici ciascuno):
        qubit 3k     -> ancilla0 del qubit k   (bit basso dell'outcome)
        qubit 3k+1   -> ancilla1 del qubit k   (bit alto dell'outcome)
        qubit 3k+2   -> qubit di sistema k

    Bit classici:
        cbit 2k      <- ancilla0 del qubit k
        cbit 2k+1    <- ancilla1 del qubit k
    => outcome a_k = anc1*2 + anc0  in {0,1,2,3}

    L'ordine [3k, 3k+1, 3k+2] passato a qc.unitary rispetta la convenzione
    di build_naimark_unitary (sistema = bit più significativo, idx = sys*4 + anc1*2 + anc0).
    """
    qr = QuantumRegister(3 * N, 'q')
    cr = ClassicalRegister(2 * N, 'c')
    qc = QuantumCircuit(qr, cr)

    # qubit di sistema alle posizioni 3k+2
    sys_qubits = [3 * k + 2 for k in range(N)]

    # prepara lo stato (entangled!) su TUTTI i qubit di sistema, stesso circuito
    qc.compose(state_circuit, qubits=sys_qubits, inplace=True)

    # dilatazione di Naimark indipendente su ogni blocco [anc0, anc1, sys]
    gate = Operator(U_naimark)
    for k in range(N):
        qc.unitary(gate, [3 * k, 3 * k + 1, 3 * k + 2])

    # misura CONGIUNTA di tutte le ancilla (le correlazioni restano nello stato)
    for k in range(N):
        qc.measure(3 * k,     2 * k)       # anc0 -> bit basso
        qc.measure(3 * k + 1, 2 * k + 1)   # anc1 -> bit alto

    return qc


# ── Decodifica counts -> lista di tuple (a_1,...,a_N) ─────────────────────────
def counts_to_samples(counts, N, shuffle=True, seed=42):
    """
    Converte i counts di Qiskit in una lista di tuple-outcome, espandendo
    ogni configurazione per il numero di shot.

    Qiskit restituisce la bitstring con il bit classico di indice più alto
    a sinistra; invertendola otteniamo rev[i] = cbit i.
    """
    samples = []
    for bitstring, cnt in counts.items():
        bs = bitstring.replace(" ", "")     # registro singolo: nessuno spazio
        rev = bs[::-1]                       # rev[i] = cbit i
        outcome = tuple(int(rev[2 * k + 1]) * 2 + int(rev[2 * k]) for k in range(N))
        samples.extend([outcome] * cnt)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(samples)
    return samples


# ── Campionamento via Naimark a N qubit (solo shots) ─────────────────────────
def sample_naimark_Nqubit(state_circuit, N, U_naimark, n_shots, seed=42):
    """
    Campiona n_shots outcome POVM congiunti via dilatazione di Naimark + AerSimulator.

    Returns:
        samples : list di tuple (a_1,...,a_N), lunghezza n_shots
        counts  : dict bitstring -> conteggio (output grezzo di Qiskit)
    """
    qc  = build_naimark_circuit_Nqubit(state_circuit, N, U_naimark)
    sim = AerSimulator()
    qc_t   = transpile(qc, sim)
    result = sim.run(qc_t, shots=n_shots, seed_simulator=seed).result()
    counts = result.get_counts()
    samples = counts_to_samples(counts, N, seed=seed)
    return samples, counts


# ── Verifica: distribuzione empirica vs distribuzione esatta algebrica ───────
def empirical_distribution(samples):
    """dict {outcome: frequenza relativa} dai campioni."""
    tot = len(samples)
    return {k: v / tot for k, v in Counter(samples).items()}


def classical_fidelity(P_emp, P_exact):
    """F_c = sum_x sqrt(P_emp(x) P_exact(x))  (Eq. 3 del paper)."""
    return float(sum(np.sqrt(P_emp.get(x, 0.0) * px) for x, px in P_exact.items()))


# ════════════════════════════════════════════════════════════════════════════
# Demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    N       = 3
    N_SHOTS = 20000

    print("=" * 60)
    print(f"Naimark a N qubit (N={N}, shots={N_SHOTS})")
    print(f"Qubit fisici simulati: 3*N = {3 * N}  (N sistema + 2N ancilla)")
    print("=" * 60)

    # 1. unitario di Naimark a 1 qubit (lo stesso per tutti i qubit)
    U = build_naimark_unitary(M1)
    print(f"U unitaria: {np.allclose(U @ U.conj().T, np.eye(8), atol=1e-10)}")

    # 2. campiona lo stato GHZ via Naimark (misura congiunta, shots)
    state = ghz_circuit(N)
    samples, counts = sample_naimark_Nqubit(state, N, U, N_SHOTS)
    print(f"\nPrimi 5 campioni: {samples[:5]}")

    # 3. one-hot pronto per il VAE (stesso formato di povm_sampling.py)
    X = samples_to_onehot(samples, N)
    print(f"Shape matrice one-hot (input VAE): {X.shape}  (atteso: ({N_SHOTS}, {4 * N}))")

    # 4. verifica contro la distribuzione esatta algebrica
    print("\n" + "=" * 60)
    print("Verifica vs distribuzione esatta (algebra pura)")
    print("=" * 60)
    rho     = ghz_state_algebra(N)
    P_exact = povm_probability(rho, N)
    P_emp   = empirical_distribution(samples)

    zero = tuple([0] * N)
    print(f"P(0,...,0)  esatta : {P_exact[zero]:.4f}   (per GHZ vale 1/2^(N+1) = {0.5 ** (N + 1):.4f})")
    print(f"P(0,...,0)  empirica: {P_emp.get(zero, 0.0):.4f}")

    Fc = classical_fidelity(P_emp, P_exact)
    print(f"\nFidelity classica F_c(empirica, esatta): {Fc:.4f}")
    print(f"(-> ~1, differenza dovuta solo allo shot noise ~1/sqrt(shots) = {1 / np.sqrt(N_SHOTS):.4f})")

    print("\nNota: i campioni sono i.i.d. dalla STESSA P(a) del metodo algebrico")
    print("di povm_sampling.py — qui ottenuti via circuito di Naimark + shots,")
    print("con misura congiunta che preserva le correlazioni del GHZ.")
