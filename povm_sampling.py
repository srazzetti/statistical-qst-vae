"""
POVM Tetraedrico: campionamento via algebra vs Qiskit
======================================================

Due approcci equivalenti per ottenere il "sample space" POVM
di uno stato quantistico (es. GHZ a N qubit).

APPROCCIO 1 — Algebra pura (numpy)
    Costruiamo rho direttamente come matrice densità,
    calcoliamo P(a) = Tr[M(a) @ rho] per ogni outcome,
    poi campioniamo da quella distribuzione.

APPROCCIO 2 — Qiskit
    Prepariamo lo stato con un circuito Qiskit,
    estraiamo rho tramite DensityMatrix,
    poi stessa procedura algebrica per P(a).

In entrambi i casi il campionamento finale è classico (numpy.random),
perché stiamo simulando — non girando su hardware reale.
Su hardware reale si userebbe il teorema di Naimark
(vedi: arxiv.org/abs/1807.08449 e qiskit-community.github.io/povm-toolbox).
"""

import numpy as np
from itertools import product as iproduct

# ── Matrici di Pauli ────────────────────────────────────────────────────────
I  = np.eye(2, dtype=complex)
sx = np.array([[0, 1 ], [1,  0 ]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0 ], [0, -1 ]], dtype=complex)

# ── Direzioni tetraedriche (vertici tetraedro nella sfera di Bloch) ─────────
# Riferimento: Renes et al. 2004, quant-ph/0310075
S_VECS = np.array([
    [ 0,             0,            1   ],   # polo nord
    [ 2*np.sqrt(2)/3, 0,          -1/3 ],
    [-np.sqrt(2)/3,  np.sqrt(2/3), -1/3],
    [-np.sqrt(2)/3, -np.sqrt(2/3), -1/3],
])


# ── Costruzione operatori POVM per un singolo qubit ──────────────────────────
def make_povm_1qubit():
    """
    Restituisce lista di 4 matrici 2x2: M[alpha] = (I + s[alpha]·sigma) / 4
    Verifica: sum(M) == I  (completezza)
    """
    M = [(I + s[0]*sx + s[1]*sy + s[2]*sz) / 4 for s in S_VECS]
    assert np.allclose(sum(M), I), "POVM non completo!"
    return M


# ── POVM per N qubit: prodotto tensoriale ────────────────────────────────────
def povm_probability(rho, N, M1=None):
    """
    Calcola la distribuzione esatta P(a) = Tr[M(a1)⊗...⊗M(aN) @ rho]
    per tutti i 4^N outcome possibili.

    Args:
        rho : matrice densità (2^N x 2^N)
        N   : numero di qubit
        M1  : lista di 4 operatori POVM per singolo qubit (default: tetraedrico)

    Returns:
        dict {(a1,...,aN): probabilità}
    """
    if M1 is None:
        M1 = make_povm_1qubit()

    P = {}
    for outcome in iproduct(range(4), repeat=N):
        # prodotto tensoriale M(a1) ⊗ M(a2) ⊗ ... ⊗ M(aN)
        op = M1[outcome[0]]
        for i in range(1, N):
            op = np.kron(op, M1[outcome[i]])
        # Born's rule: P(a) = Tr[M(a) @ rho]
        P[outcome] = float(np.real(np.trace(op @ rho)))

    # piccole correzioni numeriche (probabilità negative vicinissime a 0)
    P = {k: max(v, 0.0) for k, v in P.items()}

    # rinormalizza per sicurezza numerica
    total = sum(P.values())
    P = {k: v / total for k, v in P.items()}

    return P


# ── Campionamento da P(a) ────────────────────────────────────────────────────
def sample_povm(P_exact, n_samples, seed=42):
    """
    Simula n_samples misure POVM campionando dalla distribuzione esatta.
    Ogni campione è una tupla (a1, a2, ..., aN) con ai in {0,1,2,3}.

    In un esperimento reale questo corrisponde a eseguire
    n_samples volte la misura POVM sull'hardware.

    Returns:
        list di tuple, lunghezza n_samples
    """
    rng = np.random.default_rng(seed)
    outcomes = list(P_exact.keys())
    probs    = np.array(list(P_exact.values()))
    indices  = rng.choice(len(outcomes), size=n_samples, p=probs)
    return [outcomes[i] for i in indices]


# ── Conversione sample -> one-hot (formato input VAE) ───────────────────────
def samples_to_onehot(samples, N):
    """
    Converte lista di tuple in matrice one-hot (n_samples x 4N).
    Ogni qubit i con outcome a_i -> vettore one-hot di dim 4
    posizionato nelle colonne [4i : 4i+4].

    Es. N=3, outcome=(2,0,1) -> [0,0,1,0, 1,0,0,0, 0,1,0,0]
    """
    n = len(samples)
    X = np.zeros((n, 4 * N), dtype=np.float32)
    for row, outcome in enumerate(samples):
        for qubit, a in enumerate(outcome):
            X[row, qubit * 4 + a] = 1.0
    return X


# ════════════════════════════════════════════════════════════════════════════
# APPROCCIO 1: algebra pura
# ════════════════════════════════════════════════════════════════════════════

def ghz_state_algebra(N):
    """
    Costruisce la matrice densità dello stato GHZ a N qubit
    direttamente con numpy.

    |GHZ⟩ = (|00...0⟩ + |11...1⟩) / sqrt(2)
    rho = |GHZ⟩⟨GHZ|
    """
    dim = 2**N
    psi = np.zeros(dim, dtype=complex)
    psi[0]    = 1 / np.sqrt(2)   # |00...0⟩
    psi[-1]   = 1 / np.sqrt(2)   # |11...1⟩
    rho = np.outer(psi, psi.conj())
    return rho


def approach_algebra(N, n_samples):
    print(f"\n{'='*55}")
    print(f"APPROCCIO 1: Algebra pura  (N={N}, n_samples={n_samples})")
    print('='*55)

    # 1. stato GHZ
    rho = ghz_state_algebra(N)
    print(f"rho shape: {rho.shape},  Tr(rho) = {np.real(np.trace(rho)):.6f}")

    # 2. distribuzione esatta
    P = povm_probability(rho, N)
    print(f"Num outcome possibili: {len(P)}  (= 4^{N} = {4**N})")
    print(f"Somma P(a): {sum(P.values()):.8f}")
    print(f"Esempio P(0,0,...,0) = {P[tuple([0]*N)]:.6f}")

    # 3. campionamento
    samples = sample_povm(P, n_samples)
    print(f"Primi 5 campioni: {samples[:5]}")

    # 4. one-hot per VAE
    X = samples_to_onehot(samples, N)
    print(f"Shape matrice one-hot (input VAE): {X.shape}")

    return rho, P, samples, X


# ════════════════════════════════════════════════════════════════════════════
# APPROCCIO 2: Qiskit
# ════════════════════════════════════════════════════════════════════════════

def ghz_circuit_qiskit(N):
    """
    Costruisce il circuito GHZ a N qubit con Qiskit.
    H sul primo qubit, poi CNOT a cascata.
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N)
    qc.h(0)
    for i in range(N - 1):
        qc.cx(i, i + 1)
    return qc


def approach_qiskit(N, n_samples):
    print(f"\n{'='*55}")
    print(f"APPROCCIO 2: Qiskit        (N={N}, n_samples={n_samples})")
    print('='*55)

    from qiskit.quantum_info import DensityMatrix

    # 1. circuito e matrice densità
    qc = ghz_circuit_qiskit(N)
    print(qc.draw())
    rho = DensityMatrix(qc).data
    print(f"rho shape: {rho.shape},  Tr(rho) = {np.real(np.trace(rho)):.6f}")

    # 2. distribuzione esatta (stessa procedura algebrica)
    P = povm_probability(rho, N)
    print(f"Num outcome possibili: {len(P)}  (= 4^{N} = {4**N})")
    print(f"Somma P(a): {sum(P.values()):.8f}")
    print(f"Esempio P(0,0,...,0) = {P[tuple([0]*N)]:.6f}")

    # 3. campionamento
    samples = sample_povm(P, n_samples)
    print(f"Primi 5 campioni: {samples[:5]}")

    # 4. one-hot per VAE
    X = samples_to_onehot(samples, N)
    print(f"Shape matrice one-hot (input VAE): {X.shape}")

    return rho, P, samples, X


# ════════════════════════════════════════════════════════════════════════════
# Confronto: i due approcci devono dare P identiche
# ════════════════════════════════════════════════════════════════════════════

def compare_approaches(N):
    print(f"\n{'='*55}")
    print(f"CONFRONTO tra i due approcci (N={N})")
    print('='*55)

    rho_alg = ghz_state_algebra(N)
    P_alg   = povm_probability(rho_alg, N)

    from qiskit.quantum_info import DensityMatrix
    qc      = ghz_circuit_qiskit(N)
    rho_qk  = DensityMatrix(qc).data
    P_qk    = povm_probability(rho_qk, N)

    # confronta le distribuzioni
    max_diff = max(abs(P_alg[k] - P_qk[k]) for k in P_alg)
    print(f"Max differenza tra P_algebra e P_qiskit: {max_diff:.2e}")
    print(f"Approcci equivalenti: {np.isclose(max_diff, 0, atol=1e-10)}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N         = 3
    N_SAMPLES = 1000

    rho1, P1, s1, X1 = approach_algebra(N, N_SAMPLES)
    rho2, P2, s2, X2 = approach_qiskit(N, N_SAMPLES)
    compare_approaches(N)

    print("\nDone. X1 e X2 sono pronti come input per il VAE.")