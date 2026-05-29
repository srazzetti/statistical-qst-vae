"""
Campionamento POVM Tetraedrico via Teorema di Naimark
=====================================================

Il teorema di Naimark dice che qualsiasi POVM {M(a)} può essere realizzato
come misura proiettiva su uno spazio più grande (sistema + ancilla):

    1. Prepara ancilla in |0>
    2. Applica unitario U sul sistema+ancilla
    3. Misura l'ancilla in base Z
    4. L'outcome dell'ancilla = outcome POVM

Per il POVM tetraedrico (4 outcome) servono 2 qubit ancilla (2^2 = 4).

Convenzione Qiskit (little-endian):
    - q2 = sistema (bit più significativo)
    - q1, q0 = ancilla (q1*2 + q0 = outcome 0..3)

Riferimenti:
    - Naimark dilation: arxiv.org/abs/1807.08449
    - SIC-POVM tetraedrico: Renes et al. 2004, quant-ph/0310075
    - Qiskit POVM toolbox: qiskit-community.github.io/povm-toolbox
"""

import numpy as np
from scipy.linalg import sqrtm, null_space
from itertools import product as iproduct
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit import transpile

# ── Pauli e POVM tetraedrico ──────────────────────────────────────────────
I  = np.eye(2, dtype=complex)
sx = np.array([[0, 1 ], [1,  0 ]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0 ], [0, -1 ]], dtype=complex)

S_VECS = np.array([
    [ 0,              0,            1   ],
    [ 2*np.sqrt(2)/3, 0,           -1/3 ],
    [-np.sqrt(2)/3,   np.sqrt(2/3), -1/3],
    [-np.sqrt(2)/3,  -np.sqrt(2/3), -1/3],
])
M1 = [(I + s[0]*sx + s[1]*sy + s[2]*sz) / 4 for s in S_VECS]


# ── Costruzione unitario di Naimark ──────────────────────────────────────
def build_naimark_unitary(M):
    """
    Costruisce U (8x8) tale che:
        U |s_in>|00> = sum_a sqrt(M^a)|s_in> ⊗ |a_binario>

    Convenzione indici: |sistema, anc1, anc0> -> idx = sys*4 + anc1*2 + anc0

    Le prime due colonne (input |0>|00> e |1>|00>) sono fissate dalla
    struttura POVM. Le restanti 6 colonne completano U a matrice unitaria
    tramite QR decomposition sul null space.
    """
    import warnings
    warnings.filterwarnings('ignore')  # sqrtm di matrici singolari

    U = np.zeros((8, 8), dtype=complex)

    # colonne 0 e 4 corrispondono a input |0>|00> e |1>|00>
    for s_in in range(2):
        col_idx = s_in * 4
        for a in range(4):
            sqM_col = sqrtm(M[a])[:, s_in]
            for s_out in range(2):
                row_idx = s_out * 4 + a
                U[row_idx, col_idx] = sqM_col[s_out]

    # completa a unitaria con Gram-Schmidt (QR sul null space)
    cols_fixed = U[:, [0, 4]]
    null       = null_space(cols_fixed.conj().T)   # 8x6
    U_full     = np.hstack([cols_fixed, null])
    Q, R       = np.linalg.qr(U_full)
    for i in range(Q.shape[1]):
        if R[i, i] < 0:
            Q[:, i] *= -1

    U_final = np.zeros((8, 8), dtype=complex)
    U_final[:, 0] = Q[:, 0]
    U_final[:, 4] = Q[:, 1]
    remaining = [c for c in range(8) if c not in [0, 4]]
    for i, col in enumerate(remaining):
        U_final[:, col] = Q[:, i + 2]

    assert np.allclose(U_final @ U_final.conj().T, np.eye(8), atol=1e-10), \
        "U non è unitaria!"
    return U_final


# ── Campionamento via Naimark (statevector, esatto) ───────────────────────
def sample_naimark_statevector(state_circuit, n_samples, U_naimark, seed=42):
    """
    Campiona gli outcome POVM usando la dilatazione di Naimark
    tramite simulazione statevector (esatta, niente shot noise).

    Args:
        state_circuit : QuantumCircuit che prepara lo stato su q2
                        (q0, q1 = ancilla, devono essere |0>)
        n_samples     : numero di campioni
        U_naimark     : unitario 8x8 costruito con build_naimark_unitary()
        seed          : seed per riproducibilità

    Returns:
        list di interi in {0,1,2,3} = outcome POVM
    """
    # circuito: prepara stato su q2, applica U su q0,q1,q2
    qc = QuantumCircuit(3)
    qc.compose(state_circuit, qubits=[2], inplace=True)
    gate = Operator(U_naimark)
    qc.unitary(gate, [0, 1, 2])

    # statevector dopo U
    sv    = Statevector(qc)
    probs = sv.probabilities()   # 8 probabilità per |q2 q1 q0>

    # marginalizza su ancilla (q1, q0): somma su q2 (sistema)
    P_ancilla = np.zeros(4)
    for full_idx in range(8):
        anc = full_idx & 3          # q1*2 + q0 = outcome 0..3
        P_ancilla[anc] += probs[full_idx]

    # campiona da P_ancilla
    rng     = np.random.default_rng(seed)
    samples = rng.choice(4, size=n_samples, p=P_ancilla)
    return list(samples), P_ancilla


# ── Campionamento via Naimark (AerSimulator, con shot noise) ──────────────
def sample_naimark_shots(state_circuit, n_shots, U_naimark):
    """
    Campiona gli outcome POVM usando AerSimulator con misure reali.
    Questo simula cosa accadrebbe su hardware reale.

    Returns:
        dict {outcome: count}
    """
    qc = QuantumCircuit(3, 2)
    qc.compose(state_circuit, qubits=[2], inplace=True)
    gate = Operator(U_naimark)
    qc.unitary(gate, [0, 1, 2])
    qc.measure([0, 1], [0, 1])   # misura ancilla q0 -> cbit0, q1 -> cbit1

    sim    = AerSimulator()
    qc_t   = transpile(qc, sim)
    result = sim.run(qc_t, shots=n_shots).result()
    counts = result.get_counts()

    # converti bitstring '01' -> outcome intero (q1=0, q0=1 -> 0*2+1=1)
    outcome_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for bitstring, count in counts.items():
        # Qiskit: bitstring[0]=cbit1 (q1), bitstring[1]=cbit0 (q0)
        q1 = int(bitstring[0])
        q0 = int(bitstring[1])
        a  = q1 * 2 + q0
        outcome_counts[a] += count
    return outcome_counts


# ── Verifica: confronto con prob algebrica ────────────────────────────────
def povm_prob_algebra(rho, M):
    return np.array([np.real(np.trace(m @ rho)) for m in M])


# ════════════════════════════════════════════════════════════════════════════
# Demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 60)
    print("Costruzione U di Naimark")
    print("=" * 60)
    U = build_naimark_unitary(M1)
    print(f"U unitaria: {np.allclose(U @ U.conj().T, np.eye(8), atol=1e-10)}")
    print(f"Shape U: {U.shape}  (sistema 1q + ancilla 2q = 3q totali)")

    # ── stato di test: |+> = (|0>+|1>)/sqrt(2) ──────────────────────────
    qc_plus = QuantumCircuit(1)
    qc_plus.h(0)

    psi  = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho  = np.outer(psi, psi.conj())
    P_alg = povm_prob_algebra(rho, M1)

    print("\n" + "=" * 60)
    print("Test su stato |+>")
    print("=" * 60)
    print(f"P algebrica:          {np.round(P_alg, 4)}")

    # statevector (esatto)
    samples_sv, P_sv = sample_naimark_statevector(qc_plus, 10000, U, seed=0)
    print(f"P Naimark statevec:   {np.round(P_sv, 4)}")
    print(f"Match esatto:         {np.allclose(P_alg, P_sv, atol=1e-10)}")

    # shot simulator (con rumore statistico)
    counts = sample_naimark_shots(qc_plus, n_shots=10000, U_naimark=U)
    P_shots = np.array([counts[a] / 10000 for a in range(4)])
    print(f"P Naimark shots:      {np.round(P_shots, 4)}")
    print(f"Differenza max shots: {np.max(np.abs(P_shots - P_alg)):.4f}  "
          f"(rumore da sqrt(N)~0.01)")

    # ── stato GHZ 1 qubit (= |+>) ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Nota: per N qubit si applica Naimark su ogni qubit separatamente")
    print("Ogni qubit: 1 sistema + 2 ancilla = 3 qubit fisici")
    print(f"Per N=3 qubit GHZ: servono 3 + 3*2 = 9 qubit fisici totali")
    print("=" * 60)

    print("\nRiepilogo approcci:")
    print("  algebra pura    -> P esatta, nessun overhead hardware")
    print("  Naimark statevec-> P esatta via circuito (verifica teorica)")
    print("  Naimark shots   -> P con shot noise (simula hardware reale)")
    print("\nPer il paper: algebra pura è sufficiente e più efficiente.")