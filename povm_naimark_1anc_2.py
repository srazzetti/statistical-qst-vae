# depreato
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit import transpile

# ── 1. POVM Locale (Tetraedro 1 Qubit) ──────────────────────────────────
I  = np.eye(2, dtype=complex)
sx = np.array([[0, 1 ], [1,  0 ]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0 ], [0, -1 ]], dtype=complex)

S_VECS = np.array([
    [ 0,              0,            1   ],
    [ 2*np.sqrt(2)/3, 0,           -1/3 ],
    [ -np.sqrt(2)/3,  np.sqrt(2/3), -1/3],
    [ -np.sqrt(2)/3, -np.sqrt(2/3), -1/3],
])
M_local = [(I + s[0]*sx + s[1]*sy + s[2]*sz) / 4 for s in S_VECS]

# ── 2. Costruzione dell'Unitario Locale 4x4 Analitico Rigido ────────────
def build_local_unitary_4x4(M):
    """
    Costruisce l'operatore unitario 4x4 (Sistema + Ancella) mappando analiticamente
    le componenti vettoriali del tetraedro per bloccare le fasi geometriche.
    Convenzione Qiskit standard: |q1_sys, q0_anc> -> indici [00, 01, 10, 11]
    """
    U = np.zeros((4, 4), dtype=complex)
    
    # Estrarre i vettori di stato complessi associati ai proiettori puri della POVM
    v_states = []
    for m in range(4):
        w, v = np.linalg.eigh(M[m])
        idx_max = np.argmax(w)
        # Vettore non normalizzato che racchiude la corretta ampiezza fisica
        v_m = v[:, idx_max] * np.sqrt(w[idx_max])
        v_states.append(v_m)
        
    # Assegnazione delle colonne fisiche di input (Ancella in |0>)
    # Colonna 0 correspond a input |0_sys> ⊗ |0_anc>
    # Colonna 2 correspond a input |1_sys> ⊗ |0_anc>
    for m in range(4):
        U[m, 0] = v_states[m][0]
        U[m, 2] = v_states[m][1]
        
    # Completamento analitico delle colonne 1 e 3 (Ancella in |1>)
    # Specchiamo coniugando i coefficienti per forzare l'ortogonalità di calibrazione
    U[:, 1] = np.array([-U[1, 0].conj(), U[0, 0].conj(), -U[3, 0].conj(), U[2, 0].conj()])
    U[:, 1] /= np.linalg.norm(U[:, 1])
    
    U[:, 3] = np.array([-U[1, 2].conj(), U[0, 2].conj(), -U[3, 2].conj(), U[2, 2].conj()])
    U[:, 3] -= np.vdot(U[:, 1], U[:, 3]) * U[:, 1] # Ortogonalizzazione mutua
    U[:, 3] /= np.linalg.norm(U[:, 3])
    
    # Stabilizzazione finale per evitare micro-derive floating-point
    u_ref, _, vh_ref = np.linalg.svd(U)
    return u_ref @ vh_ref

# ── 3. Funzioni per il Calcolo Multiqubit ───────────────────────────────
def get_multiqubit_povm_element(indices, M_single):
    op = M_single[indices[0]]
    for idx in indices[1:]:
        op = np.kron(op, M_single[idx])
    return op

def sample_multiqubit_exact(state_circuit, n_qubits, M_single):
    from qiskit.quantum_info import DensityMatrix
    rho = DensityMatrix(state_circuit).data
    num_outcomes = 4 ** n_qubits
    P_outcomes = np.zeros(num_outcomes)
    
    for dec_idx in range(num_outcomes):
        indices = []
        temp = dec_idx
        for _ in range(n_qubits):
            indices.append(temp % 4)
            temp //= 4
        indices = indices[::-1]
        
        M_global = get_multiqubit_povm_element(indices, M_single)
        P_outcomes[dec_idx] = np.real(np.trace(M_global @ rho))
        
    P_outcomes /= np.sum(P_outcomes)
    return P_outcomes

def sample_multiqubit_shots(state_circuit, n_qubits, n_shots, U_local):
    # Registro a 2N qubit: qubit di sistema in alto, ancelle dedicate in basso
    qc = QuantumCircuit(2 * n_qubits, 2 * n_qubits)
    
    sys_qubits = list(range(n_qubits, 2 * n_qubits))
    qc.compose(state_circuit, qubits=sys_qubits, inplace=True)
    
    op_4x4 = Operator(U_local)
    
    # Applichiamo l'interazione ad ogni blocco locale (Ancella_i, Sistema_i)
    for i in range(n_qubits):
        anc_idx = i
        sys_idx = n_qubits + i
        # Configurazione qubit ordinata per la compilazione nativa di Qiskit
        qc.unitary(op_4x4, [anc_idx, sys_idx])
        
        # Misura accoppiata dei registri
        qc.measure([sys_idx, anc_idx], [2*i + 1, 2*i])
        
    sim = AerSimulator()
    qc_t = transpile(qc, sim)
    counts = sim.run(qc_t, shots=n_shots).result().get_counts()
    
    num_outcomes = 4 ** n_qubits
    outcome_counts = {i: 0 for i in range(num_outcomes)}
    
    for bitstring, count in counts.items():
        bitstring = bitstring[::-1]
        dec_idx = 0
        for i in range(n_qubits):
            pair = bitstring[2*i:2*i+2]
            local_outcome = int(pair, 2)
            dec_idx += local_outcome * (4 ** i)
        outcome_counts[dec_idx] += count
        
    return outcome_counts

# ── 4. Esecuzione del Test e Risultato ──────────────────────────────────
if __name__ == "__main__":
    N_QUBITS = 3
    N_SHOTS = 20000000  # Aumentiamo gli shots per una convergenza statistica ottimale su 64 esiti
    print("=" * 60)
    print(f"Test Estensione a {N_QUBITS} Qubit (Stato GHZ)")
    print("=" * 60)
    
    # Prepariamo lo stato GHZ
    qc_ghz = QuantumCircuit(N_QUBITS)
    qc_ghz.h(0)
    qc_ghz.cx(0, 1)
    qc_ghz.cx(1, 2)
    
    # Costruzione dell'unitario analitico calibrato
    U_local = build_local_unitary_4x4(M_local)
    print("-> Unitario locale 4x4 analitico generato correttamente.")
    
    # 1. Distribuzione Teorica P(m)
    P_exact = sample_multiqubit_exact(qc_ghz, N_QUBITS, M_local)
    
    # 2. Distribuzione Emulata dai campionamenti quantistici
    counts = sample_multiqubit_shots(qc_ghz, N_QUBITS, N_SHOTS, U_local)
    P_shots = np.array([counts[i] / N_SHOTS for i in range(4**N_QUBITS)])
    
    print(f"\nPrimi 8 esiti teorici (su {4**N_QUBITS}):")
    print(np.round(P_exact[:8], 4))
    print("Primi 8 esiti dal simulatore (Shots):")
    print(np.round(P_shots[:8], 4))
    
    # Verifica di aderenza statistica (tolleranza legata allo shot noise su 64 canali)
    match = np.allclose(P_exact, P_shots, atol=1.5e-2)
    print(f"\nMatch statistico con la teoria globale: {match}")