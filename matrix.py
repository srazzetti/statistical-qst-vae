import numpy as np

def get_overlap_matrix_and_inverse(M_local, n_qubits):
    """
    Calcola la matrice di overlap T globale e la sua inversa T_inv 
    sfruttando il prodotto di Kronecker per N qubit.
    """
    # 1. Calcoliamo la T locale 4x4 per 1 qubit
    T_local = np.zeros((4, 4))
    for a in range(4):
        for ap in range(4):
            # Prodotto traccia standard Tr(M_a @ M_ap)
            T_local[a, ap] = np.real(np.trace(M_local[a] @ M_local[ap]))
            
    # 2. Estendiamo a N qubit tramite prodotto tensoriale (Kronecker)
    T_global = T_local
    for _ in range(n_qubits - 1):
        T_global = np.kron(T_global, T_local)
        
    # 3. Calcoliamo l'inversa (il paper garantisce che esiste se la POVM è IC)
    T_global_inv = np.linalg.inv(T_global)
    return T_global, T_global_inv

def reconstruct_density_matrix(P_vector, M_local, n_qubits, T_global_inv):
    """
    Implementa l'Equazione (5) del paper.
    Prende in input un vettore di probabilità P (dimensione 4^N) e ricostruisce rho (2^N x 2^N).
    """
    dim_rho = 2 ** n_qubits
    rho = np.zeros((dim_rho, dim_rho), dtype=complex)
    
    # Pre-generiamo tutti gli operatori globali M^(a') per non ricalcolarli nel loop
    # Ce ne sono 4^N, ciascuno è una matrice (2^N x 2^N)
    M_global_list = []
    num_outcomes = 4 ** n_qubits
    
    for idx in range(num_outcomes):
        # Convertiamo l'indice lineare in base 4 per sapere quali operatori locali accoppiare
        # Es. per 3 qubit: idx 5 -> '011' -> M_0 x M_1 x M_1
        bitstring = format(idx, f'0{n_qubits}b' if n_qubits > 1 else '01b') # se base 4 usiamo conversioni ad hoc
        # Per sicurezza facciamo una scomposizione formale in base 4:
        temp = idx
        digits = []
        for _ in range(n_qubits):
            digits.append(temp % 4)
            temp //= 4
        # digits contiene gli indici locali [a_1, a_2, ... a_N]
        
        M_g = M_local[digits[0]]
        for d in digits[1:]:
            M_g = np.kron(M_g, M_local[d])
        M_global_list.append(M_g)
        
    # Applichiamo l'equazione (5): rho = \sum_{a, a'} P(a) * T_inv[a, a'] * M^(a')
    for a in range(num_outcomes):
        if P_vector[a] == 0: 
            continue
        for ap in range(num_outcomes):
            coeff = P_vector[a] * T_global_inv[a, ap]
            rho += coeff * M_global_list[ap]
            
    return rho

'''
esempio di come implementare rumore
from qiskit.quantum_info import DensityMatrix, PhaseDampingChannel

# 1. Crei lo stato GHZ puro di partenza (vettore di stato)
# Supponiamo tu abbia il tuo circuito 'qc_ghz' senza misure
rho_puro = DensityMatrix.from_instruction(qc_ghz)

# 2. Crei un canale di rumore algebrico (es. phase damping con parametro gamma=0.2)
noise_channel = PhaseDampingChannel(0.2)

# 3. Fai evolvere la matrice: il qubit 0 subisce il rumore
# Ripetendo l'operazione sui qubit 1 e 2, ottieni la matrice totalmente mista
rho_misto = rho_puro.evolve(noise_channel, qarg=[0])
rho_misto = rho_misto.evolve(noise_channel, qarg=[1])
rho_misto = rho_misto.evolve(noise_channel, qarg=[2])

# Ora rho_misto è una matrice 8x8 con le coerenze smorzate!
'''