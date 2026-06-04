import numpy as np
from itertools import product

# Matrici di Pauli
I  = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)

# Direzioni tetraedriche
s = np.array([
    [0, 0, 1],
    [2*np.sqrt(2)/3, 0, -1/3],
    [-np.sqrt(2)/3,  np.sqrt(2/3), -1/3],
    [-np.sqrt(2)/3, -np.sqrt(2/3), -1/3],
])

# 4 operatori POVM per un qubit
def povm_single_qubit(s_vecs):
    M = []
    for sv in s_vecs:
        m = (I + sv[0]*sx + sv[1]*sy + sv[2]*sz) / 4
        M.append(m)
    return M  # lista di 4 matrici 2x2

M1 = povm_single_qubit(s)

# POVM per N qubit: prodotto tensoriale
def povm_nqubit(M1, N):
    """
    Restituisce dict: outcome_tuple -> matrice POVM
    outcome è una tupla (a1,a2,...,aN) con ai in {0,1,2,3}
    """
    M = {}
    for outcome in product(range(4), repeat=N):
        op = M1[outcome[0]]
        for i in range(1, N):
            op = np.kron(op, M1[outcome[i]])
        M[outcome] = op
    return M

# P(a) = Tr[M(a) @ rho]
def exact_distribution(rho, N):
    M = povm_nqubit(M1, N)
    P = {}
    for outcome, op in M.items():
        P[outcome] = np.real(np.trace(op @ rho))
    return P  # 4^N valori, sommano a 1

def empirical_distribution(samples, N):
    """
    samples: array (N_e, 4N) di vettori one-hot
    Restituisce dict: outcome_tuple -> frequenza relativa
    """
    P_vae = {}
    n_samples = len(samples)
    
    for sample in samples:
        # riconverti one-hot -> tupla di outcome
        outcome = tuple(
            np.argmax(sample[i*4:(i+1)*4]) for i in range(N)
        )
        P_vae[outcome] = P_vae.get(outcome, 0) + 1
    
    # normalizza
    for k in P_vae:
        P_vae[k] /= n_samples
    
    return P_vae

def classical_fidelity(P_vae, P_exact, N):
    """
    F_c = sum_x sqrt(P_vae(x) * P(x))
    Gli outcome non osservati da P_vae hanno prob 0
    """
    fc = 0.0
    for outcome in product(range(4), repeat=N):
        p_exact = P_exact.get(outcome, 0.0)
        p_vae   = P_vae.get(outcome, 0.0)
        fc += np.sqrt(p_exact * p_vae)
    return fc

def quantum_fidelity_pure(rho_vae, psi_exact):
    """
    Calcola la fedeltà quantistica nel caso in cui lo stato esatto sia puro.
    rho_vae: matrice di densità 2^N x 2^N (Numpy array)
    psi_exact: vettore di stato 2^N (Numpy array o Qiskit Statevector)
    """
    # Trasformiamo psi_exact in un vettore colonna e riga (bra e ket)
    ket = np.array(psi_exact).reshape(-1, 1)
    bra = ket.conj().T
    
    # Calcolo del sandwich: bra @ rho @ ket
    fidelity = bra @ rho_vae @ ket
    
    # Il risultato è una matrice 1x1 complessa, estraiamo lo scalare reale
    return np.real(fidelity[0, 0])

from scipy.linalg import sqrtm

def quantum_fidelity_general(rho_1, rho_2):
    """
    Formula generale di Jozsa per la fedeltà tra due matrici di densità qualsiasi.
    F = [ Tr( sqrt( sqrt(rho_1) @ rho_2 @ sqrt(rho_1) ) ) ]^2
    """
    # Calcola la radice quadrata della prima matrice
    sqrt_rho_1 = sqrtm(rho_1)
    # Prodotto interno
    inner_matrix = sqrt_rho_1 @ rho_2 @ sqrt_rho_1
    # Radice quadrata del risultato
    sqrt_inner = sqrtm(inner_matrix)
    # Traccia al quadrato
    fidelity = np.real(np.trace(sqrt_inner)) ** 2
    return fidelity