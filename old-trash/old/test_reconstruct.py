#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test
"""

# ----------------------------------------------------------------------------------------------------------------------------
# Imports

import numpy as np
from itertools import product as iproduct
from qiskit.quantum_info import state_fidelity, DensityMatrix

# from collections import Counter

# ----------------------------------------------------------------------------------------------------------------------------
# Costants

# Pauli matrices
I  = np.eye(2, dtype=complex)
sx = np.array([[0, 1 ], [1,  0 ]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0 ], [0, -1 ]], dtype=complex)

# Tetrahedral directions (vertices of the SIC POVM on the Bloch sphere)
S_VECS = np.array([
    [ 0,             0,            1   ],   # north pole
    [ 2*np.sqrt(2)/3, 0,          -1/3 ],
    [-np.sqrt(2)/3,  np.sqrt(2/3), -1/3],
    [-np.sqrt(2)/3, -np.sqrt(2/3), -1/3],
])

# SIC-POVM single qubit operators
M1 = [(I + s[0]*sx + s[1]*sy + s[2]*sz) / 4 for s in S_VECS]

# ----------------------------------------------------------------------------------------------------------------------------
# Functions 

def build_povm(N):
    """
    Build the dictionary {outcome_tuple: POVM_operator} for N qubits.
    outcome_tuple = (a1, a2, ..., aN) with ai in {0,1,2,3}
    """
    povm_dict = {}
    # iproduct is the inter-product of {0,1,2,3}x{0,1,3,4}..{0,1,2,3} N times --> all possible outcomes 
    for outcome in iproduct(range(4), repeat=N):
        # tensor prod M(a) = M(a1) x M(a2) x ... x M(aN)
        op = M1[outcome[0]]
        for i in range(1, N):
            op = np.kron(op, M1[outcome[i]])
        povm_dict[outcome] = op
    return povm_dict


# fidelity -------------------------------------------------------------------------------------------------
def classical_fidelity(P1, P2):
    """
    Classical fidelity (Bhattacharyya) between two distributions:  Fc = sum_x sqrt(P1(x) * P2(x))
    Args:
        P1: dict{outcome: prob}
        P2: dict{outcome: prob}
    Returns: Fc
    """
    # compute the union of all observed outcomes from both probability distributions
    # "|" pipe ensures to keep all possible results
    outcomes = set(P1.keys()) | set(P2.keys())
    return sum(np.sqrt(P1.get(o, 0.0) * P2.get(o, 0.0)) for o in outcomes)

def quantum_fidelity(rho1, rho2):
    """
    Quantum fidelity (Jozsa): Fq(rho1, rho2) = [ Tr( sqrt( sqrt(rho1) @ rho2 @ sqrt(rho1) ) ) ]^2
    If one of states is pure: Fq = <psi1|rho2|psi1>, where |psi1> --> rho1 
    Args:
        rho1: denisty matrix 
        rho2: density matrix
    Returns: Fq
    """
    # state fidelity first verifies if one in rho1, rho2 is pure, then perform the relative calculation
    Fq = state_fidelity(DensityMatrix(rho1), DensityMatrix(rho2))
    return Fq


# overlap matrix and algebraic reconstruction of rho -------------------------------------------------------------------------
def get_overlap_matrix(N):
    """
    Compute global overlap matrices T and T^{-1} for an N-qubit system.
        T_{a,a'} = Tr[ M(a) @ M(a') ]   --> where a, a' = (a1, a2, ..., aN) are all possible outcomes.
    This can be written as (tensor product structure):
        T_global = T_loc x ... x T_loc  --> where T_loc[a,b] = Tr[M(a) @ M(b)] and a,b in {0,1,2,3}.
    Returns:
        T     : global overlap matrix (4^N, 4^N)
        T_inv : inverse global overlap matrix
    """
    # T local for 1 single qubit
    T_loc = np.array([
        [np.real(np.trace(M1[a] @ M1[b])) for b in range(4)]
        for a in range(4)
    ])

    # T global by Kronecker (tensor product structure)
    T = T_loc.copy()
    for _ in range(N - 1):
        T = np.kron(T, T_loc)

    # inverse by numpy
    T_inv = np.linalg.inv(T)
    return T, T_inv


def pvec_from_pdict(P_dict, N):
    """
    Transform (empirical) prob distribution dictionary into ordered numpy array.
    It uses the same order of build_povm().
    Args:
        P_dict: {outcome: prob} --> outcome are povm_tuple
        N: number of qubits
    Returns: probability numpy array (4^N,)
    """
    outcomes = list(iproduct(range(4), repeat=N))
    return np.array([P_dict.get(o, 0.0) for o in outcomes])


def reconstruct_rho(P_vec, N, T_inv, povm_dict):
    """
    Reconstruct rho density matrix using:  rho = sum_{a,a'} P(a) T_inv{a,a'} M(a')
    This can be efficiently rewritten and computed as: 
        rho = sum_{a'} c[a'] * M(a')    where    c = T_inv @ P
    Args:
        P_vec    : (empirical) prob array (4^N,)
        N        : number of qubits
        T_inv    : inverse global overlap matrix (4^N, 4^N)
        povm_dict: dict {outcome_tuple: POVM_operator} --> build_povm(N)

    Returns:
        rho : reconstructed density matrix (2^N, 2^N)
    """
    # compute the coeffs
    c   = T_inv @ P_vec

    dim = 2**N
    povm_dict = build_povm(N)
    rho = np.zeros((dim, dim), dtype=complex)
    for i, (outcome, M_g) in enumerate(povm_dict.items()):
        if abs(c[i]) < 1e-15:
            continue    # skip the negligible terms
        rho += c[i] * M_g
    return rho


def validate_rho(rho, tol=1e-6):
    """
    Verify density matrix rho is valid, print a report.
    """
    is_hermitian = np.allclose(rho, rho.conj().T, atol=tol)
    trace        = np.real(np.trace(rho))
    eigvals      = np.linalg.eigvalsh(rho)
    is_psd       = np.all(eigvals >= -tol) # positive semi def

    print(f"  Hermitian:          {is_hermitian}")
    print(f"  Tr(rho):            {trace:.8f}")
    print(f"  Pos semi-definite:  {is_psd}  (min eigval = {min(eigvals):.2e})")
    return is_hermitian and abs(trace - 1) < tol and is_psd

# ── Versione originale (per benchmark) ───────────────────────────────────────

def get_overlap_matrix_and_inverse_original(M_local, n_qubits):
    """Versione originale — doppio loop esplicito."""
    T_local = np.zeros((4, 4))
    for a in range(4):
        for ap in range(4):
            T_local[a, ap] = np.real(np.trace(M_local[a] @ M_local[ap]))
    T_global = T_local
    for _ in range(n_qubits - 1):
        T_global = np.kron(T_global, T_local)
    return T_global, np.linalg.inv(T_global)


def reconstruct_density_matrix_original(P_vector, M_local, n_qubits, T_global_inv):
    """Versione originale — doppio loop O(16^N)."""
    dim_rho      = 2 ** n_qubits
    rho          = np.zeros((dim_rho, dim_rho), dtype=complex)
    num_outcomes = 4 ** n_qubits
    M_global_list = []
    for idx in range(num_outcomes):
        temp, digits = idx, []
        for _ in range(n_qubits):
            digits.append(temp % 4); temp //= 4
        M_g = M_local[digits[0]]
        for d in digits[1:]: M_g = np.kron(M_g, M_local[d])
        M_global_list.append(M_g)
    for a in range(num_outcomes):
        if P_vector[a] == 0: continue
        for ap in range(num_outcomes):
            coeff = P_vector[a] * T_global_inv[a, ap]
            rho  += coeff * M_global_list[ap]
    return rho

# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__": 
    print('utils.py has no main')
    import time
    from qiskit.quantum_info import state_fidelity, DensityMatrix

    print("=" * 60)
    print("Ricostruzione rho da P esatta (Eq. 5)")
    print("=" * 60)

    # ── correttezza e fidelity ────────────────────────────────────────────
    for N in [2, 3, 4]:
        dim = 2**N
        print(f"\nN={N} qubit  (dim rho={dim}x{dim}, outcomes={4**N})")

        psi      = np.zeros(dim, dtype=complex)
        psi[0]   = psi[-1] = 1 / np.sqrt(2)
        rho_true = np.outer(psi, psi.conj())
        povm_dict = build_povm(N)

        P_dict = {}
        for outcome in iproduct(range(4), repeat=N):
            op = M1[outcome[0]]
            for i in range(1, N): op = np.kron(op, M1[outcome[i]])
            P_dict[outcome] = np.real(np.trace(op @ rho_true))
        P_vec    = pvec_from_pdict(P_dict, N)
        T, T_inv = get_overlap_matrix(N)

        t0      = time.time()
        rho_rec = reconstruct_rho(P_vec, N, T_inv, povm_dict)
        t1      = time.time()

        validate_rho(rho_rec)
        fq = state_fidelity(DensityMatrix(rho_true), DensityMatrix(rho_rec))
        print(f"  F_q (P esatta):      {fq:.10f}  (deve essere 1.0)")
        print(f"  Tempo:               {(t1-t0)*1000:.1f} ms")

    # ── benchmark originale vs nuova ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Benchmark: originale (doppio loop) vs nuova (loop singolo)")
    print(f"{'=' * 60}")
    print(f"{'N':>3}  {'Originale':>12}  {'Nuova':>10}  {'Speedup':>8}  {'Match':>6}")
    print("-" * 50)

    for N in [2, 3, 4, 5]:
        dim = 2**N
        psi      = np.zeros(dim, dtype=complex)
        psi[0]   = psi[-1] = 1 / np.sqrt(2)
        rho_true = np.outer(psi, psi.conj())
        povm_dict = build_povm(N)

        P_dict = {}
        for outcome in iproduct(range(4), repeat=N):
            op = M1[outcome[0]]
            for i in range(1, N): op = np.kron(op, M1[outcome[i]])
            P_dict[outcome] = np.real(np.trace(op @ rho_true))
        P_vec    = pvec_from_pdict(P_dict, N)
        T, T_inv = get_overlap_matrix(N)

        # originale (skip N=5: troppo lento)
        if N <= 5:
            t0       = time.time()
            rho_orig = reconstruct_density_matrix_original(P_vec, M1, N, T_inv)
            t_orig   = (time.time() - t0) * 1000
        else:
            t_orig   = None
            rho_orig = None

        # nuova
        t0      = time.time()
        rho_new = reconstruct_rho(P_vec, N, T_inv, povm_dict)
        t_new   = (time.time() - t0) * 1000

        match   = np.allclose(rho_orig, rho_new, atol=1e-10) if rho_orig is not None else "N/A"
        speedup = f"{t_orig/t_new:.1f}x" if t_orig is not None else ">>10x"
        orig_s  = f"{t_orig:.1f} ms" if t_orig is not None else "troppo lento"
        print(f"{N:>3}  {orig_s:>12}  {t_new:>8.1f} ms  {speedup:>8}  {str(match):>6}")