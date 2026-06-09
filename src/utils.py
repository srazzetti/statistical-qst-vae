#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: utils.py
Author: Simone Razzetti, Riccardo Ruggeri
Date: 
Description: 
    Provide all recurrent functions, constants and usefull helper.
    Usefull functions:
        build_povm()
        classical_fidelity()
        quantum_fidelity()
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


# Fidelity
def classical_fidelity(P1, P2):
    """
    Classical fidelity (Bhattacharyya) among two distributions:  Fc = sum_x sqrt(P1(x) * P2(x))

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

# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__": 
    print('utils.py has no main')