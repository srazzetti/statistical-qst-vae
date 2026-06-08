#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: utils.py
Author: Simone Razzetti, Riccardo Ruggeri
Date: 
Description: 
"""

# ----------------------------------------------------------------------------------------------------------------------------
# Imports

import numpy as np
from itertools import product as iproduct
from collections import Counter

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

# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__": 
    print('utils.py has no main')