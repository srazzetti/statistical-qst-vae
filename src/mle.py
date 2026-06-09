#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: mle.py
Author: Simone Razzetti
Date: 8-6-2026
Description: 
    Provide all necessary functions to perform quantum state tomography by MLE.
    Usefull functions:
        params_to_T()
        params_to_rho()
        make_nll()
        mle_qst()
"""

# ----------------------------------------------------------------------------------------------------------------------------
# Imports

import numpy as np
from iminuit import Minuit
from collections import Counter
from itertools import product as iproduct
# personal helper
from utils import *

# ----------------------------------------------------------------------------------------------------------------------------
# Costants

# ----------------------------------------------------------------------------------------------------------------------------
# Functions

def count_params(N):
    """
    Number of real parameters to describe the Cholesky matrix T (2^N x 2^N matrix 
    with off-diagonal complex terms in the lower triangle).

    Tot = 2^N [--> real parameters on the diag] + 2* 2^N*(2^N-1)/2 [--> real parameters off-diag (Re and Im parts)]
    """
    dim = 2**N
    return dim + dim * (dim - 1)


def params_to_T(params, N):
    """
    Transform the parameters array into the complex Cholesky matrix T.
    params contains real values scan row by row in T matrix under this convention:
        diag     --> real (params[idx])
        off-diag --> complex (params[idx] + i*params[idx+1])
    """
    dim = 2**N
    T = np.zeros((dim, dim), dtype=complex)
    idx = 0
    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                T[i, j] = params[idx]
                idx += 1
            else:
                T[i, j] = params[idx] + 1j * params[idx + 1]
                idx += 2
    return T


def params_to_rho(params, N): 
    """
    Transform real params of Cholesky matrix T into valid density matrix rho. (windows: † = ALT + 0134)
    rho = T @ T† / Tr(T @ T†)
    """
    T   = params_to_T(params, N)
    # † means complex conjugate (.conj()) and transposed (.T)
    rho = T @ T.conj().T
    rho /= np.trace(rho)
    return rho


def rho_to_params(rho, N):
    """
    Creates a params array starting from a density matrix rho.
    Usefull to initialize optimizer starting parameter.
    Se rho non è strettamente positiva, aggiunge un piccolo shift.
    """
    dim = 2**N
    # sanity check: ensuring rho is positive
    eigvals = np.linalg.eigvalsh(rho)
    if np.min(eigvals) <= 0:
        # add small quantities to make the matrix positive and prevent errors
        rho = rho + (abs(np.min(eigvals)) + 1e-8) * np.eye(dim)
        rho /= np.trace(rho)

    # Compute the Cholesky matrix T (lower triangle)
    T = np.linalg.cholesky(rho)

    params = []
    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                # diag
                params.append(np.real(T[i, j]))
            else:
                # off-diag: append both re and im part as real parameters
                params.append(np.real(T[i, j]))
                params.append(np.imag(T[i, j]))
    return np.array(params)


# build nll --> Negative Log-Likelihood  
def make_nll(samples, N, povm_dict=None):
    """
    Build the negative log-likelihood to minimize: L(rho) = -Sum_a Na*log P(a)  
    where:
        - a = (a1, a2, ..., aN) tuple with ai in {0,1,2,3} --> possible outcome
        - P(a) = Tr[M(a) @ rho]
        - Na is the frequency of outcome a in samples
    
    Args:
        - samples: list of tuple (a1, a2, ..., aN) with ai in {0,1,2,3}
        - N: number of qubits
        - povm_dict : {outcome_tuple : POVM_operator (kron prod of single qubit SIC-POVM) }

    Returns:
        nll function (Minuit callable)
    """
    if povm_dict is None:
        povm_dict = build_povm(N)

    # Counter returns dict with {outcome: counts}
    counts = Counter(samples)

    # * to pass params as (t1, t2, ..., tM) tuple
    def nll(*params):
        rho   = params_to_rho(np.array(params), N)
        total = 0.0
        for outcome, count in counts.items():
            p = np.real(np.trace(povm_dict[outcome] @ rho))
            p = max(p, 1e-10)   # prevents log(0)
            total -= count * np.log(p)
        return total

    return nll


# perform quantum state tomography by MLE
def mle_qst(samples, N, init_rho=None, verbose=False):
    """
    Performs QST by MLE method.

    Args:
        samples  : list of tuple (a1, a2, ..., aN) with ai in {0,1,2,3}
        N        : number of qubits
        init_rho : rho_0 initial matrix (default: maximally mixed state)
        verbose  : display Minuit output (default: False)

    Returns:
        rho_mle  : estimated density matrix (2^N x 2^N)
        result   : Minuit optimization object
    """
    dim    = 2**N
    # n_par  = count_params(N)

    # params initialization 
    if init_rho is None:
        init_rho = np.eye(dim, dtype=complex) / dim   # max mixed state
    p0 = rho_to_params(init_rho, N)

    # building nll
    nll = make_nll(samples, N)

    # Minuit
    m = Minuit(nll, *p0)
    # iminuit built-in to use custom likelihood cost func --> for error estimation m.minos
    m.errordef = Minuit.LIKELIHOOD   # = 0.5    
    m.print_level = 1 if verbose else 0

    m.migrad()                  
    if not m.valid:
        # second try
        m.migrad()              


    rho_mle = params_to_rho(np.array(m.values), N)
    return rho_mle, m


# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__": 
    def main () :
        print("mle.py has no main")
        return

    main ()

