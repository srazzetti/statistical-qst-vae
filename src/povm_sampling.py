#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: povm_sampling.py
Author: Simone Razzetti
Date: 6-6-2026
Description: 
    Functions for the sampling of tetraedric POVM possible outcomes, given a state rho and the exact 
    distribution P(a) = Tr[M(a) @ rho].
    Usefull functions:
        povm_probability()
        povm_probability_efficient()
        sample_povm()
        samples_to_onehot()
        onehot_to samples()
        samples_to_empirical_dist()
"""

# ----------------------------------------------------------------------------------------------------------------------------
# Imports

import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iproduct
from collections import Counter
# personal helper
from utils import *

# ----------------------------------------------------------------------------------------------------------------------------
# Costants


# ----------------------------------------------------------------------------------------------------------------------------
# Functions and classes

def povm_probability(rho, N, povm_dict=None):
    """
    Compute the exact probability of each outcome a = (a1, ..., aN) of 4^N possible: 
    P(a) = Tr[M(a1) x ... x M(aN) @ rho]

    Args:
        rho : density matrix to "measure" (2^N x 2^N) - rho.data if rho is DensityMatrix class
        N   : number of qubits
        povm_dict : {outcome_tuple : POVM_operator (kron prod of single qubit SIC-POVM) }

    Returns:
        dict {(a1,...,aN): probability}
    """
    # input sanity check 
    expected_dim = 2**N
    assert rho.shape == (expected_dim, expected_dim), f"rho dimensions {rho.shape} does not fit N = {N} qubits."

    P = {}

    if povm_dict is None:
        povm_dict = build_povm(N)
    
    for outcome, op in povm_dict.items():
        # Born's rule: P(a) = Tr[M(a) @ rho]
        P[outcome] = float(np.real(np.trace(op @ rho)))        

    # sanity check
    # clip tiny negative probabilities to 0 due to floating-point imprecision    
    P = {k: max(v, 0.0) for k, v in P.items()}
    # renormalize the distribution to ensure the total sum equals exactly 1.0
    total = sum(P.values())
    P = {k: v / total for k, v in P.items()}

    return P

def povm_probability_efficient(rho, N):
    """
    Compute the exact probability of each outcome a = (a1, ..., aN) of 4^N possible:
        P(a) = Tr[M(a1) x ... x M(aN) @ rho]

    This function implements the exact same mathematics as looping over build_povm(N), but utilizes 
    tensor contraction so that the 4^N dense operators (2^N x 2^N) are never explicitly constructed. 
    This reduces the peak memory usage, rendering simulations up to N = 8 qubits tractable. 
    Results are identical to the naive implementation up to floating-point precision.

    Args:
        rho : density matrix to "measure" (2^N x 2^N) - rho.data if rho is DensityMatrix
        N   : number of qubits

    Returns:
        dict {(a1,...,aN): probability}
    """
    # input sanity check
    rho = np.asarray(rho)
    expected_dim = 2**N
    assert rho.shape == (expected_dim, expected_dim), \
        f"rho dimensions {rho.shape} does not fit N = {N} qubits."

    # merges M1 op into a single block (3-axis tensor) with shape (4, 2, 2)
    # Axis 0: The 4 measurement outcomes (a_k). Axes 1, 2: Physical row (i) and column (j) indices
    M = np.stack(M1)

    # rewrite rho to explicitly expose the individual qubit spaces
    # rho.T implements the index swap required by the trace: Tr(M @ rho) = sum(M_ij * rho_ji).
    # reshape([2]*N + [2]*N) unravels the 2 global axes into 2N microscopic physical indices:
    # rho --> R[i_0, ..., i_{N-1}, j_0, ..., j_{N-1}] where is, js are single qubit rows, cols
    R = np.ascontiguousarray(rho.T).reshape([2] * N + [2] * N)      


    # Contraction one qubit at a time. Before step k, the axes of R are:
    #   [i_k, ..., i_{N-1}, j_k, ..., j_{N-1}, a_0, ..., a_{k-1}]
    # Contract i_k (axis 0) and j_k (axis N-k) with the (i, j) axes of M (1, 2),
    # generating the outcome axis a_k (dim 4), which is then moved to the end.
    for k in range(N):
        j_pos = N - k      
        # R_next[a_k, i_rest, j_rest, a_past] = sum_{i_k, j_k} M[a_k, i_k, j_k] * R[i_k, i_rest, j_k, j_rest, a_past]        
        R = np.tensordot(M, R, axes=([1, 2], [0, j_pos]))
        # move new axis a_k to the end: [i_k, ..., j_k, ...] -> [..., a_0, ..., a_k]
        R = np.moveaxis(R, 0, -1)              

    # R has shape (4,4,..., 4)=(4,)*N  with axes (a_0, ..., a_{N-1}) each of dim 4
    # P must be real
    P_tensor = np.real(R)

    # flatten the tensor, same order as iproduct gives
    outcomes = iproduct(range(4), repeat=N)
    P = {a: float(p) for a, p in zip(outcomes, P_tensor.ravel())}

    # sanity check
    P = {k: max(v, 0.0) for k, v in P.items()}
    total = sum(P.values())
    P = {k: v / total for k, v in P.items()}

    return P

def sample_povm(P_exact, n_samples=1, seed=42):
    """
    Simulates n_samples measurements POVM sampling from the exact distribution.
    Each sample is a tuple (a1, a2, ..., aN) with ai in {0,1,2,3}.

    This corresponds to measure POVM n_samples times on a real hardware.
    (The "true" simulation of the measurement needs ancillas qubit -2- for each data qubit;
    POVM is not a projector or a basis, need to "encode" the result in ancillas. See povm_naimark.py)

    Returns:
        list of tuple, lenght n_samples
    """
    # random number generator
    rng = np.random.default_rng(seed)
    outcomes = list(P_exact.keys())
    probs    = np.array(list(P_exact.values()))
    indices  = rng.choice(len(outcomes), size=n_samples, p=probs)
    return [outcomes[i] for i in indices]

def samples_to_onehot(samples, N):
    """
    Transform list of tuple in one-hot matrix (n_samples, 4N).
    Each qubit outcome ai --> one-hot label dim 4 in [4i : 4i+4]
    Ex: N=3, outcome=(2,0,1) -> [0,0,1,0, 1,0,0,0, 0,1,0,0]
    Args :
        samples: list of tuple (a1, a2, ..., aN) with ai in {0,1,2,3}
        N: number of qubits
    
    Returns: matrix (n_samples, 4N) of one-hot label 
    """
    n = len(samples)
    X = np.zeros((n, 4 * N), dtype=np.float32)
    for row, outcome in enumerate(samples):
        for qubit, a in enumerate(outcome):
            X[row, qubit * 4 + a] = 1.0
    return X

def onehot_to_samples(X, N, deterministic=False, seed=42):
    """
    Inverse transformation of samples_to_onehot(), supporting both deterministic (argmax) and 
    stochastic (probabilistic) sampling.
    Args:
        X : matrix (n_samples, 4N) of one-hot label (or softmax)
        N : number of qubits
        deterministic (bool): If True, uses argmax to pick the most probable outcome
                              If False (default), samples stochastically each softmax
        seed : (default=42)

    Returns:
        list: list of tuple (a1, a2, ..., aN) with ai in {0,1,2,3}
    """
    X = np.asarray(X)
    n_samples = X.shape[0]

    probs = X.reshape(n_samples, N, 4)  # (n_samples, 4N) -> (n_samples, N, 4)

    if deterministic:
        draws = probs.argmax(axis=-1) # axis=-1 : 4 outcomes
    else:
        rng = np.random.default_rng(seed)
        
        # extract a single class for each group by probs distributions
        cum = np.cumsum(probs, axis=-1)     # (n_samples, N, 4)
        u = rng.random((n_samples, N, 1))   # (n_samples, N, 1)
        # draws is boolean (n_samples, N, 4) --> argmax(-1) return position of the 1st True 
        draws = (u < cum).argmax(axis=-1)   # (n_samples, N) povm outcome

    # (n_samples, N) -> list of tuple 
    return list(map(tuple, draws.tolist()))

def samples_to_empirical_dist(samples, N):
    """
    Estimate the empirical probability distribution P_empirical(a) from the frequencies of collected 
    experimental samples. Outcomes that never occurred are explicitly assigned a probability of 0.0.
    Args :
        samples: list of tuple (a1, a2, ..., aN) with ai in {0,1,2,3}
        N: number of qubits
    
    Returns: dict {(a1,...,aN): empirical probability}
    """
    counts = Counter(samples)
    total = len(samples)
    P_empirical = {
        outcome: counts.get(outcome, 0) / total
        for outcome in iproduct(range(4), repeat=N)
    }
    return P_empirical


# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # debug - test : useless functions, only for testing
    from qiskit import QuantumCircuit
    from statesprep import create_ghz_state
    from qiskit.quantum_info import DensityMatrix
    from qiskit.visualization import plot_state_city, plot_state_hinton, plot_state_qsphere

    def debug_povm (plot=False, data_processing=False, data_dist=False) :
        qc = create_ghz_state(3)
        rho = DensityMatrix(qc)

        if plot: 
            print(rho.data)
            # possible visualizations of the density matrix
            # 3d heatmap, real and imag parts
            fig_city = plot_state_city(rho)
            fig_city.suptitle("City Plot")  

            # 2d heatmap, real and imag parts
            fig_hinton = plot_state_hinton(rho)
            fig_hinton.suptitle("Hinton Diagram")                

            # Plots the multi-qubit state on a sphere where each node is a basis state.
            # Node size is proportional to probability; node color represents phase. Link represents the entanglement.
            fig_qsphere = plot_state_qsphere(rho)
            fig_qsphere.suptitle("Q-Sphere dello Stato Globale")  

            print("Close all the figures to close the script.")
            plt.show()
        
        prob = povm_probability(rho.data, 3)

        # test povm_proba_eff()
        prob_eff = povm_probability_efficient(rho.data, 3)
        delta = [prob[esito] - prob_eff[esito] for esito in prob]
        # print(delta)
        print('max diff in exact probabilty generation methods: ', np.max(delta))

        # print(prob)
        sample = sample_povm(prob, seed=None)
        print(sample)

        if data_processing:
            one_hot = samples_to_onehot(sample, 3)
            print(one_hot)
            back_sample = onehot_to_samples(one_hot, 3)
            print(back_sample)

        if data_dist:
            samples = sample_povm(prob, 1000)
            p_emp = samples_to_empirical_dist(samples, 3)
            for k in prob.keys():
                print(prob[k], p_emp[k])
        return

    # debug main:
    debug_povm ()

