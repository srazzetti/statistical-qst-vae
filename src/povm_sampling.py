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
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit.visualization import plot_state_city, plot_state_hinton, plot_state_qsphere
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

def make_povm_1qubit():
    """
    Returns a list of 4 matrices 2x2: M[alpha] = (I + s[alpha]·sigma) / 4
    Verifies: sum(M) == I  (completeness)
    """
    M = [(I + s[0]*sx + s[1]*sy + s[2]*sz) / 4 for s in S_VECS]
    # np.allclose verifies the assertion is true with small tolerance, instead of '=='
    assert np.allclose(sum(M), I), "POVM is not complete!"
    return M

# ----------------------------------------------------------------------------------------------------------------------------
# Functions and classes

def povm_probability(rho, N, M1=None):
    """
    Compute the exact probability of each outcome a = (a1, ..., aN) of 4^N possible: 
    P(a) = Tr[M(a1) x ... x M(aN) @ rho]

    Args:
        rho : density matrix to "measure" (2^N x 2^N) - rho.data if rho is DensityMatrix class
        N   : number of qubits
        M1  : list of the 4 SIC POVM single qubit operators (default: tetraedric SIC POVM)

    Returns:
        dict {(a1,...,aN): probability}
    """
    # input sanity check 
    expected_dim = 2**N
    assert rho.shape == (expected_dim, expected_dim), f"rho dimensions {rho.shape} does not fit N = {N} qubits."
        
    if M1 is None:
        M1 = make_povm_1qubit()

    P = {}
    # iproduct is the inter-product of {0,1,2,3}x{0,1,3,4}..{0,1,2,3} N times 
    # --> all possible outcomes (they are tuple)
    for outcome in iproduct(range(4), repeat=N):
        # tensor prod M(a) = M(a1) x M(a2) x ... x M(aN)
        op = M1[outcome[0]]
        for i in range(1, N):
            op = np.kron(op, M1[outcome[i]])
        # Born's rule: P(a) = Tr[M(a) @ rho]
        P[outcome] = float(np.real(np.trace(op @ rho)))

    # sanity check
    # clip tiny negative probabilities to 0 due to floating-point imprecision    
    P = {k: max(v, 0.0) for k, v in P.items()}
    # renormalize the distribution to ensure the total sum equals exactly 1.0
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

def onehot_to_samples(X, N):
    """
    Inverse transformation of samples_to_onehot()
    Args :
        X: matrix (n_samples, 4N) of one-hot label (or softmax)
        N: number of qubits
    
    Returns: list of tuple (a1, a2, ..., aN) with ai in {0,1,2,3}
    """
    samples = []
    for row in X:
        outcome = tuple(
            np.argmax(row[qubit*4 : qubit*4 + 4])
            for qubit in range(N)
        )
        samples.append(outcome)
    return samples

def samples_to_empirical_dist(samples, N):
    """
    Estimate the empirical probability distribution P_empirical(a) from the frequencies of collected 
    experimental samples. Outcomes that never occurred are explicitly assigned a probability of 0.0.
    """
    counts = Counter(samples)
    total = len(samples)
    P_empirical = {
        outcome: counts.get(outcome, 0) / total
        for outcome in iproduct(range(4), repeat=N)
    }
    return P_empirical

# ----------------------------------------------------------------------------------------------------------------------------
# debug - test : useless functions, only for testing

def create_ghz_state(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    # qc.measure_all()
    return qc

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

# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__": 
    debug_povm ()

