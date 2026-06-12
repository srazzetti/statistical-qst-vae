#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: stateprep.py
Author: Riccardo Ruggeri, Simone Razzetti
Description: 
    Generation and validation possible states. Usefull functions to apply noise channels to 
    pure states (matrices).
    Functions:
        create_ghz_state()
        create_w_state()
        apply_phase_damping()
        apply_amplitude_damping()
        apply_depolarizing_noise()
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import DensityMatrix
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import phase_damping_error, amplitude_damping_error, depolarizing_error

# ----------------------------------------------------------------------------------------------------------------------------
# Functions

# figs = Path("figs/states")
# figs.mkdir(parents=True, exist_ok=True)

def create_ghz_state(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    return qc

def create_w_state(n_qubits):
    qc = QuantumCircuit(n_qubits)
    w_vector = np.zeros(2**n_qubits)
    for k in range(n_qubits):
        w_vector[2**k] = 1 / np.sqrt(n_qubits)
    qc.initialize(w_vector, range(n_qubits))
    return qc

# Noise channels --> necessary to study mixed states
def apply_phase_damping(rho: DensityMatrix, gamma: float) -> DensityMatrix:
    """
    Apply a Phase Damping channel (pure dephasing) to all qubits in the density matrix.
    This noise channel models the loss of quantum coherence (dephasing) without
    any energy exchange with the environment, damping the off-diagonal elements.
    Args:
        rho (DensityMatrix): The input quantum state density matrix.
        gamma (float): The noise parameter (0.0 <= gamma <= 1.0).
    Returns:
        DensityMatrix: The noisy mixed state after dephasing.
    """
    if gamma == 0.0:
        return rho
    # In Qiskit Aer si usa phase_damping_error
    noise_channel = phase_damping_error(gamma)
    for q in range(rho.num_qubits):
        rho = rho.evolve(noise_channel, qargs=[q])
    return rho

def apply_amplitude_damping(rho: DensityMatrix, gamma: float) -> DensityMatrix:
    """
    Apply an Amplitude Damping channel (energy relaxation) to all qubits in the density matrix.
    This noise channel models energy dissipation, causing the excited state 
    to decay back to the ground state with a probability gamma.
    Args:
        rho (DensityMatrix): The input quantum state density matrix.
        gamma (float): The relaxation probability (0.0 <= gamma <= 1.0).
    Returns:
        DensityMatrix: The noisy mixed state after thermal relaxation.
    """
    if gamma == 0.0:
        return rho
    # In Qiskit Aer si usa amplitude_damping_error
    noise_channel = amplitude_damping_error(gamma)
    for q in range(rho.num_qubits):
        rho = rho.evolve(noise_channel, qargs=[q])
    return rho

def apply_depolarizing_noise(rho: DensityMatrix, param: float) -> DensityMatrix:
    """
    Apply a Depolarizing noise channel to all qubits in the density matrix.
    This channel replaces the single-qubit state with the maximally mixed state 
    I/2 with probability param, acting as an isotropic error model.
    Args:
        rho (DensityMatrix): The input quantum state density matrix.
        param (float): The depolarizing error parameter (0.0 <= param <= 4/3).
    Returns:
        DensityMatrix: The noisy mixed state after depolarization.
    """
    if param == 0.0:
        return rho
    # In Qiskit Aer si usa depolarizing_error
    noise_channel = depolarizing_error(param, num_qubits=1)
    for q in range(rho.num_qubits):
        rho = rho.evolve(noise_channel, qargs=[q])
    return rho

# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__": 
    # debug-test
    n_qubits = 3
    shots = 10000
    sampler = StatevectorSampler()

    # GHZ state
    print("\n--- Processing GHZ State ---")
    qc_ghz = create_ghz_state(n_qubits)
    
    fig_ghz = qc_ghz.draw(output='mpl')
    plt.show()

    # verify state amplitude    
    qc_ghz_meas = qc_ghz.copy()
    qc_ghz_meas.measure_all()
    
    result_ghz = sampler.run([qc_ghz_meas], shots=shots).result()[0]
    counts_ghz = result_ghz.data.meas.get_counts()  
    print("GHZ State Counts:", counts_ghz)
    
    plot_histogram(counts_ghz)
    plt.show()

    # W state
    print("\n--- Processing W State ---")
    qc_w = create_w_state(n_qubits)
    
    fig_w = qc_w.draw(output='mpl')
    plt.show()
    
    # verify state amplitude    
    qc_w_meas = qc_w.copy()
    qc_w_meas.measure_all()
    
    result_w = sampler.run([qc_w_meas], shots=shots).result()[0]
    counts_w = result_w.data.meas.get_counts()      
    print("W State Counts:", counts_w)
    
    plot_histogram(counts_w)
    plt.show()

    # error channel
    from qiskit.visualization import plot_state_city
    
    print("\n--- Generating Density Matrix City Plots for GHZ State ---")
    
    # pure GHZ state den matrix
    rho_pure = DensityMatrix.from_instruction(qc_ghz)
    
    # noisy matrices
    gamma_val = 0.3
    rho_phase_noisy = apply_phase_damping(rho_pure, gamma=gamma_val)
    rho_amp_noisy   = apply_amplitude_damping(rho_pure, gamma=gamma_val)
    rho_dep_noisy   = apply_depolarizing_noise(rho_pure, param=gamma_val)
    
    small_size = (7, 5)
    # pure state
    fig1 = plot_state_city(rho_pure, title="GHZ - Ideal Pure State (Real Part)", figsize=small_size)    
    # phase damping 
    fig2 = plot_state_city(rho_phase_noisy, title=f"GHZ - Phase Damping (gamma={gamma_val})", figsize=small_size)    
    # amplitude damping
    fig3 = plot_state_city(rho_amp_noisy, title=f"GHZ - Amplitude Damping (gamma={gamma_val})", figsize=small_size)    
    # depolarizing noise
    fig4 = plot_state_city(rho_dep_noisy, title=f"GHZ - Depolarizing Noise (param={gamma_val})", figsize=small_size)
    plt.show()