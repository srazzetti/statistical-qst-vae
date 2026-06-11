#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: deprecated.py
Author: 
Date: 
Description: 
    deprecated classes / functions
"""

# ----------------------------------------------------------------------------------------------------------------------------
# Imports
import keras
from povm_sampling import *
from utils import *

# ----------------------------------------------------------------------------------------------------------------------------
# Costants


# ----------------------------------------------------------------------------------------------------------------------------
# Functions and classes

class FidelityMonitor(keras.callbacks.Callback):
    """
    Keras callback to monitor the classical fidelity at the end of each epoch
        Input args:
            P_exact: the exact probability distribution over the POVM outcomes
            n_qubits: the number of qubits in the system
            n_gen: the number of samples to generate from the model for estimating the fidelity (n_e)
            seed: the random seed for reproducibility
    """
    def __init__(self, P_exact, n_qubits, n_gen=20000, seed=0):
        super().__init__()
        self.P_exact = P_exact
        self.n_qubits = n_qubits
        self.n_gen = n_gen
        self.seed = seed

    def vae_generate(self, model, n_samples):
        '''Generates samples from vae decoder. It samples the softmax dist of each qubit'''
        rng = np.random.default_rng(self.seed)
        # sample the probability of each possible outcome
        probs = model.sample(n_samples).numpy().reshape(n_samples, self.n_qubits, 4)  # (n, N, 4)

        # extract a single class for each group by probs distributions
        # (argmax does not consider the softmax dist for onehot digits)
        cum = np.cumsum(probs, axis=-1)                 # (n, N, 4)
        u = rng.random((n_samples, self.n_qubits, 1))   # (n, N, 1)
        # draws is boolean (n, N, 4) --> argmax(-1) return position of the 1st True 
        draws = (u < cum).argmax(axis=-1)               # (n, N) povm outcome
        return [tuple(int(a) for a in row) for row in draws]
    
    # def classical_fidelity(self, P, Q):
    #     return float(sum(np.sqrt(P.get(x, 0.0) * Q.get(x, 0.0)) for x in set(P) | set(Q)))
    
    def total_variation(self, P, Q):
        '''TV = 1/2 sum|P-Q|: 0=same'''
        return float(0.5 * sum(abs(P.get(x, 0.0) - Q.get(x, 0.0)) for x in set(P) | set(Q)))
    
    def on_epoch_end(self, epoch, logs=None):
        '''Compute and log classical fidelity. Automatically called at the end of each epoch by Keras'''
        # create samples
        gen = self.vae_generate(self.model, self.n_gen)
        # compute empirical dist
        P_vae = samples_to_empirical_dist(gen, self.n_qubits)
        # compute classical fidelity
        Fc = classical_fidelity(self.P_exact, P_vae)
        # ensures logs exist, then log val_fidelity 
        logs = logs if logs is not None else {}
        logs['val_fidelity'] = Fc



# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__": 
    print('deprecate.py has no main.')
