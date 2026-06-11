#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: temporary_vae.py
Author: Riccardo Ruggeri
Description:
    Helper temporaneo per la generazione dei campioni dal VAE a N alti (6/7/8 qubit).

    Problema: a N alti N_GEN = 4^N * 500 (milioni di campioni). Generarli in un
    unico tensore fa esplodere la RAM, perche' il decoder produce un'attivazione
    di forma (N_GEN, hidden):
        N=7 -> ~46 GiB,  N=8 -> ~312 GiB  ==> crash.

    Soluzione (stesso metodo, stesso stimatore): generare e contare a BATCH,
    senza mai tenere tutti i campioni in memoria. La RAM di picco dipende solo
    da batch_size, non da N_GEN.

    Uso nel notebook (sostituisce vae_generate + samples_to_empirical_dist):
        from temporary_vae import generate_empirical_dist
        P_vae = generate_empirical_dist(model, n_qubits, N_GEN)
"""

# ----------------------------------------------------------------------------------------------------------------------------
# Imports

import numpy as np
from collections import Counter
from itertools import product as iproduct

# ----------------------------------------------------------------------------------------------------------------------------
# Functions


def generate_empirical_dist(model, n_qubits, n_samples, seed=0, batch_size=50_000):
    """
    Genera n_samples dal VAE e ne stima la distribuzione empirica P_vae sui
    4^N outcomes del POVM, lavorando a BATCH per non far esplodere la RAM.

    Equivalente (stesso stimatore Monte Carlo, stessa distribuzione) a:
        gen   = fm.vae_generate(model, n_samples)
        P_vae = samples_to_empirical_dist(gen, n_qubits)
    ma senza materializzare ne' i milioni di campioni ne' l'attivazione
    (n_samples, hidden) del decoder.

    Nota: usa un proprio RNG numpy per il latente z, quindi l'estrazione casuale
    NON e' identica bit-a-bit a model.sample(seed=...) in un colpo solo; con
    milioni di campioni le fidelity coincidono comunque a ~3-4 decimali.

    Args:
        model      : istanza VAE (espone .latent_dim e .decode())
        n_qubits   : numero di qubit N
        n_samples  : numero totale di campioni da generare (es. 4**N * 500)
        seed       : seme per la riproducibilita'
        batch_size : campioni per blocco (regola la RAM di picco)

    Returns:
        dict {(a1,...,aN): prob} su tutti i 4^N outcomes (zeri inclusi),
        stesso formato di samples_to_empirical_dist().
    """
    rng = np.random.default_rng(seed)
    counts = Counter()
    remaining = int(n_samples)

    while remaining > 0:
        b = min(batch_size, remaining)

        # z dal latente standard (numpy: riproducibile e a bassa memoria)
        z = rng.standard_normal((b, model.latent_dim)).astype('float32')
        # il decoder restituisce probabilita' softmax per gruppo: (b, 4N) -> (b, N, 4)
        probs = np.asarray(model.decode(z)).reshape(b, n_qubits, 4)

        # campiona UNA classe per gruppo secondo le prob. del decoder
        # (argmax darebbe solo la moda: perderebbe la distribuzione)
        cum = np.cumsum(probs, axis=-1)
        u = rng.random((b, n_qubits, 1))
        draws = (u < cum).argmax(axis=-1)              # (b, N): outcome per qubit

        counts.update(map(tuple, draws.tolist()))
        remaining -= b

    total = sum(counts.values())
    # stesso formato di samples_to_empirical_dist: tutti i 4^N outcomes, zeri inclusi
    return {o: counts.get(o, 0) / total
            for o in iproduct(range(4), repeat=n_qubits)}


# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("temporary_vae.py: importa generate_empirical_dist(model, n_qubits, n_samples).")
