#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: vae.py
Author: Riccardo Ruggeri
Date: 8/6/2026
Description: 
    Provides classes for the implementation of a VAE, base on Keras methods.
    Class structure is taken by Statistics and Data Analysis lecture on VAE (Prof. De Guio, Unimib).
    Decoder, Encoder architecture are based on Chen et al. (2021), as well as the default parameters.
    Usefull functions and classes:
        VAE (class)
            build, encode, reparametrize, decode, call, train_step, test_step (keras Model)
            sample()
        KLWarmup (callback)
        FidelityMonitor (callback)
            vae_generate()
            total_variation()
"""

# ----------------------------------------------------------------------------------------------------------------------------
# Imports
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
from keras import ops
import sys
sys.path.append('../src')  # (da capire se serve)
from collections import Counter
from itertools import product as iproduct
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ----------------------------------------------------------------------------------------------------------------------------
# Costants

# Paper table showing structure for the vae depending on the number of qubits
# {n_qubits: neurons_per_hidden_layer, latent_dim, batch_size, N_sample_to_estimate_fidelity}
PAPER_TABLE = {
    3: dict(hidden=96,   latent=16,  batch=100,  n_e=20000),
    4: dict(hidden=128,  latent=32,  batch=200,  n_e=100000),
    5: dict(hidden=224,  latent=64,  batch=500,  n_e=(4**5)*500),   # n_e = 4^N * 500 (ref. paper)
    6: dict(hidden=1280, latent=256, batch=600,  n_e=(4**6)*500),
    7: dict(hidden=1504, latent=512, batch=800,  n_e=(4**7)*500),
    8: dict(hidden=2560, latent=512, batch=1000, n_e=(4**8)*500),
}

# ----------------------------------------------------------------------------------------------------------------------------
# Functions and classes

class VAE(keras.Model):
    """
    VAE for Quantum State Tomography (Chen et al. 2021).
    Structure for encoder, decoder is the same as proposed by Chen et al., default parameters too.
    It consists of 4 layer encoder/decoder, hidden LeakyReLU, output sigmoid.
    It uses a loss composed of reconstruction loss (categorical_cross_entropy) and KL divergence loss.

    Input args:
        - n_qubits
        - latent_dim (default: PAPER_TABLE)
        - hidden: number of neurons for the hiddel layers (default: PAPER_TABLE)
    """
    def __init__(self, n_qubits, latent_dim=None, hidden=None, **kwargs):
        super(VAE, self).__init__(**kwargs)

        # get default config dict for n_qubits
        cfg = PAPER_TABLE.get(n_qubits, {})

        self.n_qubits = n_qubits
        self.input_dim = 4 * n_qubits
        self.latent_dim = latent_dim if latent_dim is not None else cfg.get('latent', 16)
        h = hidden if hidden is not None else cfg.get('hidden', 96)

        # KL term weight: no-trainable parameter, updated during traing by KLwarmup callback
        self.kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        # --- Encoder: input(4N) -> hidden -> hidden -> [mu, log_var](latent) 
        self.enc_dense1 = Dense(h, activation='leaky_relu', name='enc_1')
        self.enc_dense2 = Dense(h, activation='leaky_relu', name='enc_2')
        self.z_mean_dense = Dense(self.latent_dim, name='z_mean')
        self.z_log_var_dense = Dense(self.latent_dim, name='z_log_var')

        # --- Decoder: input(latent) -> hidden -> hidden -> output(4N)
        # output are 4N-dim logits representing scores for 4 POVM classes across N qubits (one-hot labels)
        self.dec_dense1 = Dense(h, activation='leaky_relu', name='dec_1')
        self.dec_dense2 = Dense(h, activation='leaky_relu', name='dec_2')
        self.dec_output = Dense(self.input_dim, name='dec_logits')   # linear

    # Explicitly implement the build method for the VAE Model subclass.
    # This ensures all internal layers are built when model.build() is called.
    def build(self, input_shape):
        # create custom dummy input for encoder/decoder to trigger building of their layers

        # input_shape encoder is (batch_size, original_dim) --> we use (1, original_dim)
        dummy_encoder_input = tf.zeros(shape=(1, self.input_dim))
        self.encode(dummy_encoder_input)

        # input_shape decoder is (batch_size, latent_dim) --> we use (1, latent_dim)
        dummy_decoder_input = tf.zeros(shape=(1, self.latent_dim))
        self.decode(dummy_decoder_input)

        # call the base class's build method to finalize the model's build state
        super().build(input_shape)

    def encode(self, inputs):
        """Encodes the input into the mean and log-variance of the latent space"""
        x = self.enc_dense1(inputs)
        x = self.enc_dense2(x)
        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        """
        Performs the reparameterization trick to sample from the latent distribution, allowing
        backprop through the sampling process.
        z = mu + exp(0.5 * log_var) * epsilon, where epsilon is a standard normal sample
        """
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        """Decodes a latent space sample back into the original data space"""
        x = self.dec_dense1(z)
        x = self.dec_dense2(x)
        logits = self.dec_output(x)     # (batch, 4N)
        # softmax activation on each 4 logits group (povm single qubit outcome into onehot label)
        logits = ops.reshape(logits, (-1, self.n_qubits, 4)) # (batch, N, 4)
        probs = ops.softmax(logits, axis=-1)                 # softmax on each 4 classes
        return ops.reshape(probs, (-1, self.input_dim))      # (batch, 4N)

    def call(self, inputs):
        """
        Forward pass through the VAE. It returns the reconstruction, along with z_mean and z_log_var
        which are needed for the KL divergence loss calculation.
        """
        # encoder
        z_mean, z_log_var = self.encode(inputs)
        # sampling latent space
        z = self.reparameterize(z_mean, z_log_var)
        # decoder
        reconstruction = self.decode(z)
        return reconstruction, z_mean, z_log_var

    def _compute_losses(self, x):
        '''
        (private function) Computes the total VAE loss, which is composed of reconstruction loss 
        and KL divergence loss. Reconstruction loss is calculated between the VAE's output and input.
        '''
        # call
        reconstruction, z_mean, z_log_var = self(x)

        # Reconstruction: categorical crossentropy on each group (N qubits, 4 classes)
        x_grp = ops.reshape(x, (-1, self.n_qubits, 4))                  # (batch, N, 4)
        rec_grp = ops.reshape(reconstruction, (-1, self.n_qubits, 4))   # (batch, N, 4)
        cce = keras.losses.categorical_crossentropy(x_grp, rec_grp)     # (batch, N)
        reconstruction_loss = ops.sum(cce, axis=1)                      # (batch,) sum over qubits

        # KL divergence: measures the difference between the learned latent distribution and N(0,1) prior
        # regularizes the latent space, encouraging it to be continuous and disentangled
        kl_loss = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)

        # kl_weight is updated at each epoch by KLWarmUp callback
        total_loss = tf.reduce_mean(reconstruction_loss + self.kl_weight * kl_loss)

        return total_loss, tf.reduce_mean(reconstruction_loss), tf.reduce_mean(kl_loss)

    def train_step(self, data):
        '''Custom training step for the VAE'''
        x = data[0] if isinstance(data, tuple) else data   # unsupervised, as vae
        # forward pass and losses
        with tf.GradientTape() as tape:
            total_loss, rec_loss, kl_loss = self._compute_losses(x)
        # backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {"loss": total_loss, 
                "reconstruction_loss": rec_loss,
                "kl_loss": kl_loss, 
                "kl_weight": self.kl_weight}

    def test_step(self, data):
        '''Custom testing (validation) step for the VAE, mirroring the train_step logic'''
        x = data[0] if isinstance(data, tuple) else data
        # no backprop
        total_loss, rec_loss, kl_loss = self._compute_losses(x)

        return {"loss": total_loss, 
                "reconstruction_loss": rec_loss, 
                "kl_loss": kl_loss}

    def sample(self, n_samples, seed=42):
        '''
        Extract n_samples latent space and decodes them.
        Args:
            n_samples: number of sample to create
            seed: (default: 42)
        Returns:
            samples: array (n_samples, 4N) (decoder output: prob dist of each qubit) 
        '''
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_samples, self.latent_dim)).astype('float32')
        samples = self.decode(z).numpy()
        # tf.random.set_seed(seed)
        # z = tf.random.normal(shape=(n_samples, self.latent_dim))
        # samples = self.decode(z)
        return samples

    def generate_empirical_dist(self, n_samples, batch_size=50_000, seed=42):
        """
        Function to calculate P_vae empirical, sampling the trained decoder.
        Generates n_sample from VAE and compute the distribution working with fixed batch, to
        prevent RAM errors.
        Args:
            n_samples  : sample to generate (ex. 4**N * 500)
            batch_size : (default = 50000)
            seed       : (default = 42)
        Returns:
            dict {(a1,...,aN): prob}
        """
        rng = np.random.default_rng(seed)
        counts = Counter()
        remaining = int(n_samples)

        while remaining > 0:
            b = min(batch_size, remaining)

            # np rng
            probs_flat = self.sample(n_samples=b, seed=rng)
            probs = probs_flat.reshape(b, self.n_qubits, 4)     # (b, 4N) -> (b, N, 4)
            # z = rng.standard_normal((b, self.latent_dim)).astype('float32')
            # probs = np.asarray(self.decode(z)).reshape(b, self.n_qubits, 4)     # (b, 4N) -> (b, N, 4)
    
            # extract a single class for each group by probs distributions
            # (argmax does not consider the softmax dist for onehot digits)  
            cum = np.cumsum(probs, axis=-1)             # (b, N, 4)
            u = rng.random((b, self.n_qubits, 1))       # (b, N, 1)
            # draws is boolean (b, N, 4) --> argmax(-1) return position of the 1st True 
            draws = (u < cum).argmax(axis=-1)           # (b, N) povm outcome

            # map outcomes into tuples and count them
            counts.update(map(tuple, draws.tolist()))
            remaining -= b

        total = sum(counts.values())

        return {o: counts.get(o, 0) / total for o in iproduct(range(4), repeat=self.n_qubits)}



class KLWarmup(keras.callbacks.Callback):
    """
    Keras callback model to implement a linear warmup of KL weight 0 to beta_max during the 
    first warmup_epochs.
    It faces the posterior collapse: vae is first trained as a quasi-autoencoder, then the
    KL divergence regularizes the latent space to N(0,1).
    Input args:
        beta_max: (default: 0.5)
        warmup_epochs: (default: 30)
    """
    def __init__(self, beta_max=0.5, warmup_epochs=30):
        super().__init__()
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        '''Update the KL weight. Automatically called at the beginnig of each epoch by Keras'''
        w = min(self.beta_max, self.beta_max * epoch / self.warmup_epochs)
        self.model.kl_weight.assign(w)


def plot_reconstruction_and_kl_divergence(history):
    """
    Plot the reconstruction loss and KL divergence over epochs from the training history.
    It extracts the relevant losses from the history object and plots them on the same graph for comparison.
    Args:
        history: Keras History object returned by model.fit(), containing the loss values for each epoch
    """
    reconstruction_losses = {key: history.history[key] for key in history.history.keys() if 'reconstruction' in key}
    kl_losses = {key: history.history[key] for key in history.history.keys() if 'kl' in key and 'weight' not in key}

    plt.figure(figsize=(8, 6))
    for key, values in reconstruction_losses.items():
        plt.plot(values, label=key)
    for key, values in kl_losses.items():
        plt.plot(values, label=key)
    plt.title('Reconstruction Loss and KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_total_loss(history):
    """
    Plot the total loss over epochs from the training history. It extracts the total loss values from the history object and plots them.
    Args:
        history: Keras History object returned by model.fit(), containing the loss values for each epoch
    """
    total_losses = {key: history.history[key] for key in history.history.keys() if 'loss' in key and 'reconstruction' not in key and 'kl' not in key}

    plt.figure(figsize=(8, 6))
    for key, values in total_losses.items():
        plt.plot(values, label=key)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_distribution_compare(self, p_true, p_gen, p_train, outcomes=None):
        """
        Plot a comparison of three distributions: the true distribution (p_true), the distribution generated by the VAE (p_gen), and the distribution from the training data (p_train).
        Args:
            p_true: list or array of true probabilities for each outcome (e.g., from GHZ state)
            p_gen: list or array of probabilities generated by the VAE
            p_train: list or array of probabilities from the training data (shot noise)
            outcomes: optional list of outcome labels (e.g., tuples of POVM outcomes) to use as x-axis ticks
        """
        p_true  = np.asarray(p_true,  dtype=float)
        p_gen   = np.asarray(p_gen,   dtype=float)
        p_train = np.asarray(p_train, dtype=float)
        n = len(p_true)
        x = np.arange(n)

        w_out = 0.95   # larghezza barra-contenitore (P esatta)
        w_in  = 0.25   # larghezza barre interne (train / VAE), due affiancate dentro w_out

        _, ax = plt.subplots(figsize=(10, 5))

        # contenitore: P esatta come barra larga, contorno netto + riempimento molto tenue
        ax.bar(x, p_true, width=w_out,
               facecolor='white', edgecolor='black', linewidth=0.8,
               label='P esatta (GHZ)', zorder=1)

        # due barre sottili affiancate dentro ogni contenitore
        ax.bar(x - w_in / 2, p_train, width=w_in, color='C0', zorder=2,
               label='P training (shot noise)')
        ax.bar(x + w_in / 2, p_gen,   width=w_in, color='C1', zorder=2,
               label='P VAE')

        ax.axhline(1 / n, ls='--', c='gray', lw=1.0, zorder=0, label='uniforme (floor)')

        # etichette = tuple degli outcome (es. "013"), ruotate per leggibilita'
        if outcomes is not None:
            ticks = [''.join(map(str, o)) if hasattr(o, '__iter__') else str(o) for o in outcomes]
            ax.set_xticks(x)
            ax.set_xticklabels(ticks, rotation=90, fontsize=6.5, family='monospace')
            ax.set_xlabel('outcome POVM  (a$_1$a$_2$a$_3$)', fontsize=11)
        else:
            ax.set_xlabel('indice outcome', fontsize=11)

        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylabel('probabilità', fontsize=11)
        ax.set_title('POVM 3 qubit: esatta (contenitore) vs training / VAE (barre interne)',
                     fontsize=12, pad=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(axis='y', ls=':', alpha=0.4)
        ax.legend(fontsize=10, framealpha=0.85, loc='upper right')

        plt.tight_layout()
        plt.show()

def plot_distribution_delta(p_true, p_gen, p_train, outcomes=None,
                            sort_by='true', annotate_metrics=True):
    """
    Compare the true distribution (p_true), the VAE-generated distribution (p_gen), and the training distribution (p_train) 
    by plotting their differences (delta) from the true distribution.
    Args:
        p_true: list or array of true probabilities for each outcome (e.g., from GHZ state)
        p_gen: list or array of probabilities generated by the VAE
        p_train: list or array of probabilities from the training data (shot noise)
        outcomes: optional list of outcome labels (e.g., tuples of POVM outcomes) to use as x-axis ticks
        sort_by: 'true' to sort by true probabilities, 'none' to keep original order (default: 'true')
        annotate_metrics: whether to calculate and display TVD metrics in the legend (default: True)
    """
    p_true  = np.asarray(p_true,  dtype=float)
    p_gen   = np.asarray(p_gen,   dtype=float)
    p_train = np.asarray(p_train, dtype=float)
    n = len(p_true)

    # ordinamento per leggibilita': outcome dominanti (P_true alta) a sinistra
    order = np.argsort(p_true)[::-1] if sort_by == 'true' else np.arange(n)
    p_true, p_gen, p_train = p_true[order], p_gen[order], p_train[order]

    # scostamenti dalla distribuzione esatta (= 0)
    d_train = p_train - p_true
    d_gen   = p_gen   - p_true
    x = np.arange(n)

    _, ax = plt.subplots(figsize=(11, 5))

    # linea di riferimento: distribuzione esatta -> Delta = 0
    ax.axhline(0.0, color='black', lw=1.2, zorder=3, label='P esatta (riferimento, Δ=0)')

    # stem (segmentini verticali) per guidare l'occhio dallo zero al punto
    ax.vlines(x, 0, d_train, color='C0', alpha=0.25, lw=0.8, zorder=1)
    ax.vlines(x, 0, d_gen,   color='C1', alpha=0.25, lw=0.8, zorder=1)

    # TVD (total variation distance) come metrica sintetica di distanza
    if annotate_metrics:
        tvd_train = 0.5 * np.sum(np.abs(d_train))
        tvd_gen   = 0.5 * np.sum(np.abs(d_gen))
        lab_train = f'Δ training (shot noise)   TVD={tvd_train:.3f}'
        lab_gen   = f'Δ VAE   TVD={tvd_gen:.3f}'
    else:
        lab_train, lab_gen = 'Δ training (shot noise)', 'Δ VAE'

    ax.scatter(x, d_train, s=28, color='C0', edgecolor='white', linewidth=0.4,
               zorder=4, label=lab_train)
    ax.scatter(x, d_gen, s=28, color='C1', edgecolor='white', linewidth=0.4,
               marker='D', zorder=4, label=lab_gen)

    # etichette = tuple degli outcome (es. "013"), ruotate per leggibilita'
    if outcomes is not None:
        outcomes = [outcomes[i] for i in order]
        ticks = [''.join(map(str, o)) if hasattr(o, '__iter__') else str(o) for o in outcomes]
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, rotation=90, fontsize=6.5, family='monospace')
        ax.set_xlabel('outcome POVM  (ordinati per P esatta decrescente)', fontsize=11)
    else:
        ax.set_xlabel('indice outcome  (ordinati per P esatta decrescente)', fontsize=11)

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylabel('Δ probabilità  (P − P esatta)', fontsize=11)
    ax.set_title('Scostamento da P esatta: training vs VAE', fontsize=12, pad=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', ls=':', alpha=0.4)
    ax.legend(fontsize=9, framealpha=0.9, loc='best')

    plt.tight_layout()
    plt.show()

def plot_fidelity_vs_samples(self, results):
    """
    Plot the classical fidelity (Fc) as a function of the number of samples used for training, for different numbers of qubits.
    Args:
        results: dict of the form {n_qubits: {'n_samples': [...], 'Fc': [...]}} containing the number of samples and corresponding fidelities for each qubit count. 
    """
    plt.figure(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(results)))

    for color, (q, r) in zip(colors, sorted(results.items())):
        ns = r["n_samples"]
        plt.plot(ns, r["Fc"], '-o', color=color, label=f'{q} qubit (VAE)')

    plt.xscale('log')
    plt.xlabel('N_SAMPLES (campioni di training)')
    plt.ylabel('Fidelity classica  $F_c$')
    plt.title('Fidelity classica vs N_SAMPLES from 3 to 8 qubit')
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__": 
    print('vae.py has no main')


'''
# === Run A — replica esatta del paper (N=3) ===
n_qubits = 3
vae = VAE(n_qubits=n_qubits)              # latente 16, hidden 96 dal PAPER_TABLE
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
vae.build(input_shape=(None, 4 * n_qubits))
vae.summary()

# === Run di griglia — esempio VAE ortodosso ===
# vae = VAE(n_qubits=3, latent_dim=4)      # latente piccolo
# poi: callbacks=[KLWarmup(beta_max=1.0)]  # KL forte

model = vae
'''