"""
Script: vae.py
Author: Riccardo Ruggeri
Date: 8/6/2026
Description: Implementazione di un VAE per tomografia di stato quantistico da sturttura de Guio.
             Funzioni principali dentro VAE:
                - encode: mappa input one-hot (4N) a media e varianza latente
                - reparameterize: campiona dal latente usando la trick di reparametrizzazione
                - decode: mappa dal latente a output (4N) con sigmoid
                - call: esegue encode + reparameterize + decode
                - _compute_losses: calcola la loss totale (ricostruzione + KL)
                - train_step: esegue un passo di training, calcola i gradienti e aggiorna i pesi
                - test_step: calcola la loss su dati di validazione
                - sample: campiona dal latente standard e decodifica (per generare nuovi campioni)
             Funzioni principali dentro KLWarmup:
                - on_epoch_begin: aggiorna il peso KL in modo lineare da 0 a beta_max sui primi warmup_epochs
"""

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
from keras import ops
import sys
sys.path.append('../src')  # per importare da src/ senza problemi
from povm_sampling import samples_to_onehot, samples_to_empirical_dist

# --- Table 1 del paper: geometria per numero di qubit ---
# (hidden per layer, latent_dim, batch_size, N_e)
PAPER_TABLE = {
    3: dict(hidden=96,   latent=16,  batch=100,  n_e=20000),
    4: dict(hidden=128,  latent=32,  batch=200,  n_e=100000),
    5: dict(hidden=224,  latent=64,  batch=500,  n_e=500),   # n_e = 4^N * 500 nel paper
    6: dict(hidden=1280, latent=256, batch=600,  n_e=500),
    7: dict(hidden=1504, latent=512, batch=800,  n_e=500),
    8: dict(hidden=2560, latent=512, batch=1000, n_e=500),
}


class VAE(keras.Model):
    """
    VAE per tomografia di stato quantistico (Chen et al. 2021).

    Baseline fedele al paper: encoder/decoder a 4 layer, hidden LeakyReLU,
    output sigmoid, reconstruction = BCE sui 4N neuroni, KL warm-up 0 -> beta_max.

    Parametri di esplorazione (per la griglia SDA):
      - latent_dim : capacita' latente. Paper = 16 (N=3). Ridotto -> VAE piu' ortodosso.
      - beta_max   : peso KL a fine warm-up. Paper = 0.85. Alzato -> regolarizzazione forte.

    Default = replica esatta del paper. Cambia solo i due assi per i run successivi.
    """
    def __init__(self, n_qubits, latent_dim=None, hidden=None, **kwargs):
        super(VAE, self).__init__(**kwargs)
        cfg = PAPER_TABLE.get(n_qubits, {})
        self.n_qubits = n_qubits
        self.input_dim = 4 * n_qubits
        # se non passati, usa i valori del paper per quel N
        self.latent_dim = latent_dim if latent_dim is not None else cfg.get('latent', 16)
        h = hidden if hidden is not None else cfg.get('hidden', 96)

        # peso KL: variabile non-trainabile, aggiornata dal callback warm-up
        self.kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        # --- Encoder: input(4N) -> hidden -> hidden -> [mu, log_var](latent) ---
        self.enc_dense1 = Dense(h, activation='leaky_relu', name='enc_1')         # Paper usa LeakyReLu ma anche ReLU va bene
        self.enc_dense2 = Dense(h, activation='leaky_relu', name='enc_2')
        self.z_mean_dense = Dense(self.latent_dim, name='z_mean')
        self.z_log_var_dense = Dense(self.latent_dim, name='z_log_var')

        # --- Decoder: simmetrico, logits su 4N -> softmax per gruppo (N gruppi da 4) ---
        # I dati sono N categoriche da 4 classi (one-hot per qubit): softmax-per-gruppo
        # + categorical crossentropy e' piu' corretto della sigmoid+BCE indipendente.
        self.dec_dense1 = Dense(h, activation='leaky_relu', name='dec_1')
        self.dec_dense2 = Dense(h, activation='leaky_relu', name='dec_2')
        self.dec_output = Dense(self.input_dim, name='dec_logits')   # linear (logits)

    # Explicitly implement the build method for the VAE Model subclass.
    # This ensures all internal layers are built when model.build() is called.
    def build(self, input_shape):
        # Create a dummy input for the encoder path to trigger building of its layers.
        # The input_shape for the VAE is (batch_size, original_dim).
        # We use a batch size of 1 for the dummy input.
        dummy_encoder_input = tf.zeros(shape=(1, self.input_dim))
        # Pass the dummy input through the encoder path to build its layers.
        self.encode(dummy_encoder_input)

        # Create a dummy input for the decoder path to trigger building of its layers.
        # The input to the decoder is a latent space sample, so its shape is (batch_size, latent_dim).
        dummy_decoder_input = tf.zeros(shape=(1, self.latent_dim))
        # Pass the dummy input through the decoder path to build its layers.
        self.decode(dummy_decoder_input)

        # Call the base class's build method to finalize the model's build state.
        super().build(input_shape)

    def encode(self, inputs):
        x = self.enc_dense1(inputs)
        x = self.enc_dense2(x)
        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        x = self.dec_dense1(z)
        x = self.dec_dense2(x)
        logits = self.dec_output(x)                          # (batch, 4N)
        # softmax indipendente su ciascun gruppo di 4 (un qubit)
        logits = ops.reshape(logits, (-1, self.n_qubits, 4))
        probs = ops.softmax(logits, axis=-1)                 # (batch, N, 4)
        return ops.reshape(probs, (-1, self.input_dim))      # (batch, 4N), prob. per gruppo

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decode(z)
        return reconstruction, z_mean, z_log_var

    def _compute_losses(self, x):
        reconstruction, z_mean, z_log_var = self(x)

        # Reconstruction: categorical crossentropy per gruppo (N categoriche da 4 classi),
        # sommata sui gruppi. reconstruction sono gia' probabilita' (softmax in decode).
        x_grp = ops.reshape(x, (-1, self.n_qubits, 4))
        rec_grp = ops.reshape(reconstruction, (-1, self.n_qubits, 4))
        cce = keras.losses.categorical_crossentropy(x_grp, rec_grp)  # (batch, N)
        reconstruction_loss = ops.sum(cce, axis=1)                  # somma sui gruppi -> (batch,)

        # KL verso N(0, I)
        kl_loss = -0.5 * ops.sum(
            1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1
        )

        # kl_weight e' aggiornato dal callback KLWarmup (0 -> beta_max).
        total_loss = tf.reduce_mean(reconstruction_loss + self.kl_weight * kl_loss)
        return total_loss, tf.reduce_mean(reconstruction_loss), tf.reduce_mean(kl_loss)

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data   # non supervisionato
        with tf.GradientTape() as tape:
            total_loss, rec_loss, kl_loss = self._compute_losses(x)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "reconstruction_loss": rec_loss,
                "kl_loss": kl_loss, "kl_weight": self.kl_weight}

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        total_loss, rec_loss, kl_loss = self._compute_losses(x)
        return {"loss": total_loss, "reconstruction_loss": rec_loss, "kl_loss": kl_loss}

    def sample(self, n_samples, seed=42):
        '''
        Campiona n_samples dal latente standard e decodifica.
        Args:
            n_samples: numero di campioni da generare
            seed: seme casuale per riproducibilità
        Returns:
            samples: array di forma (n_samples, 4N) con valori in [0,1] (output del decoder)
        '''
        tf.random.set_seed(seed)
        z = tf.random.normal(shape=(n_samples, self.latent_dim))
        samples = self.decode(z)
        return samples

class KLWarmup(keras.callbacks.Callback):
    """
    Warm-up lineare del peso KL: 0 -> beta_max sui primi warmup_epochs.
    Contrasta il posterior collapse: il modello impara prima a usare il
    latente (regime quasi-autoencoder, KL spenta), poi la KL si accende
    gradualmente e regolarizza il latente verso N(0, I) senza azzerarlo.
    """
    def __init__(self, beta_max=0.5, warmup_epochs=30):
        super().__init__()
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        w = min(self.beta_max, self.beta_max * epoch / self.warmup_epochs)
        self.model.kl_weight.assign(w)

class FidelityMonitor(keras.callbacks.Callback):
    """
    Define Fidelty monitor.
        Args:
            P_exact: the exact probability distribution over the POVM outcomes.
            n_qubits: the number of qubits in the system.
            n_gen: the number of samples to generate from the model for estimating the fidelity.
            seed: the random seed for reproducibility.
    """
    def __init__(self, P_exact, n_qubits, n_gen=20000, seed=0):
        super().__init__()
        self.P_exact = P_exact
        self.n_qubits = n_qubits
        self.n_gen = n_gen
        self.seed = seed

    def on_epoch_end(self, epoch, logs=None):
        # RNG con seed fisso -> campionamento riproducibile a ogni epoca
        rng = np.random.default_rng(self.seed)

        # Genera n_gen campioni dal VAE -> (n_gen, n_qubits, 4): per ogni qubit, prob sui 4 outcome SIC-POVM
        probs = self.model.sample(self.n_gen).numpy().reshape(self.n_gen, self.n_qubits, 4)

        # CDF lungo l'asse dei 4 outcome: [p0,p1,p2,p3] -> [p0, p0+p1, p0+p1+p2, 1.0]
        cum = np.cumsum(probs, axis=-1)

        # Uniformi in [0,1), una per (campione, qubit); ultima dim = 1 per broadcast contro cum
        u = rng.random((self.n_gen, self.n_qubits, 1))

        # Inverse-CDF sampling: primo True lungo l'asse outcome -> indice outcome estratto (0-3)
        draws = (u < cum).argmax(axis=-1)

        # Ogni campione (vettore di outcome per qubit) -> tupla di int
        gen = [tuple(int(a) for a in r) for r in draws]

        # Conta frequenze -> distribuzione empirica generata dal VAE {outcome: prob}
        P_vae = samples_to_empirical_dist(gen, self.n_qubits)

        # Classical fidelity (coeff. di Bhattacharyya): F_c = sum_x sqrt(P_exact(x) * P_vae(x))
        # .get(x, 0.0) -> 0 se l'outcome manca in una delle due distribuzioni
        Fc = float(sum(np.sqrt(self.P_exact.get(x, 0.0) * P_vae.get(x, 0.0))
                    for x in set(self.P_exact) | set(P_vae)))

        # Garantisce che logs esista, poi registra val_fidelity (monitorabile da EarlyStopping/checkpoint)
        logs = logs if logs is not None else {}
        logs['val_fidelity'] = Fc
        
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