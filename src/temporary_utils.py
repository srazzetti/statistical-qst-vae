import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Plots:

    def __init__(self, history):
        self.history = history

    def plot_reconstruction_and_kl_divergence(self):
        # Plot recostruction loss and KL divergence curves
        reconstruction_losses = {key: self.history.history[key] for key in self.history.history.keys() if 'reconstruction' in key}
        kl_losses = {key: self.history.history[key] for key in self.history.history.keys() if 'kl' in key and 'weight' not in key}

        plt.figure(figsize=(8, 6))

        # Plot reconstruction and KL losses)
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

    def plot_total_loss(self):
        # Plot total loss
        total_losses = {key: self.history.history[key] for key in self.history.history.keys() if 'loss' in key and 'reconstruction' not in key and 'kl' not in key}
        
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
        Confronto esatta (GHZ) vs VAE vs training, pensato per 3 qubit (64 outcomes POVM).

        Layout "barre annidate": per ogni outcome una barra-contenitore larga (P esatta,
        contorno colorato + riempimento tenue) e, al suo interno, due barre sottili
        affiancate (training e VAE). Cosi' si occupa lo spazio di soli 64 outcome ma si
        mostrano tutte e tre le distribuzioni; quando una barra interna sfora il contenitore
        si legge subito la sovrastima rispetto alla distribuzione esatta.

        outcomes : lista degli outcome (tuple es. (0,1,3)) usata per le etichette x.
                   Se None, si usano gli indici 0..N-1.
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

    def plot_fidelity_vs_samples(self, results):
        # Plot fidelity vs number of samples for different qubit numbers
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
