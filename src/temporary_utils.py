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

    def plot_distribution_compare(self, p_true, p_gen, p_train, idx):
        # Plot the distribution of probabilities for the true distribution, generated distribution, and training distribution
        plt.figure(figsize=(12, 5))
        ax = plt.gca()
        ax.bar(idx, p_true, width=1.0, alpha=0.55, color='C0', edgecolor='none', label='P esatta (GHZ)')
        ax.plot(idx, p_gen,   color='C1', lw=1.5, marker='.', ms=4, label='P generata (VAE)')
        ax.plot(idx, p_train, color='C2', lw=1.5, marker='.', ms=4, label='P training (shot noise)')
        ax.axhline(1/len(idx), ls='--', c='gray', lw=1.5, label='uniforme (floor)')

        ax.set_xlim(-0.5, len(idx) - 0.5)
        ax.set_xlabel(f'indice outcome', fontsize=11)
        ax.set_ylabel('probabilità', fontsize=11)
        ax.set_title(f'POVM: esatta vs VAE vs training', fontsize=11, pad=10)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10, framealpha=0.8, loc='upper right')

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
