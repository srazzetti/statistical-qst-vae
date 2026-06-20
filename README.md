# Statistical Methods for Quantum State Tomography
## Evaluating Maximum Likelihood Estimation and Variational Autoencoders

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Qiskit](https://img.shields.io/badge/Qiskit-2.4-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Repo](https://img.shields.io/badge/repo-statistical--qst--vae-lightgrey)

A systematic comparison of **Maximum Likelihood Estimation (MLE)** and **Variational Autoencoders (VAE)** for Quantum State Tomography (QST), evaluated on GHZ, W, and product states up to 8 qubits via POVM measurement simulations.

> Final project for the course *Statistics and Data Analysis* 2025–2026  
> M.Sc. in Physics — University of Milano-Bicocca

**Authors:**
- Simone Razzetti — [GitHub](https://github.com/srazzetti)
- Riccardo Ruggeri — [GitHub](https://github.com/riccardoruggeriphysics)

---

## Overview

Quantum State Tomography is a fundamental protocol in the NISQ era, used for gate benchmarking, noise mitigation verification, and fidelity estimation of prepared states. Classical reconstruction methods suffer from the **curse of dimensionality** — the Hilbert space grows exponentially with the number of qubits, making full density matrix reconstruction computationally prohibitive.

This project implements and compares two approaches:

- **MLE** — Maximum Likelihood Estimation via Cholesky parametrization, optimized with `iminuit` (Migrad algorithm)
- **VAE** — Variational Autoencoder trained on POVM measurement outcomes to learn the Born probability distribution directly, bypassing explicit density matrix reconstruction

Both methods are evaluated using **Classical Fidelity** (Bhattacharyya coefficient) as the primary metric, enabling a fair direct comparison without requiring full state reconstruction.

---

## Project Structure

```
statistical-qst-vae/
├── data/               # CSV files with simulation results
├── docs/               # Project report (LaTeX/PDF)
├── figs/               # Generated figures and plots
├── notebooks/
│   ├── 01_demo_GHZ3.ipynb      # General demo on proposed methods
│   ├── 02_MLE_analysis.ipynb   # MLE fidelities and scaling analysis
│   ├── 03_VAE_analysis.ipynb   # VAE performance and scalabing analysis
│   └── 04_comparative_analysis.ipynb  # MLE vs VAE comparison
├── src/                # Source modules (data generation, MLE, VAE, metrics)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/srazzetti/statistical-qst-vae.git
cd statistical-qst-vae
pip install -r requirements.txt
```

Or install core dependencies manually:

```bash
pip install qiskit==2.4.1 qiskit-aer==0.17.2 tensorflow==2.21.0 keras==3.14.1 \
            numpy==2.4.6 pandas==3.0.3 matplotlib==3.11.0 scipy==1.17.1 \
            seaborn==0.13.2 iminuit==2.32.0 scikit-learn==1.9.0
```
> Tested on Python 3.10+

---

## References

Key references for this project:

- Chen et al. (2021) — *Reconstructing a quantum state with a variational autoencoder*, Int. J. Quantum Inf. 19(08):2140005
- Nielsen & Chuang — *Quantum Computation and Quantum Information*, Cambridge University Press

A complete bibliography is available in the project report (`docs/`).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
