from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram

figs = Path("states/figs")
figs.mkdir(parents=True, exist_ok=True)


# Functions to create GHZ and W states, both as state preparation circuits and as full circuits with measurement.
def create_ghz_state(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    return qc

def create_ghz_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    qc.measure_all()
    return qc

def create_w_state(n_qubits):
    qc = QuantumCircuit(n_qubits)
    w_vector = np.zeros(2**n_qubits)
    for k in range(n_qubits):
        w_vector[2**k] = 1 / np.sqrt(n_qubits)
    qc.initialize(w_vector, range(n_qubits))
    return qc

def create_w_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.ry(2 * np.arccos(np.sqrt(1 / n_qubits)), 0)
    for k in range(1, n_qubits):
        angle = 2 * np.arccos(np.sqrt(1 / (n_qubits - k)))
        qc.cry(angle, k - 1, k)
        qc.cx(k, k - 1)
    qc.measure_all()
    return qc


def main():
    n_qubits = 5
    shots = 1000000
    sampler = StatevectorSampler()

    # GHZ circuit
    fig = create_ghz_circuit(n_qubits).draw(output='mpl')
    fig.savefig(figs / "ghz_state_circuit.png")
    plt.close(fig)
    # GHZ state
    qc_ghz = create_ghz_state(n_qubits)
    counts_ghz = sampler.run([qc_ghz], shots=shots).result()[0].data.meas.get_counts()
    print("GHZ State:", counts_ghz)
    plot_histogram(counts_ghz)
    plt.savefig(figs / "ghz_state_histogram.png")
    plt.close()

    # W circuit
    fig = create_w_circuit(n_qubits).draw(output='mpl')
    fig.savefig(figs / "w_state_circuit.png")
    plt.close(fig)
    # W state
    qc_w = create_w_state(n_qubits)
    counts_w = sampler.run([qc_w], shots=shots).result()[0].data.meas.get_counts()
    print("W State:", counts_w)
    plot_histogram(counts_w)
    plt.savefig(figs / "w_state_histogram.png")
    plt.close()


if __name__ == "__main__":
    main()