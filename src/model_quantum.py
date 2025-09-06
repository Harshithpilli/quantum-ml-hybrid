import pennylane as qml
import torch
from torch import nn


class QuantumModel(nn.Module):
    def __init__(self, n_qubits=4, n_features=4, n_layers=3):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)


        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.fc = nn.Linear(n_qubits, 2)


    def forward(self, x):
        x = self.qlayer(x)
        return self.fc(x)