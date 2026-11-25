"""Camada configurável de neurônios com uma ativação compartilhada."""

import numpy as np

from mlp.activations import Activation, get_activation
from mlp.initialization import initialize_biases, initialize_weights
from mlp.neuron import Neuron
from mlp.types import FloatArray


class Layer:
    """Representa uma camada totalmente conectada da MLP."""

    def __init__(
        self,
        input_size: int,
        neuron_count: int,
        activation: str,
        rng: np.random.Generator,
    ) -> None:
        """Cria neurônios usando Xavier ou He conforme a ativação.

        Args:
            input_size: Quantidade de valores vindos da camada anterior
                (ou das entradas da rede, se for a primeira camada).
            neuron_count: Quantidade de neurônios desta camada.
            activation: Nome da ativação (linear, sigmoid, tanh ou relu).
            rng: Gerador compartilhado pela rede, para reprodutibilidade.
        """
        if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("input_size deve ser um inteiro positivo.")
        if (
            isinstance(neuron_count, bool)
            or not isinstance(neuron_count, int)
            or neuron_count <= 0
        ):
            raise ValueError("neuron_count deve ser um inteiro positivo.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng deve ser uma instância de numpy.random.Generator.")
        if not isinstance(activation, str) or not activation.strip():
            raise ValueError("activation deve ser uma string não vazia.")

        self.input_size = input_size
        self.neuron_count = neuron_count
        self.activation_name = activation.strip().lower()
        self.activation: Activation = get_activation(self.activation_name)
        self.initialization_method = (
            "he" if self.activation_name == "relu" else "xavier"
        )

        weights = initialize_weights(
            input_size,
            neuron_count,
            self.activation_name,
            rng,
        )
        biases = initialize_biases(neuron_count)
        # Todos os neurônios da camada compartilham a mesma ativação.
        self.neurons = [
            Neuron(weights[index], biases[index], self.activation)
            for index in range(neuron_count)
        ]
        # Guardados para a retropropagação: os gradientes dos pesos dependem
        # das entradas que a camada recebeu no forward.
        self.last_inputs: FloatArray | None = None
        self.outputs: FloatArray | None = None

    def forward(self, inputs: FloatArray) -> FloatArray:
        """Propaga um vetor pelas somas ponderadas e pela ativação."""
        inputs_array = np.asarray(inputs, dtype=np.float64)
        if inputs_array.shape != (self.input_size,):
            raise ValueError(
                f"A camada esperava {self.input_size} entradas, "
                f"mas recebeu formato {inputs_array.shape}."
            )
        if not np.isfinite(inputs_array).all():
            raise ValueError("As entradas devem conter somente valores finitos.")

        self.last_inputs = inputs_array.copy()
        # Cada neurônio recebe o vetor completo da camada anterior
        # (camada totalmente conectada) e devolve f(w · x + b).
        outputs = np.array(
            [neuron.forward(inputs_array) for neuron in self.neurons],
            dtype=np.float64,
        )

        self.outputs = outputs
        return outputs.copy()

