"""Entidade que armazena o estado de um neurônio artificial."""

from collections.abc import Sequence

import numpy as np

from mlp.activations import Activation, get_activation
from mlp.types import FloatArray


class Neuron:
    """Armazena pesos, viés, ativação e valores intermediários de um neurônio.

    Um neurônio calcula ``z = w · x + b`` e depois ``saída = f(z)``.
    Durante o treinamento ele também guarda ``delta = ∂E/∂z``, usado pela
    retropropagação para obter os gradientes de seus pesos e viés.
    """

    def __init__(
        self,
        weights: Sequence[float] | FloatArray,
        bias: float = 0.0,
        activation: str | Activation = "linear",
    ) -> None:
        """Cria um neurônio com parâmetros e ativação fornecidos explicitamente."""
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.ndim != 1:
            raise ValueError("Os pesos do neurônio devem formar um vetor.")
        if weights_array.size == 0:
            raise ValueError("O neurônio deve possuir pelo menos um peso.")
        if not np.isfinite(weights_array).all() or not np.isfinite(float(bias)):
            raise ValueError("Pesos e viés devem conter somente valores finitos.")

        # Um peso por entrada recebida da camada anterior.
        self.weights: FloatArray = weights_array.copy()
        self.bias = float(bias)
        self.activation: Activation = (
            activation if isinstance(activation, Activation) else get_activation(activation)
        )
        # Valores do último exemplo processado. Ficam vazios até o forward
        # (z e output) e até a retropropagação (delta).
        self.z: float | None = None
        self.output: float | None = None
        self.delta: float | None = None

    def weighted_sum(self, inputs: FloatArray) -> float:
        """Calcula ``dot(weights, inputs) + bias``."""
        inputs_array = np.asarray(inputs, dtype=np.float64)
        if inputs_array.shape != self.weights.shape:
            raise ValueError(
                "Quantidade de entradas incompatível com os pesos do neurônio."
            )
        if not np.isfinite(inputs_array).all():
            raise ValueError("As entradas devem conter somente valores finitos.")

        result = float(np.dot(self.weights, inputs_array) + self.bias)
        if not np.isfinite(result):
            raise FloatingPointError("Soma ponderada não finita no neurônio.")
        return result

    def forward(self, inputs: FloatArray) -> float:
        """Calcula e armazena ``z`` e a saída ``f(z)`` do neurônio."""
        self.z = self.weighted_sum(inputs)
        self.output = self.activation.function(self.z)
        if not np.isfinite(self.output):
            raise FloatingPointError("Saída não finita no neurônio.")
        # Um novo forward invalida o delta do exemplo anterior.
        self.delta = None
        return self.output

    def activation_derivative(self) -> float:
        """Retorna ``f'(z)`` para o ``z`` do último forward."""
        if self.z is None:
            raise RuntimeError("Execute forward antes de calcular a derivada.")
        return self.activation.derivative(self.z)
