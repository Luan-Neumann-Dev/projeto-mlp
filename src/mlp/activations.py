"""Funções de ativação escalares e suas derivadas."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


ActivationFunction = Callable[[float], float]


def linear(z: float) -> float:
    """Retorna a ativação Linear de ``z``."""
    return float(z)


def linear_derivative(z: float) -> float:
    """Retorna a derivada da função Linear em ``z``."""
    del z
    return 1.0


def sigmoid(z: float) -> float:
    """Retorna a Sigmoid de ``z`` usando uma forma estável.

    ``σ(z) = 1 / (1 + e^(-z))``. Para ``z`` muito negativo, ``e^(-z)``
    estoura; por isso usamos a forma equivalente ``e^z / (1 + e^z)``.
    """
    value = float(z)
    if value >= 0.0:
        exp_neg_z = np.exp(-value)
        return float(1.0 / (1.0 + exp_neg_z))

    exp_z = np.exp(value)
    return float(exp_z / (1.0 + exp_z))


def sigmoid_derivative(z: float) -> float:
    """Retorna a derivada da Sigmoid em ``z``: ``σ(z)·(1 - σ(z))``."""
    output = sigmoid(z)
    return output * (1.0 - output)


def tanh(z: float) -> float:
    """Retorna a tangente hiperbólica de ``z``."""
    return float(np.tanh(float(z)))


def tanh_derivative(z: float) -> float:
    """Retorna a derivada da tangente hiperbólica em ``z``: ``1 - tanh(z)²``."""
    output = tanh(z)
    return 1.0 - output * output


def relu(z: float) -> float:
    """Retorna a ReLU de ``z``: ``max(0, z)``."""
    return max(0.0, float(z))


def relu_derivative(z: float) -> float:
    """Retorna a derivada da ReLU, definida como zero em ``z = 0``."""
    return 1.0 if float(z) > 0.0 else 0.0


@dataclass(frozen=True)
class Activation:
    """Agrupa uma função de ativação e sua derivada."""

    function: ActivationFunction
    derivative: ActivationFunction


ACTIVATIONS: dict[str, Activation] = {
    "linear": Activation(linear, linear_derivative),
    "sigmoid": Activation(sigmoid, sigmoid_derivative),
    "tanh": Activation(tanh, tanh_derivative),
    "relu": Activation(relu, relu_derivative),
}
SUPPORTED_ACTIVATIONS = frozenset(ACTIVATIONS)


def get_activation(name: str) -> Activation:
    """Resolve uma ativação pelo nome, sem diferenciar maiúsculas.

    Raises:
        ValueError: Se o nome não identificar uma ativação suportada.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("O nome da função de ativação deve ser uma string não vazia.")
    normalized_name = name.strip().lower()
    try:
        return ACTIVATIONS[normalized_name]
    except KeyError as error:
        available = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(
            f"Função de ativação desconhecida: {name!r}. Opções: {available}."
        ) from error


def get_activation_derivative(name: str) -> ActivationFunction:
    """Retorna somente a derivada da ativação identificada por ``name``."""
    return get_activation(name).derivative

