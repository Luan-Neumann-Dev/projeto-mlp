"""Inicialização reproduzível de pesos e vieses."""

import numpy as np

from mlp.activations import SUPPORTED_ACTIVATIONS
from mlp.types import FloatArray


def _validate_sizes(fan_in: int, fan_out: int) -> None:
    if isinstance(fan_in, bool) or not isinstance(fan_in, int) or fan_in <= 0:
        raise ValueError("fan_in deve ser um inteiro positivo.")
    if isinstance(fan_out, bool) or not isinstance(fan_out, int) or fan_out <= 0:
        raise ValueError("fan_out deve ser um inteiro positivo.")


def _validate_rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng deve ser uma instância de numpy.random.Generator.")


def initialize_xavier(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Inicializa uma matriz de pesos com Xavier/Glorot normal.

    Sorteia de ``N(0, σ²)`` com ``σ = sqrt(2 / (fan_in + fan_out))``. Essa
    escala mantém a variância dos sinais parecida entre as camadas, o que
    evita saturar Tanh e Sigmoid logo no início do treino.

    Returns:
        Matriz ``(fan_out, fan_in)``: uma linha de pesos por neurônio.
    """
    _validate_sizes(fan_in, fan_out)
    _validate_rng(rng)
    standard_deviation = np.sqrt(2.0 / (fan_in + fan_out))
    return np.asarray(
        rng.normal(0.0, standard_deviation, size=(fan_out, fan_in)),
        dtype=np.float64,
    )


def initialize_he(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Inicializa uma matriz de pesos com He normal.

    Sorteia de ``N(0, σ²)`` com ``σ = sqrt(2 / fan_in)``. O fator 2 compensa
    a ReLU zerar, em média, metade das saídas.
    """
    _validate_sizes(fan_in, fan_out)
    _validate_rng(rng)
    standard_deviation = np.sqrt(2.0 / fan_in)
    return np.asarray(
        rng.normal(0.0, standard_deviation, size=(fan_out, fan_in)),
        dtype=np.float64,
    )


def initialize_weights(
    fan_in: int,
    fan_out: int,
    activation: str,
    rng: np.random.Generator,
) -> FloatArray:
    """Seleciona He para ReLU e Xavier para as demais ativações."""
    if not isinstance(activation, str) or not activation.strip():
        raise ValueError("activation deve ser uma string não vazia.")
    normalized_activation = activation.strip().lower()
    if normalized_activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"Função de ativação desconhecida: {activation!r}.")
    if normalized_activation == "relu":
        return initialize_he(fan_in, fan_out, rng)
    return initialize_xavier(fan_in, fan_out, rng)


def initialize_biases(fan_out: int) -> FloatArray:
    """Cria vieses nulos para uma camada."""
    _validate_sizes(1, fan_out)
    return np.zeros(fan_out, dtype=np.float64)

