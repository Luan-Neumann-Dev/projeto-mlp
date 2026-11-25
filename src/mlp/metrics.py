"""Métricas do núcleo matemático."""

from typing import TYPE_CHECKING

import numpy as np

from mlp.types import FloatArray

if TYPE_CHECKING:
    from mlp.network import NeuralNetwork


def mean_squared_error(predicted: FloatArray, expected: FloatArray) -> float:
    """Calcula o MSE incluindo todas as posições dos arrays.

    ``MSE = (1/n) Σ (previsto - esperado)²``. Com várias saídas e vários
    exemplos, a média é feita sobre todos os valores.
    """
    predicted_array = np.asarray(predicted, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if predicted_array.shape != expected_array.shape:
        raise ValueError("Saída produzida e esperada possuem formatos diferentes.")
    if predicted_array.size == 0:
        raise ValueError("Não é possível calcular MSE de arrays vazios.")
    if not np.isfinite(predicted_array).all() or not np.isfinite(expected_array).all():
        raise ValueError("MSE requer somente valores finitos.")
    return float(np.mean((predicted_array - expected_array) ** 2))


def mean_squared_error_for_dataset(
    network: "NeuralNetwork",
    inputs: FloatArray,
    expected_outputs: FloatArray,
) -> float:
    """Executa forward por exemplo e calcula o MSE do conjunto."""
    input_matrix = np.asarray(inputs, dtype=np.float64)
    expected_matrix = np.asarray(expected_outputs, dtype=np.float64)
    if input_matrix.ndim != 2:
        raise ValueError("As entradas do dataset devem formar uma matriz.")
    if expected_matrix.ndim != 2:
        raise ValueError("As saídas do dataset devem formar uma matriz.")
    if input_matrix.shape[0] == 0 or input_matrix.shape[0] != expected_matrix.shape[0]:
        raise ValueError("Entradas e saídas devem possuir a mesma quantidade de exemplos.")
    predictions = np.vstack([network.forward(row) for row in input_matrix])
    return mean_squared_error(predictions, expected_matrix)

