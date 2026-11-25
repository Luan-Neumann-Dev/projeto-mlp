"""Coordenação de treinamento, avaliação e histórico de métricas."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np

from mlp.dataset import DatasetSubset
from mlp.exceptions import DatasetError
from mlp.metrics import mean_squared_error, mean_squared_error_for_dataset
from mlp.network import NeuralNetwork


@dataclass(frozen=True)
class EpochMetrics:
    """Métricas calculadas ao final de uma época."""

    epoch: int
    training_mse: float
    validation_mse: float


@dataclass(frozen=True)
class TrainingResult:
    """Resultado de uma execução de treinamento."""

    history: tuple[EpochMetrics, ...]


class Trainer:
    """Coordena SGD sem esconder os algoritmos exigidos pelo trabalho."""

    def fit(
        self,
        network: NeuralNetwork,
        training: DatasetSubset,
        validation: DatasetSubset,
        *,
        epochs: int,
        learning_rate: float,
        seed: int | None = None,
        shuffle_each_epoch: bool = True,
        on_epoch_end: Callable[[EpochMetrics], None] | None = None,
    ) -> TrainingResult:
        """Atualiza a rede por SGD e mede treino e validação ao fim de cada época.

        ``on_epoch_end``, se informado, recebe as métricas de cada época logo
        após o cálculo. A interface gráfica usa isso para mostrar o progresso.
        """
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs deve ser um inteiro positivo.")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate deve ser um número positivo e finito.")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("A seed de treinamento deve ser um inteiro não negativo ou nulo.")
        if not isinstance(shuffle_each_epoch, bool):
            raise ValueError("shuffle_each_epoch deve ser verdadeiro ou falso.")

        self._validate_subset(network, training, "treino")
        self._validate_subset(network, validation, "validação")
        # Gerador próprio do treino: a ordem dos exemplos em cada época é
        # reproduzível com a mesma seed.
        rng = np.random.default_rng(seed)
        ordered_indices = range(training.sample_count)
        history: list[EpochMetrics] = []

        # Overflow ou NaN viram exceção em vez de estragar a rede em silêncio
        # (sinal típico de taxa de aprendizado alta demais).
        with np.errstate(over="raise", invalid="raise"):
            for epoch in range(1, epochs + 1):
                indices = (
                    rng.permutation(training.sample_count)
                    if shuffle_each_epoch else ordered_indices
                )
                # SGD: um passo de atualização por exemplo de treino.
                for index in indices:
                    expected = training.expected_outputs[index]
                    try:
                        # forward guarda z e saída de cada neurônio; backward
                        # usa esses valores para os deltas e atualiza a rede.
                        prediction = network.forward(training.inputs[index])
                        # O valor não é usado: calcular o erro do exemplo aqui
                        # serve para detectar overflow (errstate acima) antes
                        # de qualquer peso ser alterado.
                        mean_squared_error(prediction, expected)
                        network.backward(expected, learning_rate)
                    except FloatingPointError as error:
                        raise FloatingPointError(
                            f"Falha numérica na época {epoch}, índice de treino {index}: {error}"
                        ) from error

                # Métricas ao fim da época, com os pesos já atualizados.
                # A validação só é medida: nunca entra no backward.
                try:
                    training_mse = self.evaluate(network, training)
                    validation_mse = self.evaluate(network, validation)
                except FloatingPointError as error:
                    raise FloatingPointError(
                        f"Falha ao calcular o MSE ao fim da época {epoch}: {error}"
                    ) from error
                metrics = EpochMetrics(epoch, training_mse, validation_mse)
                history.append(metrics)
                if on_epoch_end is not None:
                    on_epoch_end(metrics)

        return TrainingResult(tuple(history))

    @staticmethod
    def _validate_subset(network: NeuralNetwork, subset: DatasetSubset, name: str) -> None:
        if subset.sample_count == 0:
            raise DatasetError(f"O conjunto de {name} deve possuir exemplos.")
        if (
            subset.inputs.shape[1] != network.input_size
            or subset.expected_outputs.shape[1] != network.output_size
        ):
            raise DatasetError(
                f"O conjunto de {name} deve ter {network.input_size} entradas "
                f"e {network.output_size} saídas por exemplo."
            )

    def evaluate(self, network: NeuralNetwork, subset: DatasetSubset) -> float:
        """Calcula o MSE de uma rede sem atualizar seus parâmetros."""
        self._validate_subset(network, subset, "avaliação")
        with np.errstate(over="raise", invalid="raise"):
            return mean_squared_error_for_dataset(
                network, subset.inputs, subset.expected_outputs
            )


