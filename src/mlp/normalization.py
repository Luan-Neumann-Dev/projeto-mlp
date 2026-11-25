"""Normalização Z-score sem vazamento entre subconjuntos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mlp.dataset import DatasetSplit, DatasetSubset
from mlp.exceptions import NormalizationError
from mlp.types import FloatArray


@dataclass
class ZScoreNormalizer:
    """Normalizador ajustado exclusivamente com entradas de treinamento."""

    means: FloatArray | None = None
    standard_deviations: FloatArray | None = None
    zero_std_divisor: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.zero_std_divisor, bool) or not isinstance(
            self.zero_std_divisor, (int, float)
        ):
            raise NormalizationError("zero_std_divisor deve ser positivo e finito.")
        divisor = float(self.zero_std_divisor)
        if not np.isfinite(divisor) or divisor <= 0.0:
            raise NormalizationError("zero_std_divisor deve ser positivo e finito.")
        self.zero_std_divisor = divisor
        if (self.means is None) != (self.standard_deviations is None):
            raise NormalizationError("Médias e desvios devem ser fornecidos juntos.")
        if self.means is not None and self.standard_deviations is not None:
            means = np.asarray(self.means, dtype=np.float64)
            deviations = np.asarray(self.standard_deviations, dtype=np.float64)
            if (
                means.ndim != 1
                or deviations.ndim != 1
                or means.size == 0
                or means.shape != deviations.shape
                or not np.isfinite(means).all()
                or not np.isfinite(deviations).all()
                or np.any(deviations <= 0.0)
            ):
                raise NormalizationError("Parâmetros de normalização incompatíveis.")
            self.means = means.copy()
            self.standard_deviations = deviations.copy()
            self.means.setflags(write=False)
            self.standard_deviations.setflags(write=False)

    @staticmethod
    def _validate_matrix(values: object, *, allow_empty: bool = False) -> FloatArray:
        try:
            matrix = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise NormalizationError(
                "Os valores devem formar uma matriz numérica."
            ) from error
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise NormalizationError("Os valores devem formar uma matriz bidimensional.")
        if not allow_empty and matrix.shape[0] == 0:
            raise NormalizationError("Não é possível ajustar ou transformar uma matriz vazia.")
        if not np.isfinite(matrix).all():
            raise NormalizationError("A normalização requer somente valores finitos.")
        return matrix

    def fit(self, training_inputs: FloatArray) -> "ZScoreNormalizer":
        """Calcula média e desvio populacional usando apenas o treino.

        Usar só o treino evita vazamento: validação e teste não podem
        influenciar nenhum parâmetro aprendido, nem os da normalização.
        """
        values = self._validate_matrix(training_inputs)
        means = values.mean(axis=0)
        raw_standard_deviations = values.std(axis=0, ddof=0)
        standard_deviations = np.where(
            raw_standard_deviations == 0.0,
            self.zero_std_divisor,
            raw_standard_deviations,
        )
        self.means = np.asarray(means, dtype=np.float64)
        self.standard_deviations = np.asarray(
            standard_deviations, dtype=np.float64
        )
        self.means.setflags(write=False)
        self.standard_deviations.setflags(write=False)
        return self

    def transform(self, inputs: FloatArray) -> FloatArray:
        """Aplica os parâmetros ajustados sem modificar a entrada.

        ``x_normalizado = (x - média_treino) / desvio_treino``, por coluna.
        """
        if self.means is None or self.standard_deviations is None:
            raise NormalizationError(
                "O normalizador precisa ser ajustado antes da transformação."
            )
        values = self._validate_matrix(inputs)
        if values.shape[1] != self.means.shape[0]:
            raise NormalizationError(
                "Quantidade de atributos incompatível com o normalizador."
            )
        result = (values - self.means) / self.standard_deviations
        if not np.isfinite(result).all():
            raise NormalizationError("A transformação produziu valores não finitos.")
        return np.asarray(result, dtype=np.float64)

    def fit_transform(self, training_inputs: FloatArray) -> FloatArray:
        """Ajusta com o treino e retorna sua transformação."""
        return self.fit(training_inputs).transform(training_inputs)

    def to_dict(self) -> dict[str, Any]:
        """Converte os parâmetros ajustados para JSON."""
        if self.means is None or self.standard_deviations is None:
            raise NormalizationError("O normalizador ainda não foi ajustado.")
        return {
            "method": "zscore",
            "means": self.means.tolist(),
            "standard_deviations": self.standard_deviations.tolist(),
            "zero_std_divisor": self.zero_std_divisor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZScoreNormalizer":
        """Reconstrói e valida parâmetros serializados."""
        if not isinstance(data, dict) or data.get("method") != "zscore":
            raise NormalizationError("Método de normalização incompatível.")
        try:
            means = np.asarray(data["means"], dtype=np.float64)
            standard_deviations = np.asarray(
                data["standard_deviations"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as error:
            raise NormalizationError(
                "Parâmetros de normalização ausentes ou inválidos."
            ) from error
        if (
            means.ndim != 1
            or standard_deviations.ndim != 1
            or means.size == 0
            or means.shape != standard_deviations.shape
        ):
            raise NormalizationError("Parâmetros de normalização incompatíveis.")
        if not np.isfinite(means).all() or not np.isfinite(standard_deviations).all():
            raise NormalizationError("Parâmetros de normalização devem ser finitos.")
        if np.any(standard_deviations <= 0.0):
            raise NormalizationError("Desvios-padrão devem ser estritamente positivos.")

        zero_std_divisor = data.get("zero_std_divisor", 1.0)
        if isinstance(zero_std_divisor, bool):
            raise NormalizationError("zero_std_divisor serializado é inválido.")
        try:
            return cls(
                means.copy(),
                standard_deviations.copy(),
                zero_std_divisor=float(zero_std_divisor),
            )
        except (TypeError, ValueError) as error:
            raise NormalizationError(
                "zero_std_divisor serializado é inválido."
            ) from error


def normalize_split(
    split: DatasetSplit,
    normalizer: ZScoreNormalizer | None = None,
) -> tuple[DatasetSplit, ZScoreNormalizer]:
    """Ajusta no treino e transforma os três subconjuntos."""
    fitted_normalizer = normalizer or ZScoreNormalizer()
    normalized_split = DatasetSplit(
        training=DatasetSubset(
            fitted_normalizer.fit_transform(split.training.inputs),
            split.training.expected_outputs,
        ),
        validation=DatasetSubset(
            fitted_normalizer.transform(split.validation.inputs),
            split.validation.expected_outputs,
        ),
        test=DatasetSubset(
            fitted_normalizer.transform(split.test.inputs),
            split.test.expected_outputs,
        ),
        seed=split.seed,
    )
    return normalized_split, fitted_normalizer

