"""Leitura, validação e divisão reproduzível de datasets."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mlp.exceptions import DatasetError
from mlp.types import FloatArray

if TYPE_CHECKING:
    from mlp.network import NeuralNetwork


def _readonly_matrix(values: object, name: str) -> FloatArray:
    try:
        matrix = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"{name} deve conter somente valores numéricos.") from error
    if matrix.ndim != 2:
        raise DatasetError(f"{name} deve formar uma matriz bidimensional.")
    if matrix.shape[1] == 0:
        raise DatasetError(f"{name} deve possuir pelo menos uma coluna.")
    if not np.isfinite(matrix).all():
        raise DatasetError(f"{name} contém NaN ou valores infinitos.")
    matrix.setflags(write=False)
    return matrix


def _column_names(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise DatasetError(f"{name} deve ser uma sequência não vazia de nomes.")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise DatasetError(f"{name} deve conter somente nomes não vazios.")
    names = tuple(value.strip() for value in values)
    if len(names) != len(set(names)):
        raise DatasetError(f"{name} contém nomes duplicados.")
    return names


@dataclass(frozen=True)
class SupervisedDataset:
    """Dataset supervisionado com valores e ordem das colunas."""

    inputs: FloatArray
    expected_outputs: FloatArray
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        inputs = _readonly_matrix(self.inputs, "As entradas")
        outputs = _readonly_matrix(self.expected_outputs, "As saídas")
        input_columns = _column_names(self.input_columns, "input_columns")
        output_columns = _column_names(self.output_columns, "output_columns")
        if inputs.shape[0] == 0:
            raise DatasetError("O dataset não possui exemplos.")
        if inputs.shape[0] != outputs.shape[0]:
            raise DatasetError("Entradas e saídas devem possuir a mesma quantidade de exemplos.")
        if inputs.shape[1] != len(input_columns):
            raise DatasetError("A quantidade de colunas de entrada não corresponde aos dados.")
        if outputs.shape[1] != len(output_columns):
            raise DatasetError("A quantidade de colunas de saída não corresponde aos dados.")
        if set(input_columns) & set(output_columns):
            raise DatasetError("Uma coluna não pode ser entrada e saída ao mesmo tempo.")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "expected_outputs", outputs)
        object.__setattr__(self, "input_columns", input_columns)
        object.__setattr__(self, "output_columns", output_columns)

    @property
    def sample_count(self) -> int:
        """Quantidade de exemplos do dataset."""
        return self.inputs.shape[0]

    @property
    def input_size(self) -> int:
        """Quantidade de atributos de entrada."""
        return self.inputs.shape[1]

    @property
    def output_size(self) -> int:
        """Quantidade de valores esperados por exemplo."""
        return self.expected_outputs.shape[1]


@dataclass(frozen=True)
class DatasetSubset:
    """Um subconjunto emparelhado de entradas e saídas."""

    inputs: FloatArray
    expected_outputs: FloatArray

    def __post_init__(self) -> None:
        inputs = _readonly_matrix(self.inputs, "As entradas do subconjunto")
        outputs = _readonly_matrix(self.expected_outputs, "As saídas do subconjunto")
        if inputs.shape[0] != outputs.shape[0]:
            raise DatasetError("Entradas e saídas do subconjunto devem ter o mesmo tamanho.")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "expected_outputs", outputs)

    @property
    def sample_count(self) -> int:
        """Quantidade de exemplos do subconjunto."""
        return self.inputs.shape[0]


@dataclass(frozen=True)
class DatasetSplit:
    """Divisão completa em treino, validação e teste."""

    training: DatasetSubset
    validation: DatasetSubset
    test: DatasetSubset
    seed: int | None

    def __post_init__(self) -> None:
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise DatasetError("seed deve ser um inteiro ou nulo.")
        subsets = (self.training, self.validation, self.test)
        if len({subset.inputs.shape[1] for subset in subsets}) != 1:
            raise DatasetError("Os subconjuntos possuem quantidades de entradas diferentes.")
        if len({subset.expected_outputs.shape[1] for subset in subsets}) != 1:
            raise DatasetError("Os subconjuntos possuem quantidades de saídas diferentes.")

    @property
    def sample_count(self) -> int:
        """Quantidade total de exemplos nos três subconjuntos."""
        return (
            self.training.sample_count
            + self.validation.sample_count
            + self.test.sample_count
        )


def load_csv_dataset(
    path: str | Path,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> SupervisedDataset:
    """Carrega um CSV e seleciona entradas e saídas na ordem configurada."""
    dataset_path = Path(path).expanduser()
    inputs_names = _column_names(input_columns, "input_columns")
    outputs_names = _column_names(output_columns, "output_columns")
    if set(inputs_names) & set(outputs_names):
        raise DatasetError("Uma coluna não pode ser entrada e saída ao mesmo tempo.")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise DatasetError("O separador deve conter exatamente um caractere.")
    if not isinstance(encoding, str) or not encoding.strip():
        raise DatasetError("A codificação deve ser uma string não vazia.")
    if not dataset_path.exists():
        raise DatasetError(f"Arquivo não encontrado: {dataset_path}.")
    if not dataset_path.is_file():
        raise DatasetError(f"O caminho não é um arquivo: {dataset_path}.")

    try:
        frame = pd.read_csv(dataset_path, sep=delimiter, encoding=encoding)
    except pd.errors.EmptyDataError as error:
        raise DatasetError("O CSV está vazio ou não possui cabeçalho.") from error
    except (OSError, LookupError, ValueError) as error:
        raise DatasetError(
            f"Não foi possível ler o CSV '{dataset_path}': {error}."
        ) from error
    if frame.empty:
        raise DatasetError("O dataset não possui exemplos.")

    selected_columns = [*inputs_names, *outputs_names]
    missing_columns = list(dict.fromkeys(
        column for column in selected_columns if column not in frame.columns
    ))
    if missing_columns:
        raise DatasetError(f"Colunas não encontradas: {', '.join(missing_columns)}.")

    try:
        inputs = frame.loc[:, list(inputs_names)].to_numpy(dtype=np.float64)
        outputs = frame.loc[:, list(outputs_names)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DatasetError(
            "As colunas selecionadas devem conter somente valores numéricos."
        ) from error
    if not np.isfinite(inputs).all():
        raise DatasetError("As entradas contêm NaN ou valores infinitos.")
    if not np.isfinite(outputs).all():
        raise DatasetError("As saídas contêm NaN ou valores infinitos.")

    return SupervisedDataset(inputs, outputs, inputs_names, outputs_names)


def validate_dataset_compatibility(
    dataset: SupervisedDataset,
    network: "NeuralNetwork",
) -> None:
    """Valida as dimensões do dataset contra a arquitetura da rede."""
    if dataset.input_size != network.input_size:
        raise DatasetError(
            f"O dataset possui {dataset.input_size} entradas, "
            f"mas a rede espera {network.input_size}."
        )
    if dataset.output_size != network.output_size:
        raise DatasetError(
            f"O dataset possui {dataset.output_size} saídas, "
            f"mas a rede produz {network.output_size}."
        )


def _percentage(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise DatasetError(f"{name} deve ser um percentual numérico.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"{name} deve ser um percentual numérico.") from error
    if not np.isfinite(result) or result < 0.0 or result > 100.0:
        raise DatasetError(f"{name} deve estar entre 0 e 100.")
    return result


def _make_subset(dataset: SupervisedDataset, indices: np.ndarray) -> DatasetSubset:
    return DatasetSubset(
        inputs=dataset.inputs[indices],
        expected_outputs=dataset.expected_outputs[indices],
    )


def split_dataset(
    dataset: SupervisedDataset,
    train_percentage: float,
    validation_percentage: float,
    test_percentage: float,
    seed: int | None,
) -> DatasetSplit:
    """Embaralha e divide o dataset sem perder ou duplicar exemplos.

    Treino e validação usam a parte inteira das contagens. O conjunto de
    teste recebe o restante causado pelo arredondamento.
    """
    train = _percentage(train_percentage, "train_percentage")
    validation = _percentage(validation_percentage, "validation_percentage")
    test = _percentage(test_percentage, "test_percentage")
    total = train + validation + test
    if not np.isclose(total, 100.0):
        raise DatasetError(
            f"Os percentuais devem totalizar 100%, mas totalizam {total}%."
        )
    if train <= 0.0 or validation <= 0.0 or test <= 0.0:
        raise DatasetError(
            "Treino, validação e teste devem possuir percentuais maiores que zero."
        )
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise DatasetError("seed deve ser um inteiro ou nulo.")

    train_count = int(dataset.sample_count * train / 100.0)
    validation_count = int(dataset.sample_count * validation / 100.0)
    test_count = dataset.sample_count - train_count - validation_count
    if train_count == 0:
        raise DatasetError("A divisão resultou em treinamento vazio.")
    if validation_count == 0:
        raise DatasetError("A divisão resultou em validação vazia.")
    if test_count == 0:
        raise DatasetError("A divisão resultou em teste vazio.")

    # Embaralha os índices uma única vez e fatia em três partes, então
    # nenhum exemplo se repete nem fica de fora.
    indices = np.random.default_rng(seed).permutation(dataset.sample_count)
    train_end = train_count
    validation_end = train_count + validation_count
    return DatasetSplit(
        training=_make_subset(dataset, indices[:train_end]),
        validation=_make_subset(dataset, indices[train_end:validation_end]),
        test=_make_subset(dataset, indices[validation_end:]),
        seed=seed,
    )

