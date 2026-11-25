"""Carregamento e validação da configuração JSON da aplicação."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlp.activations import SUPPORTED_ACTIVATIONS
from mlp.exceptions import ConfigurationError


OUTPUT_BIAS_INIT_OPTIONS = frozenset({"zero", "train_mean"})


@dataclass(frozen=True)
class DatasetConfig:
    """Configuração da origem e das colunas do dataset."""

    path: Path
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    delimiter: str
    encoding: str


@dataclass(frozen=True)
class LayerConfig:
    """Configuração de uma camada da rede."""

    neurons: int
    activation: str


@dataclass(frozen=True)
class NetworkConfig:
    """Configuração estrutural e seed de uma rede."""

    input_size: int
    layers: tuple[LayerConfig, ...]
    seed: int | None
    output_bias_init: str = "zero"


@dataclass(frozen=True)
class TrainingConfig:
    """Parâmetros do processo de treinamento."""

    epochs: int
    learning_rate: float
    shuffle_each_epoch: bool
    seed: int | None


@dataclass(frozen=True)
class SplitConfig:
    """Percentuais e seed da divisão do dataset."""

    train: float
    validation: float
    test: float
    seed: int | None
    rounding: str


@dataclass(frozen=True)
class NormalizationConfig:
    """Política de normalização das entradas."""

    enabled: bool
    method: str
    fit_on: str
    zero_std_divisor: float
    normalize_outputs: bool


@dataclass(frozen=True)
class OutputConfig:
    """Caminhos do modelo e do histórico de treinamento."""

    model_path: Path
    history_path: Path


@dataclass(frozen=True)
class ApplicationConfig:
    """Configuração validada de toda a aplicação."""

    dataset: DatasetConfig
    network: NetworkConfig
    training: TrainingConfig
    split: SplitConfig
    normalization: NormalizationConfig
    output: OutputConfig


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"O campo '{key}' deve ser um objeto JSON.")
    return value


def _string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context}.{key} deve ser uma string não vazia.")
    return value.strip()


def _positive_int(data: dict[str, Any], key: str, context: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{context}.{key} deve ser um inteiro positivo.")
    return value


def _positive_number(data: dict[str, Any], key: str, context: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{context}.{key} deve ser um número positivo.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ConfigurationError(f"{context}.{key} deve ser um número positivo.")
    return number


def _boolean(data: dict[str, Any], key: str, context: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{context}.{key} deve ser verdadeiro ou falso.")
    return value


def _seed(data: dict[str, Any], key: str, context: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"{context}.{key} deve ser inteiro não negativo ou nulo.")
    return value


def _column_names(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"dataset.{key} deve ser uma lista não vazia.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ConfigurationError(f"dataset.{key} deve conter somente nomes válidos.")
    names = tuple(item.strip() for item in value)
    if len(names) != len(set(names)):
        raise ConfigurationError(f"dataset.{key} contém nomes duplicados.")
    return names


def _resolve_path(value: str, base_directory: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_directory / path).resolve()


def _parse_dataset(
    data: dict[str, Any],
    base_directory: Path,
) -> DatasetConfig:
    path = _resolve_path(_string(data, "path", "dataset"), base_directory)
    input_columns = _column_names(data, "input_columns")
    output_columns = _column_names(data, "output_columns")
    if set(input_columns) & set(output_columns):
        raise ConfigurationError("As colunas de entrada e saída não podem se repetir.")
    delimiter = data.get("delimiter", ",")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ConfigurationError("dataset.delimiter deve conter um único caractere.")
    encoding = data.get("encoding", "utf-8-sig")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ConfigurationError("dataset.encoding deve ser uma string não vazia.")
    return DatasetConfig(path, input_columns, output_columns, delimiter, encoding)


def _parse_network(data: dict[str, Any]) -> NetworkConfig:
    input_size = _positive_int(data, "input_size", "network")
    raw_layers = data.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ConfigurationError("network.layers deve ser uma lista não vazia.")

    layers: list[LayerConfig] = []
    for index, raw_layer in enumerate(raw_layers):
        context = f"network.layers[{index}]"
        if not isinstance(raw_layer, dict):
            raise ConfigurationError(f"{context} deve ser um objeto JSON.")
        neurons = _positive_int(raw_layer, "neurons", context)
        activation = _string(raw_layer, "activation", context).lower()
        if activation not in SUPPORTED_ACTIVATIONS:
            supported = ", ".join(sorted(SUPPORTED_ACTIVATIONS))
            raise ConfigurationError(
                f"{context}.activation desconhecida: {activation!r}. "
                f"Valores aceitos: {supported}."
            )
        layers.append(LayerConfig(neurons, activation))

    # "zero": todos os vieses começam em 0.
    # "train_mean": o viés da camada de saída começa na média das saídas de
    # treino. Só faz sentido com saída Linear, em que ŷ = z.
    output_bias_init = data.get("output_bias_init", "zero")
    if output_bias_init not in OUTPUT_BIAS_INIT_OPTIONS:
        options = ", ".join(sorted(OUTPUT_BIAS_INIT_OPTIONS))
        raise ConfigurationError(
            f"network.output_bias_init deve ser um destes valores: {options}."
        )
    if output_bias_init == "train_mean" and layers[-1].activation != "linear":
        raise ConfigurationError(
            "network.output_bias_init 'train_mean' exige ativação linear na última camada."
        )
    return NetworkConfig(
        input_size,
        tuple(layers),
        _seed(data, "seed", "network"),
        output_bias_init,
    )


def _parse_training(data: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        epochs=_positive_int(data, "epochs", "training"),
        learning_rate=_positive_number(data, "learning_rate", "training"),
        shuffle_each_epoch=_boolean(data, "shuffle_each_epoch", "training"),
        seed=_seed(data, "seed", "training"),
    )


def _percentage(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"split.{key} deve ser um percentual positivo.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigurationError(f"split.{key} deve ser um percentual positivo.")
    return result


def _parse_split(data: dict[str, Any]) -> SplitConfig:
    train = _percentage(data, "train")
    validation = _percentage(data, "validation")
    test = _percentage(data, "test")
    if not math.isclose(train + validation + test, 100.0, abs_tol=1e-9):
        raise ConfigurationError("Os percentuais de split devem totalizar 100.")
    rounding = data.get("rounding", "remainder_to_test")
    if rounding != "remainder_to_test":
        raise ConfigurationError("split.rounding deve ser 'remainder_to_test'.")
    return SplitConfig(train, validation, test, _seed(data, "seed", "split"), rounding)


def _parse_normalization(data: dict[str, Any]) -> NormalizationConfig:
    enabled = _boolean(data, "enabled", "normalization")
    method = _string(data, "method", "normalization").lower()
    if method != "zscore":
        raise ConfigurationError("normalization.method deve ser 'zscore'.")
    fit_on = data.get("fit_on", "train_only")
    if fit_on != "train_only":
        raise ConfigurationError("normalization.fit_on deve ser 'train_only'.")
    zero_std_divisor = data.get("zero_std_divisor", 1.0)
    if (
        isinstance(zero_std_divisor, bool)
        or not isinstance(zero_std_divisor, (int, float))
        or not math.isfinite(float(zero_std_divisor))
        or float(zero_std_divisor) <= 0.0
    ):
        raise ConfigurationError("normalization.zero_std_divisor deve ser positivo.")
    normalize_outputs = data.get("normalize_outputs", False)
    if normalize_outputs is not False:
        raise ConfigurationError("As saídas não podem ser normalizadas neste trabalho.")
    return NormalizationConfig(
        enabled=enabled,
        method=method,
        fit_on=fit_on,
        zero_std_divisor=float(zero_std_divisor),
        normalize_outputs=normalize_outputs,
    )


def _parse_output(
    data: dict[str, Any] | None,
    base_directory: Path,
) -> OutputConfig:
    values = data if data is not None else {
        "model_path": "outputs/model.json",
        "history_path": "outputs/training_history.csv",
    }
    return OutputConfig(
        model_path=_resolve_path(
            _string(values, "model_path", "output"), base_directory
        ),
        history_path=_resolve_path(
            _string(values, "history_path", "output"), base_directory
        ),
    )


def _parse_application(
    data: dict[str, Any],
    base_directory: Path,
) -> ApplicationConfig:
    dataset = _parse_dataset(_mapping(data, "dataset"), base_directory)
    network = _parse_network(_mapping(data, "network"))
    training = _parse_training(_mapping(data, "training"))
    split = _parse_split(_mapping(data, "split"))
    normalization = _parse_normalization(_mapping(data, "normalization"))
    raw_output = data.get("output")
    if raw_output is not None and not isinstance(raw_output, dict):
        raise ConfigurationError("O campo 'output' deve ser um objeto JSON.")
    output = _parse_output(raw_output, base_directory)

    if network.input_size != len(dataset.input_columns):
        raise ConfigurationError(
            "network.input_size deve corresponder à quantidade de colunas de entrada."
        )
    if network.layers[-1].neurons != len(dataset.output_columns):
        raise ConfigurationError(
            "A última camada deve corresponder à quantidade de colunas de saída."
        )
    return ApplicationConfig(
        dataset=dataset,
        network=network,
        training=training,
        split=split,
        normalization=normalization,
        output=output,
    )


def load_config(path: str | Path, *, relative_to: Path | None = None) -> ApplicationConfig:
    """Carrega e valida a configuração JSON da aplicação.

    Args:
        path: Caminho do arquivo JSON.
        relative_to: Base dos caminhos internos. Por padrão, a pasta do JSON.

    Returns:
        Configuração imutável e validada.

    Raises:
        ConfigurationError: Se o arquivo não existir, não for JSON válido ou
            contiver valores incompatíveis com os contratos do projeto.
    """
    config_path = Path(path).expanduser().resolve()
    try:
        text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ConfigurationError(
            f"Não foi possível ler a configuração '{config_path}': {error}."
        ) from error

    try:
        raw_data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"JSON inválido em '{config_path}', linha {error.lineno}, "
            f"coluna {error.colno}."
        ) from error
    except (ValueError, RecursionError) as error:
        raise ConfigurationError(f"JSON inválido em '{config_path}': {error}.") from error
    if not isinstance(raw_data, dict):
        raise ConfigurationError("A raiz da configuração deve ser um objeto JSON.")
    try:
        return _parse_application(raw_data, relative_to or config_path.parent)
    except (ValueError, ArithmeticError, RuntimeError) as error:
        raise ConfigurationError(f"Configuração inválida em '{config_path}': {error}.") from error

