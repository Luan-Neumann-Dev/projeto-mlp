"""Interface de linha de comando da aplicação."""

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile

from mlp.config import ApplicationConfig, load_config
from mlp.data_pipeline import load_configured_dataset, prepare_data, split_configured_dataset
from mlp.dataset import DatasetSubset, validate_dataset_compatibility
from mlp.exceptions import ConfigurationError, DatasetError, MLPError
from mlp.network import NeuralNetwork
from mlp.persistence import load_network, save_network
from mlp.trainer import EpochMetrics, Trainer


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser principal e registra os subcomandos públicos."""
    parser = argparse.ArgumentParser(
        prog="mlp",
        description="MLP configurável implementada sem frameworks de redes neurais.",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMANDO")
    inspect = subparsers.add_parser("inspect-dataset", help="Inspeciona o CSV e a divisão.")
    train = subparsers.add_parser("train", help="Treina e salva uma rede.")
    evaluate = subparsers.add_parser("evaluate", help="Mede o MSE de uma rede salva.")
    predict = subparsers.add_parser("predict", help="Prevê a partir de valores brutos.")

    for command in (inspect, train, evaluate):
        command.add_argument("--config", type=Path, required=True, help="JSON de configuração.")
        command.add_argument("--dataset", type=Path, help="Substitui o CSV da configuração.")

    inspect.set_defaults(handler=_inspect_dataset)
    train.add_argument("--output", type=Path, required=True, help="Destino do JSON do modelo.")
    train.add_argument("--history", type=Path, help="Destino do CSV com o MSE por época.")
    train.add_argument(
        "--final-test", "--final-run", action="store_true",
        help="Calcula o MSE de teste após o treinamento.",
    )
    train.set_defaults(handler=_train)
    evaluate.add_argument("--model", type=Path, required=True, help="JSON do modelo salvo.")
    evaluate.add_argument("--subset", choices=("test", "all"), default="test")
    evaluate.set_defaults(handler=_evaluate)
    predict.add_argument("--model", type=Path, required=True, help="JSON do modelo salvo.")
    predict.add_argument("--values", type=float, nargs="+", required=True, help="Valores brutos na ordem do treino.")
    predict.set_defaults(handler=_predict)
    return parser


def _absolute(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except (RuntimeError, ValueError) as error:
        raise ConfigurationError(f"Caminho inválido '{path}': {error}") from error


def _config_for_command(args: argparse.Namespace) -> ApplicationConfig:
    config = load_config(_absolute(args.config), relative_to=Path.cwd())
    if args.dataset is not None:
        config = replace(config, dataset=replace(config.dataset, path=_absolute(args.dataset)))
    return config


def _inspect_dataset(args: argparse.Namespace) -> None:
    config = _config_for_command(args)
    dataset = load_configured_dataset(config.dataset)
    split = split_configured_dataset(dataset, config.split)
    print(f"Dataset: {config.dataset.path}")
    print(f"Dimensões selecionadas: {dataset.sample_count} linhas x {dataset.input_size + dataset.output_size} colunas")
    print(f"Entradas ({dataset.input_size}): {', '.join(dataset.input_columns)}")
    print(f"Saídas ({dataset.output_size}): {', '.join(dataset.output_columns)}")
    print(
        f"Divisão: treino={split.training.sample_count}, "
        f"validação={split.validation.sample_count}, teste={split.test.sample_count}"
    )
    if split.seed is not None:
        print(f"Seed da divisão: {split.seed}")
    method = config.normalization.method if config.normalization.enabled else "desativada"
    print(f"Normalização: {method}")


def _write_history(path: Path, history: tuple[EpochMetrics, ...]) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(("epoch", "training_mse", "validation_mse"))
            writer.writerows(
                (row.epoch, row.training_mse, row.validation_mse) for row in history
            )
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _train(args: argparse.Namespace) -> None:
    config = _config_for_command(args)
    model_path = _absolute(args.output)
    history_path = _absolute(args.history) if args.history is not None else None
    destinations = (model_path,) if history_path is None else (model_path, history_path)
    inputs = {_absolute(args.config), config.dataset.path}
    if inputs.intersection(destinations):
        raise ConfigurationError("Os destinos não podem coincidir com a configuração ou o dataset.")
    if history_path is not None and (
        model_path == history_path
        or model_path in history_path.parents or history_path in model_path.parents
    ):
        raise ConfigurationError("Os destinos do modelo e do histórico devem ser arquivos distintos.")
    for path in destinations:
        if path.is_dir():
            raise ConfigurationError(f"O destino '{path}' deve ser um arquivo.")

    network = NeuralNetwork(
        config.network.input_size, config.network.layers, config.network.seed
    )
    split = prepare_data(config, network).split
    if config.network.output_bias_init == "train_mean":
        network.initialize_output_biases(split.training.expected_outputs)
    trainer = Trainer()
    settings = config.training
    result = trainer.fit(
        network, split.training, split.validation,
        epochs=settings.epochs, learning_rate=settings.learning_rate, seed=settings.seed,
        shuffle_each_epoch=settings.shuffle_each_epoch,
    )
    test_mse = trainer.evaluate(network, split.test) if args.final_test else None
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    save_network(network, model_path)
    if history_path is not None:
        _write_history(history_path, result.history)

    for row in result.history:
        print(
            f"Época {row.epoch}/{settings.epochs} | MSE treino: {row.training_mse:.6f} "
            f"| MSE validação: {row.validation_mse:.6f}"
        )
    final = result.history[-1]
    print(f"MSE final de treino: {final.training_mse:.6f}")
    print(f"MSE final de validação: {final.validation_mse:.6f}")
    if test_mse is not None:
        print(f"MSE final de teste: {test_mse:.6f} ({split.test.sample_count} exemplos)")
    print(f"Modelo: {model_path}")
    if history_path is not None:
        print(f"Histórico: {history_path}")


def _evaluate(args: argparse.Namespace) -> None:
    config = _config_for_command(args)
    network = load_network(_absolute(args.model))
    dataset = load_configured_dataset(config.dataset)
    validate_dataset_compatibility(dataset, network)
    if network.input_columns is not None and network.input_columns != dataset.input_columns:
        raise DatasetError("As colunas de entrada devem seguir a ordem registrada no modelo.")
    if network.output_columns is not None and network.output_columns != dataset.output_columns:
        raise DatasetError("As colunas de saída devem seguir a ordem registrada no modelo.")

    if args.subset == "test":
        if config.split.seed is None:
            raise DatasetError("Para reproduzir o teste, informe a split.seed da divisão original.")
        selected = split_configured_dataset(dataset, config.split).test
        label = "MSE de teste"
    else:
        selected = DatasetSubset(dataset.inputs, dataset.expected_outputs)
        label = "MSE no CSV completo"

    if network.normalizer is not None:
        selected = DatasetSubset(
            network.normalizer.transform(selected.inputs), selected.expected_outputs
        )
    mse = Trainer().evaluate(network, selected)
    print(f"{label}: {mse:.6f} ({selected.sample_count} exemplos)")


def _predict(args: argparse.Namespace) -> None:
    network = load_network(_absolute(args.model))
    try:
        prediction = network.predict(args.values)
        result = json.dumps({"prediction": prediction.tolist()}, allow_nan=False)
    except ValueError as error:
        raise MLPError(f"Falha na previsão: {error}") from error
    print(result)


def main(argv: list[str] | None = None) -> int:
    """Executa a CLI e devolve um código de saída apropriado."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        args.handler(args)
    except (MLPError, OSError, FloatingPointError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    return 0


