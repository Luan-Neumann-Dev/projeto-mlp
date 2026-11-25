import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mlp.config import load_config
from mlp.exceptions import ConfigurationError
from mlp.network import NeuralNetwork


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "example.json"


def network(output_activation="linear", outputs=2) -> NeuralNetwork:
    return NeuralNetwork(
        1,
        [{"neurons": 3, "activation": "tanh"},
         {"neurons": outputs, "activation": output_activation}],
        seed=1,
    )


def test_sets_output_biases_to_training_means() -> None:
    net = network()
    hidden_before = [neuron.bias for neuron in net.layers[0].neurons]

    net.initialize_output_biases(np.array([[10.0, -1.0], [20.0, 3.0]]))

    assert [neuron.bias for neuron in net.layers[-1].neurons] == [15.0, 1.0]
    assert [neuron.bias for neuron in net.layers[0].neurons] == hidden_before


def test_rejects_non_linear_output() -> None:
    with pytest.raises(ValueError, match="linear"):
        network("sigmoid", 1).initialize_output_biases(np.array([[0.5]]))


def test_rejects_incompatible_shape() -> None:
    with pytest.raises(ValueError, match="formato"):
        network().initialize_output_biases(np.array([[1.0]]))


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_config_defaults_to_zero(tmp_path: Path) -> None:
    data = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    data["network"].pop("output_bias_init", None)

    assert load_config(_write(tmp_path, data)).network.output_bias_init == "zero"


@pytest.mark.parametrize(
    ("value", "activation", "message"),
    [("mean", "linear", "output_bias_init"), ("train_mean", "sigmoid", "linear")],
)
def test_config_rejects_invalid_values(tmp_path: Path, value, activation, message) -> None:
    data = copy.deepcopy(json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8")))
    data["network"]["output_bias_init"] = value
    data["network"]["layers"][-1]["activation"] = activation

    with pytest.raises(ConfigurationError, match=message):
        load_config(_write(tmp_path, data))
