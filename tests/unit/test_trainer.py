import numpy as np
import pytest

from mlp.dataset import DatasetSubset
from mlp.exceptions import DatasetError
from mlp.network import NeuralNetwork
from mlp.trainer import Trainer


def linear_network(output_size=1) -> NeuralNetwork:
    network = NeuralNetwork(1, [{"neurons": output_size, "activation": "linear"}], seed=42)
    for neuron in network.layers[0].neurons:
        neuron.weights[:] = 0.0
        neuron.bias = 0.0
    return network


def subset(inputs, targets) -> DatasetSubset:
    return DatasetSubset(np.asarray(inputs).reshape(-1, 1), np.asarray(targets).reshape(-1, 1))


def test_manual_epoch_uses_updated_weights_and_end_of_epoch_mse(monkeypatch) -> None:
    network = linear_network()
    trainer = Trainer()
    neuron = network.layers[0].neurons[0]
    updates = []
    original_backward = network.backward

    def record_backward(expected, learning_rate):
        prediction = neuron.output
        original_backward(expected, learning_rate)
        updates.append((prediction, neuron.weights[0], neuron.bias))

    monkeypatch.setattr(network, "backward", record_backward)

    result = trainer.fit(
        network, subset([1, 1], [1, 0]), subset([1], [0.5]),
        epochs=1, learning_rate=0.1, shuffle_each_epoch=False,
    )

    np.testing.assert_allclose(updates, [(0.0, 0.2, 0.2), (0.4, 0.12, 0.12)])
    assert len(result.history) == 1
    metrics = result.history[0]
    assert metrics.epoch == 1
    assert metrics.training_mse == pytest.approx(0.3176)
    assert metrics.validation_mse == pytest.approx(0.0676)
    assert trainer.evaluate(network, subset([2], [1])) == pytest.approx(0.4096)
    assert len(updates) == 2


def test_evaluate_averages_all_outputs_and_preserves_parameters() -> None:
    network = linear_network(output_size=2)
    first, second = network.layers[0].neurons
    first.weights[:] = 2.0
    second.bias = -1.0
    data = DatasetSubset(np.array([[1], [2]]), np.array([[1, 1], [7, 3]]))
    before = [(neuron.weights.copy(), neuron.bias) for neuron in (first, second)]

    assert Trainer().evaluate(network, data) == pytest.approx((1 + 4 + 9 + 16) / 4)

    for neuron, (weights, bias) in zip((first, second), before, strict=True):
        np.testing.assert_array_equal(neuron.weights, weights)
        assert neuron.bias == bias


@pytest.mark.parametrize("shuffle", [False, True])
def test_training_order_pairing_and_reproducibility(monkeypatch, shuffle) -> None:
    training = subset([1, 2, 3, 4], [10, 20, 30, 40])
    validation = subset([5], [50])

    def run():
        network = linear_network()
        visits = []
        original_backward = network.backward

        def record_backward(expected, learning_rate):
            inputs = network.layers[0].last_inputs
            assert inputs is not None
            visits.append(int(inputs[0]))
            assert expected[0] == 10 * inputs[0]
            original_backward(expected, learning_rate)

        monkeypatch.setattr(network, "backward", record_backward)
        result = Trainer().fit(
            network, training, validation,
            epochs=3, learning_rate=0.01, seed=42, shuffle_each_epoch=shuffle,
        )
        neuron = network.layers[0].neurons[0]
        return visits, result, neuron.weights.tolist(), neuron.bias

    visits, result, weights, bias = run()
    assert [row.epoch for row in result.history] == [1, 2, 3]
    rng = np.random.default_rng(42)
    expected_order = []
    for _ in result.history:
        order = rng.permutation(4) + 1 if shuffle else range(1, 5)
        expected_order.extend(order)
    assert visits == expected_order
    assert run() == (visits, result, weights, bias)


def test_changing_validation_targets_does_not_change_sgd() -> None:
    training = subset([1, 2], [1, 2])
    first, second = linear_network(), linear_network()
    trainer = Trainer()
    first_result = trainer.fit(
        first, training, subset([3], [1]), epochs=2, learning_rate=0.01, seed=42
    )
    second_result = trainer.fit(
        second, training, subset([3], [100]), epochs=2, learning_rate=0.01, seed=42
    )
    first_neuron, second_neuron = first.layers[0].neurons[0], second.layers[0].neurons[0]
    np.testing.assert_array_equal(first_neuron.weights, second_neuron.weights)
    assert first_neuron.bias == second_neuron.bias
    for left, right in zip(first_result.history, second_result.history, strict=True):
        assert left.training_mse == right.training_mse
        assert left.validation_mse != right.validation_mse


@pytest.mark.parametrize("epochs,rate", [(0, 0.1), (True, 0.1), (1, 0), (1, -0.1), (1, np.inf), (1, np.nan)])
def test_fit_rejects_invalid_hyperparameters(epochs, rate) -> None:
    data = subset([1], [1])
    with pytest.raises(ValueError, match="epochs|learning_rate"):
        Trainer().fit(linear_network(), data, data, epochs=epochs, learning_rate=rate)


def test_fit_validates_subsets_before_any_update() -> None:
    network = linear_network()
    data = subset([1], [1])
    empty = subset([], [])
    wrong_outputs = DatasetSubset(np.array([[1]]), np.array([[1, 2]]))
    wrong_inputs = DatasetSubset(np.array([[1, 2]]), np.array([[1]]))
    trainer = Trainer()

    for training, validation in [(empty, data), (data, empty), (data, wrong_outputs), (wrong_inputs, data)]:
        with pytest.raises(DatasetError):
            trainer.fit(network, training, validation, epochs=1, learning_rate=0.1)
    with pytest.raises(DatasetError):
        subset([1, 2], [1])

    neuron = network.layers[0].neurons[0]
    np.testing.assert_array_equal(neuron.weights, [0])
    assert neuron.bias == 0


def test_fit_detects_sample_mse_overflow_before_updating() -> None:
    network = linear_network()
    with pytest.raises(FloatingPointError, match="época 1, índice de treino 0"):
        Trainer().fit(
            network, subset([1], [1e160]), subset([1], [0]),
            epochs=1, learning_rate=0.01,
        )
    neuron = network.layers[0].neurons[0]
    np.testing.assert_array_equal(neuron.weights, [0])
    assert neuron.bias == 0


def test_fit_reports_each_epoch_to_callback() -> None:
    received = []
    result = Trainer().fit(
        linear_network(), subset([1, 2], [1, 2]), subset([3], [3]),
        epochs=3, learning_rate=0.01, seed=42, on_epoch_end=received.append,
    )
    assert received == list(result.history)
