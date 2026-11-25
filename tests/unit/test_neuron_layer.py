import numpy as np
import pytest

from mlp.layer import Layer
from mlp.neuron import Neuron


def test_neuron_weighted_sum_matches_manual_calculation() -> None:
    neuron = Neuron(np.array([0.2, -0.4]), bias=0.1)

    result = neuron.weighted_sum(np.array([3.0, 2.0]))

    assert result == pytest.approx(-0.1)


def test_neuron_rejects_incompatible_input_shape() -> None:
    neuron = Neuron(np.array([0.2, -0.4]))
    with pytest.raises(ValueError, match="incompatível"):
        neuron.weighted_sum(np.array([1.0]))


def test_layer_forward_applies_activation_and_stores_state() -> None:
    layer = Layer(2, 2, "linear", np.random.default_rng(1))
    layer.neurons[0].weights[:] = [0.5, -0.2]
    layer.neurons[0].bias = 0.1
    layer.neurons[1].weights[:] = [0.3, 0.8]
    layer.neurons[1].bias = -0.1

    result = layer.forward(np.array([2.0, 1.0]))

    np.testing.assert_allclose(result, [0.9, 1.3])
    np.testing.assert_allclose(layer.last_inputs, [2.0, 1.0])
    assert layer.neurons[0].z == pytest.approx(0.9)
    assert layer.neurons[1].z == pytest.approx(1.3)
    assert layer.neurons[0].output == pytest.approx(0.9)
    assert layer.neurons[1].output == pytest.approx(1.3)


def test_layer_returns_copy_of_outputs() -> None:
    layer = Layer(1, 1, "linear", np.random.default_rng(1))
    result = layer.forward(np.array([2.0]))
    internal_value = float(layer.outputs[0])

    result[0] = 999.0

    assert layer.outputs[0] == internal_value


def test_layer_rejects_incompatible_shape() -> None:
    layer = Layer(2, 1, "linear", np.random.default_rng(1))
    with pytest.raises(ValueError, match="esperava 2 entradas"):
        layer.forward(np.array([[1.0, 2.0]]))



def test_neuron_applies_its_own_activation() -> None:
    neuron = Neuron(np.array([1.0, 1.0]), bias=0.0, activation="tanh")

    output = neuron.forward(np.array([0.2, 0.3]))

    assert neuron.z == pytest.approx(0.5)
    assert output == pytest.approx(np.tanh(0.5))
    assert neuron.output == pytest.approx(np.tanh(0.5))
    assert neuron.activation_derivative() == pytest.approx(1.0 - np.tanh(0.5) ** 2)


def test_neuron_derivative_requires_forward() -> None:
    with pytest.raises(RuntimeError, match="forward"):
        Neuron(np.array([1.0])).activation_derivative()


def test_layer_neurons_share_layer_activation() -> None:
    layer = Layer(2, 3, "relu", np.random.default_rng(1))

    assert all(neuron.activation is layer.activation for neuron in layer.neurons)
