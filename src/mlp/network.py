"""Composição dinâmica, forward e backpropagation de uma MLP."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mlp.layer import Layer
from mlp.normalization import ZScoreNormalizer
from mlp.types import FloatArray


@dataclass(frozen=True)
class LayerGradients:
    """Gradientes de pesos e vieses de uma camada."""

    weight_gradients: FloatArray
    bias_gradients: FloatArray


def _layer_values(layer_config: object, index: int) -> tuple[int, str]:
    context = f"layers[{index}]"
    if isinstance(layer_config, Mapping):
        neurons = layer_config.get("neurons")
        activation = layer_config.get("activation")
    else:
        neurons = getattr(layer_config, "neurons", None)
        activation = getattr(layer_config, "activation", None)

    if isinstance(neurons, bool) or not isinstance(neurons, int) or neurons <= 0:
        raise ValueError(f"{context}.neurons deve ser um inteiro positivo.")
    if not isinstance(activation, str) or not activation.strip():
        raise ValueError(f"{context}.activation deve ser uma string não vazia.")
    return neurons, activation


class NeuralNetwork:
    """Representa uma MLP com arquitetura dinâmica e SGD por exemplo."""

    def __init__(
        self,
        input_size: int,
        layers: Sequence[Any],
        seed: int | None = None,
    ) -> None:
        """Cria todas as camadas usando um único gerador aleatório."""
        if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("input_size deve ser um inteiro positivo.")
        if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence) or not layers:
            raise ValueError("A rede deve possuir pelo menos uma camada.")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed deve ser um inteiro ou nulo.")

        self.input_size = input_size
        self.seed = seed
        self.normalizer: ZScoreNormalizer | None = None
        self.input_columns: tuple[str, ...] | None = None
        self.output_columns: tuple[str, ...] | None = None
        self.layers: list[Layer] = []
        rng = np.random.default_rng(seed)
        previous_size = input_size
        for index, layer_config in enumerate(layers):
            neuron_count, activation = _layer_values(layer_config, index)
            layer = Layer(previous_size, neuron_count, activation, rng)
            self.layers.append(layer)
            previous_size = neuron_count

        # Travas de segurança: impedem calcular deltas sem um forward
        # correspondente ou aplicar gradientes sem tê-los calculado.
        self._forward_ready = False
        self._gradients_ready = False

    @property
    def output_size(self) -> int:
        """Quantidade de valores produzidos pela última camada."""
        return self.layers[-1].neuron_count

    @property
    def normalization(self) -> dict[str, Any] | None:
        """Expõe os parâmetros serializados no formato do pipeline de dados."""
        return self.normalizer.to_dict() if self.normalizer is not None else None

    @normalization.setter
    def normalization(self, data: dict[str, Any] | None) -> None:
        self.normalizer = ZScoreNormalizer.from_dict(data) if data is not None else None

    def initialize_output_biases(self, training_outputs: FloatArray) -> None:
        """Inicia os vieses da saída com a média das saídas de treino.

        Com saída Linear, ``ŷ = z``, então a rede já começa prevendo a média
        em vez de 0. Isso importa quando as saídas não são normalizadas
        (aqui, ~2.500 kg/ha): começando do zero, os primeiros erros são
        enormes, os gradientes saturam as camadas Tanh e a rede quase não
        aprende. Usa apenas o conjunto de treino, então não há vazamento.
        """
        output_layer = self.layers[-1]
        if output_layer.activation_name != "linear":
            raise ValueError("A inicialização pela média exige saída linear.")
        values = np.asarray(training_outputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != self.output_size:
            raise ValueError(
                f"As saídas de treino devem ter formato (n, {self.output_size})."
            )
        if not np.isfinite(values).all():
            raise ValueError("As saídas de treino devem conter somente valores finitos.")
        for neuron, mean in zip(output_layer.neurons, values.mean(axis=0), strict=True):
            neuron.bias = float(mean)

    def _validate_inputs(self, inputs: FloatArray) -> FloatArray:
        values = np.asarray(inputs, dtype=np.float64)
        if values.shape != (self.input_size,):
            raise ValueError(
                f"A rede esperava {self.input_size} entradas, "
                f"mas recebeu formato {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("As entradas devem conter somente valores finitos.")
        return values

    def forward(self, inputs: FloatArray) -> FloatArray:
        """Propaga um vetor de entradas já preparado para a rede.

        A saída de cada camada vira a entrada da próxima. O vetor devolvido é
        a saída da última camada. As entradas já devem estar normalizadas; para
        valores brutos, use ``predict``.
        """
        current_values = self._validate_inputs(inputs)
        self._forward_ready = False
        self._gradients_ready = False
        for layer in self.layers:
            current_values = layer.forward(current_values)
        self._forward_ready = True
        return current_values.copy()

    def predict(self, raw_inputs: FloatArray) -> FloatArray:
        """Aplica a normalização salva a um vetor bruto e executa o forward."""
        inputs = self._validate_inputs(raw_inputs)
        if self.normalizer is not None:
            inputs = self.normalizer.transform(inputs[np.newaxis, :])[0]
        return self.forward(inputs)

    def _validate_expected(self, expected: FloatArray) -> FloatArray:
        if not self._forward_ready or self.layers[-1].outputs is None:
            raise RuntimeError("Execute forward antes de calcular os deltas.")
        values = np.asarray(expected, dtype=np.float64)
        output_shape = self.layers[-1].outputs.shape
        if values.shape != output_shape:
            raise ValueError(
                f"A saída esperada deve ter formato {output_shape}, "
                f"mas recebeu {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("A saída esperada deve conter somente valores finitos.")
        return values

    def compute_deltas(self, expected: FloatArray) -> None:
        """Calcula todos os deltas sem alterar pesos ou vieses.

        O delta de um neurônio é ``δ = ∂E/∂z``: quanto o erro muda quando a
        soma ponderada ``z`` daquele neurônio muda. Ele é calculado de trás
        para frente pela regra da cadeia:

        - camada de saída: ``δ_k = ∂E/∂ŷ_k · f'(z_k)``. Com
          ``E = (1/n) Σ (ŷ_k - y_k)²``, temos ``∂E/∂ŷ_k = (2/n)(ŷ_k - y_k)``;
        - camadas ocultas: ``δ_j = f'(z_j) · Σ_k w_kj · δ_k``, somando sobre
          os neurônios ``k`` da camada seguinte que recebem a saída de ``j``.

        Nenhum peso é alterado aqui. Assim os deltas das camadas ocultas usam
        os mesmos pesos do forward deste exemplo, como exige o enunciado.
        """
        expected_values = self._validate_expected(expected)
        output_layer = self.layers[-1]
        output_count = output_layer.neuron_count

        # 1) Deltas da camada de saída: derivada do MSE vezes f'(z).
        for index, neuron in enumerate(output_layer.neurons):
            if neuron.z is None or neuron.output is None:
                raise RuntimeError("Estado incompleto: execute forward novamente.")
            error_derivative = (2.0 / output_count) * (
                neuron.output - expected_values[index]
            )
            neuron.delta = error_derivative * neuron.activation_derivative()

        # 2) Deltas das camadas ocultas, da penúltima até a primeira.
        for layer_index in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]
            for neuron_index, neuron in enumerate(current_layer.neurons):
                if neuron.z is None:
                    raise RuntimeError("Estado incompleto: execute forward novamente.")
                # Erro que "volta" para este neurônio: cada neurônio da camada
                # seguinte devolve seu delta ponderado pelo peso da ligação
                # que o conecta a este neurônio (weights[neuron_index]).
                propagated_error = 0.0
                for next_neuron in next_layer.neurons:
                    if next_neuron.delta is None:
                        raise RuntimeError("Delta ausente na camada seguinte.")
                    propagated_error += (
                        next_neuron.weights[neuron_index] * next_neuron.delta
                    )
                neuron.delta = neuron.activation_derivative() * propagated_error
        self._gradients_ready = True

    def _collect_gradients(self) -> tuple[LayerGradients, ...]:
        if not self._gradients_ready:
            raise RuntimeError("Calcule os deltas antes de obter os gradientes.")
        gradients: list[LayerGradients] = []
        for layer in self.layers:
            if layer.last_inputs is None:
                raise RuntimeError("Estado incompleto: execute forward novamente.")
            weight_gradients = np.empty(
                (layer.neuron_count, layer.input_size), dtype=np.float64
            )
            bias_gradients = np.empty(layer.neuron_count, dtype=np.float64)
            for index, neuron in enumerate(layer.neurons):
                if neuron.delta is None:
                    raise RuntimeError("Delta ausente durante o cálculo de gradientes.")
                # Como z = Σ w_i·x_i + b, temos ∂z/∂w_i = x_i e ∂z/∂b = 1.
                # Logo ∂E/∂w_i = δ·x_i e ∂E/∂b = δ.
                weight_gradients[index] = neuron.delta * layer.last_inputs
                bias_gradients[index] = neuron.delta
            gradients.append(LayerGradients(weight_gradients, bias_gradients))
        return tuple(gradients)

    def compute_gradients(
        self,
        expected: FloatArray,
    ) -> tuple[LayerGradients, ...]:
        """Calcula e retorna gradientes analíticos sem atualizar parâmetros."""
        self.compute_deltas(expected)
        return self._collect_gradients()

    @staticmethod
    def _validate_learning_rate(learning_rate: float) -> float:
        if isinstance(learning_rate, bool):
            raise ValueError("learning_rate deve ser positivo.")
        value = float(learning_rate)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("learning_rate deve ser positivo.")
        return value

    def apply_gradients(self, learning_rate: float) -> None:
        """Aplica de uma vez os gradientes calculados anteriormente.

        Descida do gradiente: ``w ← w - η·∂E/∂w`` e ``b ← b - η·∂E/∂b``,
        onde ``η`` é a taxa de aprendizado.
        """
        rate = self._validate_learning_rate(learning_rate)
        gradients = self._collect_gradients()

        # Primeiro calcula todos os novos parâmetros e confere se são finitos.
        # Só depois grava, para não deixar a rede pela metade se algo falhar.
        updated_parameters: list[tuple[FloatArray, FloatArray]] = []
        for layer, layer_gradients in zip(self.layers, gradients, strict=True):
            new_weights = np.vstack(
                [neuron.weights for neuron in layer.neurons]
            ) - rate * layer_gradients.weight_gradients
            new_biases = np.asarray(
                [neuron.bias for neuron in layer.neurons], dtype=np.float64
            ) - rate * layer_gradients.bias_gradients
            if not np.isfinite(new_weights).all() or not np.isfinite(new_biases).all():
                raise FloatingPointError("Atualização produziria parâmetros não finitos.")
            updated_parameters.append((new_weights, new_biases))

        for layer, (new_weights, new_biases) in zip(
            self.layers, updated_parameters, strict=True
        ):
            for index, neuron in enumerate(layer.neurons):
                neuron.weights = new_weights[index].copy()
                neuron.bias = float(new_biases[index])

        self._forward_ready = False
        self._gradients_ready = False

    def backward(self, expected: FloatArray, learning_rate: float) -> None:
        """Calcula todos os deltas e aplica uma atualização SGD."""
        self._validate_learning_rate(learning_rate)
        self.compute_deltas(expected)
        self.apply_gradients(learning_rate)

