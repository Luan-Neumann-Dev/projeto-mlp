"""Interface web (Streamlit) para configurar, treinar, salvar e usar a MLP.

A interface só monta a configuração a partir dos campos da tela e chama os
mesmos módulos usados pela CLI. Nenhum algoritmo da rede é reimplementado aqui.

Execução, a partir da raiz do projeto: ``streamlit run app.py``
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mlp.activations import SUPPORTED_ACTIVATIONS
from mlp.config import (
    ApplicationConfig,
    DatasetConfig,
    LayerConfig,
    NetworkConfig,
    NormalizationConfig,
    OutputConfig,
    SplitConfig,
    TrainingConfig,
    load_config,
)
from mlp.data_pipeline import load_configured_dataset, prepare_data, split_configured_dataset
from mlp.exceptions import MLPError
from mlp.network import NeuralNetwork
from mlp.persistence import load_network, save_network
from mlp.trainer import EpochMetrics, Trainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "example.json"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
ACTIVATIONS = sorted(SUPPORTED_ACTIVATIONS)
ERRORS = (MLPError, ValueError, FloatingPointError, OSError)


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _defaults() -> ApplicationConfig | None:
    """Usa o JSON de exemplo como valores iniciais dos campos, se existir."""
    try:
        return load_config(EXAMPLE_CONFIG, relative_to=PROJECT_ROOT)
    except MLPError:
        return None


def _save_upload(uploaded, suffix: str) -> Path:
    """Grava um arquivo enviado pelo navegador para os módulos lerem do disco."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as stream:
        stream.write(uploaded.getvalue())
        return Path(stream.name)


def _model_bytes(network: NeuralNetwork) -> bytes:
    """Serializa a rede com ``save_network`` e devolve o conteúdo do JSON."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        save_network(network, path)
        return path.read_bytes()


def _history_frame(history: list[EpochMetrics]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MSE treino": [row.training_mse for row in history],
            "MSE validação": [row.validation_mse for row in history],
        },
        index=pd.Index([row.epoch for row in history], name="Época"),
    )


def _architecture_label(input_size: int, layers: list[LayerConfig]) -> str:
    parts = [str(input_size)] + [f"{layer.neurons} {layer.activation}" for layer in layers]
    return " → ".join(parts)


def _config_json(config: ApplicationConfig) -> str:
    """Gera um JSON compatível com ``mlp train --config``."""
    data = {
        "dataset": {
            "path": str(config.dataset.path),
            "input_columns": list(config.dataset.input_columns),
            "output_columns": list(config.dataset.output_columns),
            "delimiter": config.dataset.delimiter,
            "encoding": config.dataset.encoding,
        },
        "network": {
            "input_size": config.network.input_size,
            "layers": [
                {"neurons": layer.neurons, "activation": layer.activation}
                for layer in config.network.layers
            ],
            "seed": config.network.seed,
            "output_bias_init": config.network.output_bias_init,
        },
        "training": {
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "shuffle_each_epoch": config.training.shuffle_each_epoch,
            "seed": config.training.seed,
        },
        "split": {
            "train": config.split.train,
            "validation": config.split.validation,
            "test": config.split.test,
            "seed": config.split.seed,
            "rounding": config.split.rounding,
        },
        "normalization": {
            "enabled": config.normalization.enabled,
            "method": config.normalization.method,
            "fit_on": config.normalization.fit_on,
            "zero_std_divisor": config.normalization.zero_std_divisor,
            "normalize_outputs": False,
        },
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------

def _data_tab(defaults: ApplicationConfig | None) -> dict | None:
    """Escolha do CSV, das colunas e da divisão. Devolve os valores escolhidos."""
    source = st.radio(
        "Origem do dataset",
        ("Dataset do projeto", "Enviar um CSV"),
        horizontal=True,
    )
    if source == "Enviar um CSV":
        uploaded = st.file_uploader("Arquivo CSV", type="csv")
        if uploaded is None:
            st.info("Envie um CSV para continuar.")
            return None
        path = _save_upload(uploaded, ".csv")
    else:
        path = DEFAULT_DATASET
        st.caption(f"Arquivo: `{path.relative_to(PROJECT_ROOT)}`")

    delimiter = st.selectbox("Separador", (",", ";", "\t"), format_func=repr)
    try:
        frame = pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")
    except (OSError, ValueError) as error:
        st.error(f"Não foi possível ler o CSV: {error}")
        return None

    st.write(f"**{len(frame)} linhas × {len(frame.columns)} colunas**")
    st.dataframe(frame.head(10), width="stretch")

    columns = list(frame.columns)
    default_inputs = [c for c in (defaults.dataset.input_columns if defaults else ()) if c in columns]
    default_outputs = [c for c in (defaults.dataset.output_columns if defaults else ()) if c in columns]
    left, right = st.columns(2)
    inputs = left.multiselect("Colunas de entrada", columns, default=default_inputs or columns[:-1])
    outputs = right.multiselect(
        "Colunas de saída",
        [c for c in columns if c not in inputs],
        default=[c for c in (default_outputs or columns[-1:]) if c not in inputs],
    )
    if not inputs or not outputs:
        st.warning("Escolha pelo menos uma coluna de entrada e uma de saída.")
        return None

    st.subheader("Divisão do dataset")
    split_default = defaults.split if defaults else None
    a, b, c, d = st.columns(4)
    train = a.number_input("Treino (%)", 1, 98, int(split_default.train) if split_default else 70)
    validation = b.number_input(
        "Validação (%)", 1, 98, int(split_default.validation) if split_default else 15
    )
    test = 100 - train - validation
    c.metric("Teste (%)", test)
    split_seed = d.number_input("Seed da divisão", 0, value=split_default.seed if split_default and split_default.seed is not None else 42,
    )
    if test <= 0:
        st.error("Treino + validação devem somar menos de 100%.")
        return None

    normalize = st.checkbox(
        "Normalizar entradas (Z-score ajustado só no treino)",
        value=defaults.normalization.enabled if defaults else True,
    )
    return {
        "path": path,
        "delimiter": delimiter,
        "inputs": tuple(inputs),
        "outputs": tuple(outputs),
        "split": SplitConfig(train, validation, test, int(split_seed), "remainder_to_test"),
        "normalize": normalize,
    }


def _architecture_and_training(defaults: ApplicationConfig | None, data: dict) -> ApplicationConfig:
    """Campos da arquitetura e dos hiperparâmetros. Devolve a configuração completa."""
    st.subheader("Arquitetura")
    default_layers = list(defaults.network.layers) if defaults else [
        LayerConfig(8, "tanh"), LayerConfig(1, "linear"),
    ]
    default_hidden = default_layers[:-1]
    hidden_count = st.number_input(
        "Quantidade de camadas ocultas", 0, 10, len(default_hidden)
    )

    layers: list[LayerConfig] = []
    for index in range(int(hidden_count)):
        base = default_hidden[index] if index < len(default_hidden) else LayerConfig(8, "tanh")
        left, right = st.columns(2)
        neurons = left.number_input(
            f"Neurônios da camada oculta {index + 1}", 1, 256, base.neurons, key=f"n{index}"
        )
        activation = right.selectbox(
            f"Ativação da camada oculta {index + 1}", ACTIVATIONS,
            index=ACTIVATIONS.index(base.activation), key=f"a{index}",
        )
        layers.append(LayerConfig(int(neurons), activation))

    # A camada de saída sempre tem um neurônio por coluna de saída.
    output_default = default_layers[-1].activation
    output_activation = st.selectbox(
        f"Ativação da camada de saída ({len(data['outputs'])} neurônio(s))",
        ACTIVATIONS, index=ACTIVATIONS.index(output_default),
    )
    layers.append(LayerConfig(len(data["outputs"]), output_activation))

    left, right = st.columns(2)
    network_seed = left.number_input(
        "Seed da inicialização dos pesos", 0,
        value=defaults.network.seed if defaults and defaults.network.seed is not None else 42,
    )
    bias_mean = right.checkbox(
        "Iniciar viés da saída com a média do treino",
        value=bool(defaults and defaults.network.output_bias_init == "train_mean"),
        disabled=output_activation != "linear",
        help="Recomendado quando a saída não é normalizada. Exige saída linear.",
    )
    st.info(f"Arquitetura: **{_architecture_label(len(data['inputs']), layers)}**")

    st.subheader("Treinamento (SGD)")
    training_default = defaults.training if defaults else None
    a, b, c, d = st.columns(4)
    epochs = a.number_input("Épocas", 1, 100_000, training_default.epochs if training_default else 500)
    learning_rate = b.number_input(
        "Taxa de aprendizado", min_value=1e-8, max_value=10.0,
        value=training_default.learning_rate if training_default else 0.01,
        step=1e-5, format="%.6f",
    )
    training_seed = c.number_input(
        "Seed do embaralhamento", 0,
        value=training_default.seed if training_default and training_default.seed is not None else 42,
    )
    shuffle = d.checkbox("Embaralhar a cada época", value=True)

    return ApplicationConfig(
        dataset=DatasetConfig(data["path"], data["inputs"], data["outputs"], data["delimiter"], "utf-8-sig"),
        network=NetworkConfig(
            len(data["inputs"]), tuple(layers), int(network_seed),
            "train_mean" if bias_mean and output_activation == "linear" else "zero",
        ),
        training=TrainingConfig(int(epochs), float(learning_rate), shuffle, int(training_seed)),
        split=data["split"],
        normalization=NormalizationConfig(data["normalize"], "zscore", "train_only", 1.0, False),
        output=OutputConfig(PROJECT_ROOT / "outputs" / "model.json", PROJECT_ROOT / "outputs" / "training_history.csv"),
    )


def _train(config: ApplicationConfig) -> None:
    """Executa o mesmo fluxo de ``mlp train`` mostrando o progresso por época."""
    network = NeuralNetwork(config.network.input_size, config.network.layers, config.network.seed)
    prepared = prepare_data(config, network)
    split = prepared.split
    if config.network.output_bias_init == "train_mean":
        network.initialize_output_biases(split.training.expected_outputs)

    total = config.training.epochs
    progress = st.progress(0.0, text="Treinando...")
    chart = st.empty()
    history: list[EpochMetrics] = []
    # Redesenhar o gráfico a cada época deixaria o treino lento.
    refresh_every = max(1, total // 100)

    def on_epoch_end(metrics: EpochMetrics) -> None:
        history.append(metrics)
        if metrics.epoch % refresh_every == 0 or metrics.epoch == total:
            progress.progress(
                metrics.epoch / total,
                text=f"Época {metrics.epoch}/{total} · MSE treino {metrics.training_mse:,.2f} "
                f"· MSE validação {metrics.validation_mse:,.2f}",
            )
            chart.line_chart(_history_frame(history))

    Trainer().fit(
        network, split.training, split.validation,
        epochs=total, learning_rate=config.training.learning_rate,
        seed=config.training.seed, shuffle_each_epoch=config.training.shuffle_each_epoch,
        on_epoch_end=on_epoch_end,
    )
    progress.empty()
    chart.empty()

    st.session_state.update(
        network=network,
        history=history,
        split=split,
        # Mesma divisão, sem normalização, para mostrar valores brutos.
        raw_test=split_configured_dataset(prepared.dataset, config.split).test,
        config=config,
        test_mse=None,
    )


def _training_tab(defaults: ApplicationConfig | None, data: dict | None) -> None:
    if data is None:
        st.info("Configure o dataset na aba **Dados** primeiro.")
        return
    config = _architecture_and_training(defaults, data)

    left, right = st.columns([1, 3])
    if left.button("Treinar rede", type="primary"):
        try:
            _train(config)
        except ERRORS as error:
            st.error(f"Falha no treinamento: {error}")
    right.download_button(
        "Baixar configuração (JSON para a CLI)", _config_json(config),
        file_name="config.json", mime="application/json",
    )

    if "network" not in st.session_state:
        return
    history = st.session_state["history"]
    trained = st.session_state["config"]
    st.divider()
    st.subheader("Resultado do treinamento")
    st.caption(
        f"{_architecture_label(trained.network.input_size, list(trained.network.layers))} · "
        f"{trained.training.epochs} épocas · taxa {trained.training.learning_rate:g}"
    )
    final = history[-1]
    a, b = st.columns(2)
    a.metric("MSE final de treino", f"{final.training_mse:,.2f}")
    b.metric("MSE final de validação", f"{final.validation_mse:,.2f}")
    st.line_chart(_history_frame(history))

    a, b = st.columns(2)
    a.download_button(
        "Baixar modelo (JSON)", _model_bytes(st.session_state["network"]),
        file_name="model.json", mime="application/json",
    )
    history_csv = _history_frame(history).to_csv().encode("utf-8")
    b.download_button("Baixar histórico (CSV)", history_csv, file_name="training_history.csv", mime="text/csv")


def _evaluation_tab() -> None:
    if "network" not in st.session_state:
        st.info("Treine uma rede na aba **Treinamento** primeiro.")
        return
    network: NeuralNetwork = st.session_state["network"]
    split = st.session_state["split"]
    st.write(
        f"O conjunto de teste ({split.test.sample_count} exemplos) ficou de fora do treino "
        "e da escolha dos hiperparâmetros. Use-o só para a avaliação final."
    )
    if st.button("Calcular MSE de teste", type="primary"):
        st.session_state["test_mse"] = Trainer().evaluate(network, split.test)

    test_mse = st.session_state.get("test_mse")
    if test_mse is None:
        return
    a, b = st.columns(2)
    a.metric("MSE de teste", f"{test_mse:,.2f}")
    b.metric("RMSE de teste (unidade da saída)", f"{np.sqrt(test_mse):,.2f}")

    # Tabela de previsões do teste com os valores originais das entradas.
    raw_test = st.session_state["raw_test"]
    config: ApplicationConfig = st.session_state["config"]
    predictions = np.vstack([network.forward(row) for row in split.test.inputs])
    table = pd.DataFrame(raw_test.inputs, columns=config.dataset.input_columns)
    for index, column in enumerate(config.dataset.output_columns):
        table[f"{column} (esperado)"] = raw_test.expected_outputs[:, index]
        table[f"{column} (previsto)"] = predictions[:, index]
        table[f"erro {column}"] = predictions[:, index] - raw_test.expected_outputs[:, index]
    st.dataframe(table, width="stretch")
    if len(config.dataset.output_columns) == 1:
        column = config.dataset.output_columns[0]
        st.caption("Previsto × esperado: quanto mais perto da diagonal, melhor.")
        st.scatter_chart(table, x=f"{column} (esperado)", y=f"{column} (previsto)")


def _prediction_tab() -> None:
    choices = ["Carregar modelo (JSON)"]
    if "network" in st.session_state:
        choices.insert(0, "Rede treinada nesta sessão")
    source = st.radio("Modelo", choices, horizontal=True)

    if source == "Rede treinada nesta sessão":
        network = st.session_state["network"]
    else:
        uploaded = st.file_uploader("Arquivo do modelo", type="json")
        if uploaded is None:
            st.info("Envie um modelo salvo pela interface ou pela CLI.")
            return
        try:
            network = load_network(_save_upload(uploaded, ".json"))
        except ERRORS as error:
            st.error(str(error))
            return

    names = network.input_columns or tuple(f"entrada {i + 1}" for i in range(network.input_size))
    outputs = network.output_columns or tuple(f"saída {i + 1}" for i in range(network.output_size))
    st.caption(
        "Informe os valores brutos. "
        + ("A normalização salva no modelo será aplicada." if network.normalizer else "O modelo não usa normalização.")
    )
    columns = st.columns(min(len(names), 4))
    values = [
        columns[index % len(columns)].number_input(name, value=0.0, format="%.4f", key=f"in_{name}")
        for index, name in enumerate(names)
    ]
    if st.button("Prever", type="primary"):
        try:
            result = network.predict(np.array(values, dtype=np.float64))
        except ERRORS as error:
            st.error(f"Falha na previsão: {error}")
            return
        for name, value in zip(outputs, result, strict=True):
            st.metric(name, f"{value:,.2f}")


def main() -> None:
    st.set_page_config(page_title="MLP configurável", layout="wide")
    st.title("MLP configurável")
    st.caption("Multilayer Perceptron implementada do zero com NumPy, treinada por SGD.")

    defaults = _defaults()
    data_tab, training_tab, evaluation_tab, prediction_tab = st.tabs(
        ["1. Dados", "2. Treinamento", "3. Avaliação", "4. Previsão"]
    )
    with data_tab:
        data = _data_tab(defaults)
    with training_tab:
        _training_tab(defaults, data)
    with evaluation_tab:
        _evaluation_tab()
    with prediction_tab:
        _prediction_tab()
