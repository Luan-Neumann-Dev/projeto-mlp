# Sistema configurável de MLP

Projeto acadêmico de uma Multilayer Perceptron configurável. Os passos 1 a 7 integram o núcleo matemático, a preparação dos dados, o treino por épocas com SGD, a persistência e os fluxos de avaliação e previsão no terminal.

O código implementa as operações da rede com Python e NumPy. Defina arquitetura, funções de ativação, hiperparâmetros e colunas no JSON de configuração.

## Configuração adotada

- Problema: regressão.
- Entradas, nesta ordem: `sementes_m2`, `nitrogenio_kg_ha`.
- Saída: `produtividade_kg_ha`.
- Arquitetura inicial: `2 -> 8 (tanh) -> 4 (tanh) -> 1 (linear)`.
- Divisão: 70% treino, 15% validação e 15% teste, com seed 42.
- Normalização: Z-score nas entradas, ajustada somente com o treino.
- Treinamento de exemplo em 3.000 épocas, taxa de aprendizado 0,00003 e seed 42.
- Viés da camada de saída iniciado com a média das saídas de treino (`network.output_bias_init: "train_mean"`). Veja [Ajuste do treinamento](docs/decisoes.md#ajuste-do-treinamento).

- Interface: CLI com `argparse`.
- Persistência em JSON UTF-8 com `format_version: 1`.

O diagnóstico e as justificativas estão em [docs/decisoes.md](docs/decisoes.md).

## Instalação

Requisito: Python 3.12 ou superior.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

O modo editável disponibiliza o pacote `mlp` e o comando de terminal de mesmo nome.

A CLI resolve os caminhos relativos pela pasta de execução, inclusive `dataset.path` do JSON. Execute os exemplos a partir da raiz do projeto. A API `load_config()` usa a pasta do JSON por padrão. Para ler `configs/example.json` pela API, informe a raiz do projeto em `relative_to`.

## Verificação

```bash
python -m pytest
python -m mlp --help
mlp --help
python scripts/validate_core.py
python scripts/validate_data_pipeline.py
```

Também permanecem disponíveis os utilitários da fase 1:

```bash
python scripts/inspect_dataset.py
python scripts/validate_phase1.py
```

Ambiente verificado nesta entrega: Python 3.12.3, NumPy 2.5.5, pandas 3.1.1 e pytest 9.2.0. A suíte contém 189 testes aprovados. Consulte as verificações do [núcleo](docs/verificacao-passo-3.md), do [pipeline de dados](docs/verificacao-passo-4.md), do [treinamento](docs/verificacao-passo-5.md), da [persistência](docs/verificacao-passo-6.md) e da [CLI](docs/verificacao-passo-7.md).

## Pipeline de dados

- seleciona entradas e saídas pelos nomes e na ordem definidos no JSON;
- converte as matrizes para `float64` e rejeita dados ausentes, infinitos ou não numéricos;
- embaralha uma vez com `numpy.random.default_rng(seed)`, preservando o pareamento entre entradas e saídas;
- usa truncamento para treino e validação, deixando o restante para teste: `125 / 26 / 28` no dataset de 179 linhas;
- ajusta média e desvio-padrão populacional somente nas entradas de treino;
- aplica os mesmos parâmetros a validação, teste e futuras previsões;
- mantém as saídas na escala original e serializa os parâmetros de normalização na rede.

## Núcleo matemático

- Linear, Sigmoid estável, Tanh e ReLU, com derivadas;
- derivada da ReLU definida como zero em `z = 0`;
- cada neurônio aplica a ativação da sua camada e guarda `z`, saída e `delta`;
- Xavier/Glorot para Linear, Sigmoid e Tanh;
- He para ReLU;
- viés da saída opcionalmente iniciado com a média das saídas de treino;
- um único `numpy.random.Generator` por rede;
- forward para qualquer sequência válida de camadas;
- MSE com derivada `2 / m`;
- deltas de saída e ocultos calculados antes de qualquer atualização;
- gradientes de pesos e vieses;
- atualização SGD após cada exemplo;
- bloqueio de backward sem um forward correspondente.

Resultados reproduzíveis com seed 42:

```text
Forward manual 2 -> 2 -> 1: 0.310000000000
Atualização manual 1 -> 1: peso=0.860000000000, viés=0.280000000000
Maior erro relativo do gradient check: 2.913141949890e-12
XOR MSE inicial: 0.299795563895
XOR MSE final: 0.000107173992
XOR previsões: [0.00959, 0.98895, 0.99129, 0.01177]
```

## CLI

Use `mlp` ou `python -m mlp`. O exemplo abaixo gera arquivos próprios para o experimento e pode ser executado na raiz do projeto.

```bash
mlp inspect-dataset --config configs/example.json

mlp train --config configs/example.json --output outputs/rede-experimento.json \
  --history outputs/historico-experimento.csv

mlp evaluate --config configs/example.json --model outputs/rede-experimento.json --subset test

mlp predict --model outputs/rede-experimento.json --values 450 120
```

- `inspect-dataset`, `train` e `evaluate` aceitam `--dataset` para substituir o caminho do CSV configurado.
- `train` exibe o MSE de treino e validação por época e grava o JSON em `--output`. Com `--history`, grava `epoch`, `training_mse` e `validation_mse` no CSV indicado. Destinos existentes são substituídos.
- `--final-test` mede o teste reservado ao fim do `train`. Use essa opção após escolher os hiperparâmetros pela validação.
- `evaluate` usa o normalizador salvo. O modo padrão `test` exige a `split.seed` da divisão original e pressupõe o mesmo CSV, ordem de linhas e percentuais. `--subset all` mede o CSV completo e aceita um conjunto externo compatível.
- `predict` recebe os valores brutos na ordem do treino, com ponto decimal, e imprime um JSON com a lista `prediction`.

Os códigos de saída são `0` para sucesso, `1` para falha de execução e `2` para erro de uso. Ajuda e resultados saem em `stdout`. Mensagens de erro saem em `stderr`.

## Modelo salvo e previsão

O modelo de exemplo treinado está em [`outputs/model.json`](outputs/model.json). A API Python disponibiliza `save_network(network, path)` e `load_network(path)`. O JSON guarda as camadas, ativações, pesos, vieses e os parâmetros do Z-score usados no treino. Basta o arquivo do modelo para prever com a rede carregada.

```python
import numpy as np
from mlp.persistence import load_network

network = load_network("outputs/model.json")
productivity = network.predict(np.array([450.0, 120.0]))
print(productivity[0])  # 2309.665609724567 kg/ha
```

Informe os valores brutos na ordem `sementes_m2`, `nitrogenio_kg_ha`. `predict` aplica o Z-score salvo. Para vetores já preparados, o treinamento e a avaliação chamam `forward` diretamente.

## Estrutura

```text
configs/          configurações reproduzíveis
data/raw/         cópia preservada do CSV original
data/processed/   dados derivados futuros
docs/             diagnóstico, arquitetura e verificações
outputs/          histórico de treinamento e modelo de exemplo
scripts/          utilitários da preparação inicial
src/mlp/          pacote instalável da aplicação
tests/            testes de ambiente, configuração, CLI e estrutura
```

Responsabilidades principais do pacote:

- `config.py`: carrega e valida o JSON em dataclasses imutáveis;
- `exceptions.py`: define erros próprios da aplicação;
- `activations.py`, `initialization.py`, `neuron.py`, `layer.py` e `network.py`: implementam o núcleo matemático;
- `metrics.py`: calcula MSE de vetores e conjuntos;
- `dataset.py`: carrega, valida e divide o dataset;
- `normalization.py`: ajusta, aplica e serializa o Z-score;
- `data_pipeline.py`: integra configuração, dataset, divisão, rede e normalização;
- `trainer.py`: coordena o SGD, mede o MSE dos subconjuntos e devolve o histórico por época;
- `persistence.py`: salva, valida e reconstrói modelos no formato JSON versionado;
- `cli.py` e `__main__.py`: expõem a interface de linha de comando.


