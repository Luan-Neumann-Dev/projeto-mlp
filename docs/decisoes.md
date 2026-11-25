# Diagnóstico e decisões técnicas

## Diagnóstico do dataset

| Item | Valor encontrado |
|---|---|
| Arquivo de trabalho | `data/raw/dataset.csv` |
| Registros de dados | 179 |
| Colunas | 3 |
| Separador | Vírgula |
| Codificação | UTF-8 com BOM, lida como `utf-8-sig` |
| Cabeçalho | Sim |
| Símbolo decimal | Ponto |
| Valores ausentes | Nenhum |
| Valores infinitos | Nenhum |
| Linhas exatamente duplicadas | Nenhuma |
| Colunas não numéricas | Nenhuma |
| Entradas | `sementes_m2`, `nitrogenio_kg_ha` |
| Saída | `produtividade_kg_ha` |
| SHA-256 da cópia | `62ab886966a86e765be07f4233d0b45584e49eee02d49877ee7c5c21ed6e434e` |

As 179 linhas representam observações completas. Há 20 combinações diferentes dos dois atributos de entrada, com repetições experimentais. As repetições de entrada não são linhas duplicadas porque a produtividade observada varia.

## Resumo numérico

| Coluna | Papel | Mínimo | Máximo | Média global | Desvio-padrão amostral |
|---|---|---:|---:|---:|---:|
| `sementes_m2` | Entrada | 150,0 | 450,0 | 300,8380 | 111,8630 |
| `nitrogenio_kg_ha` | Entrada | 0,0 | 120,0 | 60,3352 | 42,4251 |
| `produtividade_kg_ha` | Saída | 1.700,9 | 3.160,4 | 2.531,1508 | 248,9787 |

Os valores desta tabela descrevem o arquivo completo apenas para diagnóstico. A aplicação não deve usar essas médias e desvios para normalização.

## Entradas, saída e tipo de problema

O objetivo é prever uma produtividade contínua a partir da densidade de sementes e da dose de nitrogênio. O problema é de regressão.

Ordem fixa dos vetores:

```text
entrada = [sementes_m2, nitrogenio_kg_ha]
saida = [produtividade_kg_ha]
```

A rede precisa ter duas entradas e um neurônio na última camada. A ativação inicial da saída é Linear e a função de erro exigida é MSE.

## Divisão dos dados

Configuração inicial:

| Subconjunto | Percentual | Quantidade com 179 linhas |
|---|---:|---:|
| Treino | 70% | 125 |
| Validação | 15% | 26 |
| Teste | 15% | 28 |

A alocação trunca as quantidades de treino e validação e envia todas as linhas restantes ao teste. A ordem é embaralhada com `numpy.random.default_rng(42)`. Cada índice aparece uma única vez.

Como há observações repetidas para as mesmas combinações de entrada, a divisão aleatória por linha pode colocar a mesma combinação experimental em subconjuntos diferentes. Isso é compatível com o enunciado e mede principalmente interpolação dentro dos níveis observados. Uma avaliação por combinações não vistas exigiria outra estratégia e deve ser tratada como decisão posterior.

## Normalização

As entradas usam Z-score. Depois da divisão, a média e o desvio-padrão populacional de cada entrada são calculados somente no conjunto de treino e aplicados a treino, validação, teste e previsões futuras.

```text
x_normalizado = (x - media_treino) / desvio_padrao_treino
```

Se o desvio-padrão de uma entrada for zero, o divisor será `1.0`. A saída não será normalizada, conforme o enunciado.

Com a divisão reproduzível definida acima, os parâmetros obtidos somente no treino são:

| Entrada | Média do treino | Desvio-padrão do treino |
|---|---:|---:|
| `sementes_m2` | 305,2000 | 106,1742 |
| `nitrogenio_kg_ha` | 62,8800 | 42,8358 |

## Arquitetura inicial

```text
2 -> 8 (tanh) -> 4 (tanh) -> 1 (linear)
```

Essa é uma configuração inicial pequena para testar a implementação, não uma arquitetura considerada ótima. Ela fica no JSON e poderá ser alterada sem mudança no código-fonte. Xavier/Glorot será usado nas camadas Tanh e Linear quando a inicialização for implementada.

## Registro das decisões

| Decisão | Escolha inicial | Justificativa |
|---|---|---|
| Linguagem | Python 3.12+ | Atende ao guia e oferece testes e serialização simples |
| Interface | CLI com `argparse` | Menor complexidade e atende ao enunciado |
| Matemática | NumPy | Vetores e matrizes sem fornecer treinamento pronto |
| CSV | pandas | Leitura, seleção e diagnóstico diretos |
| Persistência | JSON | Formato estruturado, legível e adequado aos parâmetros da rede |
| Normalização | Z-score opcional | Coloca as duas entradas em escalas comparáveis |
| Ajuste da normalização | Apenas no treino | Evita vazamento de validação e teste |
| Divisão | 70/15/15 | Mantém conjuntos separados para ajuste, acompanhamento e avaliação final |
| Arredondamento da divisão | Restante para teste | Trunca treino e validação e usa todas as 179 linhas |
| Seed | 42 | Torna divisão e inicialização reproduzíveis |
| Arquitetura de exemplo | 2-8-4-1 | Pequena, configurável e compatível com duas entradas e uma saída |
| Ativações | Tanh/Tanh/Linear | Adequadas como ponto de partida para regressão |
| Testes | pytest | Validação automatizada dos componentes |

## Critérios de aceite da fase 1

- [x] Dataset original preservado sem modificação.
- [x] CSV lido corretamente.
- [x] Quantidades de linhas e colunas registradas.
- [x] Colunas de entrada identificadas e ordenadas.
- [x] Coluna de saída identificada e ordenada.
- [x] Tipo de problema identificado.
- [x] Quantidades de entradas e saídas definidas.
- [x] Estratégia de normalização escolhida.
- [x] Divisão, arredondamento e seed definidos.
- [x] Configuração JSON criada.
- [x] Ambiente virtual criado.
- [x] Dependências declaradas.
- [x] Estrutura inicial criada.
- [x] Testes executados com sucesso: 7 testes aprovados.

## Arquitetura inicial do passo 2

### Direção das dependências

```text
CLI -> configuração, dataset, treinamento e persistência
treinamento -> rede neural e dataset
rede neural -> camada
camada -> neurônio
neurônio -> tipos numéricos
```

Os módulos matemáticos não importam a CLI. A rede não conhece caminhos de arquivos, o dataset não conhece pesos e a persistência não executa treinamento.

### Contrato dos arrays

- uma entrada: `float64` com formato `(input_size,)`;
- uma saída: `float64` com formato `(output_size,)`;
- conjunto de entradas: `float64` com formato `(sample_count, input_size)`;
- conjunto de saídas: `float64` com formato `(sample_count, output_size)`.

O alias compartilhado `FloatArray` fica em `mlp.types`. Tipos usados por um único componente permanecem no módulo responsável.

### Fronteiras de validação

- `cli.py` interpreta argumentos e converte falhas conhecidas em códigos de saída;
- `config.py` valida tipos, valores, ativações, percentuais e compatibilidade de dimensões;
- `dataset.py` valida CSV, colunas e matrizes;
- `network.py` validará o tamanho dos vetores no núcleo matemático;
- `persistence.py` validará a versão e a estrutura de modelos salvos.

### Configuração

O JSON é convertido em dataclasses imutáveis: `DatasetConfig`, `LayerConfig`, `NetworkConfig`, `TrainingConfig`, `SplitConfig`, `NormalizationConfig`, `OutputConfig` e `ApplicationConfig`.

Caminhos relativos de dataset e saída são resolvidos em relação ao arquivo JSON, não ao diretório atual do processo. Um bloco `output` ausente recebe valores padrão; um bloco presente, porém incompleto, é rejeitado.

O carregador rejeita, entre outros casos:

- JSON malformado;
- entradas ou camadas com tamanho inválido;
- lista de camadas vazia;
- ativação fora de Linear, Sigmoid, Tanh e ReLU;
- percentuais diferentes de 100%;
- quantidade de entradas incompatível com o dataset;
- última camada incompatível com as saídas;
- normalização de saídas.

### Persistência

O formato de persistência planejado é JSON e começa em `format_version = 1`. A implementação futura deverá registrar arquitetura, pesos, vieses e parâmetros de normalização.

### Convenções

- nomes internos em inglês e `snake_case`;
- classes em `PascalCase`;
- constantes em `UPPER_SNAKE_CASE`;
- mensagens ao usuário e documentação em português;
- type hints e docstrings nas interfaces públicas;
- aleatoriedade fornecida por seed ou `numpy.random.Generator`;
- `NotImplementedError` para contratos ainda pendentes.

### Critérios de aceite do passo 2

- [x] Estrutura de diretórios criada.
- [x] Pacote `mlp` importável e instalado em modo editável.
- [x] `python -m mlp --help` funciona.
- [x] `mlp --help` funciona.
- [x] Subcomandos de treino, avaliação, previsão e inspeção registrados.
- [x] Responsabilidade de cada módulo documentada.
- [x] Núcleo matemático sem dependência da CLI.
- [x] Hierarquia de exceções do domínio criada.
- [x] JSON de exemplo preservado e válido.
- [x] Configuração carregada em objetos imutáveis e validada.
- [x] Formato dos arrays definido.
- [x] Seeds explícitas, sem estado aleatório global oculto.
- [x] Suíte com 19 testes aprovada.
- [x] README atualizado com instalação e verificação.
- [x] Nenhuma biblioteca de rede neural ou diferenciação automática adicionada.

## Núcleo matemático do passo 3

### Convenções matemáticas

- todos os arrays usam `numpy.float64`;
- pesos de cada neurônio têm formato `(input_size,)`;
- entradas e saídas individuais são vetores unidimensionais;
- o MSE é a média de `(predicted - expected)²`;
- a derivada da saída é `(2 / m) * (predicted - expected)`;
- a atualização usa `parameter -= learning_rate * gradient`;
- a derivada da ReLU em zero foi definida como `0`.

### Inicialização

Linear, Sigmoid e Tanh usam Xavier/Glorot normal com desvio `sqrt(2 / (fan_in + fan_out))`. ReLU usa He normal com desvio `sqrt(2 / fan_in)`. Todos os vieses começam em zero.

A rede cria um único `numpy.random.Generator` e o repassa às camadas. A mesma seed reproduz todos os pesos; não existe estado aleatório global oculto.

### Ordem do backpropagation

1. `forward` armazena entradas, somas ponderadas e saídas.
2. `compute_deltas` calcula primeiro a saída e depois percorre as camadas ocultas de trás para frente.
3. `compute_gradients` deriva pesos e vieses sem alterar parâmetros.
4. `apply_gradients` prepara e valida todas as atualizações antes de aplicá-las.
5. `backward` coordena uma atualização SGD completa.

Depois da atualização, o estado do forward é invalidado. Outra atualização exige um novo forward, impedindo o uso acidental de gradientes antigos.

### Evidências numéricas

| Verificação | Resultado |
|---|---:|
| Forward manual `2 -> 2 -> 1` | `0,31` |
| Peso após atualização manual `1 -> 1` | `0,86` |
| Viés após atualização manual `1 -> 1` | `0,28` |
| Maior erro relativo do gradient check | `2,913141949890e-12` |
| XOR MSE inicial | `0,299795563895` |
| XOR MSE final | `0,000107173992` |

Configuração do XOR: `2 -> 4 Tanh -> 1 Sigmoid`, taxa `0,1`, 10.000 épocas e seed 42. Previsões finais, na ordem `[00, 01, 10, 11]`: `[0,00959; 0,98895; 0,99129; 0,01177]`.

### Critérios de aceite do passo 3

- [x] Linear e derivada implementadas e testadas.
- [x] Sigmoid estável e derivada testadas.
- [x] Tanh e derivada implementadas e testadas.
- [x] ReLU e derivada implementadas e testadas.
- [x] Convenção da ReLU em zero documentada.
- [x] Xavier usado com Linear, Sigmoid e Tanh.
- [x] He usado com ReLU.
- [x] Seeds reproduzem os mesmos pesos e treinamento.
- [x] Neurônios armazenam pesos, viés, `z`, saída e delta.
- [x] Camadas e ativações são configuráveis.
- [x] Arquitetura da rede é dinâmica.
- [x] Forward manual produz `0,31`.
- [x] MSE de vetor e de conjunto validado.
- [x] Deltas de saída e ocultos validados.
- [x] Todos os deltas são calculados antes das atualizações.
- [x] Pesos e vieses são atualizados por SGD.
- [x] Atualização manual produz peso `0,86` e viés `0,28`.
- [x] Gradient check abaixo de `1e-5`.
- [x] Rede aprende XOR com seed e configuração documentadas.
- [x] Suíte completa com 69 testes aprovada.

## Pipeline de dados do passo 4

O fluxo integrado está em `mlp.data_pipeline.prepare_data`: carrega o CSV, valida as dimensões contra a rede, divide os exemplos e, quando habilitado, normaliza somente as entradas. O resultado preserva tanto o dataset original quanto os três subconjuntos preparados.

O normalizador usa desvio-padrão populacional (`ddof=0`). Seus parâmetros são imutáveis para escrita externa, podem ser convertidos para um dicionário JSON e são anexados à rede, permitindo reutilizar exatamente a mesma transformação em previsões futuras.

### Critérios de aceite do passo 4

- [x] Colunas selecionadas pelo nome e na ordem configurada.
- [x] Entradas e saídas numéricas em matrizes `float64`.
- [x] Erros de arquivo, estrutura, colunas e valores inválidos tratados.
- [x] Divisão `125 / 26 / 28` reproduzível com seed 42.
- [x] Nenhum exemplo perdido, duplicado ou desemparelhado.
- [x] Z-score ajustado exclusivamente no treino.
- [x] Validação e teste transformados com os parâmetros do treino.
- [x] Saídas não normalizadas.
- [x] Colunas constantes tratadas com divisor configurável.
- [x] Parâmetros de normalização serializáveis.
- [x] Integração validada no dataset real.
- [x] Suíte completa atual com 118 testes aprovada.




## Ajuste do treinamento

### Problema encontrado

Com a configuração anterior (taxa 0,01, 196 épocas e vieses iniciados em zero), a rede praticamente não aprendia. O MSE de teste ficava no mesmo nível de prever sempre a média do treino e acima de uma regressão linear simples:

| Seed da divisão | Configuração anterior | Prever a média | Regressão linear |
|---:|---:|---:|---:|
| 1 | 65.217 | 83.854 | 37.014 |
| 7 | 42.049 | 42.597 | 33.075 |
| 42 | 58.473 | 59.521 | 34.902 |
| 123 | 59.764 | 57.133 | 45.601 |

A causa é a escala da saída, que o enunciado não permite normalizar. A produtividade fica em torno de 2.500 kg/ha. Com vieses em zero, a rede começa prevendo cerca de 0 e os primeiros erros são da ordem de 2.500. Para produzir valores nessa escala, os pesos da camada de saída também crescem muito. Esses pesos multiplicam os deltas das camadas ocultas (`δ_j = f'(z_j) · Σ w_kj · δ_k`), e as atualizações das camadas Tanh ficam grandes demais: os neurônios saturam e o MSE oscila sem diminuir.

### Ajustes

1. `network.output_bias_init: "train_mean"`: o viés de cada neurônio de saída começa com a média daquela saída no conjunto de treino. Com saída Linear, a rede já começa prevendo a média, e o treino só precisa aprender os desvios em relação a ela, que são de algumas centenas de kg/ha. A média é calculada apenas no treino, então não há vazamento. A opção exige ativação linear na última camada. O padrão continua `"zero"`.
2. Taxa de aprendizado 0,00003 e 3.000 épocas, para que as atualizações das camadas ocultas fiquem estáveis.

### Escolha pela validação

Foram comparadas três variações com o viés iniciado pela média. A escolha usou o MSE de validação da divisão padrão (seed 42), sem consultar o teste:

| Variação | Validação (seed 42) |
|---|---:|
| `2 → 8 Tanh → 4 Tanh → 1 Linear`, taxa 0,00003, 3.000 épocas | **27.347** |
| `2 → 8 Tanh → 1 Linear`, taxa 0,00003, 3.000 épocas | 36.165 |
| `2 → 8 Tanh → 4 Tanh → 1 Linear`, taxa 0,00001, 6.000 épocas | 52.749 |

### Resultado

| Seed da divisão | Anterior | Atual | Regressão linear |
|---:|---:|---:|---:|
| 1 | 65.217 | 40.335 | 37.014 |
| 7 | 42.049 | 29.789 | 33.075 |
| 42 | 58.473 | **24.231** | 34.902 |
| 123 | 59.764 | 42.473 | 45.601 |

Na configuração padrão, o MSE de teste caiu de 58.473 para 24.231 (kg/ha)², ou seja, o erro típico (RMSE) passou de cerca de 242 para 156 kg/ha. O ruído entre repetições da mesma combinação de entradas é de cerca de 20.700 (kg/ha)², o que limita o MSE possível a esse patamar. Com apenas 28 exemplos de teste, os resultados variam bastante entre divisões.
