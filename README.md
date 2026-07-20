# GLiNER Partial Reproduction

Reprodução parcial e metodologicamente orientada do artigo **GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer** usando a biblioteca pública GLiNER e o dataset **CoNLL-2003**.

O foco deste repositório não é reproduzir exatamente os números do paper, e sim aproximar corretamente o pipeline de avaliação zero-shot:

- prompts de tipos de entidade em linguagem natural
- inferência span-based com GLiNER
- avaliação por *exact match* entre fronteira e tipo
- registro sistemático de métricas, tempos e contagens

## Limitações Declaradas

- O paper treina o modelo em **Pile-NER**. Aqui usamos **checkpoints públicos já treinados**.
- O CoNLL-2003 da biblioteca `datasets` expõe tokens e BIO tags, não o texto cru original. Por isso, o texto é reconstruído heurísticamente a partir dos tokens para alinhar a API pública do GLiNER, que retorna offsets por caractere.
- A decodificação final usa a implementação pública do GLiNER, então detalhes internos de treinamento e busca gulosa não são reimplementados do zero neste projeto.
- `spaCy` e `Flair` podem ser comparados no mesmo pipeline, mas não são equivalentes ao cenário open-type promptable do paper; entram apenas como baselines opcionais.

## Estrutura

```text
src/
  datasets/
    conll2003.py                # loader, reconstrução de texto e offsets
    explore_conll2003.py        # inspeção rápida do dataset
  evaluation/
    bio_converter.py            # spans <-> BIO e normalização de labels
    metrics.py                  # métricas de exact match + classification report
  experiments/
    common.py                   # pipeline reutilizável de experimento
    evaluate_conll2003.py       # experimento zero-shot principal
    experiment1_zero_shot.py    # alias do experimento zero-shot
    experiment2_label_prompt.py # estudo de prompt de labels
    experiment3_compare_models.py
    zero_shot.py                # inspeção qualitativa de uma sentença
  models/
    gliner_model.py             # wrapper da biblioteca pública GLiNER
  utils.py                      # paths, logging e escrita de resultados

results/
  zero_shot.csv
  label_prompt.csv
  compare_models.csv
  ood_benchmark_gliner.csv
  ood_open_source_compare.csv
  tweetner7_zero_shot.csv
  classification_report.txt
  experiments.log
```

## Ambiente

Use preferencialmente um ambiente limpo. Exemplo:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se quiser um `torch` CPU-only, é mais seguro instalar antes com o índice oficial da PyTorch e depois instalar o restante:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<2.5"
python -m pip install -r requirements.txt
```

### Baselines opcionais

`spaCy` e `Flair` não são instalados por padrão no `requirements.txt`, porque são opcionais no `experiment3_compare_models.py`.

Para habilitar:

```bash
python -m pip install spacy flair
python -m spacy download en_core_web_sm
```

## Dataset

O CoNLL-2003 é baixado automaticamente pela biblioteca `datasets` quando algum script chama `load_conll2003()`. O cache fica em `data/raw/`.

Para inspecionar o dataset:

```bash
python -m src.datasets.explore_conll2003
```

## Como Executar

### 1. Zero-shot principal

Executa o pipeline zero-shot usando o mesmo critério de avaliação do paper: *exact match* entre span e tipo.

```bash
python -m src.experiments.evaluate_conll2003 \
  --checkpoint urchade/gliner_large-v2.1 \
  --split test \
  --max-examples 100
```

Parâmetros principais:

- `--checkpoint`
- `--labels`
- `--dataset`
- `--split`
- `--max-examples`
- `--threshold`

Saídas:

- `results/zero_shot.csv`
- `results/classification_report.txt`
- `results/experiments.log`

### 2. Label prompt experiment

Compara quatro conjuntos de descrições das entidades:

1. `person organization location miscellaneous`
2. `PER ORG LOC MISC`
3. `human company country other entity`
4. `human being organization geographical location miscellaneous named entity`

Execução:

```bash
python -m src.experiments.experiment2_label_prompt \
  --checkpoint urchade/gliner_large-v2.1 \
  --split test \
  --max-examples 100
```

Saída:

- `results/label_prompt.csv`

### 3. Compare models

Compara os checkpoints GLiNER Large, Base e Small no mesmo pipeline.

```bash
python -m src.experiments.experiment3_compare_models \
  --split test \
  --max-examples 100
```

Com baselines opcionais:

```bash
python -m src.experiments.experiment3_compare_models \
  --split test \
  --max-examples 100 \
  --include-spacy \
  --include-flair
```

Saída:

- `results/compare_models.csv`

### 4. Inspeção qualitativa

```bash
python -m src.experiments.zero_shot
```

### 5. Benchmark OOD no estilo da Tabela 1

Reproduz localmente o benchmark `Movie / Restaurant / AI / Literature / Music / Politics / Science` com GLiNER e monta uma tabela comparativa com até dois baselines open-source usando os números reportados no paper.

```bash
python -m src.experiments.experiment4_ood_open_source_compare \
  --checkpoint urchade/gliner_large-v2.1 \
  --paper-baselines UniNER-7B GoLLIE
```

Saídas:

- `results/ood_benchmark_gliner.csv`
- `results/ood_open_source_compare.csv`

### 6. Zero-shot adicional em TweetNER7

```bash
python -m src.experiments.experiment5_tweetner7_zero_shot \
  --checkpoint urchade/gliner_large-v2.1
```

Saídas:

- `results/tweetner7_zero_shot.csv`
- `results/tweetner7_classification_report.txt`

## O Que Já Está Mais Próximo Do Paper

- O modelo recebe tipos de entidade como texto, não como cabeçalho fixo de um classificador supervisionado tradicional.
- A inferência é feita por spans via API pública do GLiNER.
- O limiar padrão segue `0.5`, em linha com a descrição do paper.
- A avaliação principal agora usa correspondência exata entre fronteira e tipo.

## O Que Continua Sendo Aproximação

- Não reproduzimos o treinamento do paper em Pile-NER.
- Não reconstruímos o texto cru original do CoNLL; usamos uma reconstrução determinística a partir dos tokens.
- Não reimplementamos os componentes internos do GLiNER, como encoder, span scorer e decoding guloso; usamos a biblioteca pública.

## Resultados

Os scripts salvam automaticamente:

- `results/zero_shot.csv`
- `results/label_prompt.csv`
- `results/compare_models.csv`
- `results/ood_benchmark_gliner.csv`
- `results/ood_open_source_compare.csv`
- `results/tweetner7_zero_shot.csv`
- `results/classification_report.txt`
- `results/experiments.log`

Cada linha de resultado registra, no mínimo:

- modelo
- dataset
- checkpoint
- split
- labels
- número de exemplos
- precision
- recall
- F1
- tempo total
- tempo médio
- número de entidades previstas
- número de entidades reais
