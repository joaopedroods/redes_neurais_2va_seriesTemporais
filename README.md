# Quanto o histórico importa?

**Uma comparação de cinco arquiteturas de previsão de séries temporais sob perturbações da janela de entrada**

Projeto da disciplina de Redes Neurais (UFRPE, 2026.2), com o prof. Valmir Macário.
Equipe: Giovanna Neves de Lima, João Pedro Oliveira da Silva e Letícia Barbosa de Farias.

- **Relatório:** [relatorio/relatorio.pdf](relatorio/relatorio.pdf)
- **Apresentação:** [relatorio/apresentacao.pdf](relatorio/apresentacao.pdf)

## O que foi feito

Comparamos DLinear, LSTM, TCN, Informer e PatchTST em duas bases de referência, ETTh1 e Weather, no cenário univariado e com horizonte de 96 passos. Medimos o erro e também o uso que cada modelo faz da janela de entrada. Todos os modelos passaram pelo mesmo protocolo:

- a mesma normalização por janela (RevIN), o mesmo laço de treino (Adam, MSE, lote 64, até 30 épocas, paciência 5) e o mesmo orçamento de ajuste: o learning rate escolhido pela validação entre 1e-3, 3e-4 e 1e-4;
- várias janelas de entrada (L de 96 a 720) e três seeds por configuração, num total de 140 treinamentos;
- a janela de cada modelo escolhida pela validação, nunca pelo teste;
- quatro testes de embaralhamento da janela, aplicados só na avaliação, com as mesmas permutações para todos os modelos;
- o teste de Diebold–Mariano com variância de Newey–West e correção de Holm.

## Resultados principais

MSE de teste (média ± desvio de 3 seeds) na janela escolhida pela validação:

| Modelo | ETTh1: L | ETTh1: MSE | Weather: L | Weather: MSE |
|---|---|---|---|---|
| DLinear | 512 | 0,0568 ± 0,0022 | 720 | **0,4024 ± 0,0016** |
| LSTM | 336 | 0,0546 ± 0,0014 | 336 | 0,5563 ± 0,0314 |
| TCN | 192 | 0,0547 ± 0,0004 | 336 | 0,4152 ± 0,0147 |
| Informer | 336 | 0,0688 ± 0,0055 | 96 | 0,4951 ± 0,0141 |
| PatchTST | 192 | **0,0541 ± 0,0007** | 336 | 0,5532 ± 0,0207 |
| naive (baseline) | — | 0,0693 | — | 0,5550 |

- Nenhuma arquitetura superou o DLinear com significância estatística.
- No ETTh1, DLinear, LSTM, TCN e PatchTST empatam, e os quatro superam o Informer. No Weather, DLinear e TCN empatam e superam LSTM, PatchTST e o melhor baseline.
- Todos os modelos dependem da ordem da janela, mas quase só da ordem do passado recente. Com janelas longas, o erro do Informer chega a 17 vezes o original quando a janela é embaralhada.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `00_LEIA-ME.ipynb` | instruções e ordem de execução |
| `01_dados.ipynb` | download, auditoria, limpeza, divisão e normalização das bases |
| `02_codigo_comum.ipynb` | gera `src/rn_utils.py`: janelas, RevIN, laço de treino, avaliação e embaralhamentos |
| `03_baselines.ipynb` | naive, sazonal e média da janela |
| `10_dlinear.ipynb` a `14_patchtst.ipynb` | um notebook por arquitetura |
| `90_analise.ipynb` | tabelas, figuras e testes estatísticos |
| `src/rn_utils.py` | biblioteca comum usada por todos os notebooks |
| `resultados/` | um JSON por execução (configuração, histórico por época, métricas) e os erros por amostra |
| `figuras/` | figuras do relatório |

Os notebooks ficam na raiz, como na pasta do Drive, porque o código usa a pasta do notebook como raiz do projeto quando roda fora do Colab. Eles estão salvos com as saídas da execução que gerou o relatório.

## Como reproduzir

Os notebooks rodam no Google Colab, com o projeto numa pasta `RN-Projeto` do Google Drive, ou localmente, a partir da raiz deste repositório. Rode na ordem 01, 02, 03, 10 a 14 e 90.

- As bases não estão no repositório: o notebook 01 baixa o [ETTh1](https://github.com/zhouhaoyi/ETDataset) e o [Weather](https://huggingface.co/datasets/thuml/Time-Series-Library) das fontes originais.
- Execuções que já têm resultado em `resultados/` são puladas. Para refazer só a análise, rode 01, 02 e 90.
- Os pesos treinados (`checkpoints/`) ficaram fora do repositório por causa do tamanho: só os do Informer passam de 1 GB.

Dependências principais: Python 3, PyTorch, NumPy, pandas, SciPy, statsmodels e Matplotlib.
