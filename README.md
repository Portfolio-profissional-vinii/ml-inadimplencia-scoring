# Score de Inadimplência — Sistema de Machine Learning

## Visão Geral e Problema de Negócio

Desenvolvimento de uma solução de Machine Learning com práticas de MLOps para previsão do risco de inadimplência em concessão de crédito de varejo.

O objetivo é classificar solicitações de empréstimo e calcular a probabilidade de inadimplência antes da aprovação do contrato, permitindo que a instituição financeira reduza perdas operacionais sem comprometer a esteira de aprovação.

## Resultados e Métricas

- **AUC-ROC (Validação Realista):** 0.74, obtido após sanitização rigorosa de Data Leakage, eliminando variáveis pós-concessão.
- **Tempo de Predição (SLA):** < 50 ms por requisição via API REST.
- **Regra de Decisão:** corte automatizado de aprovação com threshold de probabilidade de inadimplência em 40%.

## Estrutura do Repositório

```text
.
├── api/
│   └── main.py              # Aplicação FastAPI e rotas de inferência
├── data/
│   ├── raw/                 # Dados brutos
│   └── processed/           # Dataset tratado
├── docker/
│   └── Dockerfile           # Instruções para conteinerização da aplicação
├── models/
│   └── model_v1.pkl         # Artefato serializado do modelo (Random Forest)
├── monitoring/
│   └── check_drift.py       # Script de teste Kolmogorov-Smirnov para Data Drift
├── src/
│   ├── features.py          # Limpeza, codificação e tratamento de Data Leakage
│   ├── model.py             # Instanciação, treino e avaliação do Random Forest
│   └── pipeline.py          # Orquestrador do fluxo de treinamento
├── tests/
│   └── test_model.py        # Testes de integração da API e modelo (Pytest)
├── requirements.txt         # Dependências do projeto
└── README.md
```

## Arquitetura do Sistema

### Engenharia de Recursos e Proteção contra Leakage

**`src/features.py`**

Responsável pela limpeza dos dados, codificação das variáveis e proteção contra Data Leakage.

São excluídas explicitamente variáveis que representam informações posteriores ao momento da concessão, como:

- `last_fico_range`
- `debt_settlement_flag`
- informações relacionadas a cobranças

Também é aplicado **One-Hot Encoding** para tratamento das variáveis categóricas.

### Pipeline de Treinamento

**`src/pipeline.py`**

Orquestra o processo de treinamento do modelo utilizando divisão estratificada dos dados.

O pipeline gera o artefato:

```text
models/model_v1.pkl
```

O modelo utilizado é um **Random Forest**.

### Serviço de Predição

**`api/main.py`**

Implementa uma API REST utilizando **FastAPI**, com validação dos esquemas de entrada através do **Pydantic**.

A API recebe os dados da solicitação de crédito e retorna a previsão de risco de inadimplência.

### Garantia de Qualidade

**`tests/test_model.py`**

Suíte de testes automatizados utilizando:

- Pytest
- HTTPX

Os testes cobrem a integração entre a API e o modelo de Machine Learning.

### Conteinerização

**`docker/Dockerfile`**

Define uma imagem isolada baseada em **Python 3.11-slim**, garantindo maior padronização entre os ambientes de desenvolvimento e produção.

### Monitoramento Contínuo

**`monitoring/check_drift.py`**

Implementa um teste estatístico de **Kolmogorov-Smirnov (KS-test)** para identificar possíveis mudanças na distribuição dos dados de entrada (**Data Drift**).

## Como Rodar o Projeto

### Pré-requisito: Download do Dataset

Os dados **não estão incluídos** neste repositório (a pasta `data/` está no `.gitignore` por conter arquivos grandes).

1. **Baixe o dataset original do Kaggle:**

   👉 [Lending Club — accepted_2007_to_2018Q4.csv](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

   > É necessário ter uma conta no Kaggle. Após o login, clique em **Download** na página do dataset.

2. **Coloque o arquivo baixado na pasta correta do projeto:**

   ```text
   ml-inadimplencia-scoring/
   └── data/
       └── raw/
           └── accepted_2007_to_2018Q4.csv   ← coloque aqui
   ```

3. **Rode o notebook de exploração para gerar a base limpa automaticamente:**

   ```bash
   jupyter notebook notebooks/01_exploracao.ipynb
   ```

   Execute todas as células — o notebook irá processar o arquivo bruto e salvar o dataset tratado em:

   ```text
   data/processed/dados_tratados_model.csv
   ```

   Com a base processada em mãos, todos os demais passos (treinamento, API, testes) funcionarão normalmente.

---

## Como Executar


### 1. Configuração do Ambiente e Dependências

Crie um ambiente virtual:

```bash
python -m venv .venv
```

#### Windows

```bash
.\.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

### 2. Execução do Pipeline de Treinamento

```bash
python -m src.pipeline
```

### 3. Execução da API Localmente

```bash
uvicorn api.main:app --reload
```

Após iniciar a aplicação, acesse a documentação interativa do Swagger:

```text
http://127.0.0.1:8000/docs
```

### 4. Execução dos Testes Automatizados

```bash
python -m pytest
```

### 5. Execução via Docker

Construa a imagem:

```bash
docker build -t credit-scoring-api -f docker/Dockerfile .
```

Execute o container:

```bash
docker run -p 8000:8000 credit-scoring-api
```

A API ficará disponível em:

```text
http://127.0.0.1:8000
```

### 6. Monitoramento de Data Drift

```bash
python monitoring/check_drift.py
```

## Stack Tecnológica

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python |
| Machine Learning | scikit-learn — Random Forest |
| Manipulação de Dados | pandas, numpy |
| Estatística | scipy |
| API | FastAPI |
| Validação | Pydantic |
| Servidor ASGI | Uvicorn |
| Testes | Pytest, HTTPX |
| Conteinerização | Docker |
| Serialização | Joblib |
| Monitoramento | Kolmogorov-Smirnov Test |

## Objetivos Técnicos

Este projeto demonstra um fluxo completo de Machine Learning aplicado a um problema de negócio, contemplando:

1. Preparação e tratamento dos dados.
2. Prevenção de Data Leakage.
3. Engenharia de features.
4. Treinamento e avaliação de modelo.
5. Serialização do modelo.
6. Exposição do modelo através de API REST.
7. Validação de entradas com Pydantic.
8. Testes automatizados.
9. Conteinerização com Docker.
10. Monitoramento de Data Drift.
11. Definição de threshold para tomada de decisão.

## Fluxo da Solução

```text
Dados Brutos
     │
     ▼
Limpeza e Feature Engineering
     │
     ▼
Remoção de Data Leakage
     │
     ▼
Treinamento Random Forest
     │
     ▼
Avaliação do Modelo
     │
     ▼
model_v1.pkl
     │
     ▼
FastAPI
     │
     ▼
Predição de Risco
     │
     ▼
Threshold de 40%
     │
     ├── Risco abaixo do threshold → Aprovação
     │
     └── Risco igual/acima do threshold → Reprovação / análise
     │
     ▼
Monitoramento de Data Drift
```

## Métrica Principal

A principal métrica utilizada na avaliação é a **AUC-ROC**, com resultado de:

**0.74**

O resultado foi obtido após a remoção de variáveis que poderiam introduzir informações futuras no momento da concessão do crédito, tornando a validação mais realista para um cenário de produção.

## Licença

Este projeto é destinado a fins de estudo, portfólio e demonstração técnica.
