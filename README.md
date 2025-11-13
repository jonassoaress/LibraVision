# LibraVision: Reconhecimento de Libras

## Visão Geral:

**LibraVision** é um projeto de Visão Computacional e Inteligência Artificial que utiliza câmera do computador para reconhecer gestos do alfabeto da Língua Brasileira de Sinais (Libras) e os traduzo para texto em tempo real na tela.

## Tecnologias Utilizadas

| Parte                   | Tecnologia                       |
|-------------------------|----------------------------------|
| Detecção da mão         | **MediaPipe Hands**              |
| Captura da câmera       | **OpenCV**                       |
| Processamento de dados  | **NumPy / Pandas**               |
| Modelo de classificação | **Scikit-learn (Random Forest)** |
| Interface               | **OpenCV**                       |
| Linguagem               | **Python**                       |

## Estrutura de Pastas

```
Libras-IA/
│
├── data/
├── models/
├── scripts/
│ ├── 1_collect_data.py
│ ├── 2_train_model.py
│ ├── 3_test_model.py
│ └── 4_real_time_app.py
│
├── requirements.txt
└── README.md
```

* `data/`: Armazena o dataset coletado (`libras_data.csv`)
* `models/`: Contém o modelo treinado (`libras_model.pkl`) e a matriz de confusão
* `1_collect_data.py`: Script para coletar os dados dos gestos
* `2_train_model.py`: Script para treinar o modelo de IA
* `3_test_model.py`: Script para avaliar o desempenho do modelo
* `4_real_time_app.py`: Aplicação principal de reconhecimento em tempo real

## Como executar o projeto

### 1. Pré-Requisitos

- Python 3.8 ou superior
- 1 webcam

### 2. Instalação

**a. Clone o repositório:**
```bash
git clone https://[URL-DO-SEU-REPOSITORIO]/Libras-IA.git
cd Libras-IA
```

**b. (Recomendado) Crie e ative um ambiente virtual

```bash
# Para Windows
python -m venv venv
.\venv\Scripts\activate

# Para macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**c. Instale as dependências:**

```bash
pip install -r requirements.txt
```

### 3. Fluxo de Execução

Siga os scripts na ordem numérica;

#### **Etapa 1: Coletar Dados**
Execute o script para capturar os dados dos seus próprios gestos. Siga as instruções no terminal e na janela da câmera

```bash
python scripts/1_collect_data.py
```

#### **Etapa 2: Treinar o Modelo**
Após coletar os dados, treine o modelo de Machine Learning. Ele salvará o arquivo `libras_model.pkl` na pasta `models/`

```bash
python scripts/2_train_model.py
```

#### **Etapa 3 (Opcional): Avaliar o modelo**
Para ver um relatório detalhado de performance, execute o script de teste

```bash
python scripts/3_test_model.py
```

#### **Etapa 4: Executar a Aplicação**
Inicie a aplicação em tempo real para ver a mágia acontecer! Aponte a mão para a câmera e veja a letra sendo reconhecida

```bash
python scripts/4_real_time_app.py
```

Pressione a tecla **'q'** na janela da câmera para fechar a aplicação