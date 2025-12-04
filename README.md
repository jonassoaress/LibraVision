# LibraVision: Reconhecimento de Libras em Tempo Real

## 📋 Visão Geral

**LibraVision** é um projeto de Visão Computacional e Inteligência Artificial que utiliza a câmera do computador para reconhecer gestos do alfabeto da Língua Brasileira de Sinais (Libras) e os traduz para texto em tempo real na tela.

O sistema captura imagens da webcam, detecta a mão usando MediaPipe, extrai 21 pontos de referência em 3D, normaliza os dados e classifica o gesto usando um modelo de Machine Learning (Random Forest), exibindo a letra correspondente com confiança e suavização para evitar oscilações.

---

## 🔬 Como Funciona o Projeto

### Arquitetura e Pipeline

O LibraVision funciona em 4 etapas principais:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   1. COLETA     │ --> │  2. TREINAMENTO │ --> │    3. TESTE     │ --> │  4. TEMPO REAL  │
│   DE DADOS      │     │    DO MODELO    │     │   (OPCIONAL)    │     │   (PRODUÇÃO)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

### 1️⃣ Coleta de Dados (`1_collect_data.py`)

**Objetivo:** Capturar exemplos de gestos de Libras para treinar o modelo.

**Como funciona:**
1. Abre a webcam usando **OpenCV**
2. Usa **MediaPipe Hands** para detectar a mão na imagem
3. Extrai **21 pontos de referência (landmarks)** da mão em coordenadas 3D (x, y, z):
   - Ponto 0: Pulso
   - Pontos 1-4: Polegar
   - Pontos 5-8: Indicador
   - Pontos 9-12: Dedo médio
   - Pontos 13-16: Anelar
   - Pontos 17-20: Mindinho

4. Para cada letra do alfabeto, captura múltiplas amostras do gesto
5. Salva os dados em `data/libras_data.csv` com 64 colunas:
   - `label`: letra (A-Z)
   - `hand`: mão detectada (Left/Right)
   - 63 features: 21 pontos × 3 coordenadas (x, y, z)

**Dados coletados:**
```csv
label,hand,0_x,0_y,0_z,1_x,1_y,1_z,...,20_x,20_y,20_z
A,Right,0.5,0.6,0.1,0.52,0.58,0.09,...
```

---

### 2️⃣ Treinamento do Modelo (`2_train_model.py`)

**Objetivo:** Treinar um modelo de classificação para reconhecer as letras.

**Passos detalhados:**

#### 📊 **Pré-processamento: Normalização Relativa ao Pulso**

Para tornar o modelo **invariante à posição** da mão na tela:

1. Para cada amostra, pegamos as coordenadas do **ponto 0 (pulso)**:
   ```python
   wrist_coords = landmarks[0]  # [x_pulso, y_pulso, z_pulso]
   ```

2. **Subtraímos** as coordenadas do pulso de **todos os 21 pontos**:
   ```python
   relative_landmarks = landmarks - wrist_coords
   ```

3. Isso cria um sistema de coordenadas **relativo ao pulso**, fazendo com que:
   - O gesto "A" seja o mesmo independentemente de onde a mão está na tela
   - Reduz variação desnecessária nos dados
   - Melhora significativamente a acurácia do modelo

#### 🤖 **Modelo: Random Forest Classifier**

Configuração otimizada:
```python
RandomForestClassifier(
    n_estimators=150,      # 150 árvores de decisão
    max_depth=20,          # Profundidade máxima de 20
    min_samples_leaf=5,    # Mínimo 5 amostras por folha
    random_state=42        # Reprodutibilidade
)
```

**Por que Random Forest?**
- ✅ Alta precisão em classificação multiclasse
- ✅ Resistente a overfitting
- ✅ Rápido na inferência (importante para tempo real)
- ✅ Não requer normalização adicional
- ✅ Fornece probabilidades de classe (`predict_proba`)

#### 📈 **Divisão dos Dados**

- **80% Treino** / **20% Teste**
- Usa `stratify=y` para manter proporção de classes balanceada
- Avalia com `accuracy_score`

#### 💾 **Saída**

- Modelo salvo em: `models/libras_model.pkl`
- Precisão típica: **>90%** (depende da qualidade dos dados coletados)

---

### 3️⃣ Teste do Modelo (`3_test_model.py`)

**Objetivo:** Avaliar o desempenho do modelo treinado.

**Métricas calculadas:**

1. **Classification Report:**
   - Precision (precisão por classe)
   - Recall (revocação por classe)
   - F1-score (média harmônica)
   - Support (quantidade de amostras)

2. **Confusion Matrix:**
   - Matriz visual mostrando acertos e erros
   - Salva como imagem em `models/confusion_matrix.png`

3. **Accuracy Score:**
   - Acurácia geral do modelo

---

### 4️⃣ Aplicação em Tempo Real (`4_real_time_app.py`)

**Objetivo:** Reconhecer gestos ao vivo via webcam.

#### 🔄 **Loop Principal**

```
┌──────────────────────────────────────────────────────────┐
│  1. Captura frame da webcam (OpenCV)                     │
│  2. Converte BGR → RGB                                   │
│  3. Detecta mão com MediaPipe                            │
│  4. Extrai 21 landmarks (x, y, z)                        │
│  5. Normaliza: subtrai coordenadas do pulso              │
│  6. Passa pelo modelo Random Forest                      │
│  7. Recebe probabilidades de cada classe                 │
│  8. Sistema de suavização (buffer)                       │
│  9. Exibe resultado na tela                              │
│  10. Repete (loop)                                       │
└──────────────────────────────────────────────────────────┘
```

#### 🎯 **Sistema de Suavização Inteligente**

**Problema:** Sem suavização, a previsão oscila rapidamente entre letras (instabilidade).

**Solução:** Sistema de buffer com votação por maioria

```python
PREDICTION_BUFFER_SIZE = 10      # Armazena últimas 10 previsões
CONFIDENCE_THRESHOLD = 0.8       # Só aceita previsões com 80%+ confiança
```

**Como funciona:**

1. Modelo faz previsão e retorna probabilidades:
   ```python
   prediction_probs = model.predict_proba([landmarks])
   confidence = np.max(prediction_probs)  # Maior probabilidade
   predicted_letter = model.classes_[np.argmax(prediction_probs)]
   ```

2. **Filtro de confiança:** Só adiciona ao buffer se `confidence >= 0.8`

3. **Buffer de previsões:** Mantém as últimas 10 previsões aceitas
   ```python
   prediction_buffer = deque(maxlen=10)
   prediction_buffer.append(predicted_letter)
   ```

4. **Votação por maioria:** A letra estável é a que mais aparece no buffer
   ```python
   most_common = max(set(prediction_buffer), key=prediction_buffer.count)
   if prediction_buffer.count(most_common) > 7:  # 70% do buffer
       stable_prediction = most_common
   ```

**Resultado:**
- ✅ Elimina oscilações rápidas
- ✅ Só muda a letra exibida quando há consistência
- ✅ Melhor experiência do usuário

#### 🖥️ **Interface Visual**

- **Linha superior:** Previsão instantânea + confiança (atualiza rápido)
  - `Pred: A (0.92)` - mostra a letra e confiança atual

- **Linha inferior (destaque):** Letra estável (muda apenas com consistência)
  - `Letra Estavel: A` - resultado suavizado

- **Desenho da mão:** 21 pontos + conexões desenhados sobre a imagem

---

## 🛠️ Tecnologias Utilizadas

| Componente              | Tecnologia                       | Versão/Detalhes                    |
|-------------------------|----------------------------------|------------------------------------|
| **Detecção da mão**     | MediaPipe Hands                  | 21 landmarks em 3D                 |
| **Captura de vídeo**    | OpenCV (cv2)                     | Webcam + processamento de imagem   |
| **Processamento**       | NumPy / Pandas                   | Manipulação de arrays e DataFrames |
| **Modelo de IA**        | Scikit-learn (Random Forest)     | 150 estimadores, depth=20          |
| **Persistência**        | Joblib                           | Serialização do modelo             |
| **Visualização**        | Matplotlib / Seaborn             | Gráficos e matriz de confusão      |
| **Linguagem**           | Python                           | 3.8+                               |

---

## 📁 Estrutura de Pastas

```
LibraVision/
│
├── data/                          # 📊 Dataset coletado
│   └── libras_data.csv            # Dados dos gestos (gerado no passo 1)
│
├── models/                        # 🤖 Modelos treinados
│   ├── libras_model.pkl           # Modelo Random Forest (gerado no passo 2)
│   └── confusion_matrix.png       # Matriz de confusão (gerado no passo 3)
│
├── scripts/                       # 🐍 Scripts Python
│   ├── 1_collect_data.py          # Coleta de dados via webcam
│   ├── 2_train_model.py           # Treinamento do modelo
│   ├── 3_test_model.py            # Avaliação do modelo
│   └── 4_real_time_app.py         # Aplicação em tempo real
│
├── requirements.txt               # 📦 Dependências do projeto
├── .gitignore                     # 🚫 Arquivos ignorados pelo Git
└── README.md                      # 📖 Esta documentação
```

**Nota:** As pastas `data/` e `models/` são criadas automaticamente ao executar os scripts.

---

## 🚀 Como Executar o Projeto

### 1. Pré-Requisitos

- **Python 3.8 ou superior**
- **Webcam** funcional
- **Sistema operacional:** Windows, macOS ou Linux

---

### 2. Instalação

#### **a. Clone o repositório**

```bash
git clone https://github.com/jonassoaress/LibraVision.git
cd LibraVision
```

#### **b. (Recomendado) Crie e ative um ambiente virtual**

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### **c. Instale as dependências**

```bash
pip install -r requirements.txt
```

---

### 3. Fluxo de Execução

**Importante:** Execute os scripts na ordem numérica!

---

#### **Etapa 1: Coletar Dados** 📸

```bash
python scripts/1_collect_data.py
```

**O que fazer:**
- Siga as instruções no terminal
- Para cada letra (A-Z), faça o gesto correspondente em Libras
- Mantenha a mão estável enquanto os dados são coletados
- O programa captura múltiplas amostras de cada letra
- Os dados são salvos em `data/libras_data.csv`

**Dicas:**
- ✅ Use boa iluminação
- ✅ Mantenha o fundo limpo (sem outras mãos ou objetos)
- ✅ Varie levemente a posição/ângulo da mão entre amostras
- ✅ Colete pelo menos 100 amostras por letra para melhor precisão

---

#### **Etapa 2: Treinar o Modelo** 🤖

```bash
python scripts/2_train_model.py
```

**O que acontece:**
- Carrega os dados de `data/libras_data.csv`
- Aplica normalização (coordenadas relativas ao pulso)
- Divide em treino (80%) e teste (20%)
- Treina o Random Forest com 150 árvores
- Exibe a **precisão do modelo** no terminal
- Salva o modelo treinado em `models/libras_model.pkl`

**Saída esperada:**
```
Carregando o dataset...
Realizando engenharia de features (normalização)...
Dividindo os dados em treino e teste...
Treinando o modelo Random Forest...
Avaliando a precisão do modelo...
Precisão do modelo: 94.32%
Salvando o modelo em models/libras_model.pkl...
Treinamento concluído e modelo salvo com sucesso.
```

---

#### **Etapa 3 (Opcional): Avaliar o Modelo** 📊

```bash
python scripts/3_test_model.py
```

**O que acontece:**
- Carrega o modelo treinado
- Avalia no conjunto de teste
- Exibe **Classification Report** no terminal
- Gera e salva **Confusion Matrix** em `models/confusion_matrix.png`

**Exemplo de saída:**
```
              precision    recall  f1-score   support

           A       0.96      0.94      0.95        50
           B       0.92      0.95      0.93        48
           C       0.94      0.91      0.92        52
         ...
    accuracy                           0.94      1200
   macro avg       0.94      0.94      0.94      1200
weighted avg       0.94      0.94      0.94      1200
```

---

#### **Etapa 4: Executar a Aplicação** 🎥

```bash
python scripts/4_real_time_app.py
```

**O que acontece:**
- Abre a webcam
- Detecta sua mão em tempo real
- Desenha os 21 pontos sobre a mão
- Exibe:
  - **Pred:** Previsão instantânea com confiança
  - **Letra Estavel:** Resultado suavizado (muda apenas com consistência)

**Controles:**
- Pressione **'q'** para sair

**Dicas para melhor reconhecimento:**
- ✅ Posicione a mão no centro da tela
- ✅ Mantenha boa iluminação
- ✅ Faça o gesto de forma clara e estável
- ✅ Aguarde alguns frames para a "Letra Estavel" aparecer

---

## 🔧 Configurações Avançadas

### Ajustar Parâmetros do Modelo (`2_train_model.py`)

```python
model = RandomForestClassifier(
    n_estimators=150,      # ↑ Aumentar = mais precisão, mais lento
    max_depth=20,          # ↑ Aumentar = mais complexo, risco de overfit
    min_samples_leaf=5,    # ↓ Diminuir = mais complexo
    random_state=42
)
```

### Ajustar Suavização (`4_real_time_app.py`)

```python
PREDICTION_BUFFER_SIZE = 10      # ↑ Aumentar = mais suave, mais lento
CONFIDENCE_THRESHOLD = 0.8       # ↑ Aumentar = mais restritivo
```

### Ajustar Sensibilidade do MediaPipe (`4_real_time_app.py`)

```python
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,   # ↑ Aumentar = menos falsos positivos
    min_tracking_confidence=0.7     # ↑ Aumentar = rastreamento mais estável
)
```

---

## ❓ Troubleshooting

### **Erro: "Webcam não encontrada"**
- Verifique se há outra aplicação usando a webcam
- Tente trocar `cv2.VideoCapture(0)` para `cv2.VideoCapture(1)` em `4_real_time_app.py`

### **Erro: "libras_model.pkl não encontrado"**
- Execute o passo 2 (`2_train_model.py`) antes do passo 4

### **Erro: "libras_data.csv não encontrado"**
- Execute o passo 1 (`1_collect_data.py`) antes do passo 2

### **Precisão muito baixa (<80%)**
- Colete mais dados (>100 amostras por letra)
- Verifique se os gestos foram feitos corretamente
- Use melhor iluminação durante a coleta
- Varie a posição/ângulo da mão nas amostras

### **Letra oscila muito na tela**
- Aumente `PREDICTION_BUFFER_SIZE` (ex: 15)
- Aumente `CONFIDENCE_THRESHOLD` (ex: 0.85)

### **Mão não é detectada**
- Diminua `min_detection_confidence` para 0.5
- Melhore a iluminação
- Certifique-se que a mão está visível e aberta

---

## 📊 Resultados Esperados

Com **100+ amostras por letra** e **boa qualidade de dados**:

- ✅ **Acurácia do modelo:** 90-95%
- ✅ **FPS da aplicação:** 20-30 frames/segundo
- ✅ **Latência de reconhecimento:** <1 segundo (com suavização)
- ✅ **Taxa de falsos positivos:** <5%

---

## 🎯 Próximos Passos / Melhorias Futuras

- [ ] Adicionar suporte para **palavras e frases** (não apenas letras)
- [ ] Implementar **deep learning** (CNN/LSTM) para maior precisão
- [ ] Criar **interface gráfica** (Tkinter/PyQt)
- [ ] Desenvolver **aplicativo mobile** (Android/iOS)
- [ ] Adicionar **reconhecimento de gestos dinâmicos** (movimentos)
- [ ] Implementar **dataset público** para treino
- [ ] Adicionar **suporte multilíngue** (ASL, BSL, etc.)
- [ ] Otimizar para **edge devices** (Raspberry Pi, Jetson Nano)

---

## 👥 Contribuições

Contribuições são bem-vindas! Sinta-se livre para:
- Reportar bugs
- Sugerir novas funcionalidades
- Melhorar a documentação
- Enviar pull requests

---

## 📄 Licença

Este projeto é para **fins acadêmicos e educacionais**.

---

## 📧 Contato

- **Repositório:** [github.com/jonassoaress/LibraVision](https://github.com/jonassoaress/LibraVision)
- **Desenvolvedor:** Jonas Soares

---

## 🙏 Agradecimentos

- **MediaPipe** (Google) - Framework de detecção de mãos
- **OpenCV** - Biblioteca de Visão Computacional
- **Scikit-learn** - Framework de Machine Learning
- Comunidade surda brasileira pela importância da Libras

---

**⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!**
