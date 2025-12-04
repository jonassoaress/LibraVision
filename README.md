# LibraVision: Reconhecimento de Libras em Tempo Real

## Visão Geral

**LibraVision** é um projeto de Visão Computacional e Inteligência Artificial que utiliza a câmera do computador para reconhecer gestos do alfabeto da Língua Brasileira de Sinais (Libras) e os traduz para texto em tempo real na tela.

O projeto implementa um pipeline completo de Machine Learning, desde a coleta de dados até o reconhecimento em tempo real, utilizando MediaPipe para detecção de mãos e Random Forest para classificação.

## Tecnologias Utilizadas

| Componente              | Tecnologia                       | Descrição                                    |
|-------------------------|----------------------------------|----------------------------------------------|
| Detecção de mãos        | **MediaPipe Hands**              | Detecção e rastreamento de até 2 mãos        |
| Captura de câmera       | **OpenCV**                       | Processamento de vídeo e interface visual    |
| Processamento de dados  | **NumPy / Pandas**               | Manipulação e análise de dados               |
| Modelo de classificação | **Scikit-learn (Random Forest)** | Classificação com otimização de hiperparâmetros |
| Visualização            | **Matplotlib / Seaborn**         | Geração de gráficos e matriz de confusão     |
| Linguagem               | **Python 3.8+**                  | Linguagem base do projeto                    |

## Estrutura do Projeto

```
LibraVision/
│
├── data/
│   └── libras_data.csv          # Dataset com 63 features + handedness
│
├── models/
│   ├── libras_model.pkl         # Modelo Random Forest treinado
│   └── confusion_matrix.png     # Visualização da matriz de confusão
│
├── scripts/
│   ├── 1_collect_data.py        # Coleta de dados via webcam
│   ├── 2_train_model.py         # Treinamento com feature engineering
│   ├── 3_test_model.py          # Avaliação e métricas do modelo
│   └── 4_real_time_app.py       # Aplicação de reconhecimento em tempo real
│
├── requirements.txt             # Dependências do projeto
├── CLAUDE.md                    # Documentação técnica para IA
└── README.md                    # Este arquivo
```

## Como Executar o Projeto

### 1. Pré-requisitos

- Python 3.8 ou superior
- Webcam funcional
- 1-2 GB de espaço livre em disco

### 2. Instalação

**a. Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/LibraVision.git
cd LibraVision
```

**b. (Recomendado) Crie e ative um ambiente virtual:**

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

### 3. Fluxo de Execução Completo

Os scripts devem ser executados sequencialmente na ordem numérica:

---

## Etapa 1: Coleta de Dados

**Comando:**
```bash
python scripts/1_collect_data.py
```

### Funcionalidades
- Coleta 100 amostras por letra para 27 letras (A-Z + Ç)
- Suporta até **2 mãos simultaneamente** (esquerda e direita)
- Modo **append**: mantém dados existentes no CSV
- 10 segundos de preparação antes de cada letra

### Como Usar
1. Posicione-se em frente à câmera
2. Aguarde a contagem regressiva
3. Mostre o gesto da letra atual
4. O programa coletará automaticamente as amostras

**Saída:** `data/libras_data.csv` com 63 features (21 landmarks × 3 coordenadas) + label + handedness

### Exemplo de código-chave:

```python
# Suporte a múltiplas mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detecta até 2 mãos
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Coleta com identificação de mão
for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
    hand_label = results.multi_handedness[hand_idx].classification[0].label
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    letter_data.append([letter, hand_label] + landmarks.flatten().tolist())
```

---

## Etapa 2: Treinamento do Modelo

**Comando:**
```bash
python scripts/2_train_model.py
```

### Funcionalidades
- **Feature Engineering**: Normalização relativa ao pulso (landmark 0)
- **Validação Cruzada**: 5-fold cross-validation
- **Otimização de Hiperparâmetros**: GridSearchCV com múltiplos parâmetros
- **Encoding de Handedness**: Left=0, Right=1

### Pipeline de Processamento

```python
# Normalização relativa ao pulso
for index, row in df.iterrows():
    label = row["label"]
    hand = row["hand"]
    landmarks = row.drop(["label", "hand"]).values.reshape(21, 3)
    
    wrist_coords = landmarks[0]
    relative_landmarks = landmarks - wrist_coords  # Invariância de translação
    
    processed_data.append([label, hand] + relative_landmarks.flatten().tolist())
```

### Grid Search de Hiperparâmetros

```python
param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [15, 20, 25, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
```

**Saída:** 
- `models/libras_model.pkl` (modelo otimizado)
- Relatório de precisão e melhores hiperparâmetros no terminal

---

## Etapa 3: Avaliação do Modelo (Opcional)

**Comando:**
```bash
python scripts/3_test_model.py
```

### Funcionalidades
- Relatório completo de classificação (Precision, Recall, F1-Score)
- Matriz de confusão com heatmap visual
- Avaliação no conjunto de teste (20% dos dados)

### Métricas Geradas

```python
# Relatório de classificação
print(classification_report(y_test, y_pred))

# Matriz de confusão visual
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=model.classes_, yticklabels=model.classes_)
```

**Saída:**
- `models/confusion_matrix.png` (gráfico salvo)
- Relatório detalhado no terminal

---

## Etapa 4: Aplicação em Tempo Real

**Comando:**
```bash
python scripts/4_real_time_app.py
```

### Funcionalidades Principais

#### 1. Reconhecimento Dual-Hand
- Detecta e reconhece **mão esquerda e direita simultaneamente**
- Exibe predições separadas para cada mão

#### 2. Sistema de Suavização
- **Buffer de predições**: Armazena últimas 10 predições por mão
- **Threshold de confiança**: 80% (configurável em `CONFIDENCE_THRESHOLD`)
- **Consenso de estabilidade**: 70% das predições devem concordar

```python
PREDICTION_BUFFER_SIZE = 10
CONFIDENCE_THRESHOLD = 0.8

# Sistema de suavização por mão
if confidence >= CONFIDENCE_THRESHOLD:
    prediction_buffer.append(predicted_letter)

if len(prediction_buffer) == PREDICTION_BUFFER_SIZE:
    counter = Counter(prediction_buffer)
    most_common, count = counter.most_common(1)[0]
    if count > PREDICTION_BUFFER_SIZE * 0.7:  # 70% de consenso
        stable_prediction = most_common
```

#### 3. Modo Tela Cheia
- Pressione **'f'** para alternar entre modo janela e tela cheia
- Resolução HD (1280x720) para melhor visualização

```python
# Configuração de tela cheia
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if key == ord("f"):
    is_fullscreen = not is_fullscreen
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                         cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
```

#### 4. Interface Visual
- **Predição instável**: Amarelo com confiança em tempo real
- **Predição estável**: Verde (esquerda) / Amarelo-ciano (direita) em caixa destacada
- **Visualização de landmarks**: Pontos e conexões da mão

### Controles
- **'q'**: Sair da aplicação
- **'f'**: Alternar tela cheia

---

## Arquitetura Técnica

### Pipeline de Dados

```
Webcam → MediaPipe Hands → Landmarks (21×3) → Normalização → Features (63) 
  → Hand Encoding → Random Forest → Predição → Buffer Smoothing → Display
```

### Feature Engineering

O modelo utiliza **coordenadas relativas ao pulso** ao invés de coordenadas absolutas:

1. **Input bruto**: 21 landmarks × 3 coordenadas (x, y, z) = 63 features
2. **Normalização**: `relative_landmark[i] = landmark[i] - landmark[0]` (pulso)
3. **Resultado**: Invariância de translação e melhor generalização

**Localização no código:**
- Treinamento: `scripts/2_train_model.py:42-54`
- Inferência: `scripts/4_real_time_app.py:66-70`

### Configurações do MediaPipe

```python
# Coleta de dados (mais permissivo)
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Aplicação em tempo real (mais rigoroso)
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
```

---

## Solução de Problemas

### Câmera não abre
- Verifique as permissões de acesso à webcam
- Confirme que nenhum outro aplicativo está usando a câmera
- Tente alterar o índice em `cv2.VideoCapture(0)` para `1` ou `2`

### Modelo não encontrado
```
ERRO: Modelo não encontrado em models/libras_model.pkl
```
**Solução:** Execute `python scripts/2_train_model.py` primeiro

### Baixa acurácia no reconhecimento
- Certifique-se de ter boa iluminação
- Mantenha a mão claramente visível na câmera
- Treine o modelo com suas próprias mãos (cada pessoa tem características únicas)
- Aumente `NUM_SAMPLES` em `1_collect_data.py` para coletar mais dados (recomendado: 100+)

### Dataset muito pequeno
```
AVISO: Dataset muito pequeno (150 amostras).
```
**Solução:** Colete pelo menos 500 amostras (execute `1_collect_data.py` múltiplas vezes com modo append)

---

## Melhorias Futuras

- [ ] Suporte a gestos dinâmicos (movimentos)
- [ ] Reconhecimento de palavras completas
- [ ] Interface gráfica com Tkinter/PyQt
- [ ] Exportação de texto para arquivo
- [ ] Suporte a mais idiomas de sinais (ASL, BSL, etc.)
- [ ] Modelo de Deep Learning (LSTM/CNN)

---

## Licença

Este projeto é destinado para fins **acadêmicos e educacionais**.

---

## Contribuições

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

---

## Autores

Desenvolvido como projeto acadêmico para demonstração de conceitos de Visão Computacional e Machine Learning.

---

## Agradecimentos

- **MediaPipe** (Google) pela biblioteca de detecção de mãos
- **OpenCV** pela infraestrutura de visão computacional
- **Scikit-learn** pelo framework de Machine Learning
- Comunidade surda brasileira pelo desenvolvimento e preservação da Libras
