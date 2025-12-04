import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Caminho para o dataset e para carregar o modelo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "..", "data", "libras_data.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "libras_model.pkl")

# 1. Carregar os dados
print("Carregando o dataset...")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset carregado com sucesso: {len(df)} amostras encontradas.")
except FileNotFoundError:
    print(f"ERRO: Dataset não encontrado em: {DATA_FILE}")
    print(
        "Por favor, execute o script '1_collect_data.py' primeiro para coletar os dados."
    )
    exit(1)
except Exception as e:
    print(f"ERRO ao carregar o dataset: {e}")
    exit(1)

# 2. Engenharia de Features: Normalização (mesmo processo do treinamento)
print("Realizando engenharia de features (normalização)...")
processed_data = []
for index, row in df.iterrows():
    label = row['label']
    # Remover 'label' e 'hand' para obter apenas as coordenadas
    landmarks = row.drop(['label', 'hand']).values.reshape(21, 3)

    # Pega as coordenadas do pulso (ponto 0)
    wrist_coords = landmarks[0]

    # Subtrai as coordenadas do pulso de todos os outros pontos
    relative_landmarks = landmarks - wrist_coords

    # Achata a lista para o formato original e adiciona o label
    processed_data.append([label] + relative_landmarks.flatten().tolist())

# Cria um novo DataFrame com os dados processados
columns = ['label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
processed_df = pd.DataFrame(processed_data, columns=columns)

# 3. Preparar os dados
X = processed_df.drop('label', axis=1)
y = processed_df['label']

    # Pega as coordenadas do pulso (ponto 0)
    wrist_coords = landmarks[0]

    # Subtrai as coordenadas do pulso de todos os outros pontos
    relative_landmarks = landmarks - wrist_coords

    # Achata a lista para o formato original e adiciona o label
    processed_data.append([label] + relative_landmarks.flatten().tolist())

# Cria um novo DataFrame com os dados processados
columns = ["label"] + [f"{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
processed_df = pd.DataFrame(processed_data, columns=columns)

# 3. Preparar os dados
X = processed_df.drop("label", axis=1)
y = processed_df["label"]

# 4. Dividir os dados em treino e teste
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Carregar o modelo treinado
print(f"Carregando o modelo de {MODEL_PATH}...")
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print("Modelo carregado com sucesso!")
except FileNotFoundError:
    print(f"ERRO: Modelo não encontrado em {MODEL_PATH}")
    print(
        "Por favor, execute o script '2_train_model.py' primeiro para treinar o modelo."
    )
    exit(1)
except Exception as e:
    print(f"ERRO ao carregar o modelo: {e}")
    exit(1)

# 6. Fazer previsões no conjunto de teste
print("Fazendo previsões no conjunto de teste...")
y_pred = model.predict(X_test)

# 7. Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisão do modelo no conjunto de teste: {accuracy * 100:.2f}%\n")

# Relatório de classificação (Precisão, Recall, F1-Score)
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de confusão
print("Gerando a Matriz de Confusão...")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Matriz de Confusão')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')
plt.savefig(CONFUSION_MATRIX_PATH)
plt.show()

print(f"\nAnálise concluída. Gráfico da Matriz de Confusão salvo em '{CONFUSION_MATRIX_PATH}'.")
