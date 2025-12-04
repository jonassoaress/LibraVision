import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

# Caminho para o dataset e para salvar o modelo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "..", "data", "libras_data.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "libras_model.pkl")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

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

# Verificar se há dados suficientes
if len(df) < 50:
    print(f"AVISO: Dataset muito pequeno ({len(df)} amostras).")
    print("Recomenda-se pelo menos 500 amostras para um bom treinamento.")
    print("Execute '1_collect_data.py' para coletar mais dados.")

# 2. Engenharia de Features: Normalização
print("Realizando engenharia de features (normalização)...")
processed_data = []
for index, row in df.iterrows():
    label = row["label"]
    hand = row["hand"]
    landmarks = row.drop(["label", "hand"]).values.reshape(
        21, 3
    )  # Transforma de volta para 21x3

    # Pega as coordenadas do pulso (ponto 0)
    wrist_coords = landmarks[0]

    # Subtrai as coordenadas do pulso de todos os outros pontos
    relative_landmarks = landmarks - wrist_coords

    # Achata a lista para o formato original e adiciona o label e mão
    processed_data.append([label, hand] + relative_landmarks.flatten().tolist())

# Cria um novo DataFrame com os dados processados
columns = ["label", "hand"] + [
    f"{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]
]
processed_df = pd.DataFrame(processed_data, columns=columns)

# 3. Preparar os dados para treino e teste
# X são as features (coordenadas + hand), y é o label (letra)
# Codifica a coluna 'hand' como feature numérica (Left=0, Right=1)
X = processed_df.drop("label", axis=1).copy()
X["hand_encoded"] = (X["hand"] == "Right").astype(int)
X = X.drop("hand", axis=1)
y = processed_df["label"]

print("Dividindo os dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Realizar validação cruzada k-fold
print("Realizando validação cruzada (5-fold)...")
base_model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(base_model, X_train, y_train, cv=5, scoring="accuracy")
print(
    f"Precisão média da validação cruzada: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)"
)

# 5. Otimização de hiperparâmetros com GridSearchCV
print("\nOtimizando hiperparâmetros com GridSearchCV...")
print("(Isso pode levar alguns minutos...)")

param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [15, 20, 25, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print("\nMelhores hiperparâmetros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Melhor precisão no treino (CV): {grid_search.best_score_ * 100:.2f}%")

# Usar o melhor modelo encontrado
model = grid_search.best_estimator_

# 6. Avaliar o modelo no conjunto de teste
print("\nAvaliando a precisão do modelo no conjunto de teste...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisão do modelo no teste: {accuracy * 100:.2f}%")

# 7. Salvar o modelo treinado
print(f"\nSalvando o modelo em {MODEL_PATH}...")
try:
    joblib.dump(model, MODEL_PATH)
    print("Treinamento concluído e modelo salvo com sucesso!")
    print("\nPróximos passos:")
    print("  1. Execute '3_test_model.py' para avaliar o modelo (opcional)")
    print("  2. Execute '4_real_time_app.py' para usar o reconhecimento em tempo real")
except Exception as e:
    print(f"ERRO ao salvar o modelo: {e}")
    exit(1)
