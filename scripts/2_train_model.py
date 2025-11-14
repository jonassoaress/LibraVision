import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Caminho para o dataset e para salvar o modelo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'libras_data.csv')
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'libras_model.pkl')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 1. Carregar os dados
print("Carregando o dataset...")
df = pd.read_csv(DATA_FILE)

# 2. Engenharia de Features: Normalização
print("Realizando engenharia de features (normalização)...")
processed_data = []
for index, row in df.iterrows():
    label = row['label']
    landmarks = row.drop('label').values.reshape(21, 3) # Transforma de volta para 21x3

    # Pega as coordenadas do pulso (ponto 0)
    wrist_coords = landmarks[0]

    # Subtrai as coordenadas do pulso de todos os outros pontos
    relative_landmarks = landmarks - wrist_coords

    # Achata a lista para o formato original e adiciona o label
    processed_data.append([label] + relative_landmarks.flatten().tolist())

# Cria um novo DataFrame com os dados processados
columns = ['label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
processed_df = pd.DataFrame(processed_data, columns=columns)

# 3. Preparar os dados para treino e teste
# X são as features (coordenadas), y é o label (letra)
X = processed_df.drop('label', axis=1)
y = processed_df['label']

print("Dividindo os dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Treinar o modelo
print("Treinando o modelo Random Forest...")
model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=20, min_samples_leaf=5)
model.fit(X_train, y_train)

# 5. Avaliar o modelo
print("Avaliando a precisão do modelo...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisão do modelo: {accuracy * 100:.2f}%')

# 6. Salvar o modelo treinado
print(f"Salvando o modelo em {MODEL_PATH}...")
joblib.dump(model, MODEL_PATH)

print("Treinamento concluído e modelo salvo com sucesso.")