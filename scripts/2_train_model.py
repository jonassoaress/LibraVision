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

# 2. Preparar os dados
# X são as features (coordenadas), y é o label (letra)
X = df.drop('label', axis=1)
y = df['label']

print("Dividindo os dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Treinar o modelo
print("Treinando o modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Avaliar o modelo
print("Avaliando a precisão do modelo...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisão do modelo: {accuracy * 100:.2f}%')

# 5. Salvar o modelo treinado
print(f"Salvando o modelo em {MODEL_PATH}...")
joblib.dump(model, MODEL_PATH)

print("Treinamento concluído e modelo salvo com sucesso.")