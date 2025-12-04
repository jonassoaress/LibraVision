import os
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

# Carrega o modelo treinado
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "libras_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo carregado com sucesso de: {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERRO: Modelo não encontrado em: {MODEL_PATH}")
    print(
        "Por favor, execute o script '2_train_model.py' primeiro para treinar o modelo."
    )
    exit(1)
except Exception as e:
    print(f"ERRO ao carregar o modelo: {e}")
    exit(1)

# Inicializa o MediaPipe Hands e Webcam
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils

# Inicializa a webcam com tratamento de erros
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERRO: Não foi possível abrir a câmera.")
    print("Verifique se:")
    print("  - A webcam está conectada corretamente")
    print("  - Não há outro programa usando a câmera")
    print("  - Você tem permissões para acessar a câmera")
    exit(1)

# Testa se consegue ler um frame
ret, test_frame = cap.read()
if not ret:
    print("ERRO: Câmera aberta mas não consegue capturar frames.")
    cap.release()
    exit(1)

print("Câmera inicializada com sucesso!")
print("Pressione 'q' para sair da aplicação.")

# Lógica de suavização
PREDICTION_BUFFER_SIZE = 10  # Analisa as últimas 10 previsões
CONFIDENCE_THRESHOLD = 0.8  # Confiança mínima para considerar a previsão
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
stable_prediction = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("\nAVISO: Falha ao capturar frame da câmera. Encerrando...")
        break

    # Inverte a imagem horizontalmente para efeito de espelho
    frame = cv2.flip(frame, 1)

    # Converte a imagem para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa a imagem para detectar as mãos
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Desenha os pontos e conexões na mão
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extrai os landmarks para o modelo
            landmarks_raw = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            )
            wrist_coords = landmarks_raw[0]
            relative_landmarks = landmarks_raw - wrist_coords
            landmarks_for_model = relative_landmarks.flatten().tolist()

            # Faz a previsão com o modelo
            prediction_probs = model.predict_proba([landmarks_for_model])[0]
            confidence = np.max(prediction_probs)
            predicted_class_index = np.argmax(prediction_probs)
            predicted_letter = model.classes_[predicted_class_index]

            # Adiciona ao buffer apenas se a confiança for alta o suficiente
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_buffer.append(predicted_letter)

            # Verifica se há uma previsão estável no buffer
            if len(prediction_buffer) == PREDICTION_BUFFER_SIZE:
                # A predição estável é a que mais aparece no buffer (usando Counter para eficiência)
                counter = Counter(prediction_buffer)
                most_common, count = counter.most_common(1)[0]
                if count > PREDICTION_BUFFER_SIZE * 0.7:
                    stable_prediction = most_common

            # Exibe a confiança e a letra prevista (instável)
            cv2.putText(
                frame,
                f"Pred: {predicted_letter} ({confidence:.2f})",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
    else:
        # Limpa o buffer quando não há mão detectada
        prediction_buffer.clear()

    # Exibe o frame
    # Exibe a predição ESTÁVEL em destaque
    cv2.rectangle(frame, (40, 400), (600, 460), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Letra Estavel: {stable_prediction}",
        (50, 440),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.imshow("LibraVision", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
