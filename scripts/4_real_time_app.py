import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from collections import deque

# Carrega o modelo treinado
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'libras_model.pkl')
model = joblib.load(MODEL_PATH)

# Inicializa o MediaPipe Hands e Webcam
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Lógica de suavização
PREDICTION_BUFFER_SIZE = 10 # Analisa as últimas 10 previsões
CONFIDENCE_THRESHOLD = 0.8 # Confiança mínima para considerar a previsão
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
stable_prediction = ""

while cap.isOpened():
  ret, frame = cap.read()
  if not ret: break

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
      landmarks_raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
      wrist_coords = landmarks_raw[0]
      relative_landmarks = landmarks_raw - wrist_coords
      landmarks_for_model = relative_landmarks.flatten().toList()

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
        # A predição estável é a que mais aparece no buffer
        most_common = max(set(prediction_buffer), key=prediction_buffer.count)
        if prediction_buffer.count(most_common) > PREDICTION_BUFFER_SIZE / 0.7:
          stable_prediction = most_common

      # Exibe a confiança e a letra prevista (instável)
      cv2.putText(frame, f'Pred: {predicted_letter} ({confidence:.2f})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

  # Exibe o frame
  # Exibe a predição ESTÁVEL em destaque
  cv2.rectangle(frame, (40, 400), (600, 460), (0, 0, 0), -1)
  cv2.putText(frame, f'Letra Estavel: {stable_prediction}', (50, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

  cv2.imshow('LibraVision', frame)

  # Pressione 'q' para sair
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()