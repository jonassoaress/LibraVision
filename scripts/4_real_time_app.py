import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# Carrega o modelo treinado
MODEL_PATH = os.path.join('../models', 'libras_model.pkl')
model = joblib.load(MODEL_PATH)

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Abrea a webcam
cap = cv2.VideoCapture(0)

current_letter = ""
sentence = ""

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
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
      landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().tolist()

      # Faz a previsão com o modelo
      prediction = model.predict([landmarks])
      confidence = model.predict_proba([landmarks])

      current_letter = prediction[0]
      confidence_score = np.max(confidence) * 100

      # Exibe a letra prevista e a confiança
      cv2.putText(frame, f'Letra: {current_letter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
      cv2.putText(frame, f'Confiança: {confidence_score:.2f}%', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # Exibe o frame
  cv2.imshow('LibraVision - Reconhecimento de Libras', frame)

  # Pressione 'q' para sair
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()