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

# Inicializa o MediaPipe Hands e Webcam (agora suporta até 2 mãos)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils

# Inicializa a webcam com tratamento de erros e resolução aumentada
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERRO: Não foi possível abrir a câmera.")
    print("Verifique se:")
    print("  - A webcam está conectada corretamente")
    print("  - Não há outro programa usando a câmera")
    print("  - Você tem permissões para acessar a câmera")
    exit(1)

# Define uma resolução maior (1280x720 - HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Testa se consegue ler um frame
ret, test_frame = cap.read()
if not ret:
    print("ERRO: Câmera aberta mas não consegue capturar frames.")
    cap.release()
    exit(1)

print("Câmera inicializada com sucesso!")
print("Pressione 'q' para sair da aplicação.")
print("Pressione 'f' para alternar entre tela cheia e janela.")

# Lógica de suavização para cada mão
PREDICTION_BUFFER_SIZE = 10  # Analisa as últimas 10 previsões
CONFIDENCE_THRESHOLD = 0.8  # Confiança mínima para considerar a previsão
prediction_buffer_left = deque(maxlen=PREDICTION_BUFFER_SIZE)
prediction_buffer_right = deque(maxlen=PREDICTION_BUFFER_SIZE)
stable_prediction_left = ""
stable_prediction_right = ""

# Controle de tela cheia
is_fullscreen = False
window_name = "LibraVision"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
        # Processa todas as mãos detectadas
        for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Identifica se é mão esquerda ou direita
            hand_label = result.multi_handedness[hand_idx].classification[0].label

            # Desenha os pontos e conexões na mão
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extrai os landmarks para o modelo
            landmarks_raw = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            )
            wrist_coords = landmarks_raw[0]
            relative_landmarks = landmarks_raw - wrist_coords
            landmarks_for_model = relative_landmarks.flatten().tolist()

            # Adiciona a feature de qual mão (Left=0, Right=1)
            hand_encoded = 1 if hand_label == "Right" else 0
            landmarks_for_model.append(hand_encoded)

            # Faz a previsão com o modelo
            prediction_probs = model.predict_proba([landmarks_for_model])[0]
            confidence = np.max(prediction_probs)
            predicted_class_index = np.argmax(prediction_probs)
            predicted_letter = model.classes_[predicted_class_index]

            # Seleciona o buffer correto para a mão
            if hand_label == "Left":
                prediction_buffer = prediction_buffer_left
                y_pos = 50
            else:
                prediction_buffer = prediction_buffer_right
                y_pos = 100

            # Adiciona ao buffer apenas se a confiança for alta o suficiente
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_buffer.append(predicted_letter)

            # Verifica se há uma previsão estável no buffer
            if len(prediction_buffer) == PREDICTION_BUFFER_SIZE:
                # A predição estável é a que mais aparece no buffer
                counter = Counter(prediction_buffer)
                most_common, count = counter.most_common(1)[0]
                if count > PREDICTION_BUFFER_SIZE * 0.7:
                    if hand_label == "Left":
                        stable_prediction_left = most_common
                    else:
                        stable_prediction_right = most_common

            # Exibe a confiança e a letra prevista (instável)
            cv2.putText(
                frame,
                f"{hand_label}: {predicted_letter} ({confidence:.2f})",
                (50, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
    else:
        # Limpa os buffers quando não há mão detectada
        prediction_buffer_left.clear()
        prediction_buffer_right.clear()

    # Obtém as dimensões do frame para posicionar elementos dinamicamente
    frame_height, frame_width = frame.shape[:2]

    # Exibe as predições ESTÁVEIS em destaque (posicionamento adaptativo)
    box_y = frame_height - 180
    cv2.rectangle(
        frame, (40, box_y), (int(frame_width * 0.5), box_y + 150), (0, 0, 0), -1
    )
    cv2.putText(
        frame,
        f"Mao Esquerda: {stable_prediction_left}",
        (50, box_y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Mao Direita: {stable_prediction_right}",
        (50, box_y + 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.imshow(window_name, frame)

    # Captura teclas
    key = cv2.waitKey(1) & 0xFF

    # Pressione 'q' para sair
    if key == ord("q"):
        break

    # Pressione 'f' para alternar tela cheia
    elif key == ord("f"):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
            )

cap.release()
cv2.destroyAllWindows()
hands.close()
