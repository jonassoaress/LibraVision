import mediapipe as mp
import os
import cv2
import csv

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Diretório para salvar os dados
DATA_DIR = '../data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Letras para coletar (10 letras incluídas)
letters_to_collect = 'ABCDEFGILM'
num_samples = 40

# Arquivo CSV para salvar os dados
csv_file = os.path.join(DATA_DIR, 'libras_data.csv')

# Abre a webcam
cap = cv2.VideoCapture(0)

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Escreve o cabeçalho
    header = ['label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
    writer.writerow(header)

    for letter in letters_to_collect:
        print(f'Coletando dados para a letra: {letter}')

        # Pausa para o usuário se preparar
        for i in range(5, 0, -1):
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(frame, f'Prepare-se para a letra {letter}... {i}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Coleta de Dados', frame)
            cv2.waitKey(1000)

        # Coleta de amostras
        sample_count = 0
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Desenha os pontos na mão
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extrai as coordenadas dos pontos
                    landmark_coords = []
                    for lm in hand_landmarks.landmark:
                        landmark_coords.extend([lm.x, lm.y, lm.z])

                    # Salva os dados no CSV
                    writer.writerow([letter] + landmark_coords)
                    sample_count += 1

            # Mostra o progresso na tela
            cv2.putText(frame, f'Letra: {letter} | Amostra: {sample_count}/{num_samples}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Coleta de Dados', frame)

            # Pressione 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

print('Coleta de dados concluída!')
cap.release()
cv2.destroyAllWindows()