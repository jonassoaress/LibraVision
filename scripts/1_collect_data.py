import mediapipe as mp
import os
import cv2
import csv
import numpy as np
import time

# Configurações
# Obtém o caminho absoluto para o diretório onde o script está localizado
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

# Letras para coletar
LETTERS_TO_COLLECT = 'ABCDEFGILM'
# Número de amostras por letra
NUM_SAMPLES = 40
# Arquivo CSV para salvar os dados
CSV_FILE = os.path.join(DATA_DIR, 'libras_data.csv')

# Inicialização
# Cria o diretório de dados se não existir
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Abre a webcam
cap = cv2.VideoCapture(0)

# Flag para saída do programa
quit_program = False

# Abre o arquivo CSV em modo de escrita ('w')
try:
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Escreve o cabeçalho
        header = ['label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(header)

        # Loop principal para cada letra
        for letter in LETTERS_TO_COLLECT:
            letter_data = []
            sample_count = 0

            print(f"Coletando dados para a letra: '{letter}'")

            # Pausa para o usuário se preparar
            for i in range(5, 0, -1):
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Prepare-se para a letra '{letter}'... {i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow('Coleta de Dados', frame)
                cv2.waitKey(1000)

            # Loop de coleta para a letra atual
            while sample_count < NUM_SAMPLES:
                ret, frame = cap.read()
                if not ret:
                    quit_program = True
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # Desenha as instruções na tela
                cv2.putText(frame, f"LETRA: {letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"AMOSTRAS: {sample_count}/{NUM_SAMPLES}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "Pressione 'K' para capturar", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('Coleta de Dados', frame)

                key = cv2.waitKey(25) & 0xFF
                if key == ord('k') and results.multi_hand_landmarks:
                    # Extrai os landmarks da primeira mão detectada
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten().tolist()
                    letter_data.append([letter] + landmarks)
                    sample_count += 1

                    # Feedback visual de captura
                    frame_copy = frame.copy()
                    cv2.putText(frame_copy, "SALVO!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    cv2.imshow('Coleta de Dados', frame_copy)
                    cv2.waitKey(500)

                # Lógica para SAÍDA GRACIOSA
                elif key == ord('q'):
                    quit_program = True
                    break

            # Escreve todos os dados da letra e força o salvamento no disco
            writer.writerows(letter_data)
            f.flush() # <-- PONTO CRÍTICO: Força a escrita dos dados no disco!
            print(f"Dados da letra '{letter}' salvos com sucesso no CSV!")

            if quit_program:
                print("Coleta interrompida pelo usuário.")
                break # Sai do loop de letras

finally:
    # Garante que a câmera e as janelas sejam liberadas
    print("Finalizando o programa e liberando recursos.")
    cap.release()
    cv2.destroyAllWindows()