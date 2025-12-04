import csv
import os

import cv2
import mediapipe as mp
import numpy as np

# --- CAMINHOS E CONFIGURAÇÕES ---
# Obtém o caminho absoluto para o diretório onde o script está localizado
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

# Alfabeto completo de Libras
LETTERS_TO_COLLECT = "ABCDEFGHIJKLMNOPQRSTUVWXYZÇ"
NUM_SAMPLES = 100  # 40 amostras por letra
PREPARATION_TIME = 10  # 10 segundos para se preparar

CSV_FILE = os.path.join(DATA_DIR, "libras_data.csv")

# --- INICIALIZAÇÃO --
# Cria o diretório de dados se não existir
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Inicializa o MediaPipe Hands (agora suporta até 2 mãos)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# Abre a webcam com tratamento de erros
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
print(
    f"Coletando dados para {len(LETTERS_TO_COLLECT)} letras com {NUM_SAMPLES} amostras cada."
)
print("Pressione 'q' a qualquer momento para interromper.")

# Flag para saída do programa
quit_program = False

# Verifica se o arquivo já existe para decidir sobre o cabeçalho
file_exists = os.path.exists(CSV_FILE)

# Abre o arquivo CSV em modo de append ('a') para não sobrescrever dados
try:
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Escreve o cabeçalho apenas se o arquivo não existia
        if not file_exists:
            # agora inclui coluna 'hand' para identificar a mão
            header = ["label", "hand"] + [
                f"{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]
            ]
            writer.writerow(header)

        # Loop principal para cada letra
        for letter in LETTERS_TO_COLLECT:
            print(f"Coletando dados para a letra: '{letter}'")

            # Pausa para o usuário se preparar
            for i in range(PREPARATION_TIME, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                cv2.putText(
                    frame,
                    f"Prepare-se para a letra '{letter}'... {i}",
                    (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.imshow("Coleta de Dados", frame)
                if cv2.waitKey(1000) & 0xFF == ord("q"):
                    quit_program = True
                    break
            if quit_program:
                break

            # Loop de coleta para a letra atual
            letter_data = []
            sample_count = 0
            while sample_count < NUM_SAMPLES:
                ret, frame = cap.read()
                if not ret:
                    quit_program = True
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # Desenha as instruções na tela
                cv2.putText(
                    frame,
                    f"LETRA: {letter}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"AMOSTRAS: {sample_count}/{NUM_SAMPLES}",
                    (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                if results.multi_hand_landmarks:
                    # Processa todas as mãos detectadas
                    for hand_idx, hand_landmarks in enumerate(
                        results.multi_hand_landmarks
                    ):
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                        # Identifica se é mão esquerda ou direita
                        hand_label = (
                            results.multi_handedness[hand_idx].classification[0].label
                        )

                        # Extrai landmarks
                        landmarks = (
                            np.array(
                                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                            )
                            .flatten()
                            .tolist()
                        )

                        # Adiciona à lista com label da letra e mão
                        letter_data.append([letter, hand_label] + landmarks)
                        sample_count += 1

                        cv2.putText(
                            frame,
                            f"COLETANDO ({hand_label})...",
                            (50, 130 + hand_idx * 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                else:
                    cv2.putText(
                        frame,
                        "Mao nao detectada",
                        (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Coleta de Dados", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    quit_program = True
                    break

            if quit_program:
                break

            # Salvar os dados da letra em lote
            if letter_data:
                writer.writerows(letter_data)
                print(f"Dados para a letra '{letter}' salvos com sucesso.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nColeta de dados finalizada. Recursos liberados.")
    print(f"O dataset foi salvo em: '{CSV_FILE}'")
