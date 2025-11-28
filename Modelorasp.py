from ultralytics import YOLO
import cv2
import os

# Detecta a fonte da câmera com prioridade:
# 1) variável de ambiente `CAMERA` (p.ex. "rtsp://..." ou "/dev/video1" ou "0")
# 2) se /dev/video1 existir (câmera externa USB no Linux)
# 3) fallback para índice 0 (primeira câmera)
env_cam = os.getenv('1')
if env_cam:
    # tenta interpretar números como inteiros
    try:
        CAMERA_SOURCE = int(env_cam)
    except Exception:
        CAMERA_SOURCE = env_cam
else:
    # preferência por /dev/video1 quando presente (externa)
    if os.name == 'posix' and os.path.exists('/dev/video1'):
        CAMERA_SOURCE = '/dev/video1'
    else:
        CAMERA_SOURCE = 0

MODEL_NAME = r'C:\Users\vish8\OneDrive\Desktop\IC\ProcessamentoDeBorda\modelo.pt'

print(f"[INFO] Carregando modelo {MODEL_NAME}...")

# 1. Carrega um modelo pré-treinado do YOLOv8
# Se você tiver seu próprio modelo treinado, substitua por 'caminho/para/seu/modelo.pt'
model = YOLO(MODEL_NAME)

print("[INFO] Modelo carregado. Rodando detecção em tempo real...")

# 2. Executa a inferência em tempo real a partir da fonte da câmera
# O método predict da Ultralytics é muito flexível e aceita o índice da câmera (0)
# Os argumentos 'show=True' e 'conf=0.5' são úteis:
# - source=CAMERA_SOURCE: Usa o fluxo da câmera.
# - show=True: Exibe o resultado (com as caixas delimitadoras) em uma janela OpenCV.
# - conf=0.5: Define o limiar de confiança mínima para exibir uma detecção.
# - save=False: Não salva os frames em disco.
# - device='cpu': Força o uso da CPU (necessário no Raspberry Pi que não tem GPU NVIDIA).

results = model.predict(
    source=CAMERA_SOURCE, 
    show=True, 
    conf=0.5, 
    save=False, 
    device='cpu' # Use 'cpu' para o Raspberry Pi
)

# O loop de exibição e o gerenciamento de recursos (cv2.destroyAllWindows())
# são tratados internamente pelo método predict quando 'show=True' é usado.
# O loop será encerrado quando a janela de exibição for fechada.

# Se você precisar de controle manual sobre os frames (por exemplo, para um processamento extra):
# results = model.predict(source=CAMERA_SOURCE, stream=True)
# for r in results:
#     # r.plot() é o frame com as caixas delimitadoras e labels
#     frame = r.plot()
#     cv2.imshow("YOLOv8 Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()