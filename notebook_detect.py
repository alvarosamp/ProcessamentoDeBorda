from ultralytics import YOLO
import cv2

MODEL_PATH = r'C:\Users\vish8\OneDrive\Documentos\ProcessamentoDeBorda\best.pt'

print(f"[INFO] Carregando modelo: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("[INFO] Modelo carregado. Pressione 'Q' para sair.")

cap = cv2.VideoCapture(0)  # 0 = webcam padrão do notebook

if not cap.isOpened():
    print("[ERRO] Não foi possível abrir a câmera. Verifique se está conectada.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Falha ao capturar frame.")
        break

    results = model(frame, conf=0.5, device='cpu', verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Deteccao - best.pt  |  Q para sair", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Encerrado.")
