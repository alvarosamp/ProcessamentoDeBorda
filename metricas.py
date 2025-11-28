import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from tflite_runtime.interpreter import Interpreter

# --- Configurações e Caminhos ---
CAMERA_SOURCE = 0 
CONFIDENCE_THRESHOLD = 0.5
DEVICE = 'cpu'
INPUT_WIDTH, INPUT_HEIGHT = 640, 640 

MODELS = {
    'PyTorch (32-bit)': {'path': 'yolov8n.pt', 'type': 'pytorch', 'color': (0, 0, 255)}, # Vermelho
    'TFLite Quantizado (8-bit)': {'path': 'yolov8n_fullint.tflite', 'type': 'tflite', 'color': (0, 255, 0)} # Verde
}

# --- Funções de Preparação ---

def get_model_size(path):
    """Calcula o tamanho do arquivo em Megabytes."""
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0

def load_models():
    """Carrega os modelos PyTorch e TFLite."""
    try:
        model_pytorch = YOLO(MODELS['PyTorch (32-bit)']['path'])
        
        interpreter_tflite = Interpreter(model_path=MODELS['TFLite Quantizado (8-bit)']['path'])
        interpreter_tflite.allocate_tensors()
        
        return model_pytorch, interpreter_tflite
    except Exception as e:
        print(f"\n[ERRO FATAL] Falha ao carregar modelos: {e}")
        print("Verifique os caminhos e as bibliotecas instaladas (ultralytics, tflite-runtime).")
        exit()

# --- Função de Inferência para PyTorch (Float32) ---
def run_pytorch(model, frame):
    t_start = time.time()
    # A inferência da Ultralytics já inclui o pré-processamento.
    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, device=DEVICE, verbose=False)
    t_end = time.time()
    
    # Pegamos o frame plotado (com caixas) para comparar a Qualidade Visual.
    frame_display = results[0].plot()
    return frame_display, (t_end - t_start)

# --- Função de Inferência para TFLite (Int8) ---
def run_tflite(interpreter, frame):
    input_details = interpreter.get_input_details()
    
    # Pré-processamento: Redimensionar e UINT8
    input_tensor = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    input_tensor = input_tensor.astype(np.uint8) 
    input_tensor = np.expand_dims(input_tensor, axis=0)

    t_start = time.time()
    
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    # output_data = interpreter.get_tensor(output_details[0]['index']) # Omitido o post-processamento para foco em FPS
    
    t_end = time.time()
    
    # Retorna o frame original (sem caixas) para comparação de performance pura.
    return frame.copy(), (t_end - t_start)


# --- Loop Principal ---
if __name__ == '__main__':
    # 1. Análise de Tamanho
    print("\n--- 1. Tamanho dos Modelos (Eficiência de Armazenamento) ---")
    size_pt = get_model_size(MODELS['PyTorch (32-bit)']['path'])
    size_tflite = get_model_size(MODELS['TFLite Quantizado (8-bit)']['path'])
    print(f"  > PyTorch (32-bit): {size_pt:.2f} MB")
    print(f"  > TFLite Quantizado (8-bit): {size_tflite:.2f} MB")
    print(f"[INFO] TFLite é aproximadamente {size_pt/size_tflite:.1f}x menor.")
    
    # Carregar Modelos
    model_pytorch, interpreter_tflite = load_models()
    
    # Inicializa Câmera
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("Erro: Câmera não detectada.")
        exit()

    print("\n--- 2. Tempo e CPU (Pressione ESPAÇO para alternar, 'q' para sair) ---")
    current_model_key = 'PyTorch (32-bit)'
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        model_info = MODELS[current_model_key]
        
        # Execução
        if model_info['type'] == 'pytorch':
            display_frame, inference_time = run_pytorch(model_pytorch, frame)
            cpu_text = "CPU/Energia: ALTA (Lenta) - 32-bit Float"
        else: # TFLite
            display_frame, inference_time = run_tflite(interpreter_tflite, frame)
            cpu_text = "CPU/Energia: BAIXA (Rápida) - 8-bit Int"
        
        # Cálculo e Exibição das Métricas
        FPS = 1.0 / inference_time if inference_time > 0 else 0 
        
        cv2.putText(display_frame, model_info['path'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_info['color'], 2)
        cv2.putText(display_frame, f"FPS: {FPS:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, model_info['color'], 3)
        cv2.putText(display_frame, cpu_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_info['color'], 2)
        
        cv2.imshow('COMPARACAO DE METRICAS (YOLOv8)', display_frame)

        # Controle de Teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '): # Alterna entre os modelos
            current_model_key = 'TFLite Quantizado (8-bit)' if current_model_key == 'PyTorch (32-bit)' else 'PyTorch (32-bit)'
            print(f"Modo Alternado para: {current_model_key}")
            # Dica para a aula: Neste momento, observem o 'htop'!
            

    # Liberação de Recursos
    cap.release()
    cv2.destroyAllWindows()