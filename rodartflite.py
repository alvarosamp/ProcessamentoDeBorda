import cv2
import numpy as np
import time
# Importamos a versão otimizada do TFLite para o Raspberry Pi
from tflite_runtime.interpreter import Interpreter 

# --- Configurações ---
CAMERA_SOURCE = 0
TFLITE_MODEL_PATH = 'yolov8n_fullint.tflite' # Altere para o caminho do seu modelo .tflite
INPUT_WIDTH, INPUT_HEIGHT = 640, 640 # Dimensões esperadas pelo YOLOv8 TFLite

def rodar_tflite_quantizado(model_path, camera_source):
    """Carrega um modelo TFLite e roda a inferência em tempo real."""
    print(f"\n[INFO] Carregando modelo TFLite Quantizado: {model_path} (Int8)")
    
    try:
        # 1. Carrega o Interpreter
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Erro ao carregar o Interpreter TFLite: {e}")
        print("Certifique-se de instalar 'tflite-runtime'.")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 2. Inicializa a Câmera Manualmente
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("Erro: Câmera não detectada.")
        return

    print("Iniciando inferência TFLite. Pressione 'q' para sair.")

    # Helper functions para pós-processamento
    def xywh2xyxy(boxes):
        # boxes: [x_center, y_center, w, h]
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return xyxy

    def iou(box1, box2):
        # box: [x1,y1,x2,y2]
        xa = max(box1[0], box2[0])
        ya = max(box1[1], box2[1])
        xb = min(box1[2], box2[2])
        yb = min(box1[3], box2[3])
        inter_w = max(0, xb - xa)
        inter_h = max(0, yb - ya)
        inter = inter_w * inter_h
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def nms(boxes, scores, iou_threshold=0.45):
        idxs = np.argsort(scores)[::-1]
        keep = []
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)
            if idxs.size == 1:
                break
            rest = idxs[1:]
            rem = []
            for j in rest:
                if iou(boxes[i], boxes[j]) <= iou_threshold:
                    rem.append(j)
            idxs = np.array(rem)
        return keep

    SCORE_THRES = 0.25
    NMS_IOU = 0.45

    # 3. Loop de Processamento Manual
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h0, w0 = frame.shape[:2]

        # Pré-processamento: Redimensionamento e UINT8 (Int8 espera isso)
        input_tensor = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        input_tensor = input_tensor.astype(np.uint8)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Adiciona dimensão de batch

        # Medição de Tempo e Execução da Inferência
        t_start = time.time()

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()  # Execução da rede

        # Coleta das saídas (suporta vários formatos comuns)
        outputs = [interpreter.get_tensor(d['index']) for d in output_details]
        t_end = time.time()

        inference_time = t_end - t_start
        FPS = 1.0 / inference_time if inference_time > 0 else 0

        display_frame = frame.copy()

        # Tentativa de interpretar as saídas do TFLite (formatos comuns do YOLO exportado)
        if len(outputs) == 0:
            detections = []
        else:
            out = outputs[0]
            # Ajusta shape para (N, C)
            if out.ndim == 3 and out.shape[0] == 1:
                out = out[0]

            detections = []
            if out.ndim == 2 and out.shape[1] >= 5:
                # formato esperado: [x, y, w, h, obj, cls...] ou [x, y, w, h, conf, class]
                boxes_xywh = out[:, :4].astype(np.float32)
                # Detecta se boxes estão normalizadas (0..1) ou em pixels
                max_box_val = boxes_xywh.max()
                normalized = max_box_val <= 1.0

                obj_scores = out[:, 4]

                if out.shape[1] > 5:
                    class_probs = out[:, 5:]
                    class_ids = np.argmax(class_probs, axis=1)
                    class_confs = class_probs[np.arange(len(class_ids)), class_ids]
                    scores = obj_scores * class_confs
                else:
                    # Se não há vetor de classes, assumimos que a coluna 5 é a classe (inteiro)
                    scores = obj_scores
                    class_ids = np.zeros_like(scores, dtype=int)

                # Converte para xyxy em coordenadas do frame original
                xyxy = xywh2xyxy(boxes_xywh)
                if normalized:
                    xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * w0 / INPUT_WIDTH * INPUT_WIDTH / INPUT_WIDTH
                    xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * h0 / INPUT_HEIGHT * INPUT_HEIGHT / INPUT_HEIGHT
                    # Quando normalizado fazendo x * w0 e y * h0 é suficiente
                    xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * w0
                    xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * h0
                else:
                    # Caso as caixas já estejam no tamanho do input (INPUT_WIDTH, INPUT_HEIGHT)
                    scale_x = w0 / INPUT_WIDTH
                    scale_y = h0 / INPUT_HEIGHT
                    xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * scale_x
                    xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * scale_y

                # Filtra por score
                keep_idx = np.where(scores >= SCORE_THRES)[0]
                if keep_idx.size > 0:
                    boxes = xyxy[keep_idx]
                    sc = scores[keep_idx]
                    cls = class_ids[keep_idx]

                    # Aplica NMS por classe
                    final_dets = []
                    for c in np.unique(cls):
                        inds = np.where(cls == c)[0]
                        b = boxes[inds]
                        s = sc[inds]
                        keep = nms(b, s, iou_threshold=NMS_IOU)
                        for k in keep:
                            idx = inds[k]
                            final_dets.append((boxes[k], sc[inds][k], int(c)))

                    detections = final_dets

        # Desenha detecções
        for det in detections:
            box, score, cls_id = det
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_id}:{score:.2f}"
            cv2.putText(display_frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Demonstração das Métricas na tela
        cv2.putText(display_frame, f"TFLite Quantizado (8-bit)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"FPS: {FPS:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(display_frame, f"CPU/Energia: BAIXA (Int8)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Inferência TFLite Quantizado', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Liberação de Recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Inferência TFLite encerrada.")

if __name__ == '__main__':
    rodar_tflite_quantizado(TFLITE_MODEL_PATH, CAMERA_SOURCE)