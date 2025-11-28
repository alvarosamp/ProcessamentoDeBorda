import cv2
import numpy as np
import time
import onnxruntime as ort  # Novo: ONNX Runtime

# --- Configurações ---
CAMERA_SOURCE = 0
ONNX_MODEL_PATH = r'C:\Users\vish8\OneDrive\Desktop\IC\ProcessamentoDeBorda\modelo.onnx'  # Altere para o caminho do seu modelo .onnx
INPUT_WIDTH, INPUT_HEIGHT = 640, 640  # Dimensões esperadas pelo YOLOv8 ONNX

def rodar_onnx(model_path, camera_source):
    """Carrega um modelo ONNX e roda a inferência em tempo real."""
    print(f"\n[INFO] Carregando modelo ONNX: {model_path}")

    try:
        # 1. Cria sessão do ONNX Runtime (somente CPU por padrão)
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        print(f"[INFO] Input name: {input_name}, shape: {input_meta.shape}, type: {input_meta.type}")
    except Exception as e:
        print(f"Erro ao carregar o modelo ONNX: {e}")
        return

    # 2. Inicializa a Câmera
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("Erro: Câmera não detectada.")
        return

    print("Iniciando inferência ONNX. Pressione 'q' para sair.")

    # ----------------- Funções auxiliares -----------------
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

    # ----------------- Loop principal -----------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h0, w0 = frame.shape[:2]

        # Pré-processamento:
        # - redimensiona para 640x640
        # - BGR -> RGB
        # - normaliza 0–1
        # - HWC -> CHW
        # - adiciona batch
        img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (0, 1, 2))  # (H,W,C) -> (H,W,C) (só para clareza)
        img_chw = np.transpose(img_norm, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        input_tensor = np.expand_dims(img_chw, axis=0)  # (1,3,640,640)

        t_start = time.time()

        # Execução ONNX
        outputs = session.run(None, {input_name: input_tensor})
        t_end = time.time()

        inference_time = t_end - t_start
        FPS = 1.0 / inference_time if inference_time > 0 else 0.0

        display_frame = frame.copy()

        # ----------------- Pós-processamento -----------------
        detections = []
        if len(outputs) > 0:
            out = outputs[0]

            # Esperado comum em YOLOv8 ONNX: (1, 84, 8400) ou (1, 8400, 84)
            if out.ndim == 3 and out.shape[0] == 1:
                out = out[0]  # remove batch -> (84, 8400) ou (8400, 84)

            # Se estiver (84, 8400), transpõe para (8400,84)
            if out.shape[0] in (84, 85) and out.shape[1] > 10:
                out = out.transpose(1, 0)  # (8400,84)

            # Achata caso seja (N,8400,84) ou outro formato
            if out.ndim == 3:
                out = out.reshape(-1, out.shape[-1])  # (N,84)

            # Agora esperamos (num_boxes, num_feats>=5)
            if out.ndim == 2 and out.shape[1] >= 5:
                boxes_xywh = out[:, :4].astype(np.float32)
                obj_scores = out[:, 4].astype(np.float32)

                # Classes (probabilidades)
                if out.shape[1] > 5:
                    class_probs = out[:, 5:]
                    class_ids = np.argmax(class_probs, axis=1)
                    class_confs = class_probs[np.arange(len(class_ids)), class_ids]
                    scores = obj_scores * class_confs
                else:
                    scores = obj_scores
                    class_ids = np.zeros_like(scores, dtype=int)

                # YOLOv8 ONNX normalmente sai normalizado (0..1)
                xyxy = xywh2xyxy(boxes_xywh)
                max_box_val = max(boxes_xywh.max(), 1e-6)
                normalized = max_box_val <= 1.5  # heurística simples

                if normalized:
                    xyxy[:, [0, 2]] *= w0
                    xyxy[:, [1, 3]] *= h0
                else:
                    scale_x = w0 / INPUT_WIDTH
                    scale_y = h0 / INPUT_HEIGHT
                    xyxy[:, [0, 2]] *= scale_x
                    xyxy[:, [1, 3]] *= scale_y

                # Filtra por score
                keep_idx = np.where(scores >= SCORE_THRES)[0]
                if keep_idx.size > 0:
                    boxes = xyxy[keep_idx]
                    sc = scores[keep_idx]
                    cls = class_ids[keep_idx]

                    # NMS por classe
                    final_dets = []
                    for c in np.unique(cls):
                        inds = np.where(cls == c)[0]
                        b = boxes[inds]
                        s = sc[inds]
                        keep = nms(b, s, iou_threshold=NMS_IOU)
                        for k in keep:
                            idx = inds[k]
                            final_dets.append((boxes[idx], sc[idx], int(c)))

                    detections = final_dets

        # Desenha detecções
        for det in detections:
            box, score, cls_id = det
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_id}:{score:.2f}"
            cv2.putText(display_frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # HUD
        cv2.putText(display_frame, "ONNX (YOLOv8)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"FPS: {FPS:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        cv2.imshow('Inferência ONNX', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Liberação de Recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Inferência ONNX encerrada.")

if __name__ == '__main__':
    rodar_onnx(ONNX_MODEL_PATH, CAMERA_SOURCE)
