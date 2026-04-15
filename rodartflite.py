import argparse
import os
import time

import cv2
import numpy as np

# Importamos a versão otimizada do TFLite (comum no Raspberry Pi)
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:  # pragma: no cover
    Interpreter = None


def _parse_source(value: str):
    """Converte fonte para int quando aplicável (ex.: "0" -> 0)."""
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return value


def _build_synthetic_input(input_details, fill_mode: str = "zero"):
    """Gera um tensor de entrada compatível com o modelo, sem precisar de câmera."""
    info = input_details[0]
    shape = tuple(info["shape"])
    dtype = info["dtype"]

    quant = info.get("quantization_parameters") or {}
    zero_point = 0
    if isinstance(quant, dict) and "zero_points" in quant and len(quant["zero_points"]) > 0:
        zero_point = int(quant["zero_points"][0])
    elif isinstance(info.get("quantization"), (tuple, list)) and len(info["quantization"]) == 2:
        zero_point = int(info["quantization"][1])

    if fill_mode == "zero":
        return np.full(shape, zero_point, dtype=dtype)
    if fill_mode == "random":
        if np.issubdtype(dtype, np.integer):
            if dtype == np.int8:
                low, high = -128, 127
            elif dtype == np.uint8:
                low, high = 0, 255
            else:
                low, high = np.iinfo(dtype).min, np.iinfo(dtype).max
            return np.random.randint(low, high + 1, size=shape, dtype=dtype)
        return np.random.random(size=shape).astype(dtype)
    raise ValueError(f"fill_mode inválido: {fill_mode}")


def _preprocess_frame(frame_bgr, input_details, keep_bgr: bool = True):
    info = input_details[0]
    shape = info["shape"]
    dtype = info["dtype"]

    if len(shape) != 4:
        raise ValueError(f"Shape de entrada inesperado: {shape}")

    batch, h, w, c = shape
    if batch != 1 or c != 3:
        # mantém simples: casos mais comuns (1,H,W,3)
        raise ValueError(f"Entrada esperada (1,H,W,3); recebido: {shape}")

    img = cv2.resize(frame_bgr, (int(w), int(h)))
    if not keep_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if dtype == np.uint8:
        tensor = img.astype(np.uint8)
    elif dtype == np.int8:
        # Mapeia 0..255 -> -128..127 (aproximação comum para modelos int8)
        tensor = (img.astype(np.int16) - 128).astype(np.int8)
    else:
        # fallback: normaliza para float32 0..1
        tensor = (img.astype(np.float32) / 255.0).astype(dtype)

    return np.expand_dims(tensor, axis=0)


def rodar_tflite_quantizado(
    model_path,
    source,
    image_path=None,
    bench_iters: int = 0,
    no_display: bool = False,
    keep_bgr: bool = True,
):
    """Carrega um modelo TFLite e roda inferência por câmera, arquivo ou benchmark."""
    print(f"\n[INFO] Carregando modelo TFLite: {model_path}")

    if Interpreter is None:
        print("Erro: não foi possível importar 'tflite_runtime'.")
        print("No Windows, muitas vezes é mais simples usar TensorFlow (tensorflow.lite.Interpreter).")
        return 2

    if not os.path.exists(model_path):
        print(f"Erro: modelo não encontrado em: {model_path}")
        return 2

    try:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Erro ao carregar o Interpreter TFLite: {e}")
        return 2

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # ----------------- Benchmark sem câmera -----------------
    if bench_iters and bench_iters > 0:
        input_tensor = _build_synthetic_input(input_details, fill_mode="zero")
        interpreter.set_tensor(input_details[0]["index"], input_tensor)

        # warmup
        for _ in range(min(5, bench_iters)):
            interpreter.invoke()

        times = []
        for _ in range(bench_iters):
            t0 = time.perf_counter()
            interpreter.invoke()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        arr = np.array(times, dtype=np.float32)
        print("\n===== BENCHMARK (sem câmera) =====")
        print(f"Iterações: {bench_iters}")
        print(f"Latência média: {arr.mean():.2f} ms")
        print(f"Latência p50:  {np.percentile(arr, 50):.2f} ms")
        print(f"Latência p90:  {np.percentile(arr, 90):.2f} ms")
        print(f"Latência min/max: {arr.min():.2f}/{arr.max():.2f} ms")
        print(f"Throughput aprox.: {1000.0 / arr.mean():.2f} FPS")
        print("==================================\n")
        return 0

    # ----------------- Modo imagem única (sem câmera) -----------------
    if image_path:
        if not os.path.exists(image_path):
            print(f"Erro: imagem não encontrada em: {image_path}")
            return 2

        frame = cv2.imread(image_path)
        if frame is None:
            print("Erro: falha ao ler a imagem (formato inválido?)")
            return 2

        input_tensor = _preprocess_frame(frame, input_details, keep_bgr=keep_bgr)
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        t1 = time.perf_counter()

        outputs = [interpreter.get_tensor(d["index"]) for d in output_details]
        print(f"Inferência OK. Tempo: {(t1 - t0) * 1000.0:.2f} ms")
        for i, out in enumerate(outputs):
            print(f"Output[{i}] shape={getattr(out, 'shape', None)} dtype={getattr(out, 'dtype', None)}")

        if not no_display:
            cv2.putText(frame, "TFLite (imagem unica)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Inferencia TFLite (sem camera)", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    # ----------------- Modo câmera / vídeo -----------------
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Erro: fonte de vídeo/câmera não abriu.")
        print("Dicas: use --image <arquivo.jpg> ou --bench <iters> para rodar sem câmera.")
        return 2

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

        # Pré-processamento compatível com o dtype/shape do modelo
        input_tensor = _preprocess_frame(frame, input_details, keep_bgr=keep_bgr)

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

        if not no_display:
            cv2.imshow('Inferência TFLite Quantizado', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Liberação de Recursos
    cap.release()
    if not no_display:
        cv2.destroyAllWindows()
    print("Inferência TFLite encerrada.")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Roda um modelo TFLite (inclui modo sem câmera).")
    parser.add_argument("--model", default="yolov8n_fullint.tflite", help="Caminho do .tflite")
    parser.add_argument("--source", default="0", help="Índice da câmera (ex: 0) ou caminho de vídeo")
    parser.add_argument("--image", default=None, help="Rodar em uma imagem única (sem câmera)")
    parser.add_argument("--bench", type=int, default=0, help="Benchmark sem câmera: N iterações")
    parser.add_argument("--no-display", action="store_true", help="Não abrir janelas (headless)")
    parser.add_argument("--rgb", action="store_true", help="Converter BGR->RGB no pré-processamento")

    args = parser.parse_args()

    source = _parse_source(args.source)
    keep_bgr = not args.rgb
    raise SystemExit(
        rodar_tflite_quantizado(
            model_path=args.model,
            source=source,
            image_path=args.image,
            bench_iters=args.bench,
            no_display=args.no_display,
            keep_bgr=keep_bgr,
        )
    )