from ultralytics import YOLO
import os

# =========================
# CONFIGURAÇÕES DO USUÁRIO
# =========================
MODEL_PT = r"C:\Users\vish8\OneDrive\Desktop\IC\ProcessamentoDeBorda\modelo.pt"

# Caminho do YAML de dados (somente se for usar INT8)
# Ex.: data.yaml do seu treinamento YOLO (com train/val/test configurados)
DATA_YAML = r"C:\Users\vish8\OneDrive\Desktop\IC\ProcessamentoDeBorda\data.yaml"

# Se quiser tentar exportar INT8, coloque True (requer DATA_YAML válido)
EXPORT_INT8 = False

# =========================
# SCRIPT DE EXPORTAÇÃO
# =========================

def main():
    if not os.path.exists(MODEL_PT):
        raise FileNotFoundError(f"Arquivo .pt não encontrado em: {MODEL_PT}")

    print(f"1) Carregando modelo base: {MODEL_PT}")
    model = YOLO(MODEL_PT)

    # Diretório de saída (mesmo do modelo .pt)
    out_dir = os.path.dirname(MODEL_PT)
    base_name = os.path.splitext(os.path.basename(MODEL_PT))[0]

    # -----------------------
    # 2) Exportar para ONNX
    # -----------------------
    print("\n2) Exportando para ONNX...")
    try:
        # Ultralytics grava o arquivo e retorna o caminho
        onnx_file = model.export(
            format="onnx",
            opset=12,       # 12 é bem suportado; se quiser manter 22, pode alterar
            simplify=True
        )
        # Se o retorno for None, montamos o path manual
        if onnx_file is None:
            onnx_file = os.path.join(out_dir, f"{base_name}.onnx")

        print(f"[OK] ONNX gerado em: {onnx_file}")
    except Exception as e:
        print(f"[ERRO] Falha ao exportar ONNX: {e}")
        onnx_file = None

    # -----------------------------
    # 3) Exportar para TFLite FP32
    # -----------------------------
    print("\n3) Exportando para TFLite (Float32)...")
    try:
        tflite_fp32_file = model.export(
            format="tflite",   # por padrão, Float32
            imgsz=640          # ajuste se seu treino tiver outro tamanho
        )
        if tflite_fp32_file is None:
            tflite_fp32_file = os.path.join(out_dir, f"{base_name}_fp32.tflite")

        print(f"[OK] TFLite Float32 gerado em: {tflite_fp32_file}")
    except Exception as e:
        print(f"[ERRO] Falha ao exportar TFLite Float32: {e}")
        tflite_fp32_file = None

    # --------------------------------
    # 4) (Opcional) Exportar TFLite INT8
    # --------------------------------
    if EXPORT_INT8:
        print("\n4) Exportando para TFLite (INT8 quantizado)...")
        if not os.path.exists(DATA_YAML):
            print(f"[ERRO] DATA_YAML não encontrado: {DATA_YAML}")
            print("       Desative EXPORT_INT8 ou corrija o caminho do YAML.")
        else:
            try:
                tflite_int8_file = model.export(
                    format="tflite",
                    int8=True,
                    data=DATA_YAML,
                    imgsz=640
                )
                if tflite_int8_file is None:
                    tflite_int8_file = os.path.join(out_dir, f"{base_name}_int8.tflite")

                print(f"[OK] TFLite INT8 gerado em: {tflite_int8_file}")
            except Exception as e:
                print(f"[ERRO] Falha ao exportar TFLite INT8: {e}")
                tflite_int8_file = None
    else:
        print("\n4) Exportação INT8 desativada (EXPORT_INT8 = False).")

    # -----------------------
    # RESUMO FINAL
    # -----------------------
    print("\n===== RESUMO DOS ARQUIVOS GERADOS =====")
    print(f"- Modelo base (.pt): {MODEL_PT}")
    print(f"- ONNX:              {onnx_file}")
    print(f"- TFLite Float32:    {tflite_fp32_file}")
    if EXPORT_INT8:
        print("  (Veja logs acima para o arquivo INT8)")


if __name__ == "__main__":
    main()
