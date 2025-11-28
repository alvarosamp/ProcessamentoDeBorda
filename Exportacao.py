from ultralytics import YOLO

MODEL_NAME = 'yolov8n.pt'
model = YOLO(MODEL_NAME) 

print(f"1. Carregando modelo base: {MODEL_NAME}")

# 2. Exporta para TFLite Float32 (apenas conversão de formato)
print("\n2. Exportando para TFLite (Float32)...")
model.export(
    format='tflite', 
    int8=False,  # Sem quantização
    simplify=True 
)

# 3. Exporta para TFLite Int8 (Conversão e Quantização)
print("\n3. Exportando para TFLite (Int8 Quantizado)...")
model.export(
    format='tflite', 
    int8=True,   # Quantização de 8-bit (principal otimização)
    simplify=True 
)

print("\n[SUCESSO] Arquivos gerados:")
print(f"- {MODEL_NAME} (PyTorch Float32)")
print(f"- {MODEL_NAME.replace('.pt', '.tflite')} (TFLite Float32)")
print(f"- {MODEL_NAME.replace('.pt', '_fullint.tflite')} (TFLite Int8 Quantizado)")