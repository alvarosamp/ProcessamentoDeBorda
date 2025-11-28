import cv2
import os

print("=== Teste rápido de câmeras ===")
# Lista /dev/video*
if os.name == 'posix':
    devs = [d for d in os.listdir('/dev') if d.startswith('video')]
    print("/dev entries:", devs)

# Testa índices 0..5
for i in range(6):
    cap = cv2.VideoCapture(i)
    opened = cap.isOpened()
    ret = False
    if opened:
        ret, frame = cap.read()
    print(f"Index {i}: opened={opened}, read_ok={bool(ret)}")
    cap.release()