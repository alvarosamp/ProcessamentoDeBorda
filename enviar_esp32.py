"""Envia resultados de detecção do YOLOv8 (rodando no PC) para uma ESP32 via serial.

Requer: pip install pyserial

Uso típico (a partir de outro script, ex. rodartflite.py ou Modelorasp.py):

    from enviar_esp32 import ConexaoESP32

    esp = ConexaoESP32(porta="COM3")
    esp.enviar_deteccao(classe_id=0, score=0.87)
    esp.fechar()
"""

import serial


class ConexaoESP32:
    def __init__(self, porta="COM3", baudrate=115200, timeout=1):
        self._serial = serial.Serial(porta, baudrate=baudrate, timeout=timeout)

    def enviar_deteccao(self, classe_id: int, score: float):
        linha = f"DETECT,{classe_id},{score:.2f}\n"
        self._serial.write(linha.encode("utf-8"))

    def fechar(self):
        self._serial.close()


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Teste rápido de envio para a ESP32 via serial.")
    parser.add_argument("--porta", default="COM3", help="Porta serial da ESP32 (ex: COM3)")
    parser.add_argument("--classe", type=int, default=0, help="ID da classe a simular")
    parser.add_argument("--score", type=float, default=0.9, help="Score a simular")
    args = parser.parse_args()

    esp = ConexaoESP32(porta=args.porta)
    time.sleep(2)  # aguarda a ESP32 resetar após abrir a porta serial
    esp.enviar_deteccao(args.classe, args.score)
    print(f"Enviado: DETECT,{args.classe},{args.score:.2f}")
    esp.fechar()
