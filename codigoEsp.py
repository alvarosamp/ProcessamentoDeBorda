# Firmware MicroPython para a ESP32.
# Grave este arquivo na placa como main.py (use ampy, rshell, Thonny ou mpremote).
# A ESP32 fica ouvindo a serial (USB) esperando linhas no formato:
#   DETECT,<classe_id>,<score>\n
# e aciona o pino LED_PIN quando recebe uma detecção com score acima do limiar.

from machine import Pin
import sys

LED_PIN = 2          # GPIO do LED onboard na maioria das devkits ESP32
SCORE_THRESHOLD = 0.5

led = Pin(LED_PIN, Pin.OUT)


def processar_linha(linha):
    partes = linha.strip().split(",")
    if not partes or partes[0] != "DETECT":
        return

    if len(partes) < 3:
        return

    try:
        classe_id = int(partes[1])
        score = float(partes[2])
    except ValueError:
        return

    if score >= SCORE_THRESHOLD:
        led.value(1)
    else:
        led.value(0)


def main():
    buffer = ""
    while True:
        char = sys.stdin.read(1)
        if char in ("\n", "\r"):
            if buffer:
                processar_linha(buffer)
                buffer = ""
        else:
            buffer += char


if __name__ == "__main__":
    main()
