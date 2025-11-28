## ProcessamentoDeBorda — README


Link para pasta no drive : https://drive.google.com/drive/folders/1g3x8MJF9OEcAV2W9HhRZ2l_vymIgVsv2?usp=sharing

Este repositório mostra exemplos simples de inferência de visão computacional usando YOLOv8, com foco em Edge AI / processamento de borda. Contém scripts para: (1) exportar um modelo PyTorch para TFLite (incluindo quantizado), (2) rodar YOLOv8 diretamente com a biblioteca Ultralytics (PyTorch) e (3) rodar um modelo TFLite otimizado (Int8) — útil em dispositivos de baixa potência como Raspberry Pi.

## Arquivos principais

- `Exportacao.py`
	- Carrega um modelo YOLOv8 (`yolov8n.pt`) e exporta para TFLite.
	- Gera dois artefatos: um `.tflite` em float32 e um quantizado (Int8) tipicamente nomeado `yolov8n_fullint.tflite`.
	- Uso típico: preparar o modelo para deployment em edge devices.

- `Modelorasp.py`
	- Usa a API `ultralytics.YOLO` para carregar o `.pt` e executar inferência em tempo real a partir da câmera.
	- Indicada para ambientes onde o PyTorch e a dependência `ultralytics` podem ser rodados (ex.: PC ou dispositivos com suporte razoável de CPU).
	- Exibe janelas OpenCV com boxes e labels quando `show=True`.

- `rodartflite.py`
	- Demonstra como carregar um modelo TFLite (versão otimizada/quantizada) usando `tflite_runtime.Interpreter`.
	- Realiza leitura da câmera, pré-processamento simples (resize para 640x640), invoca o interpreter e mostra FPS e informações na tela.
	- Projetado para ser leve e apropriado para Raspberry Pi (ou outro hardware ARM sem GPU).

- `metricas.py` (se presente)
	- Arquivo indicado para funções auxiliares de cálculo de métricas (não abordado em detalhe aqui). Pode conter cálculos de FPS, latência, precisão, etc.

- `Modelo/`
	- Pasta sugerida para armazenar modelos gerados/baixados (ex.: `yolov8n.pt`, `yolov8n.tflite`, `yolov8n_fullint.tflite`).

## Como funciona (fluxo geral)

1. Treine ou obtenha um modelo YOLOv8 (`.pt`) — aqui usamos `yolov8n.pt` (nano).
2. Use `Exportacao.py` para converter o `.pt` em TFLite. Isso produz uma versão float e outra quantizada (Int8) para deployment em edge devices.
3. Para testes em um computador com recursos, rode `Modelorasp.py` que usa `ultralytics`/PyTorch para inferência e plot das detecções.
4. Para execução em um Raspberry Pi ou outro dispositivo com recursos limitados, use `rodartflite.py` com o TFLite Int8 (`yolov8n_fullint.tflite`). Esse fluxo reduz consumo de CPU e memória.

## Edge AI e Processamento de Borda — resumo e por que usar

Edge AI (ou processamento de borda) significa executar inferência de modelos de ML do lado do dispositivo (câmeras, gateways, Raspberry Pi, etc.), em vez de enviar dados para a nuvem. Vantagens principais:

- Latência reduzida: inferência local evita ida e volta para servidores remotos.
- Privacidade: imagens sensíveis podem permanecer no dispositivo.
- Economia de banda: só eventos/dados relevantes são enviados.
- Resiliência: funcionamento mesmo sem conexão de rede.

Desafios e trade-offs:

- Recursos limitados: CPU, memória e energia são restritos — por isso usamos modelos leves (yolov8n) e quantização (Int8).
- Precisão vs. desempenho: técnicas como quantização e pruning reduzem precisão levemente em troca de velocidade e economia de energia.

Como este projeto mapeia para Edge AI:

- `Exportacao.py` — preparação do modelo para edge (conversão + quantização).
- `rodartflite.py` — exemplo de runtime leve usando TFLite (ideal em edge devices).
- `Modelorasp.py` — exemplo de execução local com PyTorch quando recursos permitem.

## Requisitos e instalação

Recomendações de pacotes (Python 3.8+):

 - No desktop/PC (para `Exportacao.py` e `Modelorasp.py`):

```bash
python3 -m pip install ultralytics opencv-python numpy
```

 - No Raspberry Pi (para `rodartflite.py`) prefira `tflite-runtime` (versão compatível com sua arquitetura) e `opencv-python-headless` ou `opencv-python`:

```bash
# Exemplo genérico (ajuste a versão do tflite-runtime conforme o seu Pi):
python3 -m pip install numpy opencv-python
# Instale tflite-runtime pela wheel adequada para sua plataforma (ou via apt se disponível)
```

Observação: `ultralytics` depende de PyTorch; em muitos RPis não há suporte adequado para PyTorch — por isso, para edge real em RPi normalmente usamos TFLite (via `tflite-runtime`).

## Como executar

 - Exportar os modelos para TFLite (no PC que tem `ultralytics` e PyTorch):

```bash
python3 Exportacao.py
```

Isso deve gerar algo como `yolov8n.tflite` e `yolov8n_fullint.tflite` (Int8 quantizado).

 - Rodar inferência com a implementação PyTorch/Ultralytics (PC):

```bash
python3 Modelorasp.py
```

 - Rodar inferência TFLite (no Raspberry Pi ou local, usando `tflite-runtime`):

```bash
python3 rodartflite.py
```

Parâmetros a ajustar:

 - `CAMERA_SOURCE` (em `Modelorasp.py` e `rodartflite.py`) — índice da câmera ou caminho para arquivo/RTSP.
 - `TFLITE_MODEL_PATH` em `rodartflite.py` — coloque o caminho para o `.tflite` gerado.

## Boas práticas para quantização (Int8)

- Use um conjunto representativo de imagens (dataset) para calibrar a quantização, se a sua ferramenta suportar.
- Verifique perda de precisão avaliando o modelo quantizado em um conjunto de validação.
- Teste latência e consumo de CPU no dispositivo alvo — às vezes mudanças de pre-processing (ex.: resize, ordem de canais) impactam muito.

## Troubleshooting rápido

- Erro ao carregar `Interpreter`: verifique se `tflite-runtime` está instalado e compatível com a arquitetura (ARMv7/ARM64). Em alguns casos, é necessário instalar via wheel específica.
- Webcam não abre: confirme `CAMERA_SOURCE` correto e permissões do sistema (em macOS/raspberry use `ls /dev` para checar dispositivos de câmera no Linux).
- `ultralytics` não instala: provavelmente falta compatibilidade do PyTorch no ambiente — prefira fazer export no PC e rodar TFLite no Raspberry Pi.

