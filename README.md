
# CAM-IA-CONTROLLER

Sistema completo de controle de acesso veicular utilizando visão computacional, inteligência artificial e integração com backend.

---

## ✨ Funcionalidades

- **Detecção automática** de carros e placas usando YOLOv12 treinado.
- **Reconhecimento de placas (OCR)** Com EasyOCR.
- **Integração total** com backend: registra logs de acesso com imagem do carro.
- **Dois modos de entrada**:
  - RTSP (câmera IP ao vivo)
  - Vídeo local (ex: `inputs/1.mp4`)
- **Visualização ao vivo** opcional do processamento.
---

## 📂 Organização

```
cam-ia-controller/
├── cam-ia-controller/
│   ├── 1.mp4
│   ├── 2.mov
│   ├── train_model/
│   │   └── yolo.ipynb
│   ├── .env
│   ├── model.pt
│   ├── requirements.txt
│   ├── run_integrate.py        # Script principal: integração com backend
│   ├── test_only_model.py      # Teste offline: apenas processamento local e vídeo
```

---

## 🚀 Como Usar

### 1. Instale as dependências

Crie e ative um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate        # ou venv\Scripts\activate no Windows
pip install -r requirements.txt
```

### 2. Configure o `.env`

Copie o exemplo:

```bash
cp .env.example .env
```

Edite `.env` com seus dados:

```
CAMERA_USER=admin
CAMERA_PASSWORD=senha
CAMERA_IP=192.168.31.252
BACKEND_URL=http://localhost:3000/api
BACKEND_USER=admin
BACKEND_PASS=admin
```

### 3. Rodando em modo RTSP (câmera IP)

No `run_integrate.py`:

```python
USE_LOCAL_VIDEO = False
```

Execute:

```bash
python run_integrate.py
```

### 4. Rodando em modo vídeo local

Coloque seu vídeo em `inputs/` ou outro caminho desejado.

No `run_integrate.py`:

```python
USE_LOCAL_VIDEO = True
LOCAL_VIDEO_PATH = "inputs/1.mp4"
```

Execute:

```bash
python run_integrate.py
```

### 5. Teste Offline

Para testar só o modelo local, salve imagens/recortes/CSV:

```bash
python test_only_model.py
```

---

## ⚙️ Parâmetros principais

- `YOLO_MODEL_PATH`: caminho do modelo YOLO treinado.
- `SKIP_FRAMES`: controla a frequência de processamento para aliviar o uso da CPU/GPU.
- `SHOW_LIVE_VIEW`: ativa/desativa a janela de acompanhamento ao vivo.
- `USE_LOCAL_VIDEO`: alterna entre RTSP e arquivo local.
- `CROP_UPSCALE`: fator de upscale do recorte antes do OCR.

---

## 📝 Treinamento do Modelo

Veja o notebook em `train_model/yolo.ipynb` para reproduzir o treinamento do modelo.

---

## 📄 Licença

MIT
