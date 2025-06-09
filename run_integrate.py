import cv2
from ultralytics import YOLO
import easyocr
import os
import re
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

CAMERA_USER = os.getenv("CAMERA_USER")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD")
CAMERA_IP = os.getenv("CAMERA_IP")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000/api")
BACKEND_USER = os.getenv("BACKEND_USER", "admin")
BACKEND_PASS = os.getenv("BACKEND_PASS", "admin")

SHOW_LIVE_VIEW = True # Ativar/desativar acompanhamento ao vivo
USE_LOCAL_VIDEO = True # Se True, roda o video local em vez do stream RTSP

RTSP_URL = f"rtsp://{CAMERA_USER}:{CAMERA_PASSWORD}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=0"
LOCAL_VIDEO_PATH = "inputs/2.mov"
MODO_DIVISAO = 'vertical'
SKIP_FRAMES = 5
YOLO_CONFIDENCE = 0.7
CROP_UPSCALE = 2
YOLO_MODEL_PATH = 'model.pt'

CLASS_MAP = {0: 'mercosul', 1: 'antigo', 2: 'car'}

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def get_backend_token():
    url = f"{BACKEND_URL}/auth/login"
    try:
        resp = requests.post(url, json={"username": BACKEND_USER, "password": BACKEND_PASS})
        resp.raise_for_status()
        token = resp.json()["access_token"]
        print("\nAutenticado no backend com sucesso!")
        return token
    except Exception as e:
        print(f"Erro ao autenticar no backend: {e}")
        sys.exit(1)

def registrar_log(plate, car_crop, token):
    url = f"{BACKEND_URL}/access-history/log"
    files = {"image": ("carro.jpg", car_crop)}
    data = {"plate": plate}
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(url, data=data, files=files, headers=headers)
        if resp.ok:
            print("Realizado o log")
        else:
            print(f"Falha ao registrar log: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Erro ao registrar log: {e}")

def checar_placa(plate, token):
    url = f"{BACKEND_URL}/vehicles"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers)
        if resp.ok:
            lista = resp.json()
            return any(
                v["plate"].replace("-", "") == plate.replace("-", "") for v in lista
            )
        else:
            print(f"Falha ao consultar placas: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"Erro ao consultar placas: {e}")
        return False

def main():
    print("\nBem-vindo! Iniciando o sistema...")
    token = get_backend_token()

    if USE_LOCAL_VIDEO:
        print(f"\nAbrindo vídeo local: {LOCAL_VIDEO_PATH}")
        cap = cv2.VideoCapture(LOCAL_VIDEO_PATH)
    else:
        print(f"\nTentando abrir stream RTSP: {RTSP_URL}")
        cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Erro ao abrir vídeo/stream.")
        sys.exit(1)
    print("Stream/vídeo aberto com sucesso!")

    print(f"\nTentando abrir o modelo: {YOLO_MODEL_PATH}")
    try:
        model = YOLO(YOLO_MODEL_PATH)
        reader = easyocr.Reader(['pt', 'en'], gpu=True)
    except Exception as e:
        print(f"Erro ao carregar modelos: {e}")
        sys.exit(1)
    print(f"Modelo carregado com sucesso!")

    ultimos_logs = {}
    print(f"\nConfigs realizadas, o sistema foi iniciado!")
    idx = 0

    while True:
        ret, frame_original = cap.read()
        if not ret:
            print("Erro ao ler frame/fim do stream/vídeo.")
            break

        if idx % (SKIP_FRAMES + 1) != 0:
            idx += 1
            continue

        H, W = frame_original.shape[:2]
        if USE_LOCAL_VIDEO:
            # Se rodando vídeo local, não faz corte (processa frame inteiro)
            frame_proc = frame_original.copy()
        else:
            # Se for câmera, faz o corte configurado normalmente
            if MODO_DIVISAO == 'vertical':
                frame_proc = frame_original[H//2:H, 0:W].copy()
            elif MODO_DIVISAO == 'horizontal':
                frame_proc = frame_original[0:H, W//2:W].copy()
            else:
                frame_proc = frame_original.copy()


        h, w = frame_proc.shape[:2]
        results = model(frame_proc, verbose=False)
        car_boxes = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                if confs[i] > YOLO_CONFIDENCE and classes[i] == 2:
                    car_boxes.append(boxes[i])

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                if confs[i] <= YOLO_CONFIDENCE:
                    continue
                classe = classes[i]
                if classe not in [0, 1]:
                    continue

                x1, y1, x2, y2 = [int(a) for a in boxes[i]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame_proc[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] <= 1 or crop.shape[1] <= 1:
                    continue
                upscale_dim = (crop.shape[1] * CROP_UPSCALE, crop.shape[0] * CROP_UPSCALE)
                crop_up = cv2.resize(crop, upscale_dim, interpolation=cv2.INTER_CUBIC)

                # OCR e normalização
                txt_raw = "[NÃO LIDO]"
                conf = 0.0
                ocr_results = reader.readtext(crop_up, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                if ocr_results:
                    best_res = sorted(ocr_results, key=lambda x: x[2], reverse=True)[0]
                    if best_res[2] > 0.7:
                        txt_raw, conf = best_res[1], best_res[2]
                txt_up = ''.join(c for c in txt_raw.upper() if c.isalnum())
                txt_norm = '[NÃO LIDO]'
                if classe == 0:
                    m = re.findall(r'[A-Z]{3}\d[A-Z]\d{2}', txt_up)
                    if m: txt_norm = m[0]
                elif classe == 1:
                    m = re.findall(r'[A-Z]{3}\d{4}', txt_up)
                    if m: txt_norm = m[0]

                # Seleção do carro correspondente
                placas_cx = (x1 + x2) // 2
                placas_cy = (y1 + y2) // 2
                matched_car_box = None
                min_dist = float('inf')
                for car_box in car_boxes:
                    car_x1, car_y1, car_x2, car_y2 = [int(a) for a in car_box]
                    if car_x1 <= placas_cx <= car_x2 and car_y1 <= placas_cy <= car_y2:
                        car_cx = (car_x1 + car_x2) // 2
                        car_cy = (car_y1 + car_y2) // 2
                        dist = ((placas_cx - car_cx) ** 2 + (placas_cy - car_cy) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            matched_car_box = car_box
                car_crop_img = None
                if matched_car_box is not None:
                    car_x1, car_y1, car_x2, car_y2 = [int(a) for a in matched_car_box]
                    car_x1, car_y1 = max(0, car_x1), max(0, car_y1)
                    car_x2, car_y2 = min(w, car_x2), min(h, car_y2)
                    car_crop_img = frame_proc[car_y1:car_y2, car_x1:car_x2]
                if txt_norm not in ['[NÃO LIDO]', ""] and car_crop_img is not None:
                    agora = time.time()
                    tempo_ultimo_log = ultimos_logs.get(txt_norm, 0)
                    if agora - tempo_ultimo_log < 120:
                        continue
                    autorizado = checar_placa(txt_norm, token)
                    if not autorizado:
                        print(f"{RED}Placa {txt_norm} nao autorizada{RESET}")
                    else:
                        print(f"{GREEN}Placa {txt_norm} autorizada{RESET}")
                        _, buf = cv2.imencode(".jpg", car_crop_img)
                        registrar_log(txt_norm, buf.tobytes(), token)
                        ultimos_logs[txt_norm] = agora

        if SHOW_LIVE_VIEW:
            display_h = 600
            display_w = int(frame_proc.shape[1] * (display_h / frame_proc.shape[0]))
            frame_disp = cv2.resize(frame_proc, (display_w, display_h))
            cv2.imshow("Acompanhamento ao vivo (Q para sair)", frame_disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Finalizando por comando do usuário...")
                break

        idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
