import cv2
from ultralytics import YOLO
import easyocr
import os
import shutil
import re
import pandas as pd
import sys
import time

# === Configurações principais ===
RTSP_URL = "rtsp://admin:@192.168.31.252:554/cam/realmonitor?channel=1&subtype=0"
MODO_DIVISAO = 'vertical'  # 'vertical' (cima/baixo) ou 'horizontal' (esquerda/direita)
SKIP_FRAMES = 5            # Pula frames para economizar processamento
YOLO_CONFIDENCE = 0.7
CROP_UPSCALE = 2
YOLO_MODEL_PATH = 'model.pt'

# === Saída ===
OUTPUT_BASE = 'outputs_camera_baixo'
OUTPUT_VIDEO = os.path.join(OUTPUT_BASE, 'camera_baixo_resultado.mp4')
OUTPUT_CSV = os.path.join(OUTPUT_BASE, 'camera_baixo_deteccoes_placas.csv')
CROPS_FOLDER = os.path.join(OUTPUT_BASE, 'placas_crops')
CAR_CROPS_FOLDER = os.path.join(OUTPUT_BASE, 'carros_crops')
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
OUTPUT_FPS = 15
DISPLAY_RESIZE_WIDTH = 800

# === Diretórios de saída ===
if os.path.exists(OUTPUT_BASE):
    shutil.rmtree(OUTPUT_BASE)
os.makedirs(CROPS_FOLDER)
os.makedirs(CAR_CROPS_FOLDER)

# === Modelos ===
try:
    model = YOLO(YOLO_MODEL_PATH)
    reader = easyocr.Reader(['pt', 'en'], gpu=True)
except Exception as e:
    print(f"Erro ao carregar modelos: {e}")
    sys.exit()

CLASS_MAP = {0: 'mercosul', 1: 'antigo', 2: 'car'}

def process_frame(frame, idx):
    """
    Processa metade do frame, roda YOLO e OCR, retorna frame com anotações e resultados.
    """
    h, w = frame.shape[:2]
    results = model(frame, verbose=False)
    detections_this_frame = []
    car_boxes = []

    # Detecta carros
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            if confs[i] > YOLO_CONFIDENCE and classes[i] == 2:
                car_boxes.append(boxes[i])

    # Detecta placas
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            if confs[i] <= YOLO_CONFIDENCE or classes[i] not in [0, 1]:
                continue
            classe = classes[i]
            classe_nome = CLASS_MAP.get(classe, 'desconhecida')
            x1, y1, x2, y2 = [int(a) for a in boxes[i]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] <= 1 or crop.shape[1] <= 1:
                continue
            upscale_dim = (crop.shape[1] * CROP_UPSCALE, crop.shape[0] * CROP_UPSCALE)
            crop_up = cv2.resize(crop, upscale_dim, interpolation=cv2.INTER_CUBIC)
            crop_filename = f'frame{idx:06d}_det{i}.jpg'
            crop_path = os.path.join(CROPS_FOLDER, crop_filename)
            cv2.imwrite(crop_path, crop_up)

            # Relaciona placa ao carro mais próximo
            placas_cx = (x1 + x2) // 2
            placas_cy = (y1 + y2) // 2
            matched_car_box = None
            car_crop_path = None
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

            if matched_car_box is not None:
                car_x1, car_y1, car_x2, car_y2 = [int(a) for a in matched_car_box]
                car_x1 = max(0, car_x1)
                car_y1 = max(0, car_y1)
                car_x2 = min(w, car_x2)
                car_y2 = min(h, car_y2)
                car_crop = frame[car_y1:car_y2, car_x1:car_x2]
                if car_crop.size > 0:
                    car_crop_filename = f'frame{idx:06d}_det{i}.jpg'
                    car_crop_path = os.path.join(CAR_CROPS_FOLDER, car_crop_filename)
                    cv2.imwrite(car_crop_path, car_crop)
                    cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), (255, 128, 0), 2)

            # OCR multi-tentativas
            txt_raw = "[NÃO LIDO]"
            conf = 0.0
            ocr_results = reader.readtext(crop_up, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if ocr_results:
                best_res = sorted(ocr_results, key=lambda x:x[2], reverse=True)[0]
                if best_res[2] > 0.7:
                    txt_raw, conf = best_res[1], best_res[2]
            if conf < 0.7:
                gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gray_clahe = clahe.apply(gray)
                ocr_results = reader.readtext(gray_clahe, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                if ocr_results:
                    best_res = sorted(ocr_results, key=lambda x:x[2], reverse=True)[0]
                    if best_res[2] > conf:
                        txt_raw, conf = best_res[1], best_res[2]
            if conf < 0.7:
                _, bin_otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                ocr_results = reader.readtext(bin_otsu, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                if ocr_results:
                    best_res = sorted(ocr_results, key=lambda x:x[2], reverse=True)[0]
                    if best_res[2] > conf:
                        txt_raw, conf = best_res[1], best_res[2]
            if conf < 0.7:
                bin_adapt = cv2.adaptiveThreshold(gray_clahe,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,9)
                ocr_results = reader.readtext(bin_adapt, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                if ocr_results:
                    best_res = sorted(ocr_results, key=lambda x:x[2], reverse=True)[0]
                    if best_res[2] > conf:
                        txt_raw, conf = best_res[1], best_res[2]

            txt_up = ''.join(c for c in txt_raw.upper() if c.isalnum())
            txt_norm = '[NÃO LIDO]'
            if classe == 0:
                m = re.findall(r'[A-Z]{3}\d[A-Z]\d{2}', txt_up)
                if m: txt_norm = m[0]
            elif classe == 1:
                m = re.findall(r'[A-Z]{3}\d{4}', txt_up)
                if m: txt_norm = m[0]

            detections_this_frame.append({
                'frame': idx,
                'classe': CLASS_MAP[classe],
                'conf_yolo': confs[i],
                'placa_raw': txt_raw,
                'placa_normalizada': txt_norm,
                'conf_ocr': conf,
                'crop_placa_file': crop_path,
                'crop_carro_file': car_crop_path if matched_car_box is not None else None
            })

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            display_text = f"{txt_norm} ({conf:.2f})"
            cv2.putText(frame, display_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)

    return frame, detections_this_frame

if __name__ == "__main__":
    cap = None
    out_vid = None
    all_results = []

    try:
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir o stream RTSP em {RTSP_URL}")
            sys.exit()

        ret, f0 = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o primeiro frame.")
            cap.release()
            sys.exit()

        H, W = f0.shape[:2]

        # Define ROI da metade a ser processada
        if MODO_DIVISAO == 'vertical':
            split_h = H // 2
            roi_y_start, roi_y_end = split_h, H
            roi_x_start, roi_x_end = 0, W
            frame_size_split = (W, split_h)
        elif MODO_DIVISAO == 'horizontal':
            split_w = W // 2
            roi_y_start, roi_y_end = 0, H
            roi_x_start, roi_x_end = split_w, W
            frame_size_split = (split_w, H)
        else:
            print("MODO_DIVISAO inválido.")
            cap.release()
            sys.exit()

        out_vid = cv2.VideoWriter(OUTPUT_VIDEO, FOURCC, OUTPUT_FPS, frame_size_split)
        if not out_vid.isOpened():
            print(f"Erro ao criar VideoWriter para {OUTPUT_VIDEO}.")
            cap.release()
            sys.exit()

        idx = 0
        processed_frame_count = 0

        while True:
            ret, frame_original = cap.read()
            if not ret:
                print("Erro ao ler frame ou fim do stream.")
                break

            if idx % (SKIP_FRAMES + 1) != 0:
                idx += 1
                continue

            frame_to_process = frame_original[roi_y_start:roi_y_end, roi_x_start:roi_x_end].copy()
            processed_frame, dets_this_frame = process_frame(frame_to_process, idx)
            all_results.extend(dets_this_frame)
            out_vid.write(processed_frame)

            display_h = int(processed_frame.shape[0] * DISPLAY_RESIZE_WIDTH / processed_frame.shape[1])
            display_w = DISPLAY_RESIZE_WIDTH
            if display_h > 0 and display_w > 0:
                cv2.imshow("Camera Processada (Pressione q para sair)", cv2.resize(processed_frame, (display_w, display_h)))
            else:
                cv2.imshow("Camera Processada (Pressione q para sair)", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            idx += 1
            processed_frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    except Exception as e:
        print(f"\nErro inesperado: {e}")

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        if out_vid is not None and out_vid.isOpened():
            out_vid.release()
        cv2.destroyAllWindows()

        # Salva resultados (se houver)
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            print(f"Salvo: {OUTPUT_CSV}")

        print("Processamento finalizado.")
