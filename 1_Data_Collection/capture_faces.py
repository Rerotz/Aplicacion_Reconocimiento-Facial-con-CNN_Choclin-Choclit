"""
capture_faces.py
Script para capturar imágenes de rostros usando OpenCV.
Uso:
    python capture_faces.py --name "Nombre" --output ../dataset/raw/Nombre --count 300
"""
import cv2
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, help='Nombre de la persona (carpeta)')
parser.add_argument('--output', default='../dataset/raw', help='Directorio base de salida')
parser.add_argument('--count', type=int, default=300, help='Número de imágenes a capturar')
parser.add_argument('--cam', type=int, default=0, help='Índice de cámara')
args = parser.parse_args()

out_dir = os.path.join(args.output, args.name)
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(args.cam)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0
print("Presiona 'q' para salir antes. Comenzando captura...")

while count < args.count:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        # extraer ROIs con algo de padding
        pad = int(0.2 * w)
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
        face = frame[y1:y2, x1:x2]
        # guardar
        fname = f"{args.name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), face)
        count += 1
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        if count % 20 == 0:
            print(f"Capturadas {count}/{args.count}")
    cv2.imshow('capture', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Finalizado. {count} imágenes guardadas en {out_dir}")
