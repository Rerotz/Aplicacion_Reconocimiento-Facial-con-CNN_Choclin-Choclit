"""
preprocess.py
- Lee dataset raw (carpetas por persona)
- Detección + recorte facial (por seguridad, aplica segunda detección)
- Redimensiona a tamaño objetivo
- Guarda dataset limpio listo para training
- Opcional: realiza augmentaciones básicas y guarda en folder 'augmented'
"""
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from albumentations import Compose, HorizontalFlip, Rotate, RandomBrightnessContrast, ShiftScaleRotate, Blur

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='../dataset/raw', help='Dataset raw (carpetas por persona)')
parser.add_argument('--output', default='../dataset/processed', help='Output processed')
parser.add_argument('--img_size', type=int, default=224, help='Tamaño final (img_size x img_size)')
parser.add_argument('--augment', action='store_true', help='Generar augmentaciones')
args = parser.parse_args()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

aug = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
    RandomBrightnessContrast(p=0.5),
    Blur(blur_limit=3, p=0.2),
])

os.makedirs(args.output, exist_ok=True)

for person in os.listdir(args.input):
    in_dir = os.path.join(args.input, person)
    out_dir = os.path.join(args.output, person)
    os.makedirs(out_dir, exist_ok=True)
    for fname in tqdm(os.listdir(in_dir), desc=f"Procesando {person}"):
        path = os.path.join(in_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            # Si no detecta, usar la imagen entera resized (fallback)
            face = cv2.resize(img, (args.img_size, args.img_size))
        else:
            # Tomar la mayor detección
            x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
            pad = int(0.2 * w)
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
            face = img[y1:y2, x1:x2]
            face = cv2.resize(face, (args.img_size, args.img_size))
        # normal save
        base_out = os.path.join(out_dir, fname)
        cv2.imwrite(base_out, face)
        # augment
        if args.augment:
            for i in range(3):  # 3 augmentaciones por imagen
                augmented = aug(image=face)['image']
                aug_name = os.path.splitext(fname)[0] + f"_aug{i}.jpg"
                cv2.imwrite(os.path.join(out_dir, aug_name), augmented)

print("Preprocessing finalizado.")
