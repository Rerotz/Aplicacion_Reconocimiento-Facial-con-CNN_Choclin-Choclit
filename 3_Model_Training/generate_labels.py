"""
generate_labels.py
Genera el archivo class_labels.txt a partir de class_indices.json
"""
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--class_indices', default='./models/class_indices.json', help='Ruta al archivo JSON con índices de clases')
parser.add_argument('--output', default='../4_Mobile_App/android/app/src/main/assets/class_labels.txt', help='Ruta de salida del archivo TXT')
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

with open(args.class_indices, 'r') as f:
    class_indices = json.load(f)

# Ordenar por índice (value)
labels = [name for name, idx in sorted(class_indices.items(), key=lambda x: x[1])]

with open(args.output, 'w', encoding='utf-8') as f:
    for label in labels:
        f.write(f"{label}\n")

print(f"Archivo class_labels.txt generado en: {args.output}")
