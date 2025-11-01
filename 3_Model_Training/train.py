"""
Entrenamiento balanceado y robusto con Fine-Tuning (MobileNetV2)
- Corrige desbalance de clases con class_weight
- Reduce sobreajuste y mejora generalización
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse, os, json, numpy as np, matplotlib.pyplot as plt

# ==========================================================
# 1. PARÁMETROS
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../dataset/processed', help='Carpetas por clase')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--fine_tune', type=int, default=80, help='Número de capas a descongelar para fine-tuning')
parser.add_argument('--output', default='./models_balanced', help='Carpeta de salida')
args = parser.parse_args()

# ==========================================================
# 2. DATASET CON AUMENTACIÓN MODERADA
# ==========================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    rotation_range=35,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=0.25,
    brightness_range=[0.5, 1.5],
    shear_range=10,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_gen = train_datagen.flow_from_directory(
    args.data_dir,
    target_size=(args.img_size, args.img_size),
    batch_size=args.batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    args.data_dir,
    target_size=(args.img_size, args.img_size),
    batch_size=args.batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_gen.class_indices)
print("🔍 Clases detectadas:", train_gen.class_indices)

# ==========================================================
# 3. CÁLCULO DE PESOS POR CLASE (BALANCE)
# ==========================================================
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("⚖️ Pesos de clase aplicados:", class_weights)

# ==========================================================
# 4. MODELO BASE + CAPAS SUPERIORES
# ==========================================================
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(args.img_size, args.img_size, 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
preds = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=preds)

# --- Etapa 1: congelar base ---
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
print("🧱 Entrenando capas superiores...")
history_1 = model.fit(train_gen, validation_data=val_gen, epochs=5, class_weight=class_weights)

# ==========================================================
# 5. FINE-TUNING (descongelar últimas capas)
# ==========================================================
for layer in base.layers[-args.fine_tune:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print(f"🔧 Fine-tuning activado (últimas {args.fine_tune} capas)...")
history_2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=args.epochs,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.output, 'best_model.h5'), save_best_only=True)
    ]
)

# ==========================================================
# 6. GUARDAR Y GRAFICAR
# ==========================================================
os.makedirs(args.output, exist_ok=True)
model.save(os.path.join(args.output, 'final_model.h5'))

with open(os.path.join(args.output, 'class_indices.json'), 'w') as f:
    json.dump(train_gen.class_indices, f)

# --- Graficar resultados combinados ---
def plot_history(h1, h2):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(h1.history['accuracy'] + h2.history['accuracy'], label='Entrenamiento')
    plt.plot(h1.history['val_accuracy'] + h2.history['val_accuracy'], label='Validación')
    plt.legend(); plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(h1.history['loss'] + h2.history['loss'], label='Entrenamiento')
    plt.plot(h1.history['val_loss'] + h2.history['val_loss'], label='Validación')
    plt.legend(); plt.title("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "training_curves.png"))
    plt.show()

plot_history(history_1, history_2)

# ==========================================================
# 7. MATRIZ DE CONFUSIÓN
# ==========================================================
y_true = val_gen.classes
y_pred = np.argmax(model.predict(val_gen), axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=list(train_gen.class_indices.keys()))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Matriz de Confusión (Validación)")
plt.savefig(os.path.join(args.output, "confusion_matrix.png"))
plt.show()

print("✅ Entrenamiento balanceado completado correctamente.")
