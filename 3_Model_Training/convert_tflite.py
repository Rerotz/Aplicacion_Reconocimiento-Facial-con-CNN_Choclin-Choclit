"""
convert_tflite.py
Convierte el modelo Keras (.h5) a TFLite.
Soporta:
 - Float32 (default)
 - Post-training quantization (dynamic)
 - Full integer (requiere representante)
"""
import tensorflow as tf
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--keras_model', default='./models/final_model.h5')
parser.add_argument('--output_tflite', default='./models/model.tflite')
parser.add_argument('--quantize', choices=['none', 'dynamic', 'int8'], default='dynamic')
parser.add_argument('--representative_dir', default='../dataset/processed')
args = parser.parse_args()

model = tf.keras.models.load_model(args.keras_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
if args.quantize == 'none':
    tflite_model = converter.convert()
elif args.quantize == 'dynamic':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
elif args.quantize == 'int8':
    # representative dataset generator
    def representative_gen():
        import numpy as np
        import os
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        # recorrer algunas imágenes
        cnt = 0
        for cls in os.listdir(args.representative_dir):
            cls_dir = os.path.join(args.representative_dir, cls)
            for fname in os.listdir(cls_dir):
                img = load_img(os.path.join(cls_dir, fname), target_size=(224,224))
                arr = img_to_array(img)/255.0
                arr = arr.astype('float32')
                yield [np.expand_dims(arr, axis=0)]
                cnt += 1
                if cnt > 100:
                    return
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

os.makedirs(os.path.dirname(args.output_tflite), exist_ok=True)
with open(args.output_tflite, 'wb') as f:
    f.write(tflite_model)
print("TFLite convertido:", args.output_tflite)
