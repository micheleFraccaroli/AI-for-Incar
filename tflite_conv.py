from tensorflow.contrib import lite

converter = lite.TFLiteConverter.from_keras_model_file('accelerometer_inference.h5')
tfmodel = converter.convert()
open ("accelerometer.tflite","wb").write(tfmodel)