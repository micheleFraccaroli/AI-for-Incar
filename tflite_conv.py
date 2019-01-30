from tensorflow.contrib import lite

converter = lite.TFLiteConverter.from_keras_model_file('ANN_inference.h5')
tfmodel = converter.convert()
open ("RNN.tflite","wb").write(tfmodel)