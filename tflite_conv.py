from tensorflow.contrib import lite

converter = lite.TFLiteConverter.from_keras_model_file('LSTM_inference.h5')
tfmodel = converter.convert()
open ("LSTM.tflite","wb").write(tfmodel)