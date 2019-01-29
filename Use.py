import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from labels_gen import labels_gen as lg

# hide AVX2 warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Use:
	def usage(self):
		model = load_model('accelerometer_inference.h5')

		lgen = lg()
		test_data = np.loadtxt("Test/test_data.txt", delimiter=" ")
		
		for i in test_data:
			data = [i]
			#data = [[7.202, -0.306, 6.129]]
			data = np.array(data)
			data = np.expand_dims(data,axis=2)

			#print(data.shape)

			res = model.predict(data)
			if(np.argmax(res) == 0):
				res = 'no auto'
			else:
				res = 'auto'

			print(res)

if __name__ == '__main__':
	u = Use()
	u.usage()