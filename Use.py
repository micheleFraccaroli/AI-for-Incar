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
		model = load_model('ANN_inference.h5')

		lgen = lg()
		test_data = np.loadtxt("Test/test_data.txt", delimiter=" ")
		
		test_data = test_data[98:]

		list_res = []
		for i in range(20):
			data = [test_data[i]]
			#data = [[7.202, -0.306, 6.129]]
			data = np.array(data)
			#data = np.expand_dims(data,axis=2)

			#print(data.shape)

			res = model.predict(data)
			if(np.argmax(res) == 0):
				res = 'no auto'
			else:
				res = 'auto'

			list_res.append(res)

		return list_res

	def verdict(self, list_res):
		print("Auto 	→ " ,list_res.count("auto"))
		print("No auto  → " ,list_res.count("no auto"))

if __name__ == '__main__':
	u = Use()
	l = u.usage()
	#print(l)
	u.verdict(l)