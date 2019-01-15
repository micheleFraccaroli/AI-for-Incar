import numpy as np
import random

dataset = np.loadtxt("dataset.txt", delimiter=" ")
X_train_CCC = dataset[:70]
X_train = np.random.random((70, 3))

print(type(X_train_CCC))
print(X_train_CCC.shape)
print(X_train_CCC)
print("###############################################")
print(type(X_train))
print(X_train.shape)
print(X_train)

