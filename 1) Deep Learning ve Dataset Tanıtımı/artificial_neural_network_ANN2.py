import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Veri setini yükle
x_l = np.load('X.npy')
Y_l = np.load('Y.npy')
img_size = 64

plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')

# Veri Hazırlığı
X = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0)
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0], 1)
print("X shape:", X.shape)
print("Y shape:", Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# Verileri düzleştir
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)
print("X train flatten", X_train_flatten.shape)
print("X test flatten", X_test_flatten.shape)

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train:", x_train.shape)
print("x test:", x_test.shape)
print("y train:", y_train.shape)
print("y test:", y_test.shape)

import tensorflow as tf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Modeli tanımla
def build_classifier():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=8, activation="relu", input_shape=(x_train.shape[0],)),
        tf.keras.layers.Dense(units=4, activation="relu"),
        tf.keras.layers.Dense(units=1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# K-fold cross-validation ayarla
kf = KFold(n_splits=3, shuffle=True, random_state=42)
accuracies = []

# K-fold cross-validation uygula
for train_index, test_index in kf.split(x_train.T):
    model = build_classifier()
    history = model.fit(x_train.T[train_index], y_train.T[train_index], epochs=100, batch_size=10, verbose=0)
    _, accuracy = model.evaluate(x_train.T[test_index], y_train.T[test_index], verbose=0)
    accuracies.append(accuracy)

mean = np.mean(accuracies)
variance = np.std(accuracies)

print("Accuracy mean: " + str(mean))
print("Accuracy variance: " + str(variance))
