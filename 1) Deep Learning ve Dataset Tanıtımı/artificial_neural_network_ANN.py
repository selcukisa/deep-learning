
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load dataset
x_l = np.load('X.npy')
Y_l = np.load('Y.npy')
img_size = 64

plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')

# Data Preparation
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0)
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# Flatten the data
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)
print("X train flatten", X_train_flatten.shape)
print("X test flatten", X_test_flatten.shape)

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)

def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head

def parameters_weights_and_bias(x_train, y_train):
    n_x = x_train.shape[0]
    n_h = 3  # Hidden layer size (you can adjust this)
    n_y = y_train.shape[0]
    
    parameters = {
        "weight1": np.random.randn(n_h, n_x) * 0.1,
        "bias1": np.zeros((n_h, 1)),
        "weight2": np.random.randn(n_y, n_h) * 0.1,
        "bias2": np.zeros((n_y, 1))
    }
    return parameters

def forward_propagation(x_train, parameters):
    Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"], A1) + parameters["bias2"]
    A2 = sigmoid(Z2)
    
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache

def compute_cost(A2, y_train, parameters):
    logprobs = np.multiply(np.log(A2), y_train) + np.multiply(np.log(1 - A2), 1 - y_train)
    cost = -np.sum(logprobs) / y_train.shape[1]
    return cost

def backward_propagation(parameters, cache, x_train, y_train):
    m = x_train.shape[1]
    dZ2 = cache["A2"] - y_train
    dW2 = np.dot(dZ2, cache["A1"].T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(parameters["weight2"].T, dZ2) * (1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1, x_train.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    grads = {
        "dweight1": dW1,
        "dbias1": db1,
        "dweight2": dW2,
        "dbias2": db2
    }
    return grads

def update(parameters, grads, learning_rate=0.01):
    parameters = {
        "weight1": parameters["weight1"] - learning_rate * grads["dweight1"],
        "bias1": parameters["bias1"] - learning_rate * grads["dbias1"],
        "weight2": parameters["weight2"] - learning_rate * grads["dweight2"],
        "bias2": parameters["bias2"] - learning_rate * grads["dbias2"],
    }
    return parameters

def predict(x_test, parameters):
    A2, _ = forward_propagation(x_test, parameters)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    
    for i in range(A2.shape[1]):
        if A2[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    
    return Y_prediction

def two_layer_NN(x_train, y_train, x_test, y_test, num_iterations):
    cost_list = []
    index = []
    
    parameters = parameters_weights_and_bias(x_train, y_train)
    
    for i in range(num_iterations):
        A2, cache = forward_propagation(x_train, parameters)
        cost = compute_cost(A2, y_train, parameters)
        grads = backward_propagation(parameters, cache, x_train, y_train)
        parameters = update(parameters, grads)
        
        if i % 100 == 0:
            cost_list.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" % (i, cost))
    
    plt.plot(index, cost_list)
    plt.xticks(index, rotation="vertical")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()
    
    y_prediction_test = predict(x_test, parameters)
    y_prediction_train = predict(x_train, parameters)
    
    print("Train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
    return parameters

# Train the model
parameters = two_layer_NN(x_train, y_train, x_test, y_test, num_iterations=2500)
"""
#%% NN WİTH KERAS
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load dataset
x_l = np.load('X.npy')
Y_l = np.load('Y.npy')
img_size = 64

plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')

# Data Preparation
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0)
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# Flatten the data
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)
print("X train flatten", X_train_flatten.shape)
print("X test flatten", X_test_flatten.shape)

x_train = X_train_flatten
x_test = X_test_flatten
y_train = Y_train.ravel()  # Flatten y_train
y_test = Y_test.ravel()    # Flatten y_test
print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)

from scikeras.wrappers import KerasClassifier 
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library

def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))

