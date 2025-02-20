import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# load data set
x_l = np.load('X.npy')
Y_l = np.load('Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')

# Join a sequence of arrays along an row axis.
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205) # 0 lar için label
o = np.ones(205) # 1 ler için label
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1) #☺ label leri birleştir
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15 ,random_state=42)
number_of_train = X_train.shape[0]
number_of_test =X_test.shape[0]

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


def weight_and_bias(boyut):
    w = np.full((boyut,1),0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1 / (1+ np.exp(-z))
    return y_head

def forward_backward_propagation(w,b,x_train,y_train):
    #forward
    z = np.dot(w.T,x_train) +b
    y_head = sigmoid(z)
    loss = -(1-y_train) * np.log(1-y_head) - (y_train * np.log(y_head))
    cost = sum(loss) / x_train.shape[1]
    
    #backward
    turev_weight = (np.dot(x_train,(y_head-y_train).T)) / x_train.shape[1]
    turev_bias = (sum(y_head-y_train)) / x_train.shape[1]
    turev = {"turev_weight": turev_weight, "turev_bias": turev_bias}
    
    return turev,cost

def update(w,b,x_train,y_train,number_of_iteration,learning_rate):
    cost_list = []
    cost_list2 = []
    index = []

    
    for i in range(int(1/number_of_iteration)):
        turev,cost  = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        w = w - learning_rate *number_of_iteration * turev["turev_weight"]
        b = b - learning_rate *number_of_iteration * turev["turev_bias"]
        if i % 10 == 0 :
            cost_list2.append(cost)
            index.append(i)
            cost_mean = cost.mean()
            print("yinelemeden sonraki maliyet %i: %f" %(i,cost_mean))

    parameters = {"weights": w , "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index,rotation= "vertical" )
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters,turev, cost_list

def tahmin(w,b,x_test):
    b = b.reshape(-1,1)
    z = sigmoid(np.dot(w.T,x_test) + b)
    y_tahmin = np.zeros((1, z.shape[1]))  # Burada z.shape[1] kullanarak doğru boyutta diziyi oluşturuyoruz
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            y_tahmin[0, i] = 0
        else:
            y_tahmin[0, i] = 1
    return y_tahmin



def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    boyut =  x_train.shape[0]  # that is 4096
    w,b = weight_and_bias(boyut)
    # do not change learning rate
    parameters, turev, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
           
   
    y_tahmin_test = tahmin(parameters["weights"],parameters["bias"],x_test)
    y_tahmin_train = tahmin(parameters["weights"],parameters["bias"],x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_tahmin_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_tahmin_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 100)
"""
from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
"""