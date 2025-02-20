import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(train.shape)
print(test.shape)

Y_train = train["label"]
X_train = train.drop(labels =["label"],axis=1)

#plt.figure(figsize=(15,7))
#sns.countplot(Y_train,palette="icefire")
#plt.title("number of digit classes")
#Y_train.value_counts()

#normalize
X_train = X_train / 255
test = test / 255
#reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
#label encoding
from tensorflow.keras.utils import to_categorical
Y_train =to_categorical(Y_train,num_classes=10)

#train test split
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.1,random_state=2)
print("x_train shape: " ,X_train.shape)
print("x_val shape: " ,X_val.shape)
print("y_train shape: " ,Y_train.shape)
print("y_val shape: " ,Y_val.shape)

from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau


model = Sequential()
#
model.add(Conv2D(filters=8,kernel_size = (5,5),padding= "Same",activation = "relu",input_shape=(28,28,1) ))
model.add(MaxPool2D(pool_size=(2,2) ))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters=16,kernel_size = (3,3),padding= "Same",activation = "relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2) ))
model.add(Dropout(0.25))
#fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
#define the optimizer (cost u min yapacak parametrelere hızlı giderse learnin rateyi azaltıyo veya yavaş giderse hızlandırıyo)
optimizer = Adam(learning_rate =0.001,beta_1=0.9,beta_2=0.999 )
#model compile
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"] )

epochs = 10  
batch_size = 250

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.05, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

#fit the model
history = model.fit(datagen.flow(X_train,Y_train,batch_size=batch_size),epochs=epochs,validation_data=(X_val,Y_val),steps_per_epoch=X_train.shape[0]//batch_size)

# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()












