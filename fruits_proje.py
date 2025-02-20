#%% library and data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Dense,Flatten,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array,load_img
from glob import glob
import matplotlib.pyplot as plt

train_path = "fruits-360_dataset_100x100/fruits-360/Training"
test_path =  "fruits-360_dataset_100x100/fruits-360/Test"

img = load_img(train_path + "/Apple Braeburn 1/0_100.jpg" )
plt.imshow(img), plt.axis("off"), plt.show()

x = img_to_array(img)
print(x.shape)

className = glob(train_path + '/*' )
numberOfClass = len(className)
print("NumberOfClass: ",numberOfClass)
#%%gpu ile eğitim
pd= tf.config.experimental.list_physical_devices("GPU")
if len(pd) > 0:
    tf.config.experimental.set_memory_growth(pd[0],True)
    print("gpu ile eğitiliyor")
    
#%% CNN Model
model = Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="Same",activation="relu",input_shape = (100,100,3)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding="Same",activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))
          
model.add(Conv2D(filters=256,kernel_size=(3,3),padding="Same",activation="relu",))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=512,kernel_size=(3,3),padding="Same",activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))
#fully connected
model.add(Flatten())
model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass,activation="softmax"))

#model compile
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics= ["accuracy"] )
batch_size = 32

#%% Data Generation(agumention) - Train - Test
train_datagen = ImageDataGenerator( 
    rotation_range=40,        #max dönüş açısı
    brightness_range=[0.8, 1.2],  # Parlaklık aralığı
    width_shift_range=0.2,  # Yatay kaydırma
    height_shift_range=0.2,  # Dikey kaydırma
    rescale=1./255,             # Normalizasyon
    shear_range=0.2,            # dönme (shear) işlemi
    zoom_range=0.2,             # Yakınlaştırma (zoom)
    horizontal_flip=True)       # Yatay döndürme)

test_datagen = ImageDataGenerator(rescale=1./255 )

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(100, 100),      # Resim boyutu
    batch_size=32,               # Batch boyutu
    color_mode ="rgb",
    class_mode='categorical')    # Sınıflandırma tipi (örneğin, 'categorical'))

test_generator = test_datagen.flow_from_directory(
        test_path, 
        target_size=x.shape[:2],
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")

hist = model.fit(
        train_generator,
        steps_per_epoch = 6400 // batch_size,
        epochs=100,
        validation_data = test_generator,
        validation_steps = 800 // batch_size)


#%% model save
#☻model.save("deneme.h5")
model.save_weights("deneme.weights.h5")
#%% model evaluation
print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"], label = "Train acc")
plt.plot(hist.history["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()


#%% save history
import json
with open("deneme_gpu.json","w") as f:
    json.dump(hist.history, f)


#%% load history
import codecs
with codecs.open("deneme_gpu.json", "r",encoding = "utf-8") as f:
    h = json.loads(f.read())
plt.plot(h["loss"], label = "Train Loss")
plt.plot(h["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["accuracy"], label = "Train acc")
plt.plot(h["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()   