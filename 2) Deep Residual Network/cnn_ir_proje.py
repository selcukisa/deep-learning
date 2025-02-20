"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
print("DEVİCE: ",device)

#%%
def read_images(path,num_img):
    array = np.zeros([num_img,64*32])
    i = 0
    for img in os.listdir(path):
        img_path = path + "\\" + img
        img = Image.open(img_path,mode ="r")
        data = np.asarray(img,dtype="uint8" )
        data = data.flatten()
        array[i,:]= data
        i += 1
    return array
#%% x_train,x_test,y_train,y_test hazırlığı
#read train negative
train_neg_path = r"C:/Users/Administrator/Desktop/deep learning/X) Kaynaklar/2) Deep Residual Network/LSIFIR/Classification/Train/neg"
num_train_neg_img = 43390
train_negative_array = read_images(train_neg_path,num_train_neg_img )

x_train_neg_tensor = torch.from_numpy(train_negative_array)
print("x_train_neg_tensor: ",x_train_neg_tensor.size())

y_train_neg_tensor = torch.zeros(num_train_neg_img ,dtype=torch.long)
print("y_train_neg_tensor: ",y_train_neg_tensor.size())

#read train positive
train_pos_path =r"C:/Users/Administrator/Desktop/deep learning/X) Kaynaklar/2) Deep Residual Network/LSIFIR/Classification/Train/pos"
num_train_pos_img= 10208
train_pos_array = read_images(train_pos_path, num_train_pos_img)

x_train_pos_tensor = torch.from_numpy(train_pos_array)
print("x_train_pos_tensor: ",x_train_pos_tensor.size())

y_train_pos_tensor = torch.ones(num_train_pos_img,dtype=torch.long )
print("y_train_pos_tensor: ",y_train_pos_tensor.size())

#concat train
x_train = torch.cat((x_train_neg_tensor,x_train_pos_tensor),0)
y_train = torch.cat((y_train_neg_tensor,y_train_pos_tensor),0)
print("x_train: ",x_train.size())
print("y_train: ",y_train.size())

#read test negative
test_neg_path = r"C:/Users/Administrator/Desktop/deep learning/X) Kaynaklar/2) Deep Residual Network/LSIFIR/Classification/Test/neg"
num_test_neg_img = 22050
test_neg_array = read_images(test_neg_path,num_test_neg_img )

x_test_neg_tensor = torch.from_numpy(test_neg_array)
print("x_train_neg_tensor: ",x_test_neg_tensor.size())

y_test_neg_tensor = torch.zeros(num_test_neg_img ,dtype=torch.long)
print("y_train_neg_tensor: ",y_test_neg_tensor.size())

#read test positive
test_pos_path =r"C:/Users/Administrator/Desktop/deep learning/X) Kaynaklar/2) Deep Residual Network/LSIFIR/Classification/Test/pos"
num_test_pos_img= 5944
test_pos_array = read_images(test_pos_path, num_test_pos_img)

x_test_pos_tensor = torch.from_numpy(test_pos_array)
print("x_test_pos_tensor: ",x_test_pos_tensor.size())

y_test_pos_tensor = torch.ones(num_test_pos_img,dtype=torch.long )
print("y_test_pos_tensor: ",y_test_pos_tensor.size())

#concat test
x_test = torch.cat((x_test_neg_tensor,x_test_pos_tensor),0)
y_test = torch.cat((y_test_neg_tensor,y_test_pos_tensor),0)
print("x_train: ",x_test.size())
print("y_train: ",y_test.size())

#%% visualize
plt.imshow(x_train[45001,:].reshape(64,32),cmap="gray")

#%% CNN
#hyperparameter
num_epochs = 5000
num_classes = 2
batch_size = 8933
learning_rate = 0.00001

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10, 16, 5)
        
        self.fc1 = nn.Linear(16*13*5, 520)
        self.fc2 = nn.Linear(520, 130)      
        self.fc3 = nn.Linear(130,num_classes)
    
    def forward(self, x ):
        
        x = self.pool(F.relu((self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1,16*13*5)  #flatten işi görür
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
import torch.utils.data
    
    
train = torch.utils.data.TensorDataset(x_train,y_train)
trainloader = torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=True)

test = torch.utils.data.TensorDataset(x_test,y_test)
testloader = torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=False)

#gpu
net = Net().to(device)
#cpu
#net = Net()

#%% loss and optimizer

criterion = nn.CrossEntropyLoss()

import torch.optim as optim
optimizer = optim.SGD(net.parameters(),lr= learning_rate,momentum=0.8)

#%% train a network
start = time.time()
train_acc = []
test_acc = []
loss_list = []
use_gpu = True

for epoch in range(num_epochs):
    for i,data in enumerate(trainloader,0):
        
        inputs,labels = data
        inputs = inputs.view(batch_size,1,64,32) #reshape
        inputs = inputs.float()
        
        #use gpu
        if use_gpu:
            if torch.cuda.is_available():
                inputs,labels = inputs.to(device),labels.to(device)
         
        #zero gradient
        optimizer.zero_grad()
        
        #forward
        outputs = net(inputs)
        
        #loss
        loss = criterion(outputs,labels)
        
        #back
        loss.backward()
        
        #update weights
        optimizer.step()
        
    #test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images,labels = data
            
            images = images.view(batch_size,1,64,32)
            images = images.float()
            
            #use gpu
            if use_gpu:
                if torch.cuda.is_available():
                    inputs,labels = images.to(device),labels.to(device)
                    
            outputs = net(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc1 = 100*correct/total        
    print("acc test: ",acc1)
    test_acc.append(acc1)
    
    
    # train
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels= data
            
            images = images.view(batch_size,1,64,32)
            images = images.float()
            
            # gpu
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)
            
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data,1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    acc2 = 100*correct/total
    print("accuracy train: ",acc2)
    train_acc.append(acc2)
    
            
print("train is done.")                  
            
end = time.time()
process_time = (end-start)/60
print("process time: ", process_time)

#%% visualize
fig, ax1 = plt.subplots()

plt.plot(loss_list,label = "Loss",color = "black")

ax2 = ax1.twinx()

ax2.plot(np.array(test_acc)/100,label = "Test Acc",color="green")
ax2.plot(np.array(train_acc)/100,label = "Train Acc",color= "red")
ax1.legend()
ax2.legend()
ax1.set_xlabel('Epoch')
fig.tight_layout()
plt.title("Loss vs Test Accuracy")
plt.show()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE: ", device)

#%%
def read_images(path, num_img):
    array = np.zeros([num_img, 64*32])
    i = 0
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img = Image.open(img_path, mode="r")
        data = np.asarray(img, dtype="uint8")
        data = data.flatten()
        array[i, :] = data
        i += 1
    return array

#%% x_train, x_test, y_train, y_test hazırlığı
# Read train negative
train_neg_path = r"C:/Users/Administrator/Desktop/deep learning/X) Kaynaklar/2) Deep Residual Network/LSIFIR/Classification/Train/neg"
num_train_neg_img = 43390
train_negative_array = read_images(train_neg_path, num_train_neg_img)

x_train_neg_tensor = torch.from_numpy(train_negative_array)
print("x_train_neg_tensor: ", x_train_neg_tensor.size())

y_train_neg_tensor = torch.zeros(num_train_neg_img, dtype=torch.long)
print("y_train_neg_tensor: ", y_train_neg_tensor.size())

# Read train positive
train_pos_path = r"C:/Users/Administrator/Desktop/deep learning/X) Kaynaklar/2) Deep Residual Network/LSIFIR/Classification/Train/pos"
num_train_pos_img = 10208
train_pos_array = read_images(train_pos_path, num_train_pos_img)

x_train_pos_tensor = torch.from_numpy(train_pos_array)
print("x_train_pos_tensor: ", x_train_pos_tensor.size())

y_train_pos_tensor = torch.ones(num_train_pos_img, dtype=torch.long)
print("y_train_pos_tensor: ", y_train_pos_tensor.size())

# Concatenate train
x_train = torch.cat((x_train_neg_tensor, x_train_pos_tensor), 0)
y_train = torch.cat((y_train_neg_tensor, y_train_pos_tensor), 0)
print("x_train: ", x_train.size())
print("y_train: ", y_train.size())

# Read test negative
test_neg_path = r"C:/Users/Administrator/Desktop/deep learning/X) Kaynaklar/2) Deep Residual Network/LSIFIR/Classification/Test/neg"
num_test_neg_img = 22050
test_neg_array = read_images(test_neg_path, num_test_neg_img)

x_test_neg_tensor = torch.from_numpy(test_neg_array)
print("x_test_neg_tensor: ", x_test_neg_tensor.size())

y_test_neg_tensor = torch.zeros(num_test_neg_img, dtype=torch.long)
print("y_test_neg_tensor: ", y_test_neg_tensor.size())

# Read test positive
test_pos_path = r"C:/Users/Administrator/Desktop/deep learning/X) Kaynaklar/2) Deep Residual Network/LSIFIR/Classification/Test/pos"
num_test_pos_img = 5944
test_pos_array = read_images(test_pos_path, num_test_pos_img)

x_test_pos_tensor = torch.from_numpy(test_pos_array)
print("x_test_pos_tensor: ", x_test_pos_tensor.size())

y_test_pos_tensor = torch.ones(num_test_pos_img, dtype=torch.long)
print("y_test_pos_tensor: ", y_test_pos_tensor.size())

# Concatenate test
x_test = torch.cat((x_test_neg_tensor, x_test_pos_tensor), 0)
y_test = torch.cat((y_test_neg_tensor, y_test_pos_tensor), 0)
print("x_test: ", x_test.size())
print("y_test: ", y_test.size())

#%% Visualize
plt.imshow(x_train[45001, :].reshape(64, 32), cmap="gray")

#%% CNN
# Hyperparameters
num_epochs = 5000
num_classes = 2
batch_size = 8933
learning_rate = 0.00001

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 5)

        self.fc1 = nn.Linear(16*13*5, 520)
        self.fc2 = nn.Linear(520, 130)
        self.fc3 = nn.Linear(130, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16*13*5)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import torch.utils.data

train = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

test = torch.utils.data.TensorDataset(x_test, y_test)
testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

# GPU
net = Net().to(device)
# CPU
# net = Net()

#%% Loss and optimizer
criterion = nn.CrossEntropyLoss()

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.8)

#%% Train a network
start = time.time()
train_acc = []
test_acc = []
loss_list = []

for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.view(-1, 1, 64, 32)  # Use -1 for automatic size calculation
        inputs = inputs.float()

        # Use GPU
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Forward
        outputs = net(inputs)

        # Loss
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # Update weights
        optimizer.step()

    # Test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images = images.view(-1, 1, 64, 32)  # Use -1 for automatic size calculation
            images = images.float()

            # Use GPU
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc1 = 100 * correct / total
    print("Test Accuracy: ", acc1)
    test_acc.append(acc1)

    # Train
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data

            images = images.view(-1, 1, 64, 32)  # Use -1 for automatic size calculation
            images = images.float()

            # Use GPU
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc2 = 100 * correct / total
    print("Train Accuracy: ", acc2)
    train_acc.append(acc2)

print("Training is done.")

end = time.time()
process_time = (end - start) / 60
print("Process time: ", process_time)

#%% visualize
fig, ax1 = plt.subplots()

plt.plot(loss_list,label = "Loss",color = "black")

ax2 = ax1.twinx()

ax2.plot(np.array(test_acc)/100,label = "Test Acc",color="green")
ax2.plot(np.array(train_acc)/100,label = "Train Acc",color= "red")
ax1.legend()
ax2.legend()
ax1.set_xlabel('Epoch')
fig.tight_layout()
plt.title("Loss vs Test Accuracy")
plt.show()









