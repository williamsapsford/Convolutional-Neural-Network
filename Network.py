import os
import cv2 #used for image processing
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

REBUILD_DATA = False

class DogsVSCats():
    IMG_SIZE = 50
    CATS = "kagglecatsanddogs_3367a/PetImages/Cat"
    DOGS = "kagglecatsanddogs_3367a/PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)): #creates list of items in that directory
                try:
                    path = os.path.join(label, f) #joins labels to items in the directory
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #loads images of a certain file in grayscale
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("CATS: ", self.catcount)
        print("DOGS: ", self.dogcount)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) #creating convolutional layers
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #creating linear layers with flattened input
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1) #normalizes data to prob dist

net = Net()

if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True) #“Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy.

optimizer = optim.Adam(net.parameters(), lr=0.001) #.parameters contains all learnable parameters of the model
loss_function = nn.MSELoss() #mean squared error loss

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50) #seperating X and y's from training data
X = X/255.0 #scaling images so pixel values are in range (0, 1)
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:] #2494 images, each in a list of lists with 50 lists of size 50. Representing each row of pixels.
test_y = y[-val_size:]


def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad() #if training data then zero out the gradients before back propogation
    outputs = net(X) #turns data from structure mentioned on line 81 to tuple of 2 numbers (prob dist)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs,y)] #adds True if torch.argmax(i) = torch.argmax(j), False if not
                                                                              # y[0] = tensor([0., 1.]), torch.argmax(y[0]) = tensor(1)
                                                                              #torch.argmax(X[0]) = tensor(1)
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y) #returns the mean squared error for that batch

    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

def test(size=32):
    random_start = np.random.randint(len(test_X)-size) #gets a random slice from test_X to use
    X, y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1,1,50,50), y) #validation set is from data that has never been 'seen' by the network before
    return val_acc, val_loss

val_acc, val_loss = test(size=32)
print(val_acc, val_loss)

MODEL_NAME = f"model-{int(time.time())}"

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001) #.parameters contains all learnable parameters of the model
loss_function = nn.MSELoss() #mean squared error loss

print(MODEL_NAME)

def train():
    BATCH_SIZE = 100
    EPOCHS = 8
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
                batch_y = train_y[i:i+BATCH_SIZE]

                acc, loss = fwd_pass(batch_X, batch_y, train=True)
                if i % 50 == 0: #while training the CNN, test it every 50th iteration to check progress
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME}, {round(time.time(), 3)}, {round(float(acc), 2)}, {round(float(loss), 4)}, {round(float(val_acc), 2)}, {round(float(val_loss), 4)\n") #will log every 50th data reading to file model.log

train()