import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x=np.load('scaled_spec_resampled_array.npy')
y=np.load('labels_array.npy')-1
x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = running_loss / len(test_loader)
    val_acc = correct / total
    return val_loss, val_acc

def train(model, train_loader, test_loader, criterion, optimizer, epochs):
    model.train()
    running_loss = 0.0
    epoch_bar = tqdm(range(epochs), position=0)
    for epoch in epoch_bar:
        batch_bar=tqdm(enumerate(train_loader, 0), total=len(train_loader), position=1, leave=False)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_bar.set_description('Train loss: %.3f' % (loss.item()))
        train_loss = running_loss / len(train_loader)
        epoch_bar.set_description('Train loss: %.3f' % train_loss)
        val_loss, val_acc = test(model, test_loader, criterion)
        print('Epoch: %d, Train Loss: %.3f, Val Loss: %.3f, Val Acc: %.3f' % (epoch, train_loss, val_loss, val_acc))
    return model

train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

resnet18 = models.resnet18(pretrained=True)

resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #change input channels to 1
resnet18.fc = nn.Linear(in_features=512, out_features=6, bias=True)#change output classes to 6 

resnet18 = resnet18.to(device)

#freeze every layer except conv1 and fc layers of resnet18
for name, param in resnet18.named_parameters():
    if 'conv1' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = optim.Adam(resnet18.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model=train(resnet18, train_loader,test_loader, criterion, optimizer, 10)