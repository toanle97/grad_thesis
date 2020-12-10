#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


# In[2]:


#Creat Bearing Dataset
class bearingStates(Dataset):
    def __init__(self, csv_file, transform, local):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        self.local = local
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        image = io.imread(os.path.join(self.local, self.annotations.iloc[index, 0]), plugin="matplotlib")
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        if self.transform:
            image = self.transform(image)
        return (image, label)


# In[3]:


dataset = bearingStates(csv_file = './python_data/test_labels.csv', 
                       transform = transforms.ToTensor(),
                       local = 'python_data')
train_set, test_set = torch.utils.data.random_split(dataset, [1500, 569])


# In[4]:


train_loader = DataLoader(dataset = train_set, batch_size = 100, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size = 100, shuffle = True)


# In[5]:


class modified_CNN(nn.Module):
    def __init__(self):
        super(modified_CNN, self).__init__()
        self.dw1 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, groups = 3)
        self.pw1 = nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = 1, stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        self.bn1 = nn.BatchNorm2d(12)
        
        self.dw2 = nn.Conv2d(in_channels = 12, out_channels = 12, kernel_size = 3, stride = 1, padding = 1, groups = 12)
        self.pw2 = nn.Conv2d(in_channels = 12, out_channels = 32, kernel_size = 1, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3)
        
        self.fc1 = nn.Linear(in_features = 2048, out_features = 4)
        #self.fc2 = nn.Linear(in_features = 2048, out_features = 4)
        #self.dr = nn.Dropout(0.5)
        
    def forward(self, x):
        out = self.dw1(x)
        out = self.pw1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = self.dw2(out)
        out = self.pw2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool3(out)
        out = self.maxpool2(out)
        out = out.view(-1, 2048)
        out = self.fc1(out)
        #out = self.relu(out)
        #out = self.dr(out)
        #out = self.fc2(out)
        
        return out


# In[6]:


model = modified_CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


# In[7]:


total_original_params = 0
for parameter in model.parameters():
    if parameter.requires_grad:
        total_original_params += parameter.data.nelement()
print(total_original_params)


# In[8]:


#Training the network
epochs = 10
train_loss = []
train_acc = []
test_loss = []
test_acc = []

#Training Phase
for epoch in range(epochs):
    iter_loss = 0
    correct = 0
    iterations = 0
    model.train()
    train_start = time.time()
    for idx, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_fn(outputs, label)
        iter_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == label).sum().item()
        iterations += 1
    train_end = time.time()
    train_loss.append(iter_loss / iterations)
    train_acc.append(correct / len(train_set))
#test phase
    iter_loss = 0
    correct  = 0
    iteration = 0
    model.eval()
    for i, (inputs, labels) in enumerate(test_loader):
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        iter_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        iteration += 1
    test_end = time.time()
    test_loss.append(iter_loss / iteration)
    test_acc.append(correct / len(test_set))
    print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Accuracy: {:.3f}'.format(epoch+1, 
                                                                                                                                  epochs, 
                                                                                                                                  train_loss[-1], 
                                                                                                                                  train_acc[-1],
                                                                                                                                  test_loss[-1],
                                                                                                                                  test_acc[-1]))
    


# In[9]:


print(train_start - train_end)


# In[10]:


f = plt.figure(figsize=(5, 5))
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Testing Loss')
plt.legend()
plt.show()
f2 = plt.figure(figsize = (5, 5))
plt.plot(train_acc, label = 'Training Accuracy')
plt.plot(test_acc, label = 'Testing Accuracy')
plt.legend()
plt.show()


# In[11]:


#torch.save(model.state_dict(), 'bearing_weights.pt')
img = Image.open('Capture.png').convert('RGB')
img = transforms.ToTensor()(img).unsqueeze_(0)
output = model(img)
print(output)
print(torch.max(output, 1))
_, predicted = torch.max(output,1)
print("Prediction is: {}".format(predicted.item()))
from thop import profile
macs, params = profile(model, inputs=(img, ))
print(macs, params)


# In[ ]:





# In[12]:


nb_classes = 4
train_loader = DataLoader(dataset = train_set, batch_size = 1, shuffle = True)
confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(train_loader):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)


# In[13]:


import itertools
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    
    accuracy = np.trace(cm) / float(cm.sum().item())
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = pd.DataFrame(cm).astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[14]:


plot_confusion_matrix(confusion_matrix, 
                      normalize    = False,
                      target_names = ['Normal', 'Inner', 'Outer', 'Balls'],
                      title        = "Confusion Matrix")


# In[15]:


torch.save(model.state_dict(), 'bearing_weights.pt')


# In[ ]:




