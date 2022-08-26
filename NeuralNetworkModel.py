import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def get_data_loader(training = True):
  transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
  train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
  test_set = datasets.FashionMNIST('./data', train=False, transform=transform)
  if(training):
    return torch.utils.data.DataLoader(train_set, batch_size = 64)
  else:
    return torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False)



def build_model():
  model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28*28, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 10),
  )
  return model

def train_model(model, train_loader, criterion, T):
  model.train()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  for epoch in range(T):  # loop over the dataset multiple times
    running_loss = 0.0
    total = 0
    correct = 0
    for inputs, labels in train_loader:
      # get the inputs; data is a list of [inputs, labels]

      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      #print(outputs[0])
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      # print statistics
      running_loss += loss.item()

    print(f'Training Epoch: {epoch}  Accuracy: {correct}/{total}({(correct/total)*100:.2f}%) '+  f'Loss: {(running_loss / 938):.3f}')


def evaluate_model(model, test_loader, criterion, show_loss = True):
  model.eval()
  with torch.no_grad():
    running_loss = 0.0
    total = 0
    correct = 0
    for inputs, labels in test_loader:
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      running_loss += loss.item()

  if(show_loss):
    print(f'Average loss: {running_loss / total:.4f}')
    print(f'Accuracy: {(correct/total)*100:.2f}%')
  else:
    print(f'Accuracy: {(correct/total)*100:.2f}%')

def predict_label(model, test_images, index):
  class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt'
                ,'Sneaker','Bag','Ankle Boot']
  model.eval()
  with torch.no_grad():
    total = 0
    correct = 0
    inputs = test_images[index]
    outputs = model(inputs)
    prob = F.softmax(outputs, dim=1)

    l = list(prob[0])
    l.sort(reverse=True)

    ind1 = list(prob[0]).index(l[0])
    ind2 = list(prob[0]).index(l[1])
    ind3 = list(prob[0]).index(l[2])
    print(class_names[ind1] + f': {prob[0][ind1]*100:.2f}%')
    print(class_names[ind2] + f': {prob[0][ind2]*100:.2f}%')
    print(class_names[ind3] + f': {prob[0][ind3]*100:.2f}%')


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions.
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()

#train_loader = get_data_loader()
#print(type(train_loader))
#print(train_loader.dataset)
#test_loader = get_data_loader(False)
#print(type(test_loader))
#print(test_loader.dataset)
#model = build_model()
#print(model)
#criterion = nn.CrossEntropyLoss()
#train_model(model, train_loader, criterion, T = 5)
#evaluate_model(model, test_loader, criterion, show_loss = True)
#pred_set = next(iter(train_loader))[0]
#from matplotlib import pyplot
#pyplot.imshow(pred_set[10][0], cmap=pyplot.get_cmap('gray'))
#pyplot.show()
#predict_label(model, pred_set, 10)

