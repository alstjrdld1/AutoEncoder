import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

from torchvision import datasets, transforms 

import numpy as np 
from random import sample 
import matplotlib.pyplot as plt
from Models.AutoEncoder import AutoEncoder
from Models.NoiseAutoEncoder import NoiseAutoEncoder 

from Models.NoiseEdgeAutoEncoder import *

import sys 
def train(model_name, num_epochs):
  train_dataset = datasets.MNIST(root='./data', train=True, transform = transforms.ToTensor(), download = True)
  test_dataset = datasets.MNIST(root='./data' , train=False, transform = transforms.ToTensor())
  
  batch_size = 512

  train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False)
  # About Model
  if(model_name == "AE"):
    model = AutoEncoder(28*28, 64, 32)
  elif (model_name == "NAE"):
    model = NoiseAutoEncoder(28*28, 64, 32)
  elif (model_name == "NEAE"):
    model = NoiseEdgeAutoEncoder(28*28, 64, 32)

  model = model.cuda()
  learning_rate = 0.01
  optimizer = optim.Adam(model.parameters(), lr = learning_rate)
  Loss = nn.MSELoss()

  train_loss_arr = []
  test_loss_arr =[]

  best_test_loss = 999999999
  early_stop, early_stop_max = 0., 3.

  for epoch in range(num_epochs):
    epoch_loss = 0.

    for batch_X, _ in train_loader:
      batch_X = torch.Tensor(batch_X).cuda()
      optimizer.zero_grad()

      model.train()
      outputs = model(batch_X)
      train_loss = Loss(outputs, batch_X)
      epoch_loss += train_loss.data

      train_loss.backward()
      optimizer.step()

    train_loss_arr.append(epoch_loss / len(train_loader.dataset))

    if epoch % 10 == 0 :
      model.eval()
      today = sys.argv[3]
      torch.save(model.state_dict(), './ptfiles/'+ today + "_"+model_name+f'_20220520_{epoch}.pt')
      test_loss = 0.

      for batch_X, _ in test_loader:
        batch_X = torch.Tensor(batch_X).cuda()

        outputs = model(batch_X)
        batch_loss = Loss(outputs, batch_X)
        test_loss += batch_loss.data 

      test_loss = test_loss 
      test_loss_arr.append(test_loss)

      if best_test_loss > test_loss:
        beset_test_loss = test_loss
        early_stop = 0

        print('Epoch [{}/{}], Train LOsos : {:.4f}, Test Loss : {:.4f} *'. format(epoch, num_epochs, epoch_loss, test_loss))
      else:
        early_stop += 1
        print('Epoch [{}/{}], Train LOsos : {:.4f}, Test Loss : {:.4f} *'. format(epoch, num_epochs, epoch_loss, test_loss))
    if early_stop >= early_stop_max:
      break

if __name__ == "__main__":
    model_name = sys.argv[1]
    epochs = sys.argv[2]


    train(model_name, int(epochs))