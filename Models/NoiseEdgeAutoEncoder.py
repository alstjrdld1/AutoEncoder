import cv2

import torch 
import torch.nn as nn 

import numpy as np 

class NoiseEdgeAutoEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dim1, hidden_dim2):
    super(NoiseEdgeAutoEncoder, self).__init__()

    self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim1),
        nn.ReLU(),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.ReLU()
    )
    self.noise = np.random.normal(0, 1, hidden_dim2)
    self.decoder = nn.Sequential(
        nn.Linear(hidden_dim2, hidden_dim1),
        nn.ReLU(),
        nn.Linear(hidden_dim1, input_dim),
        nn.ReLU()
    )

  def edgeDetect(self, x):
    edges = []
    
    for item in x:
        edge =cv2.Laplacian(item, -1)
        edges.append(edge)
    edges = torch.Tensor(np.array(edges))
    return edges

  def forward(self, x):
    out = x.view(x.size(0), -1)
    
    edge = x.cpu().numpy()
    edge = self.edgeDetect(edge)
    edge = edge.view(edge.size(0), -1)
    edge = edge.cuda()

    out = self.encoder(out)

    out = out + torch.Tensor(self.noise).cuda()

    out = self.decoder(out)
    out = out.view(x.size())
    return out

  def get_codes(self, x):
    return self.encoder(x)