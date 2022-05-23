from re import L
import torch.nn as nn 

class FCAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(FCAutoEncoder, self).__init__()
        
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.tran_cnn_layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2, padding=0),
            nn.ReLU()
        )

        self.tran_cnn_layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride = 2, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        output = self.cnn_layer1(x)
        output = self.cnn_layer2(output)
        output = self.tran_cnn_layer1(output)
        output = self.tran_cnn_layer2(output)

        return output
    
    def get_codes(self, x):
        output = self.cnn_layer1(x)
        return self.cnn_layer2(output)