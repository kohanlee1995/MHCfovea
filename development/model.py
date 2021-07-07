import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Block, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
    def forward(self, x):
        return self.main(x)
    

class MHCModel(nn.Module):
    def __init__(self, in_channel):
        super(MHCModel, self).__init__()
        self.main = nn.Sequential(
            Block(in_channel, 128),
            nn.MaxPool1d(3, 1, 1),
            Block(128, 128),
            nn.MaxPool1d(5, 3, 0),
            Block(128, 128),
            nn.MaxPool1d(4, 2, 1),
            Block(128, 128),
            nn.MaxPool1d(4, 2, 1)
        )
    def forward(self, x):
        return self.main(x)
    

class EpitopeModel(nn.Module):
    def __init__(self, in_channel):
        super(EpitopeModel, self).__init__()
        self.main = nn.Sequential(
            Block(in_channel, 128),
            nn.MaxPool1d(3, 1, 1),
            Block(128, 128),
            nn.MaxPool1d(3, 1, 1),
            Block(128, 128),
            nn.MaxPool1d(3, 1, 1)
        )
    def forward(self, x):
        return self.main(x)
    

class CombineModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(CombineModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.main = nn.Sequential(
            Block(256, 256),
            nn.MaxPool1d(3, 1, 1),
            Block(256, 64),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1,x2), dim=1)
        return self.main(x)