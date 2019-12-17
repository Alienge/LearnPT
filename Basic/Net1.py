import torch
import torch.nn.functional as F
from collections import OrderedDict

class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,32,3,1,1)
        self.dense1 = torch.nn.Linear(32*3*3,128)