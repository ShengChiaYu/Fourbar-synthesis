import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from utils import parse_args


class Net_1(nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(240, 80),
            nn.Hardtanh(),
            # nn.Dropout(0.5)
            nn.Linear(80, 20),
            nn.Hardtanh(),
            # nn.Dropout(0.5)
            nn.Linear(20, 5),
            nn.Hardtanh(),
            # nn.Dropout(0.5)
            nn.Linear(5,5)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


class Net_2(nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(60, 30),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(30, 15),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(15,5)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


class Net_3(nn.Module):
    def __init__(self):
        super(Net_3, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(30, 15),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(15,5)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


class Net_4(nn.Module):
    def __init__(self):
        super(Net_4, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(48, 240),
            nn.ReLU(),

            nn.Linear(240, 120),
            nn.ReLU(),

            nn.Linear(120, 60),
            nn.ReLU(),

            nn.Linear(60, 30),
            nn.ReLU(),

            nn.Linear(30, 5),
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(240, 120),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(120, 60),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(60, 30),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(30, 15),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(15,8)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


def test():
    args = parse_args()

    model = Net_1().cuda()

    input = torch.rand(4,240).cuda()
    output = model(input)
    print('Tensor size and type:', output.size(), output.dtype)
    print(model)


if __name__ == '__main__':
    test()
