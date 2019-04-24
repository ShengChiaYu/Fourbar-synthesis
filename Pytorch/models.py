import torch.nn as nn
import math
import torch

class Net_1(nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(22, 22),
            nn.Hardtanh(),
            # nn.Dropout(0.5)
            nn.Linear(22, 5),
            nn.Hardtanh(),
            # nn.Dropout(0.5)
            nn.Linear(5, 5),
            nn.Hardtanh(),
            # nn.Dropout(0.5)
            nn.Linear(5,5)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x

def test():
    model = Net_1()
    input = torch.rand(2,22)
    output = model(input)
    print('Tensor size and type after fc1:', output.size(), output.dtype)
    print(model)

if __name__ == '__main__':
    test()
