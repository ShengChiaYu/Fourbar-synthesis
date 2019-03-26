import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.preprocessing import MinMaxScaler
import glob
import os
import numpy as np
np.random.seed(1216)

# Creating a custom dataset
class Fourbar(Dataset):
    def __init__(self, root, x_file, y_file, transform=None):
        """ Intialize the Fourbar dataset """
        self.fourier_descriptors = None
        self.parameters = None
        self.positions = None
        self.x_file = x_file
        self.y_file = y_file
        self.transform = transform
        self.x_scaler = None
        self.y_scaler = None

        # read x_file
        print('Loading ', x_file)
        self.fourier_descriptors = np.genfromtxt(os.path.join(root, x_file), delimiter=',')
        self.x_scaler = MinMaxScaler(copy=False, feature_range=(-1, 1))
        self.x_scaler.fit_transform(self.fourier_descriptors)
        self.fourier_descriptors = torch.squeeze(transform(self.fourier_descriptors))

        self.len = self.fourier_descriptors.shape[0]
        # read y_file
        if 'param' in y_file:
            print('Loading ', y_file)
            self.parameters = np.genfromtxt(os.path.join(root, y_file), delimiter=',')
            self.y_scaler = MinMaxScaler(copy=False, feature_range=(-1, 1))
            self.y_scaler.fit_transform(self.parameters)
            self.parameters = torch.squeeze(transform(self.parameters))
        elif 'pos' in y_file:
            print('Loading ', y_file)
            self.positions = np.genfromtxt(os.path.join(root, y_file), delimiter=',')
            self.positions = torch.squeeze(transform(self.positions))

        self.len = self.fourier_descriptors.shape[0]

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        pass

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

# Load the images into custom-created Dataset.
root = os.path.join('data','60positions')
y_file = '_param.csv'
print('root:',root)
trainset = Fourbar(root=root, x_file='x_train.csv', y_file='y_train'+y_file, transform=transforms.ToTensor())
testset = Fourbar(root=root, x_file='x_test.csv', y_file='y_test'+y_file, transform=transforms.ToTensor())

print('Fourier descriptors tensor in each epoch:', trainset.fourier_descriptors.shape, trainset.fourier_descriptors.dtype)
if 'param' in y_file:
    print('Target tensor in each epoch:', trainset.parameters.shape, trainset.parameters.dtype)
elif 'pos' in y_file:
    print('Target tensor in each epoch:', trainset.positions.shape, trainset.positions.dtype)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

# Creating a Neural Network
class Net_1(nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(22, 22),
            nn.Hardtanh(),
            # nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(22, 5),
            nn.Hardtanh(),
            # nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(22, 5),
            nn.Hardtanh(),
            # nn.Dropout(0.5)
        )
        self.fc4 = nn.Linear(5,5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

model = Net_1().to(device) # Remember to move the model to "device"
print(model)

# Train the network.
def train(model, epoch, batch_size=64, log_interval=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()  # Important: set training mode

    x_batches = BatchSampler(trainset.fourier_descriptors, batch_size=batch_size, drop_last=True)
    if 'param' in y_file:
        y_batches = BatchSampler(trainset.parameters, batch_size=batch_size, drop_last=True)
    elif 'pos' in y_file:
        y_batches = BatchSampler(trainset.positions, batch_size=batch_size, drop_last=True)

    iteration = 0
    for ep in range(epoch):
        for batch_idx in range(len(x_batches)):
            data, target = x_batches[batch_idx], y_batches[batch_idx]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset),
                    100. * batch_idx / len(x_batches), loss.item()))
            iteration += 1

        test(model) # Evaluate at the end of each epoch

def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))
