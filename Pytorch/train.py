import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import glob
import os
import numpy as np

# Creating a custom dataset
class Fourbar(Dataset):
    def __init__(self, root, x_file, y_file, transform=None):
        """ Intialize the Fourbar dataset """
        self.root = root
        self.x_file = x_file
        self.y_file = y_file
        self.transform = transform
        self.x_scaler = None
        self.y_scaler = None

        # read files to get scalers
        x_data = np.genfromtxt(os.path.join(self.root, self.x_file), delimiter=',')
        y_data = np.genfromtxt(os.path.join(self.root, self.y_file), delimiter=',')

        self.x_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
        self.x_scaler.fit(x_data)
        self.y_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
        self.y_scaler.fit(y_data)

        self.len = x_data.shape[0]

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        with open(os.path.join(self.root, self.x_file)) as fpx:
            x_lines = fpx.readlines()
        with open(os.path.join(self.root, self.y_file)) as fpy:
            y_lines = fpy.readlines()

        inputs = np.array([float(i) for i in x_lines[index].split(',')]).reshape(1,-1)
        inputs = self.x_scaler.transform(inputs)
        targets = np.array([float(i) for i in y_lines[index].split(',')]).reshape(1,-1)
        targets = self.y_scaler.transform(targets)

        if self.transform is not None:
            inputs = self.transform(inputs)
            targets = self.transform(targets)

        return inputs, targets

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

# Creating a Neural Network
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
        # print('Tensor size and type after fc1:', x.shape, x.dtype)
        return x

# Train the network.
def train_save(model, trainset_loader, device, epoch, save_interval, log_interval=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()  # Important: set training mode

    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.float().to(device), target.float().to(device)
            # print(data.shape)
            # print(target.shape)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            if iteration % save_interval == 0 and iteration > 0:
                save_path = os.path.join('models', 'fourbar-%i.pth' % iteration)
                save_checkpoint(save_path, model, optimizer)
            iteration += 1

    # save the final model
    save_path = os.path.join('models', 'fourbar-%i.pth' % iteration)
    save_checkpoint(save_path, model, optimizer)

def test(model, testset_loader, device):
    criterion = nn.MSELoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in testset_loader:
            print('Testset data tensor in each batch:', data.shape, data.dtype)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

# Save the model
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

if __name__ == "__main__":
    # Load the images into custom-created Dataset.
    root = os.path.join('data','60positions')
    y_file = '_param.csv'
    print('root:', root, ', y file:', y_file)
    trainset = Fourbar(root=root, x_file='x_train.csv', y_file='y_train'+y_file, transform=transforms.ToTensor())
    testset = Fourbar(root=root, x_file='x_test.csv', y_file='y_test'+y_file, transform=transforms.ToTensor())

    print('# Fourier descriptors in trainset:', len(trainset)) # Should print 60000
    print('# Fourier descriptors in testset:', len(testset)) # Should print 10000

    # Use the torch dataloader to iterate through the dataset
    trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)
    testset_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)

    # get some random training images
    dataiter = iter(trainset_loader)
    inputs, targets = dataiter.next()

    print('Trainset inputs tensor in each batch:', inputs.shape, inputs.dtype)
    print('Trainset targets tensor in each batch:', targets.shape, targets.dtype)

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    # Training
    model = Net_1().to(device) # Remember to move the model to "device"
    print(model)
    train_save(model, trainset_loader, device, epoch=10, save_interval=500, log_interval=100)

    # create a new model to test final checkpoint
    # model = Net().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # load from the final checkpoint
    # save_path = os.path.join('models', 'fourbar-.pth')
    # load_checkpoint(save_path, model, optimizer)

    # should give you the final model accuracy
    # test(model)
