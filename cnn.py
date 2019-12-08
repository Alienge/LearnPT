import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 

# You can then use the prebuilt data loader. 
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=True)
'''
# hyper parameters
num_epoches = 5
num_classes = 10
batch_size = 100
learning_rate = 0.01

# mnist data
train_data = torchvision.datasets.MNIST(root="./data/",train=True,transform=transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.MNIST(root="./data/",train=False,transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size=batch_size,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_data,batch_size=batch_size,shuffle = False)


class ConvNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ConvNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=5,stride=1,padding=2),#same
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)

        )
        self.fc = nn.Linear(7*7*32,num_classes)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)
print(model)
criterion = nn.CrossEntropyLoss()

optimiter = torch.optim.Adam(model.parameters(),lr = learning_rate)

total_step = len(train_loader)

for epoch in range(num_epoches):
    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimiter.zero_grad()
        loss.backward()
        optimiter.step()
        if (i + 1) % 100 == 0:
            print("Epoch{}/{},Step{}/{},Loss{:.4f}".format(epoch+1,num_epoches,i,total_step,
                                                           loss.item()))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicts = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicts==labels).sum().item()
    print("test accurcy is {}".format(correct/total))
