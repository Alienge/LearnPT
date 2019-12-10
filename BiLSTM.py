import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameter
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epoches = 10
learning_rate = 0.01

train_dataset = torchvision.datasets.MNIST(root="./data",train=True,
                                           transform = transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root="./data",train=False,
                                          transform= transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size=batch_size,
                                          shuffle = False)

class BiLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(BiLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,num_classes) #concat the output(hidden_size)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)

        out, _ = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out

# construct network
model = BiLSTM(input_size,hidden_size,num_layers,num_classes)

model = model.to(device)

# loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

total_step = len(train_loader)

for epoch in range(num_epoches):
    for i, (images,labels) in enumerate(train_loader):
        images = images.reshape(-1,sequence_length,input_size)
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)

        loss = criterion(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch {}/{}, Step {}/{}, Loss {:.4f}".format(epoch,num_epoches,
                                                                  i,total_step,loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))




