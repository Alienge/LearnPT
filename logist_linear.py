import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0')

# hyper pamameters
input_size = 784 # image is 28*28*1
num_class = 10
num_epochs = 5
batch_size = 10
learning_rate = 0.001

# mnist data
train_dataset = torchvision.datasets.MNIST(root="./data",train=True,
                                           transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root="./data",train=False,
                                          transform=transforms.ToTensor())

# data loader
test_num = len(test_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=test_num,
                                           shuffle=False)
model = nn.Linear(input_size,num_class)
model = model.to(device)
#model = nn.DataParallel(model)
#model = model.cuda()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1,28*28).cuda()
        outputs = model(images)
        labels = labels.cuda()
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%100 ==0:
            print("Epoch{}/{},Step{}/{},Loss{:.4f}".format(epoch+1,num_epochs,i,total_step,loss.item()))

with torch.no_grad():
    correct = 0
    total=0
    for images,labels in test_loader:
        images = images.reshape(-1,28*28).cuda()
        print("==================> image_size")
        print(images.size())
        outputs = model(images)
        print("====================> output_size")
        print(outputs.size())
        outputs = outputs.cpu()
        _,predict = torch.max(outputs.data,1)
        total += labels.size(0)
        print("====================>total")
        print(total)
        correct += (predict==labels).sum()
        print("==================>")
        print(correct)
        print("Accuracy is {:.2f}".format(correct.numpy()/total))