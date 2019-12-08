import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bactch_size = 5
x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)
torch_dataset = Data.TensorDataset(x,y)
print(torch_dataset)

loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=bactch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

for num, (x_, y_) in enumerate(loader):
        print(num)
        print("x:", x_)
        print("y:", y_)
