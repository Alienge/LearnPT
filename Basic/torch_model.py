import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

input_size = 1
output_size = 1
num_epoch = 60
learning_rate = 0.01

# 没有设置bacth_size
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]],dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

model = nn.Linear(input_size,output_size)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


for epoch in range(num_epoch):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    outputs = model(inputs)
    loss = criterion(outputs,targets)
    optimizer.zero_grad()#把存储在每个变量中的grad强制为0
    loss.backward()#caculate grad
    optimizer.step()#update value grad

    if(epoch+1)%5 == 0:
        print("Epoch{}/{}, Loss:{:.4f}".format(epoch+1,num_epoch,loss.item()))

predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train,y_train,'ro',label='original data')
plt.plot(x_train,predicted,label="Fitted line")
plt.legend()
plt.show()





