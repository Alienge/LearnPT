import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyMul(nn.Module):
    def forward(self, input):
        out = input * 4
        return out

class MyMean(nn.Module):            # 自定义除法module
    def forward(self, input):
        out = input/4
        return out

def tensor_hook(grad):
    print("tensor hook")
    print("grad : ",grad )
    return grad

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.f1 = nn.Linear(4,1,bias=True)
        self.f2 = MyMean()
        self.weight_init()

    def forward(self,input):
        self.input = input
        output = self.f1(input)
        output = self.f2(output)
        return output

    def weight_init(self):
        self.f1.weight.data.fill_(8.0)  # 这里设置Linear的权重为8
        self.f1.bias.data.fill_(2.0)

    def my_hook(self,module,grad_input,grad_output):
        print("doing my hook")
        print("original grad :", grad_input)
        print("original outgrad :", grad_output)
        return grad_input

if __name__=="__main__":
    input = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True).to(device)
    net = MyNet()
    # cpu to gpu
    net.to(device)
    # 将net中的参数设置成可导
    net.register_backward_hook(net.my_hook)
    input.register_hook(tensor_hook)

    # forward 方法在 __call__ 方法中被调用，继承来自Module模块
    # def __call__(self, *input, **kwargs):
    #  for hook in self._forward_pre_hooks.values():
    #     hook(self, input)
    #   if torch._C._get_tracing_state():
    #     result = self._slow_forward(*input, **kwargs)
    #   else:
    #     result = self.forward(*input, **kwargs)
    result = net(input)

    print('result =', result)
    result.backward()

    # print('input.grad:', input.grad)
    # for param in net.parameters():
    #     print('{}:grad->{}'.format(param, param.grad))
