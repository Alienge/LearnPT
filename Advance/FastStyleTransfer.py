from __future__ import division
from torchvision import transforms
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
import glob
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path,transform=None,max_size = None,shape=None):
    image = cv2.imread(img_path)
    image = cv2.resize(image,(512, 512))

    # if max_size:
    #     scale = max_size/max(image.size)
    #     size = np.array(image.size)*scale
    #     image = image.resize(size.astype(int),Image.ANTIALIAS)
    # if shape:
    #     image = image.resize(shape,Image.LANCZOS)
    #     #image = image.resize((shape,320))

    if transform:
        image = transform(image).unsqueeze(0)

    return image.to(device)


# E:\datasets\SUN2012\Images\misc *.jpg
class CustomerDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir,transform=None,max_size = None,shape=None):
        #super(CustomerDataset,self).__init__()
        self.root_dir = root_dir
        self.max_size = max_size
        self.shape = shape
        self.transform = transform
        path_list = glob.glob(os.path.join(root_dir,"*.jpg"))
        self.image_paths = list()
        for path in path_list:
            image = Image.open(path)
            if len(image.split()) == 3:
                self.image_paths.append(path)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = Image.open(image_path)
        image.resize((512,512))
        #image = np.asarray(image)/255.0-0.5



        if self.max_size:
            scale = self.max_size / max(image.size)
            size = np.array(image.size) * scale
            image = image.resize(size.astype(int), Image.ANTIALIAS)
        if self.shape:
            image = image.resize(self.shape, Image.LANCZOS)

        #sample = {"image": image}
        if self.transform:
            sample = self.transform(image)
        return sample



def conv(in_channel,out_channel,stride=1):
    return nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False).to(device)


def deconv(in_channel,out_channel,stride=2):
    return nn.ConvTranspose2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,output_padding=1,bias=False).to(device)


class FasterStyleTransfer(nn.Module):
    def __init__(self):
        super(FasterStyleTransfer,self).__init__()
        self.select = ["0","5","10","19","28"] # content_loss + style_loss
        self.vgg = models.vgg19(pretrained=True).features
        self.conv1 = conv(3,16)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2,stride=2)

        self.conv2 = conv(16,32)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = conv(32, 64)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.deconv1 = deconv(64,32)
        self.derelu1 = nn.ReLU()
        self.debn1 = nn.BatchNorm2d(32)

        self.deconv2 = deconv(32, 16)
        self.derelu2 = nn.ReLU()
        self.debn2 = nn.BatchNorm2d(16)

        self.deconv3 = deconv(16, 3)
        self.derelu3 = nn.Sigmoid()
        #self.debn2 = nn.BatchNorm2d(3)




    def resblock(self,channel,x): #res模块固定channel
        #print("===========res1 size===========>",x.size())
        out = conv(channel,channel)(x)
        #print("===========res2 size===========>", out.size())
        out = nn.BatchNorm2d(channel).to(device)(out)
        out = F.relu(out)

        out = conv(channel, channel)(x)
        out = nn.BatchNorm2d(channel).to(device)(out)
        out = out + x
        out = F.relu(out)
        return out

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = self.resblock(64,out).to(device)

        out = self.resblock(64,out).to(device)

        out = self.deconv1(out)
        out = self.debn1(out)
        out = self.derelu1(out)
        out = self.deconv2(out)
        out = self.debn2(out)
        out = self.derelu2(out)
        out = self.deconv3(out)
        out = self.derelu3(out)
        return out



class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet,self).__init__()
        self.select = ["0","5","10","19","28"]
        self.vgg = models.vgg19(pretrained=True).features
    def forward(self, x):
        features=[]
        for name,layer in self.vgg._modules.items():
            x=layer(x)
            if name in self.select:
                features.append(x)
        return features


def main(config):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
                                    ])
    style = load_image(config.style, transform, max_size = config.max_size)
    #print("===============style size============>",style.size())
    train_data = CustomerDataset(root_dir="E:\datasets\SUN2012\Images\misc",
                                 transform=transforms.ToTensor(),shape=[512, 512])

    train_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size=2,shuffle=True)

    model_generate = FasterStyleTransfer().to(device)
    model_content_style = VGGNet().to(device)
    model_style = VGGNet().to(device)
    model_content = VGGNet().to(device)

    style_result = model_style(style)
    optimizer = torch.optim.Adam(model_generate.parameters(), lr=config.lr)

    total_step = len(train_loader)
    for epoch in range(config.epoches):
        for step, batch_image in enumerate(train_loader):
            #print(sample.size())
            #print(batch_image[0,1])
            batch_image = batch_image.to(device)

            content_result = model_content(batch_image)

            generate_result = model_generate(batch_image)

            g_loss =  torch.mean((generate_result - batch_image) ** 2)
            #print("===================>")
            #print(generate_result.size())
            content_style_result = model_content_style(generate_result)

            content_loss = 0
            style_loss = 0
            for f1,f2,f3 in zip(content_result,content_style_result,style_result):
                content_loss += torch.mean((f1 - f2) ** 2)
                _, c1, h1, w1 = f3.size()
                _,c2,h2,w2 = f2.size()
                # batch_size * c * h * w
                #print("=============f3 shape==========>",f3.size())
                f3 = f3.view(c1, h1 * w1)
                f3 = torch.mm(f3, f3.t())
                #print("=============f3==========>", f3.size())
                #print("==========f2=====>", f2.size())
                for i in range(2):
                    #print("=========f2[i]======>",f2[i].size())
                    f2_ = f2[i]. view(c2,h2*w2)
                    f2_ = torch.mm(f2_, f2_.t())
                    #print("=============f2_size============>",f2_.size())
                    style_loss += torch.mean((f2_ - f3) ** 2) / (c2 * h2 * w2)
            g_loss = 0
            loss = content_loss + config.style_weight * style_loss + g_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if (step + 1) % config.log_step == 0:
                print('Epoch: [{}/{}], Step: [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}, Generate loss: {:.4f}'
                      .format(epoch+1,config.epoches,step + 1, total_step, content_loss.item(), style_loss.item(),g_loss.item()))

            if (step + 1) % config.sample_step == 0:
                # Save the generated image
                content_img_ = load_image(config.content, transform, max_size = config.max_size)
                #denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
                result_img = model_generate(content_img_)
                torchvision.utils.save_image(result_img, 'style/output-{}-{}.png'.format(epoch,step + 1))

            #
            #     pass
                #content_loss = torch.mean(( - f2) ** 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # this is test image
    parser.add_argument('--content', type=str, default='image/content.png')
    parser.add_argument('--style', type=str, default='image/style.png')
    parser.add_argument('--max_size', type=int, default=512)
    parser.add_argument('--epoches', type=int, default=20)
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--style_weight', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.003)
    config = parser.parse_args()
    print(config)
    main(config)
