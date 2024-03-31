import torch
from torch import nn

class Resblock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 stride:int=1,
                 downsample:nn.Module=None):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride=stride
        self.downsample=downsample
        self.batchnorm=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()
        self.conv_1=nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1)
        self.conv_2=nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1)
    def forward(self, x:torch.Tensor):
        out=self.conv_1(x)
        out=self.batchnorm(out)
        if self.downsample is not None:
            I=self.downsample(x)
        else:
            I=x
        out+=I
        out=self.relu(out)
        return out
    

class Resnet(nn.Module):
    def __init__(self,
                 img_channels:int,
                 n_classes):
        super().__init__()
        
        self.batchnorm=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_layer_in=nn.Conv2d(
            in_channels=img_channels,
            out_channels=64,
            kernel_size=7,
            stride=2
        )

        self.conv_layer_1=self.build_layer(in_channels=64, out_channels=64, stride=1)
        self.conv_layer_2=self.build_layer(in_channels=64, out_channels=128, stride=2)
        self.conv_layer_3=self.build_layer(in_channels=128, out_channels=256, stride=2)
        self.conv_layer_4=self.build_layer(in_channels=256, out_channels=512, stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc_layer_out=nn.Linear(512, n_classes)

    def build_layer(self,
                    in_channels:int,
                    out_channels:int,
                    stride:int)->nn.Sequential:
        downsample=None
        if stride!=1 or in_channels!=out_channels:
            downsample=nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm2d(out_channels)
            )
        return nn.Sequential(
            Resblock(in_channels, out_channels, stride, downsample),
            Resblock(out_channels, out_channels)
        )
    
    def forward(self, x:torch.Tensor):
        x=self.conv_layer_in(x)
        x=self.batchnorm(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.conv_layer_1(x)
        x=self.conv_layer_2(x)
        x=self.conv_layer_3(x)
        x=self.conv_layer_4(x)

        x=self.avgpool(x)
        x=torch.flatten(x, start_dim=1)
        x=self.fc_layer_out(x)

        return x
