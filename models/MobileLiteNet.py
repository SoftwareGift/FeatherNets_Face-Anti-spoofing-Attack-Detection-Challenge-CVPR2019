import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
         
# reference form : https://github.com/moskomule/senet.pytorch  
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
#reference from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py    
class InvertedResidual(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(InvertedResidual, self).__init__()
        self.expansion = 6
        self.conv1 = nn.Conv2d(inplanes, inplanes*self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes*self.expansion)
        self.conv2 = nn.Conv2d(inplanes*self.expansion, inplanes*self.expansion, kernel_size=3, stride=stride,
                               padding=1, groups=inplanes, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes*self.expansion)
        self.conv3 = nn.Conv2d(inplanes*self.expansion, planes , kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU6(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = out
        
        return out
    
class MobileLiteNet(nn.Module):
    def __init__(self,block,layers,num_classes = 2,se = False):
        
        super(MobileLiteNet, self).__init__()
        self.se = se
        self.channels = [32, 16, 32, 48, 64]
        self.inplanes = self.channels[1]
        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=3, stride=2, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.relu = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(self.channels[1])
        
        self.layer1 = self._make_layer(block, self.channels[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, self.channels[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.channels[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.channels[4], layers[3], stride=2)
        self.final_DW = nn.Conv2d(self.channels[4], self.channels[4], kernel_size=4, 
                                  groups=self.channels[4], bias=False)
        self.do = nn.Dropout(0.2) 
        self.linear = nn.Linear(self.channels[4]*16, num_classes)
        if self.se:
            self.layer1_se1= SELayer(self.channels[1])
            self.layer2_se2= SELayer(self.channels[2])
            self.layer3_se3= SELayer(self.channels[3])
            self.layer4_se4= SELayer(self.channels[4])
        
        self._initialize_weights()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 :
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, self.inplanes ,
                          kernel_size=3, stride=stride,padding=1, groups=self.inplanes, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.Conv2d(self.inplanes, planes , kernel_size=1, bias=False)
            )


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.se:
            x = self.layer1_se1(self.layer1(x))
            x = self.layer2_se2(self.layer2(x))
            x = self.layer3_se3(self.layer3(x))
            x = self.layer4_se4(self.layer4(x))
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.final_DW(x)
        x = x.view(x.size(0), -1)

        x = self.do(x)
        x = self.linear(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
def MobileLiteNet54( **kwargs):
    model = MobileLiteNet(InvertedResidual, [4, 4, 6, 3], num_classes=2,se = False, **kwargs)
    return model 
def MobileLiteNet54_se( **kwargs):
    model = MobileLiteNet(InvertedResidual, [4, 4, 6, 3], num_classes=2,se = True, **kwargs)
    return model
def MobileLiteNet102( **kwargs):
    model = MobileLiteNet(InvertedResidual, [3, 4, 23, 3], num_classes=2,se = False, **kwargs)
    return model 
def MobileLiteNet105_se( **kwargs):
    model = MobileLiteNet(InvertedResidual, [4, 4, 23, 3], num_classes=2,se = True, **kwargs)
    return model 
def MobileLiteNet153( **kwargs):
    model = MobileLiteNet(InvertedResidual, [3, 8, 36, 3], num_classes=2,se = False, **kwargs)
    return model 
def MobileLiteNet156_se( **kwargs):
    model = MobileLiteNet(InvertedResidual, [4, 8, 36, 3], num_classes=2,se = True, **kwargs)
    return model 
