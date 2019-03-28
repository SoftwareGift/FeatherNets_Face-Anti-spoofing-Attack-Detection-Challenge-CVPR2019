import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, mode='NORM', k=1, dilation=1):
        """
        Pre-act residual block, the middle transformations are bottle-necked
        :param inplanes:
        :param planes:
        :param stride:
        :param downsample:
        :param mode: NORM | UP
        :param k: times of additive
        """

        super(Bottleneck, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace=True)
        self.k = k

        btnk_ch = planes // 4
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, btnk_ch, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(btnk_ch)
        self.conv2 = nn.Conv2d(btnk_ch, btnk_ch, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)

        self.bn3 = nn.BatchNorm2d(btnk_ch)
        self.conv3 = nn.Conv2d(btnk_ch, planes, kernel_size=1, bias=False)

        if mode == 'UP':
            self.shortcut = None
        elif inplanes != planes or stride > 1:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                self.relu,
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = None

    def _pre_act_forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.mode == 'UP':
            residual = self.squeeze_idt(x)
        elif self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual

        return out

    def squeeze_idt(self, idt):
        n, c, h, w = idt.size()
        return idt.view(n, c // self.k, self.k, h, w).sum(2)

    def forward(self, x):
        out = self._pre_act_forward(x)
        return out
