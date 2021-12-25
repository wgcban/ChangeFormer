import math
import torch
import torch.nn as nn
from torchvision.models import ResNet
import torch.nn.functional as F
from functools import partial


nonlinearity = partial(F.relu,inplace=True)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out
class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.scse = SCSEBlock(in_channels // 4)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        y = self.scse(x)
        x = x + y
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        '''self.channel_excitation = nn.Sequential(nn.(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())'''
        self.channel_excitation = nn.Sequential(nn.Conv2d(channel, int(channel//reduction), kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(int(channel // reduction), channel,kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x)
        chn_se = self.channel_excitation(chn_se)
        chn_se = torch.mul(x, chn_se)
        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class CDNet_model(nn.Module):
    def __init__(self, in_channels=3, block=SEBasicBlock, layers=[3, 4, 6, 3], num_classes=2):
        super(CDNet_model, self).__init__()

        filters = [64, 128, 256, 512]
        self.inplanes = 64
        self.firstconv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.firstbn = nn.BatchNorm2d(64)
        self.firstrelu = nn.ReLU(inplace=True)
        self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder1 = self._make_layer(block, 64, layers[0])
        self.encoder2 = self._make_layer(block, 128, layers[1], stride=2)
        self.encoder3 = self._make_layer(block, 256, layers[2], stride=2)
        self.encoder4 = self._make_layer(block, 512, layers[3], stride=2)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.dblock_master = Dblock(512)
        self.dblock = Dblock(512)

        self.decoder4_master = DecoderBlock(filters[3], filters[2])
        self.decoder3_master = DecoderBlock(filters[2], filters[1])
        self.decoder2_master = DecoderBlock(filters[1], filters[0])
        self.decoder1_master = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1_master = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1_master = nonlinearity
        self.finalconv2_master = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2_master = nonlinearity
        self.finalconv3_master = nn.Conv2d(32, num_classes, 3, padding=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, y):
        # Encoder_1
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1_x = self.encoder1(x)
        e2_x = self.encoder2(e1_x)
        e3_x = self.encoder3(e2_x)
        e4_x = self.encoder4(e3_x)

        # # Center_1
        # e4_x_center = self.dblock(e4_x)

        # # Decoder_1
        # d4_x = self.decoder4(e4_x_center) + e3_x
        # d3_x = self.decoder3(d4_x) + e2_x
        # d2_x = self.decoder2(d3_x) + e1_x
        # d1_x = self.decoder1(d2_x)

        # out1 = self.finaldeconv1(d1_x)
        # out1 = self.finalrelu1(out1)
        # out1 = self.finalconv2(out1)
        # out1 = self.finalrelu2(out1)
        # out1 = self.finalconv3(out1)

        # Encoder_2
        y = self.firstconv(y)
        y = self.firstbn(y)
        y = self.firstrelu(y)
        y = self.firstmaxpool(y)

        e1_y = self.encoder1(y)
        e2_y = self.encoder2(e1_y)
        e3_y = self.encoder3(e2_y)
        e4_y = self.encoder4(e3_y)

        # # Center_2
        # e4_y_center = self.dblock(e4_y)

        # # Decoder_2
        # d4_y = self.decoder4(e4_y_center) + e3_y
        # d3_y = self.decoder3(d4_y) + e2_y
        # d2_y = self.decoder2(d3_y) + e1_y
        # d1_y = self.decoder1(d2_y)
        # out2 = self.finaldeconv1(d1_y)
        # out2 = self.finalrelu1(out2)
        # out2 = self.finalconv2(out2)
        # out2 = self.finalrelu2(out2)
        # out2 = self.finalconv3(out2)

        # center_master
        e4 = self.dblock_master(e4_x - e4_y)
        # decoder_master
        d4 = self.decoder4_master(e4) + e3_x - e3_y
        d3 = self.decoder3_master(d4) + e2_x - e2_y
        d2 = self.decoder2_master(d3) + e1_x - e1_y
        d1 = self.decoder1_master(d2)

        out = self.finaldeconv1_master(d1)
        out = self.finalrelu1_master(out)
        out = self.finalconv2_master(out)
        out = self.finalrelu2_master(out)
        out = self.finalconv3_master(out)

        output = []
        output.append(out)

        return output



def CDNet34(in_channels, **kwargs):

    model = CDNet_model(in_channels, SEBasicBlock, [3, 4, 6, 3], **kwargs)

    return model