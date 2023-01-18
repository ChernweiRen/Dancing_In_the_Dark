from builtins import isinstance
from numpy.core.shape_base import block
from numpy.lib import twodim_base
import torch
import torch.nn as nn
import math
from torch.nn.modules import padding

__all__ = ['LSID', 'lsid']

def pixel_shuffle(input, upscale_factor, depth_first=False):
    r"""Rearranges elements in a tensor of shape :math:`[*, C*r^2, H, W]` to a
    tensor of shape :math:`[C, H*r, W*r]`.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Tensor): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples::

        >>> ps = nn.PixelShuffle(3)
        >>> input = torch.empty(1, 9, 4, 4)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])
    """
    batch_size, channels, in_height, in_width = input.size()
    channels //= upscale_factor ** 2

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    if not depth_first:
        input_view = input.contiguous().view(
            batch_size, channels, upscale_factor, upscale_factor,
            in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
        return shuffle_out.view(batch_size, channels, out_height, out_width)
    else:
        input_view = input.contiguous().view(batch_size, upscale_factor, upscale_factor, channels, in_height, in_width)
        shuffle_out = input_view.permute(0, 4, 1, 5, 2, 3).contiguous().view(batch_size, out_height, out_width,
                                                                             channels)
        return shuffle_out.permute(0, 3, 1, 2)


class TwoLayerConvWithBN(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size=3, stride=1,padding=1,bias=True):
        super(TwoLayerConvWithBN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )

        # self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
        #                     kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
        # self.bn1 = nn.BatchNorm2d(out_channel)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        # self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
        #                     kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
        # self.bn2 = nn.BatchNorm2d(out_channel)
    
    def forward(self,x):
        # x = self.lrelu(self.bn1(self.conv1(x)))
        # x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.conv(x)

        return x
        

class LSID(nn.Module):
    def __init__(self, inchannel=4, block_size=2):
        super(LSID, self).__init__()
        self.block_size = block_size

        self.conv1_1 = nn.Conv2d(inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        out_channel = 1 * self.block_size * self.block_size
        self.conv10 = nn.Conv2d(32, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.lrelu(x)

        x = self.up6(x)
        x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)

        x = self.up7(x)
        x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)

        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)

        x = self.up8(x)
        x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)

        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)

        x = self.up9(x)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)

        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.conv10(x)
        # x = nn.functional.sigmoid(x)

        depth_to_space_conv = pixel_shuffle(x, upscale_factor=self.block_size, depth_first=True)

        return depth_to_space_conv


class LSID_npm(nn.Module):
    def __init__(self, inchannel=4, block_size=2):
        super(LSID_npm, self).__init__()
        self.block_size = block_size

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv1_1 = nn.Conv2d(inchannel+inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        
        self.conv2_1 = nn.Conv2d(32+inchannel, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(64+inchannel, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(128+inchannel, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv5_1 = nn.Conv2d(256+inchannel, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.conv6_1 = nn.Conv2d(512+inchannel, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(256+inchannel, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(128+inchannel, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(64+inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        out_channel = 1 * self.block_size * self.block_size
        self.conv10 = nn.Conv2d(32+inchannel, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x, npm):
        x = self.conv1_1(torch.cat([x, npm],1))
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)
        npm_2 = self.avgpool(npm)

        x = self.conv2_1(torch.cat([x,npm_2],1))
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        x = self.maxpool(x)
        npm_3 = self.avgpool(npm_2)

        x = self.conv3_1(torch.cat([x,npm_3],1))
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        x = self.maxpool(x)
        npm_4 = self.avgpool(npm_3)

        x = self.conv4_1(torch.cat([x,npm_4],1))
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        x = self.maxpool(x)
        npm_5 = self.avgpool(npm_4)

        x = self.conv5_1(torch.cat([x,npm_5],1))
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.lrelu(x)

        x = self.up6(x)
        x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4, npm_4), 1)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)

        x = self.up7(x)
        x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3, npm_3), 1)

        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)

        x = self.up8(x)
        x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2, npm_2), 1)

        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)

        x = self.up9(x)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1, npm), 1)

        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.conv10(torch.cat([x,npm],1))
        # x = nn.functional.sigmoid(x)

        depth_to_space_conv = pixel_shuffle(x, upscale_factor=self.block_size, depth_first=True)

        return depth_to_space_conv



class LSID_2layer(nn.Module):
    def __init__(self, inchannel=4, block_size=2):
        super(LSID_2layer, self).__init__()
        self.block_size = block_size

        self.conv1_1 = nn.Conv2d(inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        out_channel = 1 * self.block_size * self.block_size
        self.conv10 = nn.Conv2d(32, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)

        x = self.up9(x)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)

        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.conv10(x)
        # x = nn.functional.sigmoid(x)

        depth_to_space_conv = pixel_shuffle(x, upscale_factor=self.block_size, depth_first=True)

        return depth_to_space_conv



class LSID_3layer(nn.Module):
    def __init__(self, inchannel=4, block_size=2):
        super(LSID_3layer, self).__init__()
        self.block_size = block_size

        self.conv1_1 = nn.Conv2d(inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        out_channel = 1 * self.block_size * self.block_size
        self.conv10 = nn.Conv2d(32, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)

        x = self.up8(x)
        x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)

        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)

        x = self.up9(x)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)

        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.conv10(x)
        # x = nn.functional.sigmoid(x)

        depth_to_space_conv = pixel_shuffle(x, upscale_factor=self.block_size, depth_first=True)

        return depth_to_space_conv


class LSID_6layer(nn.Module):
    def __init__(self, inchannel=4, block_size=2):
        super(LSID_6layer, self).__init__()
        self.block_size = block_size

        self.conv1_1 = nn.Conv2d(inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv6_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True)

        # self.conv7_1 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv7_2 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=True)

        # self.up8 = nn.ConvTranspose2d(1024, 512, 2, stride=2, bias=False)
        # self.conv8_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv8_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True)

        self.up9 = nn.ConvTranspose2d(1024, 512, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.up10 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.conv10_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv10_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.up11 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv11_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.up12 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.conv12_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv12_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.up13 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.conv13_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv13_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        out_channel = 1 * self.block_size * self.block_size
        self.conv14 = nn.Conv2d(32, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.lrelu(x)
        conv5 = x
        x = self.maxpool(x)

        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)
        # conv6 = x
        # x = self.maxpool(x)

        # x = self.conv7_1(x)
        # x = self.lrelu(x)
        # x = self.conv7_2(x)
        # x = self.lrelu(x)

        # x = self.up8(x)
        # x = torch.cat((x[:, :, :conv6.size(2), :conv6.size(3)], conv6), 1)
        # x = self.conv8_1(x)
        # x = self.lrelu(x)
        # x = self.conv8_2(x)
        # x = self.lrelu(x)

        x = self.up9(x)
        x = torch.cat((x[:, :, :conv5.size(2), :conv5.size(3)], conv5), 1)
        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.up10(x)
        x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        x = self.conv10_1(x)
        x = self.lrelu(x)
        x = self.conv10_2(x)
        x = self.lrelu(x)

        x = self.up11(x)
        x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)

        x = self.conv11_1(x)
        x = self.lrelu(x)
        x = self.conv11_2(x)
        x = self.lrelu(x)

        x = self.up12(x)
        x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)

        x = self.conv12_1(x)
        x = self.lrelu(x)
        x = self.conv12_2(x)
        x = self.lrelu(x)

        x = self.up13(x)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)

        x = self.conv13_1(x)
        x = self.lrelu(x)
        x = self.conv13_2(x)
        x = self.lrelu(x)

        x = self.conv14(x)
        # x = nn.functional.sigmoid(x)

        depth_to_space_conv = pixel_shuffle(x, upscale_factor=self.block_size, depth_first=True)

        return depth_to_space_conv


class LSID_bn(nn.Module):
    def __init__(self, inchannel=4, block_size=2):
        super(LSID_bn, self).__init__()
        self.block_size = block_size

        self.conv1_1 = nn.Conv2d(inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_2 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4_2 = nn.BatchNorm2d(256)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5_2 = nn.BatchNorm2d(512)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.bn_up6 = nn.BatchNorm2d(256)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6_1 = nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6_2 = nn.BatchNorm2d(256)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.bn_up7 = nn.BatchNorm2d(128)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7_2 = nn.BatchNorm2d(128)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.bn_up8 = nn.BatchNorm2d(64)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn8_2 = nn.BatchNorm2d(64)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.bn_up9 = nn.BatchNorm2d(32)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn9_2 = nn.BatchNorm2d(32)

        out_channel = 1 * self.block_size * self.block_size
        self.conv10 = nn.Conv2d(32, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.lrelu(x)

        x = self.up6(x)
        x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)

        x = self.up7(x)
        x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)

        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)

        x = self.up8(x)
        x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)

        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)

        x = self.up9(x)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)

        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.conv10(x)
        # x = nn.functional.sigmoid(x)

        depth_to_space_conv = pixel_shuffle(x, upscale_factor=self.block_size, depth_first=True)

        return depth_to_space_conv



class LSID_bn2(nn.Module):
    def __init__(self, inchannel=4, block_size=2):
        super(LSID_bn2, self).__init__()
        self.block_size = block_size

        self.conv1_1 = nn.Conv2d(inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_2 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4_2 = nn.BatchNorm2d(256)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5_2 = nn.BatchNorm2d(512)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.bn_up6 = nn.BatchNorm2d(256)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6_1 = nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6_2 = nn.BatchNorm2d(256)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.bn_up7 = nn.BatchNorm2d(128)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7_2 = nn.BatchNorm2d(128)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.bn_up8 = nn.BatchNorm2d(64)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn8_2 = nn.BatchNorm2d(64)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.bn_up9 = nn.BatchNorm2d(32)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn9_2 = nn.BatchNorm2d(32)

        out_channel = 1 * self.block_size * self.block_size
        self.conv10 = nn.Conv2d(32, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.lrelu(x)
        conv2 = x
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.lrelu(x)
        conv3 = x
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.lrelu(x)
        conv4 = x
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.lrelu(x)

        x = self.up6(x)
        x = self.bn_up6(x)
        # import pdb; pdb.set_trace()
        x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        x = self.conv6_1(x)
        x = self.bn6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.bn6_2(x)
        x = self.lrelu(x)

        x = self.up7(x)
        x = self.bn_up7(x)
        x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)

        x = self.conv7_1(x)
        x = self.bn7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.bn7_2(x)
        x = self.lrelu(x)

        x = self.up8(x)
        x = self.bn_up8(x)
        x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)

        x = self.conv8_1(x)
        x = self.bn8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.bn8_2(x)
        x = self.lrelu(x)

        x = self.up9(x)
        x = self.bn_up9(x)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)

        x = self.conv9_1(x)
        x = self.bn9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.bn9_2(x)
        x = self.lrelu(x)

        x = self.conv10(x)
        # x = nn.functional.sigmoid(x)

        depth_to_space_conv = pixel_shuffle(x, upscale_factor=self.block_size, depth_first=True)

        return depth_to_space_conv

def lsid(batch_norm=False,layer_num=5,input_npm=False,**kwargs):
    if batch_norm==0:
        if layer_num==5:
            if input_npm:
                model = LSID_npm(**kwargs)
            else:
                model = LSID(**kwargs)
        elif layer_num==2:
            if input_npm:
                raise NotImplementedError
            model = LSID_2layer(**kwargs)
        elif layer_num==3:
            if input_npm:
                raise NotImplementedError
            model = LSID_3layer(**kwargs)
        elif layer_num==6:
            if input_npm:
                raise NotImplementedError
            model = LSID_6layer(**kwargs)
    elif batch_norm==1:
        if input_npm:
            raise NotImplementedError
        model = LSID_bn2(**kwargs)
    elif batch_norm==2:
        if input_npm:
            raise NotImplementedError
        model = U_Net(**kwargs)
    return model



class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False),
            # nn.BatchNorm2d(out_ch),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, inchannel=4, block_size=2):
        in_ch = inchannel
        self.block_size = block_size

        out_ch = 1 * self.block_size * self.block_size
        super(U_Net, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, conv_block):
                # import pdb; pdb.set_trace()
                for item in m.conv:
                    if isinstance(item, nn.Conv2d):
                        n = item.kernel_size[0] * item.kernel_size[1] * item.out_channels
                        item.weight.data.normal_(0, math.sqrt(2. / n))
                        item.bias.data.zero_()
                    elif isinstance(item, nn.ConvTranspose2d):
                        n = item.kernel_size[0] * item.kernel_size[1] * item.out_channels
                        item.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # import pdb; pdb.set_trace()

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        ########################################## Important! ##############################
        ## We omit the sigmoid here. Make sure that you add sigmoid in the following stages:
        ## - e.g. image_write; loss_computing
        ####################################################################################
        
        depth_to_space_conv = pixel_shuffle(out, upscale_factor=self.block_size, depth_first=True)

        return depth_to_space_conv

        # return out
