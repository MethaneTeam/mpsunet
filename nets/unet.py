import torch.nn as nn

from thop import profile
from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class PSA(nn.Module):
    def __init__(self, channel=512, reduction=4, S=4):
        super(PSA, self).__init__()
        self.S = S
        self.convs = nn.ModuleList([
            nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1)
            for i in range(S)
        ])
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ) for i in range(S)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        SPC_out = x.view(b, self.S, c // self.S, h, w)
        conv_out = []
        for idx, conv in enumerate(self.convs):
            conv_out.append(conv(SPC_out[:, idx, :, :, :]))
        SPC_out = torch.stack(conv_out, dim=1)
        se_out = [se(SPC_out[:, idx, :, :, :]) for idx, se in enumerate(self.se_blocks)]
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand(-1, -1, -1, h, w)
        softmax_out = self.softmax(SE_out)
        PSA_out = SPC_out * softmax_out
        PSA_out = torch.sum(PSA_out, dim=1)
        return PSA_out

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

        self.cbma64 = CBAM(64)
        self.cbma128 = CBAM(128)
        self.cbma256 = CBAM(256)
        self.cbma512 = CBAM(512)

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        feat1 = self.cbma64(feat1)
        feat2 = self.cbma128(feat2)
        feat3 = self.cbma256(feat3)
        feat4 = self.cbma512(feat4)
        feat5 = self.cbma512(feat5)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True

if __name__ == '__main__':
    model = Unet(2, False, "vgg")
    print(model)
    input = torch.randn(1, 4, 512, 512)

    flops, params = profile(model, inputs=(input,))

    print(f"FLOPs: {flops / 1e9:.2f}")
    print(f"Params: {params / 1e6:.2f}")