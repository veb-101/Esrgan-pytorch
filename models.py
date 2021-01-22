# import os
# import numpy as np

import math as m
import torch
import torch.nn as nn
from torchvision.models import vgg19

# from torchsummary import summary


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(
            self.loss_network(high_resolution), self.loss_network(fake_high_resolution)
        )
        return perception_loss


class DenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super().__init__()
        # gc: growth channel, i.e. intermediate channels # x + previous layers output channels
        # 64  -> 32
        self.conv1 = nn.Conv2d(nf + 0 * gc, gc, 3, 1, 1)
        # self.conv1.
        # 64+ 32 -> 32
        self.conv2 = nn.Conv2d(nf + 1 * gc, gc, 3, 1, 1)
        # 64+ 32  + 32-> 32
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        # 64+ 32  + 32+ 32 -> 32
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        # 64+ 32  + 32 + 32 + 32 -> 32
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.res_scale = res_scale

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def weights_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # module.weight = torch.nn.Parameter(data=nn.init.kaiming_normal_(module.weight, 0.2) * 0.1)
                module.weight.data = nn.init.kaiming_normal_(module.weight, 0.2) * 0.1

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        out = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super().__init__()
        self.layer1 = DenseBlock(nf, gc, res_scale)
        self.layer2 = DenseBlock(nf, gc, res_scale)
        self.layer3 = DenseBlock(nf, gc, res_scale)
        self.res_scale = res_scale

        self.layer1.weights_init()
        self.layer2.weights_init()
        self.layer3.weights_init()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.res_scale) + x


class Generator(nn.Module):
    def __init__(self, channels=3, nf=64, gc=32, num_res_blocks=16, scale=4):
        super().__init__()

        assert scale != 0 and (scale & scale - 1) == 0, "not a power of 2"
        scale_factor = int(m.log(scale) / m.log(2.0))
        # scale_factor.

        # first layer
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, nf, kernel_size=3, stride=1)
        )

        # trunk
        self.res_blocks = nn.Sequential(
            *[ResidualInResidualDenseBlock(nf=nf, gc=gc) for _ in range(num_res_blocks)]
        )

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1)
        )

        # Upsampling layers
        upsample_layers = []

        for _ in range(scale_factor):
            upsample_layers.append(
                nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(nf, nf * 4, kernel_size=3, stride=1),
                    nn.LeakyReLU(),
                    nn.PixelShuffle(upscale_factor=2),
                )
            )

        self.upsampling = nn.Sequential(*upsample_layers)

        # Final block
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf, channels, kernel_size=3, stride=1),
        )

        self.layers_ = [self.conv1, self.conv2, *upsample_layers, self.conv3]

    def _mrsa_init(self, layers_):
        for layer in layers_:
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    # print("here")
                    # module.weight = torch.nn.Parameter(data=nn.init.kaiming_normal_(module.weight, 0.2) * 0.1)
                    module.weight.data = nn.init.kaiming_normal_(module.weight) * 0.1

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape=(3, 512, 512)):
        super().__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, bias=False))
            )
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, bias=False)
            ))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([32, 64, 128, 256]):
            layers.extend(
                discriminator_block(in_filters, out_filters, first_block=(i == 0))
            )
            in_filters = out_filters

        layers.append(nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, bias=False))
        )

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

    # def _weights_init(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             # print("here")
    #             # module.weight = torch.nn.Parameter(data=nn.init.kaiming_normal_(module.weight, 0.2) * 0.1)
    #             module.weight.data = nn.init.kaiming_normal_(module.weight) * 0.1


if __name__ == "__main__":
    from torchsummary import summary

    # model = Generator(num_res_blocks=12, nf=32, gc=32)
    # model._mrsa_init(model.layers_)
    # summary(model, (3, 128, 128))

    # model = Discriminator()
    # print(model.output_shape)
    # summary(model, (3, 512, 512))
