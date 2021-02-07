import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, input_shape=(3, 512, 512)):
        super().__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(
                nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(
                        in_filters, out_filters, kernel_size=3, stride=1, bias=False
                    ),
                )
            )
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(
                nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(
                        out_filters, out_filters, kernel_size=3, stride=2, bias=False
                    ),
                )
            )
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(
                discriminator_block(in_filters, out_filters, first_block=(i == 0))
            )
            in_filters = out_filters

        layers.append(
            nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, bias=False),
            )
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

    # model = Generator(num_res_blocks=23, nf=64, gc=32)
    # model.load_state_dict(torch.load("Gen_GAN.pth"), strict=True)
    # model._mrsa_init(model.layers_)

    # summary(model, (3, 64, 64))

    model = Discriminator()
    summary(model, (3, 256, 256))
