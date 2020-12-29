import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



### model1: Unet with based on pretrained resnet18

class decoder_Res18(nn.Module):
    def __init__(self, inC, MiddleC, outC):
        super(decoder_Res18, self).__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(inC, MiddleC, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(MiddleC),
            nn.ConvTranspose2d(MiddleC, outC, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outC)
        )

    def forward(self, x):
        return self.sequence(x)




class Generator_resnet_unet(nn.Module):
    def __init__(self, input_nc, output_nc, pretrained=True, Dropout=0.3):

        super(Generator_resnet_unet, self).__init__()
        if pretrained:
            self.encoder = models.resnet18(pretrained=pretrained)
        else:
            print('please defined your own model here')

        # fix the parameters in resnet18
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.number_class = output_nc
        self.Dropout = Dropout
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_2d = nn.Dropout2d(p=Dropout)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool
                                   )
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.decov = decoder_Res18(512, 512, 256)
        self.decov5 = decoder_Res18(512 + 256, 512, 256)
        self.decov4 = decoder_Res18(256 + 256, 512, 256)
        self.decov3 = decoder_Res18(128 + 256, 256, 64)
        self.decov2 = decoder_Res18(64 + 64, 128, 128)
        self.decov1 = decoder_Res18(128, 128, 32)
        self.decov0 = nn.Sequential(
            nn.Conv2d(32, self.number_class, kernel_size=1)
        )

    def Pad_Same(self, x1, x2):
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.dropout_2d(self.conv2(conv1))
        conv3 = self.dropout_2d(self.conv3(conv2))
        conv4 = self.dropout_2d(self.conv4(conv3))
        conv5 = self.dropout_2d(self.conv5(conv4))

        mid = self.decov(self.pool(conv5))

        decov5 = self.decov5(self.Pad_Same(mid, conv5))
        decov4 = self.decov4(self.Pad_Same(decov5, conv4))
        decov3 = self.decov3(self.Pad_Same(decov4, conv3))
        decov2 = self.decov2(self.Pad_Same(decov3, conv2))
        decov1 = self.decov1(self.dropout_2d(decov2))

        output = self.decov0(decov1)

        return output


### model 2 resnet self defined
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator_resnet(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator_resnet, self).__init__()

        # Initial convolution block       
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


