import torch
import torch.nn as nn
import torch.nn.functional as F

# Unet 的 的卷积块 都是进行了两次same的卷积操作，same表示padding后的尺寸和输入尺寸相同
class Block(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.features = features
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x
    

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.conv_encoder_1 = Block(in_channels, features)
        self.conv_encoder_2 = Block(features, features*2)
        self.conv_encoder_3 = Block(features*2, features*4)
        self.conv_encoder_4 = Block(features*4, features*8) #256

        # 256 * 2 = 512
        # 简化了直接3次block操作
        self.bottleneck = Block(features*8, features*16)

        self.upconv4 = nn.ConvTranspose2d(
                            in_channels=features*16,
                            out_channels=features*8,
                            kernel_size=2,
                            stride=2,
                        )
        
        self.conv_decoder_4 = Block(features*16, features*8) # 因为有skip connection，所以输入通道数是features*16
        self.upconv3 = nn.ConvTranspose2d(
                            in_channels=features*8,
                            out_channels=features*4,
                            kernel_size=2,
                            stride=2,
                        )
        self.conv_decoder_3 = Block(features*8, features*4)
        self.upconv2 = nn.ConvTranspose2d(
                            in_channels=features*4,
                            out_channels=features*2,
                            kernel_size=2,
                            stride=2,
                        )
        self.conv_decoder_2 = Block(features*4, features*2)
        self.upconv1 = nn.ConvTranspose2d(
                            in_channels=features*2,
                            out_channels=features,
                            kernel_size=2,
                            stride=2,
                        )
        # stride is 2, so the feature map size will be doubled
        self.decoder1 = Block(features*2, features)
        self.conv = nn.Conv2d(
                            in_channels=features,
                            out_channels=out_channels,
                            kernel_size=1,
                        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_encoder_1_1 = self.conv_encoder_1(x)
        conv_encoder_1_2 = F.max_pool2d(conv_encoder_1_1, kernel_size=2, stride=2)

        conv_encoder_2_1 = self.conv_encoder_2(conv_encoder_1_2)
        conv_encoder_2_2 = F.max_pool2d(conv_encoder_2_1, kernel_size=2, stride=2)

        conv_encoder_3_1 = self.conv_encoder_3(conv_encoder_2_2)
        conv_encoder_3_2 = F.max_pool2d(conv_encoder_3_1, kernel_size=2, stride=2)

        conv_encoder_4_1 = self.conv_encoder_4(conv_encoder_3_2)
        conv_encoder_4_2 = F.max_pool2d(conv_encoder_4_1, kernel_size=2, stride=2)

        bottleneck = self.bottleneck(conv_encoder_4_2)

        conv_decoder_4_1 = self.upconv4(bottleneck)
        conv_decoder_4_2 = torch.cat((conv_decoder_4_1, conv_encoder_4_1), dim=1)
        conv_decoder_4_3 = self.conv_decoder_4(conv_decoder_4_2)

        conv_decoder_3_1 = self.upconv3(conv_decoder_4_3)
        conv_decoder_3_2 = torch.cat((conv_decoder_3_1, conv_encoder_3_1), dim=1)
        conv_decoder_3_3 = self.conv_decoder_3(conv_decoder_3_2)

        conv_decoder_2_1 = self.upconv2(conv_decoder_3_3)
        conv_decoder_2_2 = torch.cat((conv_decoder_2_1, conv_encoder_2_1), dim=1)
        conv_decoder_2_3 = self.conv_decoder_2(conv_decoder_2_2)

        conv_decoder_1_1 = self.upconv1(conv_decoder_2_3)
        conv_decoder_1_2 = torch.cat((conv_decoder_1_1, conv_encoder_1_1), dim=1)
        conv_decoder_1_3 = self.decoder1(conv_decoder_1_2)

        output = torch.sigmoid(self.conv(conv_decoder_1_3))
        return output
