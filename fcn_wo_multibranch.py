import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(FCN, self).__init__()

        self.encoder_1 = self.encoder_block_2(in_channels, 64)   # i/p = 256, o/p = 128
        self.encoder_2 = self.encoder_block_2(64, 128)           # i/p = 128, o/p = 64
        self.encoder_3 = self.encoder_block_3(128, 256)          # i/p = 64, o/p = 32
        self.encoder_4 = self.encoder_block_3(256, 512)          # i/p = 32, o/p = 16

        self.mid = self.mid_block(512, 1024)                     # i/p = 16, o/p = 16

        self.conv_t_32s = nn.ConvTranspose2d(1, 1, 2, 2)         # i/p = 16  , o/p = 32
        self.conv_t_16s = nn.ConvTranspose2d(2, 1, 2, 2)         # i/p = 32,  o/p = 64
        self.conv_t_8s  = nn.ConvTranspose2d(2, 1, 4, 4)         # i/p = 64, o/p = 256

        self.x3_conv_1x1 = nn.Conv2d(256, 1, 1, 1)
        self.x2_conv_1x1 = nn.Conv2d(128, 1, 1, 1)
        
        self.output = nn.Sigmoid()


    def encoder_block_2(self, in_channels, out_channels):
        return  nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
    
    def encoder_block_3(self, in_channels, out_channels):
        return  nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
    
    def mid_block(self, in_channels, out_channels):
        return  nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.5),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.5),
                    nn.Conv2d(out_channels, 1, 3, 1, 'same'),
                    nn.ReLU(inplace=True),
                )

    
    def forward(self, x):

        # Encoder
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)    # conv_1x1
        x3 = self.encoder_3(x2)    # conv_1x1
        x4 = self.encoder_4(x3)

        # Conv_1x1
        x3_1x1 = self.x3_conv_1x1(x3)
        x2_1x1 = self.x2_conv_1x1(x2)

        # Mid-Block
        x4 = self.mid(x4)

        # FCN-32s output
        x4 = self.conv_t_32s(x4)
        x5 = torch.cat([x4, x3_1x1], dim=1)
        x5 = self.conv_t_16s(x5)
        x6 = torch.cat([x5, x2_1x1], dim=1)
        x6 = self.conv_t_8s(x6)
        x6 = self.output(x6)
        
        return x6
