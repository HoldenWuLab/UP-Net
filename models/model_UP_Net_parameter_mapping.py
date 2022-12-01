import torch.nn as nn
import torch

class UP_Net_parameter_mapping(nn.Module):
    def __init__(self):
        super(UP_Net_parameter_mapping, self).__init__()
        
        num_feat = [64, 128, 256, 512, 1024]
        num_channels = 12
        out_channels = 6
        
        self.down1_img = nn.Sequential(Conv3x3(num_channels, num_feat[0]))
        self.down2_img = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[0], num_feat[1]))
        self.down3_img = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[1], num_feat[2]))
        self.down4_img = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[2], num_feat[3]))
        self.bottom_img = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[3], num_feat[4]))
        self.upconcat4_img = UpConcat(num_feat[4], num_feat[3])
        self.up4_img = Conv3x3(num_feat[4], num_feat[3])
        self.upconcat3_img = UpConcat(num_feat[3], num_feat[2])
        self.up3_img = Conv3x3(num_feat[3], num_feat[2])
        self.upconcat2_img = UpConcat(num_feat[2], num_feat[1])
        self.up2_img = Conv3x3(num_feat[2], num_feat[1])
        self.upconcat1_img = UpConcat(num_feat[1], num_feat[0])
        self.up1_img = Conv3x3(num_feat[1], num_feat[0])
        self.final_img = nn.Conv2d(num_feat[0], 12, kernel_size=1)
                
        self.down1_map = nn.Sequential(Conv3x3(num_channels, num_feat[0]))
        self.down2_map = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[0], num_feat[1]))
        self.down3_map = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[1], num_feat[2]))
        self.down4_map = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[2], num_feat[3]))
        self.bottom_map = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[3], num_feat[4]))
        self.upconcat4_map = UpConcat(num_feat[4], num_feat[3])
        self.up4_map = Conv3x3(num_feat[4], num_feat[3])
        self.upconcat3_map = UpConcat(num_feat[3], num_feat[2])
        self.up3_map = Conv3x3(num_feat[3], num_feat[2])
        self.upconcat2_map = UpConcat(num_feat[2], num_feat[1])
        self.up2_map = Conv3x3(num_feat[2], num_feat[1])
        self.upconcat1_map = UpConcat(num_feat[1], num_feat[0])
        self.up1_map = Conv3x3(num_feat[1], num_feat[0])
        self.final_map = nn.Conv2d(num_feat[0], out_channels, kernel_size=1)

    def forward(self, inputs):
        
        
        down1_feat_map = self.down1_map(inputs)
        down2_feat_map = self.down2_map(down1_feat_map)
        down3_feat_map = self.down3_map(down2_feat_map)
        down4_feat_map = self.down4_map(down3_feat_map)
        bottom_feat_map = self.bottom_map(down4_feat_map)
        
        up4_feat_map = self.upconcat4_map(bottom_feat_map, down4_feat_map)
        up4_feat_map = self.up4_map(up4_feat_map)
        up3_feat_map = self.upconcat3_map(up4_feat_map, down3_feat_map)
        up3_feat_map = self.up3_map(up3_feat_map)
        up2_feat_map = self.upconcat2_map(up3_feat_map, down2_feat_map)
        up2_feat_map = self.up2_map(up2_feat_map)
        up1_feat_map = self.upconcat1_map(up2_feat_map, down1_feat_map)
        up1_feat_map = self.up1_map(up1_feat_map)
        output_map = self.final_map(up1_feat_map) 
        
        
        return output_map

class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.PReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.PReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs



class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        
        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):

        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out

