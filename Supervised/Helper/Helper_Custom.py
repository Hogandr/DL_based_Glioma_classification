import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel_Pooling(nn.Module):
    def __init__(self, dropout_prob=0.0):
        super(CustomModel_Pooling, self).__init__()

        self.conv1 = self._conv_block(1, 32, kernel_size=3, stride=2)
        self.conv2 = self._conv_block(32, 64, kernel_size=3, stride=1)
        self.conv3 = self._conv_block(64, 128, kernel_size=3, stride=2)
        self.conv4 = self._conv_block(128, 256, kernel_size=3, stride=1)
        self.conv5 = self._conv_block(256, 256, kernel_size=3, stride=2)

        self.conv6 = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=0)  # No BN here
        self.downsample6 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        self.bn_after_concat6 = nn.BatchNorm3d(256)

        self.conv7 = nn.Conv3d(256, 64, kernel_size=3, stride=2, padding=0)  # No BN here
        self.downsample7 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.bn_after_concat7 = nn.BatchNorm3d(128)

        self.final_conv = nn.Conv3d(128, 32, kernel_size=3, stride=1, padding=0)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten(1)

        self.fc1 = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Dropout(dropout_prob))
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout_prob))
        self.fc3 = nn.Linear(256, 1)

    def _conv_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        conv6_out = self.conv6(x)
        down6 = self.downsample6(conv6_out)
        print(conv6_out.shape)
        print(down6.shape)
        print(conv6_out.shape[2:])
        print(down6.shape[2:])

        if down6.shape[2:] != conv6_out.shape[2:]:
            down6 = F.interpolate(down6, size=conv6_out.shape[2:], mode='trilinear', align_corners=False)

        x = torch.cat((conv6_out, down6), dim=1)
        x = self.bn_after_concat6(x)

        conv7_out = self.conv7(x)
        down7 = self.downsample7(conv7_out)

        if down7.shape[2:] != conv7_out.shape[2:]:
            down7 = F.interpolate(down7, size=conv7_out.shape[2:], mode='trilinear', align_corners=False)

        x = torch.cat((conv7_out, down7), dim=1)
        x = self.bn_after_concat7(x)

        x = self.final_conv(x)
        print(x.shape)
        x = self.global_avg_pool(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
