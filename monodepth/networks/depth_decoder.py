# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from ..layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.upconv_4_0 = ConvBlock(self.num_ch_enc[-1], self.num_ch_dec[4])
        num_ch_in = self.num_ch_enc[3] if self.use_skips else 0
        self.upconv_4_1 = ConvBlock(self.num_ch_dec[4] + num_ch_in, self.num_ch_dec[4])

        self.upconv_3_0 = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])
        num_ch_in = self.num_ch_enc[2] if self.use_skips else 0
        self.upconv_3_1 = ConvBlock(self.num_ch_dec[3] + num_ch_in, self.num_ch_dec[3])

        self.upconv_2_0 = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])
        num_ch_in = self.num_ch_enc[1] if self.use_skips else 0
        self.upconv_2_1 = ConvBlock(self.num_ch_dec[2] + num_ch_in, self.num_ch_dec[2])

        self.upconv_1_0 = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])
        num_ch_in = self.num_ch_enc[0] if self.use_skips else 0
        self.upconv_1_1 = ConvBlock(self.num_ch_dec[1] + num_ch_in, self.num_ch_dec[1])

        self.upconv_0_0 = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
        self.upconv_0_1 = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])

        self.dispconv_0 = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        if 1 in self.scales:
            self.dispconv_1 = Conv3x3(self.num_ch_dec[1], self.num_output_channels)
        if 2 in self.scales:
            self.dispconv_2 = Conv3x3(self.num_ch_dec[2], self.num_output_channels)
        if 3 in self.scales:
            self.dispconv_3 = Conv3x3(self.num_ch_dec[3], self.num_output_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = []
        x = input_features[-1]
        
        x = self.upconv_4_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.upconv_4_1(x)

        x = self.upconv_3_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [input_features[2]]
        x = torch.cat(x, 1)
        x = self.upconv_3_1(x)
        if 3 in self.scales:
            self.outputs.append(self.sigmoid(self.dispconv_3(x)))
            
        
        x = self.upconv_2_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [input_features[1]]
        x = torch.cat(x, 1)
        x = self.upconv_2_1(x)
        if 2 in self.scales:
            self.outputs.append(self.sigmoid(self.dispconv_2(x)))
        
        x = self.upconv_1_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [input_features[0]]
        x = torch.cat(x, 1)
        x = self.upconv_1_1(x)
        if 1 in self.scales:
            self.outputs.append(self.sigmoid(self.dispconv_1(x)))
        
        x = self.upconv_0_0(x)
        x = [upsample(x)]
        x = torch.cat(x, 1)
        x = self.upconv_0_1(x)
        self.outputs.append(self.sigmoid(self.dispconv_0(x)))

        return self.outputs
