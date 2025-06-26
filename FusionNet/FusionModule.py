import torch
import torch.nn as nn
from FusionNet.MoE import MMoE
import torch.nn.functional as F
import numpy as np
from FusionNet.TGFM import MultiTaskGeneralFusion
from einops import rearrange
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last == False:
            out = F.relu(out, inplace=True)
        return out


class FusionBlock(nn.Module):

    def __init__( self,channel=1024,r=16,task_num=3):
        super().__init__()
        self.channel = channel
        self.fc = nn.Linear(channel*2, channel, bias=False)
        self.norm = nn.LayerNorm(channel)
        self.act = nn.ReLU()
        self.fusion1 = MultiTaskGeneralFusion(channel, r)  #TGFM
        self.fusion2 = MMoE(channel, channel, 4 ,channel//2, noisy_gating=True, k=2,task_num=task_num) #TSFM
        self.init_scale_shift()

    def init_scale_shift(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
       
    def forward(self, x,t,task_index):
        B,C,H,W = x.shape
        N=H*W
        z = self.fusion1(x, t)
        x=to_3d(x)
        t=to_3d(t)
        y = torch.cat([x, t], dim=-1)
        y = self.norm(self.act(self.fc(y)))
        y, aux_loss = self.fusion2(y.reshape(B*N, C), task_index)
        y = to_4d(y.reshape(B,N,C),H,W)+z
        return y,aux_loss

