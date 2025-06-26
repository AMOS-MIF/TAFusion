
from torch import nn
from FusionNet.FusionModule import FusionBlock

class Fusion_network(nn.Module):
    def __init__(self, nC=[256,256,256,256,256]):
        super(Fusion_network, self).__init__()
        self.nC=nC
        task_num=3

        self.fusion_block1 = FusionBlock(nC[0], r=256, task_num=task_num)
        self.fusion_block2 = FusionBlock(nC[1], r=128, task_num=task_num)
        self.fusion_block3 = FusionBlock(nC[2], r=64, task_num=task_num)
        self.fusion_block4 = FusionBlock(nC[3], r=32, task_num=task_num)

    def forward(self, x1, x2,type=0):
        f1_0, a = self.fusion_block1(x1[0], x2[0], type)
        f2_0, b = self.fusion_block2(x1[1], x2[1], type)
        f3_0, c = self.fusion_block3(x1[2], x2[2], type)
        f4_0, d = self.fusion_block4(x1[3], x2[3], type)
        return [f1_0, f2_0, f3_0, f4_0],a+b+c+d



