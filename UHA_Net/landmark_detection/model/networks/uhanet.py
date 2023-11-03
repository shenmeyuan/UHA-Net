import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn


class dwise(nn.Module):
    def __init__(self, inChans, kernel_size=3, stride=1, padding=1):
        super(dwise, self).__init__()
        self.conv1 = nn.Conv2d(inChans, inChans, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=inChans)

    def forward(self, x):
        out = self.conv1(x)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=16):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,dropout=0)
        
    def forward(self,input, mask=None):
        batch_size = input.size(0)
        n,c,h,w = input.shape
        input = torch.reshape(input,(n,c,h*w))
        input = input.transpose(1,2)
        
        # Perform linear operation and split into h heads
        # Q = self.q_linear(input).view((-1, self.num_heads, self.head_dim))
        # K = self.k_linear(input).view((-1, self.num_heads, self.head_dim))
        # V = self.v_linear(input).view((-1, self.num_heads, self.head_dim))
        output = self.attn(input,input,input)[0]
        output = output.transpose(1,2)
        output = torch.reshape(output,(n,c,h,w))
        
        return output
   
class pwise(nn.Module):
    def __init__(self, inChans, outChans, kernel_size=1, stride=1, padding=0):
        super(pwise, self).__init__()
        self.conv1 = nn.Conv2d(
            inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv1(x)
        return out


class MHAConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, task_num=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.mha1 = nn.ModuleList([MultiHeadAttention(in_channels)
                                     for i in range(task_num)])
        self.mha2 = nn.ModuleList([MultiHeadAttention(mid_channels)
                                     for i in range(task_num)])
        self.pwise1 = pwise(in_channels, mid_channels)
        self.pwise2 = pwise(mid_channels, out_channels)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(mid_channels)
                                  for i in range(task_num)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_channels)
                                  for i in range(task_num)])
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x, task_idx=0):
        x = self.pwise1(self.mha1[task_idx](x))
        x = self.relu1(self.bn1[task_idx](x))
        x = self.pwise2(self.mha2[task_idx](x))
        x = self.relu2(self.bn2[task_idx](x))
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, task_num=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.dwise1 = nn.ModuleList([dwise(in_channels)
                                     for i in range(task_num)])
        self.dwise2 = nn.ModuleList([dwise(mid_channels)
                                     for i in range(task_num)])
        self.pwise1 = pwise(in_channels, mid_channels)
        self.pwise2 = pwise(mid_channels, out_channels)
        # self.bn1 = nn.ModuleList([nn.BatchNorm2d(mid_channels)
        #                           for i in range(task_num)])
        # self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_channels)
        #                           for i in range(task_num)])
        self.bn1 = nn.ModuleList([nn.InstanceNorm2d(mid_channels)
                                  for i in range(task_num)])
        self.bn2 = nn.ModuleList([nn.InstanceNorm2d(out_channels)
                                  for i in range(task_num)])    #换成实例归一化
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x, task_idx=0):
        x = self.pwise1(self.dwise1[task_idx](x))
        x = self.relu1(self.bn1[task_idx](x))
        x = self.pwise2(self.dwise2[task_idx](x))
        x = self.relu2(self.bn2[task_idx](x))
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, task_num=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = MHAConv(in_channels, out_channels, task_num=task_num)

    def forward(self, x, task_idx=0):
        return self.conv(self.maxpool(x), task_idx)

#通道融合
class SKFusion(nn.Module):
        def __init__(self, dim, height=2, reduction=8):
            super(SKFusion, self).__init__()
            
            #height表示输入的heatmap数量
            self.height = height

            d = max(int(dim/reduction), 4)
            
            #d = max(int(dim/reduction), 4) 表示先将输入通道数 dim 除以一个缩小因子 reduction，
            #得到的结果再向下取整，作为 MLP（多层感知器）中间层的通道数 d。
            #这么做的目的是减少 MLP 中间层的通道数，以降低计算量和模型复杂度。具体而言，
            #将输入通道数除以一个较大的 reduction 值，再向下取整，可以使得 d 不至于太小，
            #同时能够在一定程度上减少计算量和模型复杂度。
            #同时为了防止 d 取值太小而导致模型性能下降，d 的最小值被限定为 4，即使得 MLP 中间层的通道数
            #最小为 4。

            self.avg_pool = nn.AdaptiveMaxPool2d(1)
            self.mlp = nn.Sequential(
                nn.Conv2d(dim, d, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(d, dim*height, 1, bias=False)  #dim*height
            )
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, in_feats):
            B, C, H, W = in_feats[0].shape
            in_feats = torch.cat(in_feats, dim=1)
            #两张图拼接起来了
            in_feats = in_feats.view(B, self.height, C, H, W)
            #print("in_feats2",in_feats.size())
            
            feats_sum = torch.sum(in_feats, dim=1)
            #print("feats_sum",feats_sum.size())
            
            attn = self.mlp(self.avg_pool(feats_sum))
           # print("attn",attn.size())
            attn = self.softmax(attn.view(B,self.height, C, 1, 1)) #(B, self.height, C, 1, 1)

            out = torch.sum(in_feats*attn, dim=1)
            #print("out",out.size())

            return out      
    #通道融合
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, task_num=1,dims=[512,256,128,64]):
        super().__init__()
        
        # Add a SKFusion module
        self.fusion1 = SKFusion(dims[0])
        self.fusion2 = SKFusion(dims[1])
        self.fusion3 = SKFusion(dims[2])
        self.fusion4 = SKFusion(dims[3])
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, task_num)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels, out_channels, task_num=task_num)

    def forward(self, x1, x2, task_idx=0):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        # return self.conv(x, task_idx)
        
        
        # Apply SKFusion to the concatenated feature maps
        if x1.size()[1]==512:
            x=torch.cat([self.fusion1([x1, x2]),x1],dim=1)
        if x1.size()[1]==256:
            x=torch.cat([self.fusion2([x1, x2]),x1],dim=1)
        if x1.size()[1]==128:
            x=torch.cat([self.fusion3([x1, x2]),x1],dim=1)
        if x1.size()[1]==64:
            x=torch.cat([self.fusion4([x1, x2]),x1],dim=1)
        
        up_feature=self.conv(x, task_idx)
       # print("卷积后的维度",up_feature.size())
        return up_feature


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class uhanet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(uhanet, self).__init__()
        if not isinstance(in_channels, list):
            in_channels = [in_channels]
        if not isinstance(out_channels, list):
            out_channels = [out_channels]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.task_num = len(in_channels)

        for i, (n_chan, n_class) in enumerate(zip(in_channels, out_channels)):
            setattr(self, 'in{i}'.format(i=i), OutConv(n_chan, 64))
            setattr(self, 'out{i}'.format(i=i), OutConv(64, n_class))
        self.conv = MHAConv(64, 64, task_num=self.task_num)
        self.down1 = Down(64, 128, task_num=self.task_num)
        self.down2 = Down(128, 256, task_num=self.task_num)
        self.down3 = Down(256, 512, task_num=self.task_num)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, task_num=self.task_num)
        self.up1 = Up(1024, 512 // factor, bilinear, task_num=self.task_num)
        self.up2 = Up(512, 256 // factor, bilinear, task_num=self.task_num)
        self.up3 = Up(256, 128 // factor, bilinear, task_num=self.task_num)
        self.up4 = Up(128, 64, bilinear, task_num=self.task_num)

    def forward(self, x, task_idx=0):
        x1 = getattr(self, 'in{}'.format(task_idx))(x)
        x1 = self.conv(x1, task_idx)
        x2 = self.down1(x1, task_idx)
        x3 = self.down2(x2, task_idx)
        x4 = self.down3(x3, task_idx)
        x5 = self.down4(x4, task_idx)
        x = self.up1(x5, x4, task_idx)
        x = self.up2(x, x3, task_idx)
        x = self.up3(x, x2, task_idx)
        x = self.up4(x, x1, task_idx)
        logits = getattr(self, 'out{}'.format(task_idx))(x)
        return {'output': torch.sigmoid(logits)}
