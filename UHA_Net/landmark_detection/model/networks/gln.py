from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class myConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(myConv2d, self).__init__()
        padding = (kernel_size-1)//2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


class dilatedConv(nn.Module):
    ''' stride == 1 '''

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(dilatedConv, self).__init__()
        # f = (kernel_size-1) * d +1
        # new_width = (width - f + 2 * padding)/stride + stride
        padding = (kernel_size-1) * dilation // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

#通道融合
class SKFusion(nn.Module):
        def __init__(self, dim, height=2, reduction=8):
            super(SKFusion, self).__init__()
            
            #height表示输入的特征图数量，这里是两张特征图
            self.height = height

            d = max(int(dim/reduction), 4)
            
            #d = max(int(dim/reduction), 4) 表示先将输入通道数 dim 除以一个缩小因子 reduction，
            #得到的结果再向下取整，作为 MLP（多层感知器）中间层的通道数 d。
            #这么做的目的是减少 MLP 中间层的通道数，以降低计算量和模型复杂度。具体而言，
            #将输入通道数除以一个较大的 reduction 值，再向下取整，可以使得 d 不至于太小，
            #同时能够在一定程度上减少计算量和模型复杂度。
            #同时为了防止 d 取值太小而导致模型性能下降，d 的最小值被限定为 4，即使得 MLP 中间层的通道数
            #最小为 4。

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            
            #######这里的MLP设置不符合#######
            #512 256 128 64
            #print("dim",dim)
            #print("d",d)
            self.mlp = nn.Sequential(
                #nn.Conv2d(512, 4, 1, bias=False),
                nn.Conv2d(dim, d, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(d, dim*height, 1, bias=False)  #dim*height
                #nn.Conv2d(4, 4, 1, bias=False)
            )
            self.softmax = nn.Softmax(dim=1)
            

        def forward(self, in_feats):
            
            B, C, H, W = in_feats[0].shape
            in_feats = torch.cat(in_feats, dim=1)
            #print("in_feats",in_feats.size())
            #两张图拼接起来了
    
            #print("in_feats.size()",in_feats.size())
            in_feats = in_feats.view(B, self.height, C, H, W)
            #print("in_feats2",in_feats.size())
            
            feats_sum = torch.sum(in_feats, dim=1)
            #print("feats_sum",feats_sum.size())
            
            #这里报错
            #print("平均池化",(self.avg_pool(feats_sum)).size())
            
            attn = self.mlp(self.avg_pool(feats_sum))
           # print("attn",attn.size())
            attn = self.softmax(attn.view(B,self.height, C, 1, 1)) #(B, self.height, C, 1, 1)

            out = torch.sum(in_feats*attn, dim=1)
            #print("out",out.size())

            return out      
    #通道融合

class globalNet(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=0.25, kernel_size=3, dilations=None, dims=[512,256,128,64]):
        super(globalNet, self).__init__()
        self.scale_factor = scale_factor
	# Add a SKFusion module
        
        
        self.fusion1 = SKFusion(dims[0])
        self.fusion2 = SKFusion(dims[1])
        self.fusion3 = SKFusion(dims[2])
        self.fusion4 = SKFusion(dims[3])
        
        if not isinstance(in_channels, list):
            in_channels = [in_channels]
        if not isinstance(out_channels, list):
            out_channels = [out_channels]
        mid_channels = 128
        if dilations is None:
            dilations = [1, 2, 5]
        for i, n_chan in enumerate(in_channels):
            setattr(self, 'in{i}'.format(i=i),
                    myConv2d(n_chan, mid_channels//2, 3))
        for i, n_chan in enumerate(out_channels):
            setattr(self, 'in_local{i}'.format(i=i),
                    myConv2d(n_chan, (mid_channels+1)//2, 3))
            setattr(self, 'out{i}'.format(i=i),
                    myConv2d(mid_channels, n_chan, 1))
            convs = [dilatedConv(mid_channels, mid_channels,
                                 kernel_size, dilation) for dilation in dilations]
            convs = nn.Sequential(*convs)
            setattr(self, 'convs{}'.format(i), convs)

    def forward(self, x, local_feature, task_idx=0):
        size = x.size()[2:]
        sf = self.scale_factor
        x = F.interpolate(x, scale_factor=sf)
        local_feature = F.interpolate(local_feature, scale_factor=sf)
        x = getattr(self, 'in{}'.format(task_idx))(x)
        local_feature = getattr(
            self, 'in_local{}'.format(task_idx))(local_feature)
        #fuse = torch.cat((x, local_feature), dim=1)
      	# Apply SKFusion to the concatenated feature maps
        if local_feature.size()[1]==512:
            fuse=torch.cat([self.fusion1([local_feature,x]),local_feature],dim=1)
        if local_feature.size()[1]==256:
            fuse=torch.cat([self.fusion2([local_feature,x]),local_feature],dim=1)
        if local_feature.size()[1]==128:
            fuse=torch.cat([self.fusion3([local_feature,x]),local_feature],dim=1)
        if local_feature.size()[1]==64:
            fuse=torch.cat([self.fusion4([local_feature,x]),local_feature],dim=1)
        x = getattr(self, 'convs{}'.format(task_idx))(fuse)
        x = getattr(self, 'out{}'.format(task_idx))(x)
        x = F.interpolate(x, size=size)
        return torch.sigmoid(x)


class GLN(nn.Module):
    ''' global and local net '''

    def __init__(self, localNet, localNet_params, globalNet_params):
        super(GLN, self).__init__()
        self.localNet = localNet(**localNet_params)
        in_channels = localNet_params['in_channels']
        out_channels = localNet_params['out_channels']
        globalNet_params['in_channels'] = in_channels
        globalNet_params['out_channels'] = out_channels
        self.globalNet = globalNet(**globalNet_params)

    def forward(self, x, task_idx=0):
        local_feature = self.localNet(x, task_idx)['output']
        global_feature = self.globalNet(x, local_feature, task_idx)
        return {'output': global_feature*local_feature}
