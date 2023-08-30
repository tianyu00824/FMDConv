import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import math



class conv_dy(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(conv_dy, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)  # 实现了普通的卷积
        self.dim = int(math.sqrt(inplanes))
        squeeze = max(inplanes, self.dim ** 2) // 16

        print("2inplanes: "+str(inplanes), " planes: "+str(planes), " dim: " + str(self.dim), " squeeze: " + str(squeeze))
        #inplanes: 64  planes: 64  dim: 8, squeeze: 4 start

        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)  # 降维

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)   # 升维
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(inplanes, squeeze, bias=False),
            SEModule_small(squeeze),   # 这里加了一个小的注意力模块
        )
        self.fc_phi = nn.Linear(squeeze, self.dim ** 2, bias=False)  # 论文中的phi矩阵
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()

    def forward(self, x):
        r = self.conv(x)  # 先进行卷积，后对卷积结果加权
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 平均池化，self.avg_pool(x)尺寸为[b,c,1,1]，故加view
        y = self.fc(y)   # fc层
        phi = self.fc_phi(y).view(b, self.dim, self.dim)  # phi 即φ，这一步即为φ(x)
        scale = self.hs(self.fc_scale(y)).view(b, -1, 1, 1)  # hs 即Hsigmoid()激活函数
        r = scale.expand_as(r) * r   # 这里就是加权操作，从公式来看这里应该是A*W0
                                  # 实际上这里是 A*W0*x，即把参数的获取和参数计算融合到一块
                                  # fc_scale实现了A*W0

        out = self.bn1(self.q(x))  # q的话就是压缩通道数
        _, _, h, w = out.size()
                                    # 这里操作的顺序和示意图不太一样
        out = out.view(b, self.dim, -1)
        out = self.bn2(torch.matmul(phi, out)) + out  # +out是做类似残差的处理
        out = out.view(b, -1, h, w)
        out = self.p(out) + r   # p是把通道维进行升维
        return out


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        #kernel_num = 1
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            print("zheer")
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        #print(channel_attention.size())
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        #print(filter_attention.size())
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        #print(spatial_attention.size())
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        #print(kernel_attention.size())
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_num == 1:
            print("zhe")
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)




class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.


class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Attention2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention2, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention


        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        #print(channel_attention.size())
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        #print(filter_attention.size())
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        #print(spatial_attention.size())
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        #print(kernel_attention.size())
        return kernel_attention


    def forward(self, x):
        #print("here!")
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)



class conv_basic_dy(nn.Module):
    def __init__(self, inplanes, planes, stride, kernel_size, groups=1, reduction=0.0625, kernel_num=4, padding=0, dilation=1):
        super(conv_basic_dy, self).__init__()

        self.in_planes = inplanes
        self.out_planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention2(inplanes, planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)

        self.weight = nn.Parameter(torch.randn(kernel_num, planes, inplanes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()


        self.conv = conv3x3(inplanes, planes, stride)  # 实现了普通的卷积
        self.dim = int(math.sqrt(inplanes * 4))
        squeeze = max(inplanes * 4, self.dim ** 2) // 16

        print("inplanes "+str(inplanes)," planes: "+str(planes), " dim: "+ str(self.dim), " squeeze: "+str(squeeze))
        #inplanes 64  planes: 64  dim: 16  squeeze: 16 start
        #inplanes 512  planes: 512  dim: 45  squeeze: 128

        if squeeze < 4:
            squeeze = 4
        #print("stride: "+ str(stride))
        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)  # 降维

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)  # 升维
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(2)

        self.fc2 = nn.Sequential(
            nn.Linear(inplanes * 4, squeeze, bias=False),
            SEModule_small(squeeze),   # 这里加了一个小的注意力模块
        )


        self.fc_phi = nn.Linear(squeeze, self.dim ** 2, bias=False)  # 论文中的phi矩阵
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention


        x2 = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x2, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        #output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        #x = output * filter_attention
        print(output.size())



        r = self.conv(x)   # 先进行卷积，后对卷积结果加权
        b, c, _, _ = x.size()
        #print("size: ", x.size())  # size:  torch.Size([64, 64, 8, 8])
        y = self.avg_pool(x).view(b, c * 4)  # 平均池化，self.avg_pool(x)尺寸为[b,c,1,1]，故加view
        #print("ysize: ",y.size()) # ysize:  torch.Size([64, 256])


        y = self.fc2(y)   # fc层





        phi = self.fc_phi(y).view(b, self.dim, self.dim)  # phi 即φ，这一步即为φ(x)
        scale = self.hs(self.fc_scale(y)).view(b, -1, 1, 1)  # hs 即Hsigmoid()激活函数




        r = scale.expand_as(r) * r   # 这里就是加权操作，从公式来看这里应该是A*W0
                                  # 实际上这里是 A*W0*x，即把参数的获取和参数计算融合到一块
                                  # fc_scale实现了A*W0

        out = self.bn1(self.q(x))  # q的话就是压缩通道数
        _, _, h, w = out.size()
                                    # 这里操作的顺序和示意图不太一样
        out = out.view(b, self.dim, -1)
        out = self.bn2(torch.matmul(phi, out)) + out  # +out是做类似残差的处理
        out = out.view(b, -1, h, w)
        out = self.p(out) + r  # p是把通道维进行升维

        #out = out.view(batch_size, self.out_planes, out.size(-2), out.size(-1))
        #out = out * filter_attention


        return out

