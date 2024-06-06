import torch
from torch import nn
from torch.nn.utils import weight_norm

class SELayer(nn.Module):
  def __init__(self, channel, reduction=16):
      super(SELayer, self).__init__()
      self.avg_pool = nn.AdaptiveAvgPool3d(1)
      self.fc = nn.Sequential(
          nn.Linear(channel, channel // reduction),
          nn.ReLU(inplace=True),
          nn.Linear(channel // reduction, channel),
          nn.Sigmoid()
      )

  def forward(self, x):
      b, c, _, _,_ = x.size()
      y = self.avg_pool(x).view(b, c)
      y = self.fc(y).view(b, c, 1, 1, 1)
      return x * y.expand_as(x)


class Chomp3d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp3d, self).__init__()
        self.chomp_size = chomp_size
        self.forward = self.chomp if chomp_size else self.skip

    def chomp(self, X):
        return X[:, :, :-self.chomp_size, :, :].contiguous()

    def skip(self, X):
        return X


class Bite3d(nn.Module):
    def __init__(self, bite_size):
        super(Bite3d, self).__init__()
        self.bite_size = bite_size

    def forward(self, X):
        return X[:, :, self.bite_size:, :, :].contiguous()


class TemporalBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation, padding, groups=1, dropout=0.2, activation=nn.ReLU, reduction=16):
        super(TemporalBlock3d, self).__init__()
        conv1 = weight_norm(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )
        )
        chomp1 = Chomp3d(padding[0])
        activation1 = activation()
        dropout1 = nn.Dropout(dropout)
        conv2 = weight_norm(
            nn.Conv3d(
                out_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )
        )
        chomp2 = Chomp3d(padding[0])
        activation2 = activation()
        dropout2 = nn.Dropout(dropout)
        self.main_branch = nn.Sequential(
            conv1, chomp1, activation1, dropout1,
            conv2, chomp2, activation2, dropout2
        )
        self.se = SELayer(out_channels, reduction)
        self.downsample = nn.Conv3d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None
        self.activation = activation()
        self.init_weights()

    def init_weights(self):
        self.main_branch[0].weight.data.normal_(0, 0.01)  # conv1
        self.main_branch[4].weight.data.normal_(0, 0.01)  # conv2
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, X):
        out = self.main_branch(X)
        out = self.se(out)
        res = X if self.downsample is None else self.downsample(X)
        return self.activation(out + res)


class TemporalConv3dStack(nn.Module):
    def __init__(self, in_channels, block_channels, kernel_size,
                 space_dilation, groups, dropout, activation):
        super(TemporalConv3dStack, self).__init__()
        blocks = []
        for i, out_channels in enumerate(block_channels):
            time_dilation = 2**i
            padding = (
                (kernel_size[0]-1)*time_dilation,
                (kernel_size[1]*space_dilation - 1)//2,
                (kernel_size[2]*space_dilation - 1)//2
            )
            blocks.append(
                TemporalBlock3d(
                    in_channels, out_channels, kernel_size, stride=1,
                    dilation=(time_dilation, space_dilation, space_dilation),
                    padding=padding, groups=1, dropout=dropout
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*blocks)

    def forward(self, X):
        return self.network(X)


class CausalTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, groups=1,
                 bias=True, dilation=(1, 1, 1)):
        super(CausalTranspose3d, self).__init__()
        d, h, w = kernel
        st_d, st_h, st_w = stride
        dil_d, dil_h, dil_w = dilation
        padding = (0, h*dil_h//2, w*dil_w//2)
        out_padding = (st_d//2, st_h//2, st_w//2)
        self.network = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, (d, h, w), (st_d, st_h, st_w),
                padding, out_padding, groups, bias, dilation,
            ),
            Chomp3d((d-1)*dil_d)
        )

    def forward(self, X):
        return self.network(X)


class CausalPool3d(nn.Module):
    def __init__(self, op, kernel, stride=None):
        super(CausalPool3d, self).__init__()
        stride = kernel if stride is None else stride
        padding = (kernel[0]-1, 0, 0)
        if op == 'avg':
            pool = nn.AvgPool3d(
                kernel, (1, *stride[1:]), padding, count_include_pad=False
            )
        elif op == 'max':
            pool = nn.MaxPool3d(kernel, (1, *stride[1:]), padding)
        chomp = Chomp3d(padding[0])
        bite = Bite3d(padding[0])
        downsample = nn.AvgPool3d((1, 1, 1), (stride[0], 1, 1))
        self.network = nn.Sequential(pool, chomp, bite, downsample)

    def forward(self, X):
        return self.network(X)


class ATCNConv3d(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, space_dilation, groups, dropout, activation):
        super(ATCNConv3d, self).__init__()
        self.tcn = TemporalConv3dStack(input_size, num_channels, kernel_size=kernel_size, space_dilation=space_dilation, groups=groups, dropout=dropout, activation=activation)
        self.pool = CausalPool3d('avg', (2, 2, 2))
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, inputs):
        """Inputs have to have dimension (N, C, T, D, H, W), batch_size, channels, time_steps, depth, height, width"""
        batch_size, channels, time_steps, depth, height, width = inputs.size()
        inputs = inputs.view(batch_size, channels, time_steps, depth, height, width)
        x = self.tcn(inputs)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
