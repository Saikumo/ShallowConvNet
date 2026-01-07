import torch
from torch.nn import init


# for (B,1,22,1118) crop=500
class ShallowConvNetSpeedup(torch.nn.Sequential):
    def __init__(self):
        super(ShallowConvNetSpeedup, self).__init__()

        self.add_module('ensure4d', Ensure4d())
        self.add_module("dimshuffle", DimShuffle())
        self.add_module("time_conv", TimeConv())
        self.add_module("spat_conv", SpatConv())
        self.add_module("batch_norm", BatchNorm())
        self.add_module("square", Square())  # (B,40,1094,1)

        self.add_module("mean_pool_speedup", MeanPoolSpeedup())  # (B*stride,40,68,1)
        self.add_module("safe_log", SafeLog())
        self.add_module("dropout", Dropout())

        self.add_module("final_conv", FinalConv())  # (B*stride,4,42,1)
        self.add_module("final_squeeze", FinalSqueeze())

        init.xavier_uniform_(self.time_conv.conv_time.weight)
        init.constant_(self.time_conv.conv_time.bias, 0)
        init.xavier_uniform_(self.spat_conv.conv_spat.weight)
        init.constant_(self.batch_norm.bn.weight, 1)
        init.constant_(self.batch_norm.bn.bias, 0)
        init.xavier_uniform_(self.final_conv.conv_final.weight)
        init.constant_(self.final_conv.conv_final.bias, 0)


class Ensure4d(torch.nn.Module):
    def forward(self, x):
        while x.dim() < 4:
            x = x.unsqueeze(-1)
        return x


# batch C T 1 -> batch 1 T C
class DimShuffle(torch.nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 2, 1)


class TimeConv(torch.nn.Module):
    def __init__(self,
                 n_filters_time=40,
                 filter_time_length=25):
        super(TimeConv, self).__init__()
        self.conv_time = torch.nn.Conv2d(1, n_filters_time, (filter_time_length, 1))

    def forward(self, x):
        return self.conv_time(x)


class SpatConv(torch.nn.Module):
    def __init__(self,
                 in_channels=22,
                 n_filters_time=40,
                 n_filters_spat=40,
                 bias_spat=False):
        super(SpatConv, self).__init__()
        self.conv_spat = torch.nn.Conv2d(n_filters_time, n_filters_spat, (1, in_channels), bias=bias_spat)

    def forward(self, x):
        return self.conv_spat(x)


class BatchNorm(torch.nn.Module):
    def __init__(self,
                 n_filters_conv=40,
                 batch_norm_alpha=0.1,
                 affine=True):
        super(BatchNorm, self).__init__()
        self.bn = torch.nn.BatchNorm2d(n_filters_conv, momentum=batch_norm_alpha, affine=affine)

    def forward(self, x):
        return self.bn(x)


class Square(torch.nn.Module):
    def forward(self, x):
        return x ** 2


class MeanPoolSpeedup(torch.nn.Module):
    def __init__(self,
                 pool_time_length=75,
                 pool_time_stride=15):
        super(MeanPoolSpeedup, self).__init__()
        self.stride = pool_time_stride

        self.mean_pool = torch.nn.AvgPool2d(kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1))

    # each fragment must have same shape
    def forward(self, x):
        fragments = []
        for offset in range(self.stride):
            frag = x[:, :, offset:, :]
            fragments.append(self.mean_pool(frag))
        return torch.cat(fragments, 0)


class SafeLog(torch.nn.Module):
    def forward(self, x, eps=1e-6):
        return torch.log(torch.clamp(x, min=eps))


class Dropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.dropout = torch.nn.Dropout(p=p)

    def forward(self, x):
        return self.dropout(x)


class FinalConv(torch.nn.Module):
    def __init__(self,
                 final_conv_length=27,
                 n_filters_conv=40,
                 n_classes=4):
        super(FinalConv, self).__init__()
        self.conv_final = torch.nn.Conv2d(n_filters_conv, n_classes, (final_conv_length, 1))

    def forward(self, x):
        return self.conv_final(x)


class FinalSqueeze(torch.nn.Module):
    def forward(self, x):
        crop_num = 619  # 1118 - 500 + 1
        batch = x.shape[0] // 15
        mean_logits = []
        for batch_no in range(batch):
            crop_mean_pool_logits = []
            for i in range(crop_num):
                index = i % 15
                skip = i // 15
                batch_index = batch_no * 15 + index
                crop = x[batch_index, :, skip + 0: skip + 1, :]
                crop = crop.squeeze(-1)
                crop = crop.squeeze(-1)
                crop_mean_pool_logits.append(crop)
                mean_logits.append(torch.stack(crop_mean_pool_logits, dim=0).mean(dim=0))

        return torch.stack(mean_logits, dim=0)
