import paddle.nn as nn
import paddle
from paddle.vision.transforms import transforms


def SA_calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.shape
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape([N, C, -1]).var(axis=2) + eps
    feat_std = feat_var.sqrt().reshape([N, C, 1, 1])
    feat_mean = feat.reshape([N, C, -1]).mean(axis=2).reshape([N, C, 1, 1])
    return feat_mean, feat_std

def SA_mean_variance_norm(feat):
    size = feat.shape
    mean, std = SA_calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def SANETmoedl():
    decoder = nn.Sequential(
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(512, 256, [3, 3]),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(256, 256, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(256, 256, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(256, 256, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(256, 128, [3, 3]),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(128, 128, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(128, 64, [3, 3]),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(64, 64, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(64, 3, [3, 3]),
    )

    vgg = nn.Sequential(
        nn.Conv2D(3, 3, [1, 1]),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(3, 64, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(64, 64, [3, 3]),
        nn.ReLU(),
        nn.MaxPool2d([2, 2], [2, 2], [0, 0], ceil_mode=True),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(64, 128, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(128, 128, [3, 3]),
        nn.ReLU(),
        nn.MaxPool2d([2, 2], [2, 2], [0, 0], ceil_mode=True),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(128, 256, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(256, 256, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(256, 256, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(256, 256, [3, 3]),
        nn.ReLU(),
        nn.MaxPool2d([2, 2], [2, 2], [0, 0], ceil_mode=True),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(256, 512, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(512, 512, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(512, 512, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(512, 512, [3, 3]),
        nn.ReLU(),
        nn.MaxPool2d([2, 2], [2, 2], [0, 0], ceil_mode=True),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(512, 512, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(512, 512, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(512, 512, [3, 3]),
        nn.ReLU(),
        nn.ReflectionPad2d([1, 1, 1, 1]),
        nn.Conv2D(512, 512, [3, 3]),
        nn.ReLU()
    )

    return vgg,decoder

class SANet(nn.Layer):

    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2D(in_planes, in_planes, [1, 1])
        self.g = nn.Conv2D(in_planes, in_planes, [1, 1])
        self.h = nn.Conv2D(in_planes, in_planes, [1, 1])
        self.sm = nn.Softmax(axis=-1)
        self.out_conv = nn.Conv2D(in_planes, in_planes, [1, 1])

    def forward(self, content, style):
        F = self.f(SA_mean_variance_norm(content))
        G = self.g(SA_mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.shape
        F = F.reshape([b, -1, w * h]).transpose([0, 2, 1])
        b, c, h, w = G.shape
        G = G.reshape([b, -1, w * h])
        S = paddle.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.shape
        H = H.reshape([b, -1, w * h])
        O = paddle.bmm(H, S.transpose([0, 2, 1]))
        b, c, h, w = content.shape
        O = O.reshape([b, c, h, w])
        O = self.out_conv(O)
        O += content
        return O


class SANET_Transform(nn.Layer):
    def __init__(self, in_planes):
        super(SANET_Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes=in_planes)
        self.sanet5_1 = SANet(in_planes=in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d([1, 1, 1, 1])
        self.merge_conv = nn.Conv2D(in_planes, in_planes, [3, 3])

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        return self.merge_conv(self.merge_conv_pad(
            self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))

def SA_test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


