import torch
import torchvision.transforms
from torch import nn
from torch.nn import functional as F
from src.utils import (
    drop_connect,
    Swish,
    MemoryEfficientSwish,
)
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]


def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size) for j in
          range(size)] for i in range(size)]
    return m


class Filter(nn.Module):
    def __init__(self, size,
                 band_start,
                 band_end,
                 use_learnable=False,
                 norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                                       requires_grad=False)

    def norm_sigma(self, x):
        return 2. * torch.sigmoid(x) - 1.

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + self.norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class DCT(nn.Module):
    def __init__(self, size):
        # size -> img_size
        super().__init__()
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)
        self.high_filter = Filter(size, size // 8, size)

    def forward(self, x):
        x_freq = self._DCT_all @ x @ self._DCT_all_T
        y = self.high_filter(x_freq)
        y = self._DCT_all_T @ y @ self._DCT_all
        return y


class ConstrainCNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.const_weight = nn.Parameter(torch.randn(size=[out_channel, in_channel, kernel_size, kernel_size]),
                                         requires_grad=True)
        central_pixel = (self.const_weight.data[:, 0, 2, 2])
        for i in range(out_channel):
            sumed = self.const_weight.data[i].sum() - central_pixel[i]
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, 2, 2] = -1.0
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = F.conv2d(x, self.const_weight, stride=self.stride)
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthSeperabelConv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class DepthSeperabelConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size, groups=input_channels, **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, drop_rate=0., if_se=False):
        super().__init__()
        self._bn_mom = 0.01  # pytorch's difference from tensorflow
        self._bn_eps = 1e-3
        self.has_se = if_se
        self.id_skip = True  # whether to use skip connection and drop connect
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        # Expansion phase (Inverted Bottleneck)
        self._expand_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0,
                                      bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channel, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel, groups=out_channel,  # groups makes it depthwise
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channel, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(in_channel * 0.25))
            self._se_reduce = nn.Conv2d(in_channels=out_channel, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=out_channel, kernel_size=1)

        # Pointwise convolution phase
        self._project_conv = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=out_channel, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()
        self.drop_rate = drop_rate

    def forward(self, inputs):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        x = self._expand_conv(inputs)
        x = self._bn0(x)
        x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self.in_channel, self.out_channel
        if self.id_skip and self.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if self.drop_rate:
                x = drop_connect(x, p=self.drop_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViTlayer(nn.Module):
    # depth是Transformer里EncoderLayer的数量,dim_head是每个头的维度
    def __init__(self, *,  dim, depth, heads, mlp_dim, num_patches,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, dim, channel):
        super().__init__()
        # self.img_h, self.img_w = img_size
        # assert self.img_h % patch_size == 0 and self.img_w % patch_size == 0, "img_size should be divisible by patch_size "
        # self.num_patches = self.img_h * self.img_w // (patch_size * patch_size)
        # self.patch_dim = patch_size * patch_size * channel
        # self.patch_embedding = nn.Sequential(
        #     Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
        #     nn.LayerNorm(self.patch_dim),
        #     nn.Linear(self.patch_dim, dim),
        #     nn.LayerNorm(dim)
        # )
        self.patch_embdding=nn.Conv2d(channel,dim,kernel_size=patch_size,stride=patch_size)
        # self.inverse_emb = nn.Sequential(
        #     nn.Linear(dim,self.patch_dim),
        #     nn.LayerNorm(self.patch_dim),
        #     Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=patch_size, p2=patch_size,
        #               h=self.img_h // patch_size),
        #     nn.Conv2d(channel, channel, 3, 1, 1)
        # )

    # def inverse_embedding(self, x):
    #     return self.inverse_emb(x)

    def forward(self, x):
        return self.patch_embedding(x)  # b n c


class CrossAttention(nn.Module):
    def __init__(self, num_patches,dim, num_head=8):
        super().__init__()

        self.pos_embedding1 = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cross_msa1 = nn.MultiheadAttention(dim, num_heads=num_head, batch_first=True)
        self.cross_msa2 = nn.MultiheadAttention(dim, num_heads=num_head, batch_first=True)

    def forward(self, x_A, x_B):
        x_A += self.pos_embedding1
        x_B += self.pos_embedding2
        x_B, _ = self.cross_msa1(x_B, x_A, x_A)
        x_A, _ = self.cross_msa1(x_A, x_B, x_B)
        return x_A, x_B


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    from skimage import io, data, color
    from skimage.util import img_as_float

    image_path = r"C:\Users\林嘉曦\Desktop\AI_FaceForgery\B_test\0brndnBoY53jVyE4.jpg"  # 替换为你本地图像的路径
    image = io.imread(image_path)

    # 转换为灰度图像并标准化为浮点数
    # image = img_as_float(color.rgb2gray(image))
    img = torchvision.transforms.ToTensor()(image).to(torch.float32).unsqueeze(0).cuda()
    img = torchvision.transforms.Resize((512, 512))(img)
    print(img.shape)
    x = torch.randn((1, 256, 32, 32)).cuda()
    y=torch.randn((1,256,32,32)).cuda()
    # emb=PatchEmbedding(32,4,)
    # vit=ViTlayer(image_size=32,patch_size=32,dim=64,depth=2,heads=2,
    #              mlp_dim=512,channels=512)
    # conv = MBConvBlock(3, 32, 3, 1, 1, 0.2, True).cuda()
    # img = conv(img)
    # dct=DCT(512).cuda()
    # img=dct(img)
    print(img.shape)
