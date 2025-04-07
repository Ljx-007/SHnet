import torch
import torchvision.transforms
from thop import profile, clever_format
from torch import nn
from torch.nn import functional as F
from src.module import BasicBlock, MBConvBlock, ConstrainCNN, DCT, CrossAttention, pair, PatchEmbedding, ViTlayer
from src.filters import HighPass
from einops.layers.torch import Rearrange


class GatingCNN(nn.Module):
    def __init__(self, in_channel, output_dim=1280):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channel, 16, 3, 2, 1, bias=False),  # 256
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.stem = nn.Sequential(
            BasicBlock(16, 24, stride=2),
            BasicBlock(24, 36, stride=2),  # 128
            BasicBlock(36, 48, stride=2),  # 64
            BasicBlock(48, 128, stride=2),  # 32
            BasicBlock(128, 512, stride=2),
        )
        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        self.mean_fc = nn.Linear(512, output_dim)
        self.logvar_fc = nn.Linear(512, output_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return std + mean

    def forward(self, x):
        x = self.head(x)
        x = self.stem(x)
        x = self.adaptivepool(x)
        x = torch.flatten(x, start_dim=1)
        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)
        x = self.reparameterize(mean, logvar)
        return torch.sigmoid(mean), torch.sigmoid(logvar), x


class SHnet(nn.Module):
    def __init__(self, img_size, patch, cnn_dim, vit_dim, depth,
                 vit_heads, cross_head, drop_rate, num_classes):
        # cnn_dim=8
        super().__init__()
        self.cross1_patch = patch

        self.constrain = ConstrainCNN(3, 3, 5, 1, 2)
        self.highpass = HighPass()
        self.dct = DCT(img_size)

        self.sem_head = nn.Sequential(
            nn.Conv2d(3, cnn_dim * 2, kernel_size=3, stride=2, padding=1),  # 16x256x256
            nn.BatchNorm2d(cnn_dim * 2),
            nn.ReLU(),
            MBConvBlock(cnn_dim * 2, cnn_dim * 4, 3, 1, 1, if_se=True)  # 32x256x256
        )
        self.hf_head = nn.Sequential(
            nn.Conv2d(7, cnn_dim * 2, kernel_size=3, stride=2, padding=1),  # 16x256x256
            nn.BatchNorm2d(cnn_dim * 2),
            nn.ReLU(),
            MBConvBlock(cnn_dim * 2, cnn_dim * 4, 3, 1, 1, if_se=True)
        )
        self.sem_body1 = nn.Sequential(
            MBConvBlock(cnn_dim * 4, cnn_dim * 6, 5, 2, 2),  # 48x128x128
            MBConvBlock(cnn_dim * 6, cnn_dim * 6, 3, 1, 1, if_se=True),
            MBConvBlock(cnn_dim * 6, cnn_dim * 6, 5, 1, 2),
            MBConvBlock(cnn_dim * 6, cnn_dim * 12, 3, 2, 1),  # 96x64x64
            MBConvBlock(cnn_dim * 12, cnn_dim * 12, 3, 1, 1, if_se=True),
            MBConvBlock(cnn_dim * 12, cnn_dim * 12, 3, 1, 1)
        )
        self.hf_body1 = nn.Sequential(
            MBConvBlock(cnn_dim * 4, cnn_dim * 6, 5, 2, 2),  # 48x128x128
            MBConvBlock(cnn_dim * 6, cnn_dim * 6, 3, 1, 1, if_se=True),
            MBConvBlock(cnn_dim * 6, cnn_dim * 6, 5, 1, 2),
            MBConvBlock(cnn_dim * 6, cnn_dim * 12, 3, 2, 1),  # 96x64x64
            MBConvBlock(cnn_dim * 12, cnn_dim * 12, 3, 1, 1, if_se=True),
            MBConvBlock(cnn_dim * 12, cnn_dim * 12, 3, 1, 1)
        )
        self.sem_emb1 = nn.Sequential(
            nn.Conv2d(cnn_dim * 12, vit_dim, kernel_size=self.cross1_patch, stride=self.cross1_patch),
            Rearrange('b c h w -> b (h w) c')
        )
        self.hf_emb1 = nn.Sequential(
            nn.Conv2d(cnn_dim * 12, vit_dim, kernel_size=self.cross1_patch, stride=self.cross1_patch),
            Rearrange('b c h w -> b (h w) c')
        )
        self.cross_attn1 = CrossAttention(dim=vit_dim, num_head=cross_head,
                                          num_patches=(img_size // 8 // self.cross1_patch) ** 2)
        self.sem_body2 = nn.Sequential(
            ViTlayer(dim=vit_dim, depth=depth[0], heads=vit_heads, mlp_dim=vit_dim * 2,
                     num_patches=((img_size // 8 // self.cross1_patch) ** 2), dim_head=vit_dim // vit_heads,
                     dropout=drop_rate),
            nn.Linear(vit_dim, vit_dim * 2),
            nn.LayerNorm(vit_dim * 2),
            nn.GELU(),
            ViTlayer(dim=vit_dim * 2, depth=depth[1], heads=vit_heads, mlp_dim=vit_dim * 3,
                     num_patches=((img_size // 8 // self.cross1_patch) ** 2), dim_head=vit_dim // vit_heads,
                     dropout=drop_rate)
        )
        self.hf_body2 = nn.Sequential(
            ViTlayer(dim=vit_dim, depth=depth[0], heads=vit_heads, mlp_dim=vit_dim * 2,
                     num_patches=((img_size // 8 // self.cross1_patch) ** 2), dim_head=vit_dim // vit_heads,
                     dropout=drop_rate),
            nn.Linear(vit_dim, vit_dim * 2),
            nn.LayerNorm(vit_dim * 2),
            nn.GELU(),
            ViTlayer(dim=vit_dim*2,depth=depth[1],heads=vit_heads,mlp_dim=vit_dim*3,
                     num_patches=((img_size // 8 // self.cross1_patch) ** 2),dim_head=vit_dim//vit_heads,
                     dropout=drop_rate)
        )

        self.cross_attn2 = CrossAttention(num_patches=((img_size // 8 // self.cross1_patch) ** 2),
                                          dim=vit_dim*2, num_head=cross_head)
        self.sem_body3 = nn.Sequential(
            nn.Linear(vit_dim*2, vit_dim * 3),
            nn.LayerNorm(vit_dim * 3),
            nn.GELU(),
            ViTlayer(dim=vit_dim * 3, depth=depth[2], heads=vit_heads, mlp_dim=vit_dim * 4,
                     num_patches=((img_size // 8 // self.cross1_patch) ** 2), dim_head=vit_dim // vit_heads,
                     dropout=drop_rate),
            nn.Linear(vit_dim *3, 1280),
            nn.LayerNorm(1280),
            nn.GELU()
        )
        self.hf_body3 = nn.Sequential(
            nn.Linear(vit_dim*2, vit_dim * 3),
            nn.LayerNorm(vit_dim * 3),
            nn.GELU(),
            ViTlayer(dim=vit_dim * 3, depth=depth[2], heads=vit_heads, mlp_dim=vit_dim * 4,
                     num_patches=((img_size // 8 // self.cross1_patch) ** 2), dim_head=vit_dim // vit_heads,
                     dropout=drop_rate),
            nn.Linear(vit_dim *3, 1280),
            nn.LayerNorm(1280),
            nn.GELU()
        )

        self.gate_net = GatingCNN(3, 1280)
        self.tail = nn.Sequential(
            nn.Linear(2560, 1280),
            nn.LayerNorm(1280),
            nn.Dropout(drop_rate),
            # nn.GELU(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x_resize, x_crop):
        weight_sem, weight_hf, rebuild = self.gate_net(x_resize)
        constrain = self.constrain(x_crop)
        highpass = self.highpass(x_crop)
        dct = self.dct(x_crop)
        hf = torch.cat((constrain,highpass,dct), dim=1)  # 高频分量
        # 处理语义
        sem = self.sem_head(x_resize)
        sem = self.sem_body1(sem)
        sem = self.sem_emb1(sem)  # b c h w -> b h d
        # 处理高频
        hf = self.hf_head(hf)
        hf = self.hf_body1(hf)
        hf = self.hf_emb1(hf)  # b c h w -> b h d
        # 第一次cross attn
        # print(hf.shape)
        # print(sem.shape)
        sem, hf = self.cross_attn1(sem, hf)

        sem = self.sem_body2(sem)
        hf = self.hf_body2(hf)

        # 第二次cross attn
        sem, hf = self.cross_attn2(sem, hf)

        sem = self.sem_body3(sem)
        hf = self.hf_body3(hf)
        _sem=sem.clone().mean(dim=1)
        _hf=hf.clone().mean(dim=1)
        final = torch.cat((_sem * weight_sem, _hf * weight_hf), dim=-1)
        # final=_sem * weight_sem+_hf * weight_hf
        final = self.tail(final)

        return final, rebuild


if __name__ == '__main__':
    inputs = torch.rand(1, 3, 512, 512).cuda()
    # model = GatingCNN(3, 1280).cuda()
    model = SHnet(512, 8, 8, 256, [3,4,2], 8, 8,
                  0.2, 2).cuda()
    model.eval()
    # mean, logvar, output = model(inputs)
    output, rebuild = model(inputs, inputs)
    # print(model)
    flops, params = profile(model, inputs=(inputs, inputs))
    flops, params = clever_format([flops, params])
    # torch.save(model.state_dict(), "model.pth")
    print(flops)
    print(params)
    print(output.shape)
