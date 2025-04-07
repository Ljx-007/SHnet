import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms
from PIL import Image
from torch import nn

f1 = torch.tensor([[[0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, -1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, -1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, -1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, -1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]],dtype=torch.float32).cuda()

f2 = torch.tensor([[[0, 0, 0, 0, 0],
                    [0, 2, 1, 0, 0],
                    [0, 1, -3, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, -1, 0, 0],
                    [0, 0, 3, 0, 0],
                    [0, 0, -3, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 1, 2, 0],
                    [0, 0, -3, 1, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -3, 3, -1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, -3, 1, 0],
                    [0, 0, 1, 2, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, -3, 0, 0],
                    [0, 0, 3, 0, 0],
                    [0, 0, -1, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 1, -3, 0, 0],
                    [0, 2, 1, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [-1, 3, -3, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]],dtype=torch.float32).cuda()

f3 = torch.tensor([[[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, -2, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, -2, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, -2, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0]]],dtype=torch.float32).cuda()

f4 = torch.tensor([[[0, 0, 0, 0, 0],
                    [0, -1, 2, -1, 0],
                    [0, 2, -4, 2, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, -1, 2, 0, 0],
                    [0, 2, -4, 0, 0],
                    [0, -1, 2, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 2, -4, 2, 0],
                    [0, -1, 2, -1, 0],
                    [0, 0, 0, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 2, -1, 0],
                    [0, 0, -4, 2, 0],
                    [0, 0, 2, -1, 0],
                    [0, 0, 0, 0, 0]]],dtype=torch.float32).cuda()

f6 = torch.tensor([[[1, 2, -2, 2, 1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                   [[1, 2, -2, 0, 0],
                    [2, -6, 8, 0, 0],
                    [-2, 8, -12, 0, 0],
                    [2, -6, 8, 0, 0],
                    [1, 2, -2, 0, 0]],

                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [1, 2, -2, 2, 1]],

                   [[0, 0, -2, 2, 1],
                    [0, 0, 8, -6, 2],
                    [0, 0, -12, 8, -2],
                    [0, 0, 8, -6, 2],
                    [0, 0, -2, 2, 1]]],dtype=torch.float32).cuda()

f7 = torch.tensor([[0, 0, 0, 0, 0],
                     [0, -1, 2, -1, 0],
                     [0, 2, -4, 2, 0],
                     [0, -1, 2, -1, 0],
                     [0, 0, 0, 0, 0]],dtype=torch.float32).cuda()

f8 = torch.tensor([[-1, 2, -2, 2, -1],
                     [2, -6, 8, -6, 2],
                     [-2, 8, -12, 8, -2],
                     [2, -6, 8, -6, 2],
                     [-1, 2, -2, 2, -1]],dtype=torch.float32).cuda()

def apply_filter_a(src,conv):
    conv.weight=nn.Parameter(f1[0].expand(3,3,-1,-1),requires_grad=False)
    img=conv(src)
    for filter in f1[1:]:
        conv.weight=nn.Parameter(filter.expand(3,3,-1,-1),requires_grad=False)
        img = torch.add(img, conv(src))

    return img // 8


def apply_filter_b(src,conv):
    conv.weight=nn.Parameter(f2[0].expand(3,3,-1,-1),requires_grad=False)
    img = conv(src)
    for filter in f2[1:]:
        conv.weight=nn.Parameter(filter.expand(3,3,-1,-1),requires_grad=False)
        img = torch.add(img, conv(src))

    return img // 8


def apply_filter_c(src,conv):
    conv.weight=nn.Parameter(f3[0].expand(3,3,-1,-1),requires_grad=False)
    img=conv(src)
    for filter in f3[1:]:
        conv.weight = nn.Parameter(filter.expand(3, 3, -1, -1), requires_grad=False)
        img = torch.add(img, conv(src))

    return img // 4


def apply_filter_d(src,conv):
    conv.weight=nn.Parameter(f4[0].expand(3,3,-1,-1),requires_grad=False)
    img = conv(src)
    for filter in f4[1:]:
        conv.weight = nn.Parameter(filter.expand(3, 3, -1, -1), requires_grad=False)
        img = torch.add(img, conv(src))

    return img // 4


def apply_filter_e(src,conv):
    conv.weight=nn.Parameter(f6[0].expand(3,3,-1,-1),requires_grad=False)
    img = conv(src)
    for filter in f6[1:]:
        conv.weight=nn.Parameter(filter.expand(3,3,-1,-1),requires_grad=False)
        img = torch.add(img, conv(src))

    return img // 4


def apply_filter_f(src,conv):
    conv.weight=nn.Parameter(f7.expand(3,3,-1,-1),requires_grad=False)
    img = conv(src)
    return img


def apply_filter_g(src,conv):
    conv.weight=nn.Parameter(f8.expand(3,3,-1,-1),requires_grad=False)
    img = conv(src)
    return img


class HighPass(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_gray=torchvision.transforms.Grayscale()
        self.conv=nn.Conv2d(3,3,5,1,2,bias=False,padding_mode="reflect")

    def forward(self,src):
        img=(apply_filter_a(src,self.conv)+apply_filter_b(src,self.conv)+apply_filter_c(src,self.conv)+
             apply_filter_d(src,self.conv)+apply_filter_e(src,self.conv)+ apply_filter_e(src,self.conv)+
             apply_filter_g(src,self.conv))
        img=self.to_gray(img/7)
        return img



if __name__ == '__main__':
    from patch_generator import smash_n_reconstruct
    img="../test_dataset_label/fake/0f2aXtqrne0RkkfZ.jpg"
    img2="../test_dataset_label/real/0baJnz2AfuFkXwFe.jpeg"
    img=Image.open(img).convert("RGB").resize((256,256))
    img2=Image.open(img2).convert("RGB").resize((256,256))
    # img.show()
    start=time.time()
    img=torchvision.transforms.ToTensor()(img).unsqueeze(0).cuda()
    img2=torchvision.transforms.ToTensor()(img2).unsqueeze(0).cuda()
    img_rich_fake,img_poor_fake=smash_n_reconstruct(img)
    img_rich_real,img_poor_real=smash_n_reconstruct(img2)

    img_rich_fake=apply_all_filters(img_rich_fake)
    img_poor_fake=apply_all_filters(img_poor_fake)
    # img=apply_all_filters(img)
    img_rich_real=apply_all_filters(img_rich_real)
    img_poor_real=apply_all_filters(img_poor_real)
    # img2=apply_all_filters(img2)
    end=time.time()
    show=torch.cat((img_rich_fake,img_poor_fake,img_rich_real,img_poor_real),dim=0)
    # fake=img_rich_fake-img_poor_fake
    # real=img_rich_real-img_poor_real
    print(end-start)
    # show=torch.cat((fake,real),dim=0)
    torchvision.utils.save_image(show,"show.png")
