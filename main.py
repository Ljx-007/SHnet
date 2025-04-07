import csv
import os
import random

import cv2
from skimage import io
import numpy as np
import torch.nn.functional as F
import torch
import tqdm
from torchvision.transforms import ToTensor,Resize,Compose,CenterCrop,Normalize
from PIL import Image
from src.SHnet import SHnet

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_path="./testdata"
checkpoint_path="checkpoint/best_checkpoint_acc_AI.pth"
file_list=os.listdir(data_path)
img_list=[]
id_list=[]
to_tensor=ToTensor()
input_size=(256,256)
resize=Resize(input_size)
crop=CenterCrop(input_size)
for img_name in file_list:
    img_path=os.path.join(data_path,img_name)
    name=os.path.splitext(img_name)[0]
    id_list.append(name)
    img_list.append(img_path)

checkpoint=torch.load(checkpoint_path)
model=SHnet(img_size=input_size[0],patch=input_size[0]//64,cnn_dim=8,vit_dim=256,depth=[3,4,2],vit_heads=8,cross_head=8,drop_rate=0,num_classes=2).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

label_list=[]
for i in tqdm.trange(len(img_list)):
    img=io.imread(img_list[i])
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h,w=img.shape[0],img.shape[1]
    img=to_tensor(img)
    img_resize=resize(img)
    if h <= 512 or w <= 512:
        img_crop = img_resize
    else:
        img_crop = crop(img)
    img_resize=img_resize.unsqueeze(0).cuda()
    img_crop=img_crop.unsqueeze(0).cuda()

    with torch.no_grad():
        output,_=model(img_resize,img_crop)
        label=output.argmin(1)
    label_list.append(label.item())

with open("cla_pre.csv",'w',newline='',encoding='utf-8') as csvfile:
    csvwriter=csv.writer(csvfile)
    for id,label in zip(id_list,label_list):
        csvwriter.writerow([id,label])

print("done")
