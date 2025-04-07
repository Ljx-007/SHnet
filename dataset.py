import cv2
import numpy as np
import torch
import torchvision.utils
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TF
from torchvision.datasets import ImageFolder


class TrainDataset(Dataset):
    def __init__(self, data_path, resize=(256, 256), transform=None):
        super().__init__()
        self.data = ImageFolder(data_path)
        self.transform = transform
        self.crop = TF.RandomCrop(resize)
        self.resize = TF.Resize(resize)
        self.size = resize
        self.to_tensor = TF.ToTensor()


    def __len__(self):
        return len(self.data.imgs)

    def __getitem__(self, index):
        img_path = self.data.imgs[index][0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)
        h,w=img.shape[0],img.shape[1]
        img=self.to_tensor(img)
        img_resize = self.resize(img)
        if h <= self.size[0] or w<=self.size[1]:
            img_crop = img_resize
        else:
            img_crop = self.crop(img)

        return (img_resize,img_crop), self.data.imgs[index][1]


class TestDataset(Dataset):
    def __init__(self, data_path, resize=(256, 256), transform=None):
        super().__init__()
        self.data = ImageFolder(data_path)
        self.transform = transform
        self.crop = TF.RandomCrop(resize)
        self.resize = TF.Resize(resize)
        self.size = resize
        self.to_tensor = TF.ToTensor()

    def __len__(self):
        return len(self.data.imgs)

    def __getitem__(self, index):
        img_path = self.data.imgs[index][0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)
        h, w = img.shape[0], img.shape[1]
        img=self.to_tensor(img)
        img_resize = self.resize(img)
        if h <= self.size[0] or w<=self.size[1]:
            img_crop = self.resize(img)
        else:
            img_crop = self.crop(img)

        return (img_resize,img_crop), self.data.imgs[index][1]


if __name__ == '__main__':
    # dataset = TrainDataset("./dataset", transform=transform)
    dataset = TestDataset("./test_dataset_label")
    (img_resize,img_crop),label=dataset[2]
    # loader = DataLoader(dataset, 4)
    # pred = torch.tensor([1, 1, 0, 0])
    # for img, label in loader:
    #     print(torch.eq(pred, label).sum())
    # print("done")
    torchvision.utils.save_image(img_resize, 'img_resize.png')
    torchvision.utils.save_image(img_crop, 'img_crop.png')
