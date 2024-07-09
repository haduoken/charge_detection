import os
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.ops import box_convert
from helpers import plot
from matplotlib import pyplot as plt
from box_ops import box_cxcywh_to_xyxy
from torch.utils.data import DataLoader, random_split, Subset
import cv2


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.files = []
        for file in os.listdir(root):
            self.files.append(os.path.join(root, file))

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        data['img'] = data['img'][None, ...]
        return data

    def __len__(self):
        return len(self.files)


class ChargeDataset(torch.utils.data.Dataset):

    def __init__(self, root):
        self.root = root

        self.target_img_w = 64
        self.target_img_h = 64

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annos = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        anno_path = os.path.join(self.root, "annotations", self.annos[idx])
        # img = Image.open(img_path).convert("RGB")
        img = read_image(img_path, ImageReadMode.GRAY)

        # resize
        img = img / 255
        img = F.resize(img, [self.target_img_h, self.target_img_w], antialias=True)

        has_ob = False
        cx = 0
        with open(anno_path, 'r') as f:
            anno_arr = f.read().splitlines()
            for anno in anno_arr:
                anno = [x for x in anno.strip().split(' ')]
                cx = float(anno[1])
                has_ob = True
                break

        return {
            'img': img,
            'cls': torch.tensor(has_ob, dtype=torch.long),
            'pt': torch.tensor(cx, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.imgs)


class TrackingDataset(Dataset):

    def __init__(self, root) -> None:
        super().__init__()
        self.files = [os.path.join(root, x) for x in os.listdir(root)]

    def __getitem__(self, index):
        data = torch.load(self.files[index])
        return data

    def __len__(self):
        return len(self.files)


def get_loader(BS):
    dataset = ChargeDataset('/home/kilox/data/charge_station/ori')
    tracking_root = '/home/kilox/data/charge_station/tracking'
    for folder in os.listdir(tracking_root):
        dir = os.path.join(tracking_root, folder)
        dataset += TrackingDataset(dir)

    positive_len = len(dataset)

    empty_root = '/home/kilox/data/charge_station/empty'
    for folder in os.listdir(empty_root):
        dir = os.path.join(empty_root, folder)
        dataset += TrackingDataset(dir)

    negtive_len = len(dataset) - positive_len

    print(f'dataset size {len(dataset)} positive {positive_len} negtive {negtive_len} p/n {positive_len/negtive_len}')

    train_set, val_set = random_split(dataset, [0.7, 0.3])

    train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=BS, shuffle=True, drop_last=False)

    return train_loader, val_loader


if __name__ == '__main__':

    def register_torch():
        import torch

        try:
            if torch.already_registed:
                pass
        except:
            print('register print')
            torch.already_registed = True
            original_repr = torch.Tensor.__repr__
            torch.Tensor.__repr__ = lambda self: f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

    register_torch()

    # dataset = CustomDataset('/home/kilox/data/test_charge')
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)

    # for data in dataloader:
    #     print(data)
    loader1, loader2 = get_loader(1)
    for data in loader1:
        img = data['img']
        cls = data['cls']
        pt = data['pt']

        img = img[0]
        img = F.to_dtype(img, torch.uint8, scale=True)

        img = img.permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w, _ = img.shape
        if cls.item() == 1:
            pt = pt.item()
            x = int(pt * w)
            cv2.line(img, (x, 0), (x, h - 1), (255, 0, 0), 1)
        else:
            cv2.putText(img, 'not found', (w // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.15, (0, 0, 255), 2)

        cv2.imshow('sdf', img)
        cv2.waitKey(0)

        # _, axs = plt.subplots(nrows=1, ncols=1, squeeze=False)
        # ax = axs[0, 0]
        # ax.imshow(img.permute(1, 2, 0).numpy())
        # ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        # plt.show()
