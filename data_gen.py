import torch
from torch import nn
import os
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
import random


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


def assign_2d_region(a: torch.TensorType, cx, cy, b: torch.TensorType):
    h_b, w_b = b.shape[0], b.shape[1]

    lt_x = cx - w_b // 2
    lt_y = cy - h_b // 2

    pad_x, pad_y = w_b // 2 + 1, h_b // 2 + 1
    pad_a = F.pad(a, (pad_x, pad_x, pad_y, pad_y), value=0)

    pad_a[lt_y + pad_y : lt_y + pad_y + h_b, lt_x + pad_x : lt_x + pad_x + w_b] = b

    a_ret = pad_a[pad_y:-pad_y, pad_x:-pad_x]

    return a_ret


IMG_SIZE = (16, 16)
BS = 16

torch.manual_seed(1)
random.seed(1)

folder = '/home/kilox/data/test_charge'
for i in range(10000):

    img = torch.rand((IMG_SIZE[0], IMG_SIZE[1]))

    has_center = random.randint(0, 1)

    center_x = random.randint(0, IMG_SIZE[0] - 1)
    center_y = random.randint(0, IMG_SIZE[1] - 1)

    roi = torch.tensor([[1, 0, 0, 0, 1], [1, 1, 0, 0, 1], [1, 1, 0, 1, 1], [0, 0, 0, 1, 0], [1, 0, 0, 0, 1]])

    img_fusion = assign_2d_region(img, center_x, center_y, roi)

    label = torch.tensor([has_center, center_x / IMG_SIZE[0]])

    file = os.path.join(folder, f'{i}.pth')
    torch.save(
        {
            'img': img_fusion,
            'cls': torch.tensor(has_center, dtype=torch.long),
            'pt': torch.tensor(center_x / IMG_SIZE[0], dtype=torch.float32),
        },
        file,
    )

    print(f'dump file {file}')
