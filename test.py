import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from torch import nn
from data_loader import *


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

# deteministic
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from model import ChargeDetection
from data_loader import ChargeDataset

model = ChargeDetection()
model.load_state_dict(torch.load('latest.pth')['model'])
model.to(device)


train_loader, val_loader = get_loader(1)

train_loss_arr = []
val_loss_arr = []

best_val_loss = 0.01

IMG_SIZE = [64, 64]

# cur loss in train set
model.eval()
with torch.no_grad():
    for batch_data in train_loader:
        img = batch_data['img'].to(device)
        pred = model(img)

        img = img.detach().cpu()[0]
        img = img.permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        pred = pred[0]

        if pred[1] > pred[0]:
            cx = pred[2].item()
            x = int(cx * IMG_SIZE[1])
            cv2.line(img, (x, 0), (x, IMG_SIZE[0] - 1), (255, 0, 0), 1)
        else:
            cv2.putText(
                img,
                'not found',
                (IMG_SIZE[1] // 2, IMG_SIZE[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.15,
                (0, 0, 255),
                2,
            )

        cv2.imshow('Tracking', img)
        ret = cv2.waitKey(1000)
        if ret & 0xFF == ord('q'):
            break

# cur loss in val set
val_set_loss = 0
with torch.no_grad():
    for batch_data in val_loader:
        img = batch_data['img'].to(device)
        pred = model(img)

        img = img.detach().cpu()[0]
        img = img.permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        pred = pred[0]

        if pred[1] > pred[0]:
            cx = pred[2].item()
            x = int(cx * IMG_SIZE[1])
            cv2.line(img, (x, 0), (x, IMG_SIZE[0] - 1), (255, 0, 0), 1)
        else:
            cv2.putText(
                img,
                'not found',
                (IMG_SIZE[1] // 2, IMG_SIZE[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.15,
                (0, 0, 255),
                2,
            )

        cv2.imshow('Tracking', img)
        ret = cv2.waitKey(1000)
        if ret & 0xFF == ord('q'):
            break
