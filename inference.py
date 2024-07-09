import cv2
import torch
from torchvision.transforms.v2 import functional as v2F


import os


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import ChargeDetection

model = ChargeDetection()
model.load_state_dict(torch.load('latest.pth')['model'])
model.to(device)

print('model load ok')

cap = cv2.VideoCapture('/dev/video2')

save = False

IMG_SIZE = [64, 64]

while True:
    ok, img = cap.read()
    if not ok:
        continue

    with torch.no_grad():
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.from_numpy(img)
        img = img / 255
        img = v2F.resize(img[None, ...], IMG_SIZE, antialias=True)

        viz_img = img.permute(1, 2, 0).numpy()
        viz_img = cv2.cvtColor(viz_img, cv2.COLOR_GRAY2BGR)

        img = img.unsqueeze(0)
        img = img.to(device)

        pred = model(img)

        pred = pred[0]

        if pred[1] > pred[0]:
            cx = pred[2].item()
            x = int(cx * IMG_SIZE[1])
            cv2.line(viz_img, (x, 0), (x, IMG_SIZE[0] - 1), (255, 0, 0), 1)
        else:
            cv2.putText(
                viz_img,
                'not found',
                (IMG_SIZE[1] // 2, IMG_SIZE[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.15,
                (0, 0, 255),
                2,
            )

    cv2.imshow('Tracking', viz_img)

    ret = cv2.waitKey(1)
    if ret & 0xFF == ord('q'):
        break
