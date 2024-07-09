import os
import cv2
import torch

import time
import numpy as np
from torchvision.transforms.v2 import functional as F
import shutil


def create_folder(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)


class DatasetWriter:
    cnt = 1
    data_dir: str = ''

    last_save_tms = -1
    last_cx, last_cy = -1, -1
    target_img_h = 64
    target_img_w = 64

    @classmethod
    def cls_init(cls, root='/home/kilox/data/charge_station/tracking1', start_cnt=1):
        cls.data_dir = root
        create_folder(cls.data_dir)

        cls.cnt = start_cnt

    @classmethod
    def save_frame_change(cls, img: np.ndarray, xywh):
        h, w, _ = img.shape
        cx = (xywh[0] + int(xywh[2] / 2)) / w
        cy = (xywh[1] + int(xywh[3] / 2)) / h

        diff = abs(cx - cls.last_cx) + abs(cy - cls.last_cy)
        if diff < 0.1:
            return

        cls.last_cx = cx
        cls.last_cy = cy

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.from_numpy(img)
        img = img / 255

        img = F.resize(img[None, ...], [cls.target_img_h, cls.target_img_w], antialias=True)

        has_ob = True

        data = {
            'img': img,
            'cls': torch.tensor(has_ob, dtype=torch.long),
            'pt': torch.tensor(cx, dtype=torch.float32),
        }

        file = os.path.join(cls.data_dir, f'{cls.cnt}.pth')
        torch.save(data, file)
        print(f'save frame {file}')
        cls.cnt += 1

    @classmethod
    def save_empty(cls, img: np.ndarray):
        cur_time = time.time()
        if abs(cur_time - cls.last_save_tms) < 0.3:
            return
        cls.last_save_tms = cur_time

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.from_numpy(img)
        img = img / 255

        img = F.resize(img[None, ...], [cls.target_img_h, cls.target_img_w], antialias=True)

        has_ob = False

        data = {
            'img': img,
            'cls': torch.tensor(has_ob, dtype=torch.long),
            'pt': torch.tensor(0, dtype=torch.float32),
        }

        file = os.path.join(cls.data_dir, f'{cls.cnt}.pth')
        torch.save(data, file)
        print(f'save frame {file}')
        cls.cnt += 1
