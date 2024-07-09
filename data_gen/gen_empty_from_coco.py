import cv2
from dataset_writer import *
import torch

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

# cap = cv2.VideoCapture('rtsp://admin:kilox1234@172.16.2.253:554/Streaming/Channels/101?transportmode=unicast')
# cap = cv2.VideoCapture('rtsp://admin:kilox1234@192.168.8.15:554/Streaming/Channels/101?transportmode=unicast')

cap = cv2.VideoCapture('/dev/video2')
DatasetWriter.cls_init('/home/kilox/data/charge_station/empty/2')


save = False

while True:
    ok, img = cap.read()
    if not ok:
        continue
    cv2.imshow('Tracking', img)

    ret = cv2.waitKey(1)
    if ret & 0xFF == ord('q'):
        break
    if ret & 0xFF == ord(' '):
        save = not save
    if save:
        DatasetWriter.save_empty(img)
