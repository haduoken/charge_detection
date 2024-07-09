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
DatasetWriter.cls_init()


while True:
    print('press space to select a region')
    tracker = cv2.TrackerKCF_create()
    region_ok = False
    while True:
        ok, img = cap.read()
        if not ok:
            continue
        cv2.imshow('Tracking', img)

        ret = cv2.waitKey(1)
        if ret & 0xFF == ord(' '):
            bbox = cv2.selectROI('Tracking', img, False)
            tracker.init(img, bbox)
            region_ok = True
            break
        if ret & 0xFF == ord('q'):
            break
    if not region_ok:
        break

    while True:
        ok, img = cap.read()
        if ok:
            ret, bbox = tracker.update(img)
            if ret:
                x, y, w, h = [int(i) for i in bbox]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                DatasetWriter.save_frame_change(img, [x, y, w, h])

            else:
                cv2.putText(img, "Lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                break

            cv2.imshow('Tracking', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
