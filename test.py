import torch
import torch.nn.functional as F
import cv2

img = cv2.imread('/home/kilox/2.png')

cv2.circle(img, (10, 100), 3, (255, 0, 0), thickness=-1)

cv2.imshow('sdf', img)
cv2.waitKey(0)
