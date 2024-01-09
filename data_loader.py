import os

import cv2
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class YOLODataSet(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = os.listdir(os.path.join(root, "images"))

    def show(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        # image name xxx.jpg
        pure_name = os.path.splitext(os.path.basename(img_path))[0]
        anno_path = os.path.join(self.root, "annotations", f'{pure_name}.txt')

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        h, w, _ = img.shape

        with open(anno_path, 'r') as f:
            for line in f.readlines():
                data = line.split(' ')
                cx, cy, bw, bh = float(data[1]), float(data[2]), abs(float(data[3])), abs(float(data[4]))
                x1, y1, x2, y2 = w * (cx - bw * 0.5), h * (cy - bh * 0.5), w * (cx + bw * 0.5), h * (cy + bh * 0.5)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255))

                cv2.imshow('sdf', img)
                cv2.waitKey(-1)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        # image name xxx.jpg
        pure_name = os.path.splitext(os.path.basename(img_path))[0]
        anno_path = os.path.join(self.root, "annotations", f'{pure_name}.txt')

        img = read_image(img_path)

        _, h, w = img.size()

        # read anno file
        #
        bbox_data = []
        label = []
        area = []
        iscrowd = []
        with open(anno_path, 'r') as f:
            line_cnt = 0
            for line in f.readlines():
                data = line.split(' ')
                cx, cy, bw, bh = float(data[1]), float(data[2]), abs(float(data[3])), abs(float(data[4]))

                x1, y1, x2, y2 = w * (cx - bw * 0.5), h * (cy - bh * 0.5), w * (cx + bw * 0.5), h * (cy + bh * 0.5)
                bbox_data.append([x1, y1, x2, y2])
                label.append(1)
                area.append(bw * bh * h * w)
                iscrowd.append(False)
                line_cnt += 1
            if line_cnt >= 2:
                print(f'file {anno_path} has multi center')

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(bbox_data, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = torch.tensor(label, dtype=torch.int64)
        target["image_id"] = idx
        target["area"] = torch.tensor(area, dtype=torch.float)
        target["iscrowd"] = torch.tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = YOLODataSet('/home/kilox/workspace/charge_detection/data', None)

    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=1,
    # )
    #
    # for data in data_loader:
    #     pass
    # print(data)

    for i in range(1000):
        dataset.show(i)
    # print(dataset.__len__())
    # print(dataset.__getitem__(0))
