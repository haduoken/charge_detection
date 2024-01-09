import cv2
import torchvision
from data_loader import YOLODataSet, PennFudanDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.transforms import v2 as T
import torch
from torch import nn
import torchvision

import vision_helper.utils as utils
import torch.nn.functional as F


class DetectionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # backbone
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # convert first conv in-channel to 1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # without avg_pool and fc
        self.back_bone = nn.Sequential(*list(resnet.children())[:-1])

        # head
        self.cls_head = nn.Linear(512, 1)
        self.pt_head = nn.Linear(512, 2)

    def forward(self, batch_img, target=None, mode='train'):
        _, h, w = batch_img[0].shape
        batch_img = [img[None] for img in batch_img]
        concat_img = torch.concat(batch_img, dim=0)
        feat = self.back_bone(concat_img)
        feat = torch.flatten(feat, 1)

        cls_pred = self.cls_head(feat)
        pt_pred = self.pt_head(feat)

        if mode == 'train' or mode == 'val':
            return self.loss(cls_pred, pt_pred, target, batch_img[0].device, batch_img[0].dtype, w, h)
        if mode == 'predict':
            return cls_pred, pt_pred

    def loss(self, cls_pred, pt_pred, target, device, dtype, w, h):
        target_label = []
        target_pt = []
        for target_pre_img in target:
            if target_pre_img['labels'].numel() == 0:
                target_label.append(torch.tensor([0], device=device, dtype=dtype))
                target_pt.append(torch.tensor([0, 0], device=device, dtype=dtype))
            else:
                bbox = target_pre_img['boxes']

                # implementation starts here
                ex_widths = bbox[:, 2] - bbox[:, 0]
                ex_heights = bbox[:, 3] - bbox[:, 1]

                ex_ctr_x = (bbox[:, 0] + 0.5 * ex_widths) / w
                ex_ctr_y = (bbox[:, 1] + 0.5 * ex_heights) / h

                pt = torch.cat([ex_ctr_x, ex_ctr_y], dim=0)

                target_label.append(torch.tensor([1], device=device, dtype=dtype)[None])
                target_pt.append(pt[None])

        target_label = torch.concat(target_label, dim=0)
        target_pt = torch.concat(target_pt, dim=0)

        cls_loss = F.binary_cross_entropy_with_logits(cls_pred, target_label)
        pt_loss = F.smooth_l1_loss(pt_pred, target_pt)

        return {'cls_loss': cls_loss, 'pt_loss': pt_loss}


def get_transform(train):
    transforms = []
    transforms.append(T.Grayscale())
    transforms.append(T.Resize((576, 704), antialias=True))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


dataset = YOLODataSet('data', get_transform(train=True))

from vision_helper.engine import train_one_epoch, eval_custom

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset_train = torch.utils.data.Subset(dataset, indices[:-50])
dataset_val = torch.utils.data.Subset(dataset, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn
)

# move model to the right device
model = DetectionModel()
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)


# let's train it just for 2 epochs

def train():
    num_epochs = 25

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # eval_custom(model, data_loader_test, device=device, print_freq=10)
        if epoch % 3 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_{epoch}.pth")

    print("That's it!")


def predict():
    model.to(device)
    model.load_state_dict(torch.load("checkpoints/model_24.pth"))
    model.eval()

    test_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=utils.collate_fn
    )

    for images, targets in test_data_loader:
        _, h, w = images[0].shape

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        cls_pred, pt_pred = model(images, mode='predict')

        cls_pred = torch.sigmoid(cls_pred)

        img = torch.permute(images[0], (1, 2, 0))

        img = img.cpu().detach().numpy().copy()

        print(img.shape)

        if cls_pred[0] > 0.5:
            pt = pt_pred.cpu().detach().numpy()

            ptx = int(pt[0, 0] * w)
            pty = int(pt[0, 1] * h)

            print(ptx, pty)

            cv2.circle(img, (ptx, pty), 3, 255, thickness=-1)

        cv2.imshow('sdf', img)
        cv2.waitKey(-1)


if __name__ == '__main__':
    predict()
    # train()
