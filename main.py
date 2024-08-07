import torch
from torch import nn
from torchvision.models.resnet import resnet18


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


IMG_SIZE = (16, 16)
BS = 16

torch.manual_seed(1)


class BasicBlock(nn.Module):
    def __init__(self, in_feature, out_feature, downsample=False) -> None:
        super().__init__()
        self.downsample = downsample
        self.in_feature = in_feature
        self.out_feature = out_feature

        if not downsample:
            self.conv1 = nn.Conv2d(in_feature, out_feature, kernel_size=(3, 3), padding=1)
        else:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_feature, out_feature, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.BatchNorm2d(out_feature),
            )
            self.conv1 = nn.Conv2d(in_feature, out_feature, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.bn1 = nn.BatchNorm2d(out_feature)

        self.conv2 = nn.Conv2d(out_feature, out_feature, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_feature)

        self.relu = nn.ReLU(False)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.downsample_layer(identity)

        x += identity
        x = self.relu(x)

        return x


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l1 = BasicBlock(1, 8, False)
        self.l2 = BasicBlock(8, 16, True)
        self.l3 = BasicBlock(16, 32, True)
        self.l4 = BasicBlock(32, 64, True)
        self.l5 = BasicBlock(64, 128, True)

        self.fc = nn.Linear(128, 3)

        self.label_loss = nn.CrossEntropyLoss()
        self.pt_loss = nn.SmoothL1Loss()

    def loss(self, pred, target):
        target_cls = target['cls']
        target_pt = target['pt']
        # 分类误差
        loss_label = self.label_loss(pred[:, :2], target_cls)
        # 回归误差
        keep = target_cls == 1
        pos_target = target_pt[keep]  # BS * 1
        pos_pred = pred[keep][:, 2]

        loss_pt = self.pt_loss(pos_pred, pos_target)

        return loss_label + loss_pt

    def forward(self, x, target=None):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        x = torch.flatten(x, 1)
        pred = self.fc(x)
        if target is not None:
            loss = self.loss(pred, target)
            return loss
        return pred
