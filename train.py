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
model.to(device)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

EPOCH = 40
BS = 128
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCH // 3, gamma=0.1)

train_loader, val_loader = get_loader(BS)

train_loss_arr = []
val_loss_arr = []

best_val_loss = 0.01

for epoch in range(EPOCH + 1):
    loss = {"loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "total": 0.0}
    iter = 0

    if epoch > 0:
        model.train()
        for batch_data in train_loader:
            img = batch_data['img'].to(device)
            target = {'cls': batch_data['cls'].to(device), 'pt': batch_data['pt'].to(device)}

            loss = model(img, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

    # cur loss in train set
    model.eval()
    train_set_loss = 0
    with torch.no_grad():
        total_loss = 0.0
        loss_cnt = 0

        for batch_data in train_loader:
            img = batch_data['img'].to(device)
            target = {'cls': batch_data['cls'].to(device), 'pt': batch_data['pt'].to(device)}

            loss = model(img, target)

            total_loss += loss.item()
            loss_cnt += 1

        train_set_loss = total_loss / loss_cnt

    # cur loss in val set
    val_set_loss = 0
    with torch.no_grad():
        total_loss = 0.0
        loss_cnt = 0
        for batch_data in train_loader:
            img = batch_data['img'].to(device)
            target = {'cls': batch_data['cls'].to(device), 'pt': batch_data['pt'].to(device)}

            loss = model(img, target)

            total_loss += loss.item()
            loss_cnt += 1

        val_set_loss = total_loss / loss_cnt

        print(f'{epoch=} train loss {train_set_loss} val loss {val_set_loss} lr {lr_scheduler.get_last_lr()}')
        train_loss_arr.append(train_set_loss)
        val_loss_arr.append(val_set_loss)

    if val_set_loss < best_val_loss:
        best_val_loss = val_set_loss
        torch.save({'model': model.state_dict()}, 'best.pth')

fig, axs = plt.subplots(1, 2)
axs[0].plot(range(len(train_loss_arr)), train_loss_arr)
axs[1].plot(range(len(val_loss_arr)), val_loss_arr)
plt.show()
