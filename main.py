# Import libraries

import matplotlib.pyplot as plt
import gc
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

from config import ModelConfig
from fcn_multibranch import FCN_Multibranch
from functions import *
from train_eval import train_and_eval


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
config = ModelConfig()

# Run process_data.py
exec(open('process_data.py').read())

# Load created images and masks
train_images = torch.load('/kaggle/working/train_images.pt')
val_images = torch.load('/kaggle/working/val_images.pt')
train_masks = torch.load('/kaggle/working/train_masks.pt')
val_masks = torch.load('/kaggle/working/val_masks.pt')

train_dataset = TensorDataset(train_images, train_masks)
val_dataset = TensorDataset(val_images, val_masks)

del train_images, train_masks, val_images, val_masks
gc.collect()

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, pin_memory=True)

# Initialise model

model = FCN_Multibranch(9, 1)
model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

print(summary(model, (9, 256, 256)))

# Model Training and Evalaution

train_loss, val_loss, dice_values, iou_values = train_and_eval(model, criterion, optimizer, config.num_epochs, device, train_dataloader, val_dataloader)

# Plot metrics

x_values = range(config.num_epochs)

plt.plot(x_values, train_loss, label='Train Loss')
plt.plot(x_values, val_loss, label='Validation Loss')
plt.show()