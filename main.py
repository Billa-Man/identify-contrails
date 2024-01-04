# Import libraries

import matplotlib.pyplot as plt
import gc
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

from config import ModelConfig
from fcn_wo_multibranch import FCN
from fcn_multibranch import FCN_Multibranch
from functions import *


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# Set hyperparameters
config = ModelConfig()

LEARNING_RATE = config.learning_rate
NUM_EPOCHS = config.num_epochs
BATCH_SIZE = config.batch_size
NUM_CLASSES = config.num_classes
WEIGHT_DECAY = config.weight_decay

# Run process_data.py
exec(open('process_data.py').read())


# Load created images and masks
train_images = torch.load('/kaggle/working/train_images.pt')
val_images = torch.load('/kaggle/working/val_images.pt')
train_masks = torch.load('/kaggle/working/train_masks.pt')
val_masks = torch.load('/kaggle/working/val_masks.pt')

train_dataset = TensorDataset(train_images, train_masks)
val_dataset = TensorDataset(val_images, val_masks)

del train_images
del train_masks
del val_images
del val_masks

gc.collect()

train_dataloader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              drop_last=True,
                              pin_memory=True,
                              )


val_dataloader = DataLoader(val_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            )

gc.collect()

model = FCN_Multibranch(9, 1)
model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(summary(model, (9, 256, 256)))

# Model Training and Evalaution

train_loss = []
val_loss = []
dice_values = []
iou_values = []

for i in range(NUM_EPOCHS):
    
    # Train
    model.train()
    train_running_loss = 0.0
    
    for inputs, targets in tqdm(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.float()
        targets = targets.permute(0, 3, 1, 2)
        targets = targets.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.cpu()
        targets = targets.cpu()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item()
        
    # Validation
    model.eval()
    val_running_loss = 0.0
    dice_scores = []
    jaccard_scores = []
    
    for inputs, targets in val_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.float()
        targets = targets.permute(0, 3, 1, 2)
        targets = targets.float()
        outputs = model(inputs)
        outputs = outputs.detach().cpu()
        outputs = outputs.permute(0, 2, 3, 1)
        targets = targets.cpu()
        targets = targets.permute(0, 2, 3, 1)
        loss = criterion(outputs, targets)
        
        val_running_loss += loss.item()
        
        for batch_idx in range(len(outputs)):
            dice = dice_coefficient(outputs[batch_idx], targets[batch_idx])
            jaccard = jaccard_index(outputs[batch_idx], targets[batch_idx])
            dice_scores.append(dice.item())
            jaccard_scores.append(jaccard.item())
            
#         visualize_segmentation(targets[0], outputs[0])
        
    train_running_loss /= len(train_dataloader)
    val_running_loss /= len(val_dataloader)
    avg_dice_score = sum(dice_scores) / len(dice_scores)
    avg_jaccard_score = sum(jaccard_scores) / len(jaccard_scores)
    
    train_loss.append(train_running_loss)
    val_loss.append(val_running_loss)
    dice_values.append(avg_dice_score)
    iou_values.append(avg_jaccard_score)
        
    print(f"Epoch: {i+1}/{NUM_EPOCHS} | Training Loss: {train_running_loss:.5f} | Validation Loss: {val_running_loss:.5f} | Dice: {avg_dice_score:.5f} | IoU: {avg_jaccard_score:.5f}")


# Plot metrics

x_values = range(NUM_EPOCHS)

plt.plot(x_values, train_loss, label='Train Loss')
plt.plot(x_values, val_loss, label='Validation Loss')
plt.show()