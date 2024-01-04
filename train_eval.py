import tqdm
from functions import *


# Model Training and Evalaution

def train_and_eval(model, criterion, optimizer, NUM_EPOCHS, device, train_dataloader, val_dataloader):

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

    return train_loss, val_loss, dice_values, iou_values