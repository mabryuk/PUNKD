import torch
import torch.nn as nn
import numpy as np
from monai.transforms import AsDiscrete
from tqdm.notebook import tqdm
import torch.nn.functional as F
from punkd.diceMetrices import DiceLoss

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self.conv_block(x)
        return x

# Define the building blocks of the U-Net model
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        self.conv_block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        pooled = self.pool(x)
        return x, pooled

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=(2, 2, 2), stride=2)
        self.conv_block = ResidualBlock(in_channels, out_channels)
    
    def forward(self, skip, x):
        x = self.upsample(x)
        x = torch.cat((skip, x), 1)
        x = self.conv_block(x)
        return x
        


# Define the U-Net model
class FlexUNet(nn.Module):
    def __init__(self, num_classes=4, in_channels=4, model_size=16):
        super(FlexUNet, self).__init__()
        
        layer_sizes = [model_size * 2**i for i in range(5)]
        
        self.encoder1 = EncoderBlock(in_channels, layer_sizes[0])
        self.encoder2 = EncoderBlock(layer_sizes[0], layer_sizes[1])
        self.encoder3 = EncoderBlock(layer_sizes[1], layer_sizes[2])
        self.encoder4 = EncoderBlock(layer_sizes[2], layer_sizes[3])
        
        self.bottleneck = ResidualBlock(layer_sizes[3], layer_sizes[4])
        
        self.decoder1 = DecoderBlock(layer_sizes[4], layer_sizes[3])
        self.decoder2 = DecoderBlock(layer_sizes[3], layer_sizes[2])
        self.decoder3 = DecoderBlock(layer_sizes[2], layer_sizes[1])
        self.decoder4 = DecoderBlock(layer_sizes[1], layer_sizes[0])
        
        self.conv = nn.Conv3d(layer_sizes[0], num_classes, kernel_size=1, stride=1)
    
    def forward(self, x):
        e1, p1 = self.encoder1(x)
        e2, p2 = self.encoder2(p1)
        e3, p3 = self.encoder3(p2)
        e4, p4 = self.encoder4(p3)
        
        b = self.bottleneck(p4)
        
        d1 = self.decoder1(e4, b)
        d2 = self.decoder2(e3, d1)
        d3 = self.decoder3(e2, d2)
        d4 = self.decoder4(e1, d3)
        
        out = self.conv(d4)
        
        return out


class FlexUNetStudent(FlexUNet):
    def __init__(self, model_size=32):  
        super().__init__(model_size=model_size)      

# Define the training function



def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    loss_item = 0.0
    one_hot_transform = AsDiscrete(to_onehot=4, dim=1)
    for data in tqdm(train_loader, desc="Training", unit="batch", position=1, leave=False):
        inputs = torch.cat([
            data["t1"], data["t2"], data["t1ce"], data["flair"]
        ], dim=1).to(device)
        
        targets = data["seg"].to(device)
        targets[targets == 4] = 3
        
        if targets.shape[1] != 1:
            targets = targets.unsqueeze(1)
            
        targets = one_hot_transform(targets)
        
        optimizer.zero_grad()

        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Apply softmax if not included in the model
        outputs = torch.softmax(outputs, dim=1)

        # Ensure the outputs have the correct shape
        if outputs.shape != targets.shape:
            outputs = outputs.permute(0, 2, 1, 3, 4)
        
        # Make sure the shapes match before loss calculation
        assert outputs.shape == targets.shape, f"Shape mismatch: outputs {outputs.shape}, labels {targets.shape}"
        
        loss.backward()
        optimizer.step()
        
        loss_item += loss.item()
    
    scheduler.step()

    return loss_item / len(train_loader)

# Define the validation function
def validate_epoch(model, val_loader, score_fn, device):
    model.eval()
    one_hot_transform = AsDiscrete(to_onehot=4, dim=1)
    score_fn.reset()  # Clear previous scores

    class_scores = [[] for _ in range(4)]  # To store scores for each class (background, tumor, enhancing, core)
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation", unit="batch", position=1, leave=False):
            inputs = torch.cat([
                data["t1"], data["t2"], data["t1ce"], data["flair"]
            ], dim=1).to(device)
            
            targets = data["seg"].to(device)
            targets[targets == 4] = 3  # Remap class 4 to 3
            
            if targets.shape[1] != 1:
                targets = targets.unsqueeze(1)
                
            targets = one_hot_transform(targets)
            
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1)  # Apply softmax
            
            if outputs.shape != targets.shape:
                outputs = outputs.permute(0, 2, 1, 3, 4)
            
            for i in range(4):  # Loop through each class
                class_score = score_fn(outputs[:, i], targets[:, i])  # Dice score per class
                class_scores[i].append(class_score.item())
        
    # Aggregate class-wise scores
    class_means = [np.mean(scores) if scores else 0.0 for scores in class_scores]
    total_score = np.mean(class_means)  # Mean score across all classes

    return total_score, class_means  # Return total and individual class scores

    
    
  
    
def student_train_epoch(student_model, teacher_model, train_loader, optimizer, scheduler, device, temperature=3, alpha=0.5):
   
    student_model.train()
    teacher_model.eval()  # Teacher model in evaluation mode
    loss_item = 0.0
    one_hot_transform = AsDiscrete(to_onehot=4, dim=1)

    for data in tqdm(train_loader, desc="Training", unit="batch", position=1, leave=False):
        # Prepare inputs and targets
        inputs = torch.cat([
            data["t1"], data["t2"], data["t1ce"], data["flair"]
        ], dim=1).to(device)

        targets = data["seg"].to(device)
        targets[targets == 4] = 3  # Map class 4 to class 3

        if targets.shape[1] != 1:
            targets = targets.unsqueeze(1)

        targets = one_hot_transform(targets)
        
        optimizer.zero_grad()

        # Forward pass through student and teacher models
        student_logits = student_model(inputs)
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        
        # Compute distillation loss
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        soft_outputs = F.log_softmax(student_logits / temperature, dim=1)

        distillation_loss = F.kl_div(
            soft_outputs, soft_targets, reduction="batchmean"
        ) * (temperature**2)

        # Compute hard loss (Dice loss)
        dice_loss = DiceLoss()
        hard_loss = dice_loss(student_logits, targets)

        # Total loss: weighted sum of distillation and hard loss
        total_loss = alpha * distillation_loss + (1 - alpha) * hard_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        loss_item += total_loss.item()

    scheduler.step()
    return loss_item / len(train_loader)
    