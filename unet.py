import torch
import torch.nn as nn
from monai.transforms import AsDiscrete
from tqdm.notebook import tqdm


# Define the building blocks of the U-Net model
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        self.conv_block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_channels, affine=False, track_running_stats=False),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_channels, affine=False, track_running_stats=False),
        nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv_block(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True))
        
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=2, stride=2)
    
    def forward(self, skip, x):
        x = self.upsample(x)
        x = torch.cat((x, skip), 1)
        x = self.conv_block(x)
        return x
        

def pool3d(x):
    return nn.MaxPool3d(kernel_size=2, stride=2)(x)

# Define the U-Net model
class FlexUNet(nn.Module):
    def __init__(self, num_classes=4, in_channels=4, model_size=16):
        super(FlexUNet, self).__init__()
        
        layer_sizes = [model_size * 2**i for i in range(5)]
        
        self.encoder1 = EncoderBlock(in_channels, layer_sizes[0])
        self.encoder2 = EncoderBlock(layer_sizes[0], layer_sizes[1])
        self.encoder3 = EncoderBlock(layer_sizes[1], layer_sizes[2])
        self.encoder4 = EncoderBlock(layer_sizes[2], layer_sizes[3])
        self.encoder5 = EncoderBlock(layer_sizes[3], layer_sizes[4])
        
        self.decoder1 = DecoderBlock(layer_sizes[4], layer_sizes[3])
        self.decoder2 = DecoderBlock(layer_sizes[3], layer_sizes[2])
        self.decoder3 = DecoderBlock(layer_sizes[2], layer_sizes[1])
        self.decoder4 = DecoderBlock(layer_sizes[1], layer_sizes[0])
        
        self.conv = nn.Conv3d(layer_sizes[0], num_classes, kernel_size=1, stride=1)
    
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(pool3d(e1))
        e3 = self.encoder3(pool3d(e2))
        e4 = self.encoder4(pool3d(e3))
        e5 = self.encoder5(pool3d(e4))
        
        d1 = self.decoder1(e4, e5)
        d2 = self.decoder2(e3, d1)
        d3 = self.decoder3(e2, d2)
        d4 = self.decoder4(e1, d3)
        
        out = self.conv(d4)
        
        return out
    

# Define the training function

one_hot_transform = AsDiscrete(to_onehot=4, dim=1)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    loss_item = 0.0
    
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
    
    return loss_item / len(train_loader)

# Define the validation function
def validate_epoch(model, val_loader, criterion_all, criterion_sep, device):
    model.eval()
    criterion_all.reset()
    criterion_sep.reset()
    loss_item = 0.0
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation", unit="batch", position=1, leave=False):
            inputs = torch.cat([
            data["t1"], data["t2"], data["t1ce"], data["flair"]
            ], dim=1).to(device)
            
            targets = data["seg"].to(device)
            targets[targets == 4] = 3
            
            if targets.shape[1] != 1:
                targets = targets.unsqueeze(1)
                
            targets = one_hot_transform(targets)
            
            outputs = model(inputs)
            
            # Apply softmax if not included in the model
            outputs = torch.softmax(outputs, dim=1)

            # Ensure the outputs have the correct shape
            if outputs.shape != targets.shape:
                outputs = outputs.permute(0, 2, 1, 3, 4)
            
            val_outputs = outputs.cpu()
            val_labels = targets.cpu()
            combined_val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
            combined_val_labels = torch.argmax(val_labels, dim=1, keepdim=True)
            
            criterion_all.update(outputs=val_outputs, targets=val_labels)
            criterion_sep.update(combined_val_outputs,combined_val_labels)
    
    return criterion_all.aggregate().item(), criterion_sep.aggregate()