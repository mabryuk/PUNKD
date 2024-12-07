import torch
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def calculate_dice(self, predict, target, reduction='all'):
        dice_coef = 0
        
        if reduction == 'all':
            predict = predict.view(-1)  
            target = target.view(-1)
            intersection = torch.sum(predict * target)  
            union = torch.sum(predict.pow(2)) + torch.sum(target)
        else:
            predict = predict.view(predict.size(0), predict.size(1), -1)
            target = target.view(target.size(0), target.size(1), -1)
            intersection = torch.sum(predict * target, dim=2)
            union = torch.sum(predict.pow(2), dim=2) + torch.sum(target, dim=2)
        
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)
        return dice_coef

    def forward(self, predict, target):
        dice_coef = self.calculate_dice(predict, target)
        dice_coef = 1 - torch.mean(dice_coef)
        return dice_coef





if __name__ == '__main__':
      y_target = torch.Tensor([[[0, 1], [1, 0]]]).long()  # Create a target tensor with the same shape as y_predict
      y_predict = torch.Tensor([[[[1.5, 1.0], [0.2, 1.6]]]]).float()
                          
      criterion = DiceLoss()
      loss = criterion(y_predict, y_target)




class DiceScore(DiceLoss):
    def __init__(self, smooth=1e-6):
        super(DiceScore, self).__init__(smooth)
        self.reduction = 'channel'
        self.score = None

    def forward(self, predict, target):
        dice_coef = self.calculate_dice(predict, target, reduction=self.reduction)
        if self.score is None:
            self.score = dice_coef
        else:
            self.score = torch.cat((self.score, dice_coef), 0)
            
        return dice_coef  
    
    def aggregate(self):
        return torch.mean(self.score, 0)

    def reset(self):
        self.score = None
        

