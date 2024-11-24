import torch
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
     def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-6
     def forward(self, predict, target):
        predict = predict.view( -1)  
        target = target.view(-1)  
        
        intersection = torch.sum(predict * target)  
        union = torch.sum(predict.pow(2)) + torch.sum(target) 
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coef 

        return dice_loss
     

if __name__ == '__main__':
      y_target = torch.Tensor([[[0, 1], [1, 0]]]).long()  # Create a target tensor with the same shape as y_predict
      y_predict = torch.Tensor([[[[1.5, 1.0], [0.2, 1.6]]]]).float()
                          
      criterion = DiceLoss()
      loss = criterion(y_predict, y_target)
      print(loss)




class DiceScore(nn.Module):
     def __init__(self):
        super(DiceScore, self).__init__()
        self.smooth = 1e-6
     def forward(self, predict, target):
        predict = predict.view( -1)  
        target = target.view(-1)  

        intersection = torch.sum(predict * target)  
        union = torch.sum(predict.pow(2)) + torch.sum(target) 
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return dice_score

if __name__ == '__main__':
      y_target = torch.Tensor([[[0, 1], [1, 0]]]).long()  # Create a target tensor with the same shape as y_predict
      y_predict = torch.Tensor([[[[1.5, 1.0], [0.2, 1.6]]]]).float()
                          
      criterion = DiceScore()
      score = criterion(y_predict, y_target)
      print(score)
