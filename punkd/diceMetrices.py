import torch
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def calculate_dice(self, predict, target):
        predict = predict.reshape(-1)  # Use reshape instead of view
        target = target.reshape(-1)   # Use reshape instead of view
        
        intersection = torch.sum(predict * target)  
        union = torch.sum(predict.pow(2)) + torch.sum(target) 
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)
        return dice_coef

    def forward(self, predict, target):
        dice_coef = self.calculate_dice(predict, target)
        return 1 - dice_coef  # Dice loss





if __name__ == '__main__':
      y_target = torch.Tensor([[[0, 1], [1, 0]]]).long()  # Create a target tensor with the same shape as y_predict
      y_predict = torch.Tensor([[[[1.5, 1.0], [0.2, 1.6]]]]).float()
                          
      criterion = DiceLoss()
      loss = criterion(y_predict, y_target)
      print(loss)




class DiceScore(DiceLoss):
    def __init__(self, smooth=1e-6):
        super(DiceScore, self).__init__(smooth)
        self.scores = []  # List to store batch-wise scores

    def forward(self, predict, target):
        dice_coef = self.calculate_dice(predict, target)
        self.scores.append(dice_coef.item())  # Store the score as a float
        return dice_coef

    def aggregate(self):
        # Compute the mean Dice score across all stored scores
        if not self.scores:
            raise ValueError("No scores to aggregate. Ensure forward() is called.")
        mean_score = np.mean(self.scores)
        return torch.tensor(mean_score)  # Return as a torch tensor for consistency

    def reset(self):
        # Clear the stored scores for the next epoch
        self.scores = []

    

if __name__ == '__main__':
      y_target = torch.Tensor([[[0, 1], [1, 0]]]).long()  # Create a target tensor with the same shape as y_predict
      y_predict = torch.Tensor([[[[1.5, 1.0], [0.2, 1.6]]]]).float()
                          
      criterion = DiceScore()
      score = criterion(y_predict, y_target)
      print(score)
