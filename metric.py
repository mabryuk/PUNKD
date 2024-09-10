import numpy as np
import torch

# Define the binary dice coefficient for 3D volumes
def binary_dice3d(s, g):
    num = np.sum(np.multiply(s, g))
    denom = s.sum() + g.sum()
    if denom == 0:
        return 1
    else:
        return 2.0 * num / denom

# Define the Dice coefficient for the whole tumor
def DSC_whole(pred, orig_label):
    return binary_dice3d(pred > 0, orig_label > 0)

# Define the Dice coefficient for the enhancing region
def DSC_en(pred, orig_label):
    # return binary_dice3d(pred == 4, orig_label == 4)
    return binary_dice3d(pred == 3, orig_label == 3)

# Define the Dice coefficient for the core region
def DSC_core(pred, orig_label):
    seg_ = np.copy(pred)
    ground_ = np.copy(orig_label)
    seg_[seg_ == 2] = 0
    ground_[ground_ == 2] = 0
    return binary_dice3d(seg_ > 0, ground_ > 0)

# Define the custom Dice metric class
class SeparateDiceMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.whole_dice_scores = []
        self.enhancing_dice_scores = []
        self.core_dice_scores = []

    def update(self, outputs, targets):
        assert outputs.shape == targets.shape, "Outputs and targets must have the same shape"
        outputs = outputs.numpy() if isinstance(outputs, torch.Tensor) else outputs
        targets = targets.numpy() if isinstance(targets, torch.Tensor) else targets

        whole_dice = DSC_whole(outputs, targets)
        enhancing_dice = DSC_en(outputs, targets)
        core_dice = DSC_core(outputs, targets)

        self.whole_dice_scores.append(whole_dice)
        self.enhancing_dice_scores.append(enhancing_dice)
        self.core_dice_scores.append(core_dice)

    def aggregate(self):
        mean_whole_dice = np.mean(self.whole_dice_scores)
        mean_enhancing_dice = np.mean(self.enhancing_dice_scores)
        mean_core_dice = np.mean(self.core_dice_scores)

        return mean_whole_dice, mean_enhancing_dice, mean_core_dice


# Define the custom Dice metric
class AllDiceMetric:
    def __init__(self, smooth=1e-6, include_background=True):
        self.smooth = smooth
        self.include_background = include_background
        self.reset()

    def reset(self):
        self.intersections = 0.0
        self.unions = 0.0

    def update(self, outputs, targets):
        assert outputs.shape == targets.shape, "Outputs and targets must have the same shape"

        if not self.include_background:
            outputs = outputs[:, 1:]  # Exclude background class
            targets = targets[:, 1:]  # Exclude background class

        outputs = outputs.contiguous()
        targets = targets.contiguous()

        intersection = (outputs * targets).sum(dim=(2, 3, 4))
        union = outputs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))

        self.intersections += intersection.sum(dim=1)
        self.unions += union.sum(dim=1)

    def aggregate(self):
        dice_score = (2.0 * self.intersections + self.smooth) / (self.unions + self.smooth)
        return dice_score.mean()