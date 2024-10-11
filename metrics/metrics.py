import torch
from torch.nn.functional import mse_loss
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass


@dataclass
class MetricTracker:
    losses: list = None
    accuracies: list = None
    precisions: list = None
    recalls: list = None
    f1_scores: list = None
    dices: list = None
    ious: list = None
    sensitivities: list = None
    specificities: list = None
    ssims: list = None
    psnrs: list = None
    aucs: list = None

    def __post_init__(self):
        self.losses = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.dices = []
        self.ious = []
        self.sensitivities = []
        self.specificities = []
        self.ssims = []
        self.psnrs = []
        self.aucs = []


def calculate_dice(pred, true):
    intersection = (pred * true).sum()
    return (2.0 * intersection) / (pred.sum() + true.sum())


def calculate_iou(pred, true):
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum() - intersection
    return intersection / union


def calculate_sensitivity_specificity(pred, true):
    TP = ((pred == 1) & (true == 1)).sum()
    TN = ((pred == 0) & (true == 0)).sum()
    FP = ((pred == 1) & (true == 0)).sum()
    FN = ((pred == 0) & (true == 1)).sum()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity.item(), specificity.item()


def calculate_ssim_psnr(pred, true, data_range):
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    ssim_index = ssim(pred, true, data_range=data_range)
    mse = mse_loss(torch.tensor(pred), torch.tensor(true), reduction="mean")
    psnr_value = 20 * torch.log10(torch.tensor(data_range) / torch.sqrt(mse))
    return ssim_index, psnr_value.item()


def calculate_dice(pred, true):
    # Implementation of Dice Coefficient
    intersection = (pred * true).sum()
    return (2.0 * intersection) / (pred.sum() + true.sum())


def calculate_iou(pred, true):
    # Implementation of Intersection over Union
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum() - intersection
    return intersection / union
