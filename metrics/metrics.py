import torch
from torch.nn.functional import mse_loss
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass, field


@dataclass
class MetricTracker:
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
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


def mse_loss(pred, true):
    return ((pred - true) ** 2).mean()


def calculate_ssim_psnr(pred, true, data_range):
    # Assuming pred and true are PyTorch tensors in the shape [1, 1, 150, 256, 256]
    ssim_total = 0.0
    psnr_total = 0.0
    num_images = pred.size(2)

    for i in range(num_images):
        pred_i = pred[0, 0, i].detach().cpu().numpy()
        true_i = true[0, 0, i].detach().cpu().numpy()

        ssim_total += ssim(pred_i, true_i, data_range=data_range)

        mse = mse_loss(
            torch.tensor(pred_i, dtype=torch.float32),
            torch.tensor(true_i, dtype=torch.float32),
        )
        psnr_total += 20 * torch.log10(torch.tensor(data_range) / torch.sqrt(mse))

    # Average the SSIM and PSNR values over all images
    avg_ssim = ssim_total / num_images
    avg_psnr = psnr_total / num_images

    return avg_ssim, avg_psnr.item()


def calculate_dice(pred, true):
    # Implementation of Dice Coefficient
    intersection = (pred * true).sum()
    return (2.0 * intersection) / (pred.sum() + true.sum())


def calculate_iou(pred, true):
    # Implementation of Intersection over Union
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum() - intersection
    return intersection / union
