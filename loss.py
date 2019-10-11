from torch import nn
import torch


class AvgMeter:
    def __init__(self):
        self.average = 0
        self.num_averaged = 0

    def update(self, loss, size):
        n = self.num_averaged
        m = n + size
        self.average = ((n * self.average) + float(loss)) / m
        self.num_averaged = m

    def reset(self):
        self.average = 0
        self.num_averaged = 0


class SiameseCriterion(nn.Module):
    def __init__(self, device, lmbda=1e-4):
        super().__init__()
        self.lmbda = lmbda
        self.criterion = nn.MSELoss()
        self.MSEloss = AvgMeter()
        self.total_loss = None
        self.device = device

    def forward(self, prediction1, prediction2, gt1, gt2, weight):
        basic_loss = self.criterion(prediction1, gt1) + self.criterion(prediction2, gt2)
        additional_loss = self.lmbda * torch.det((torch.mm(weight, torch.transpose(weight, 0, 1).contiguous()) - torch.eye(weight.size(0)).to(self.device)))
        self.total_loss = basic_loss + additional_loss
        self.MSEloss.update(self.total_loss * 100, prediction1.size(0))
        return self.total_loss

    def reset(self):
        self.MSEloss.reset()
