from torch import nn
import torch
from copy import deepcopy


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
    def __init__(self, device, lmbda=1e-2, beta=1e-6):
        super().__init__()
        self.lmbda = lmbda
        self.beta = beta
        self.criterion = nn.MSELoss()
        self.count_criterion = nn.L1Loss()
        self.MSEloss = AvgMeter()
        self.additional_loss = AvgMeter()
        self.count_loss = AvgMeter()
        self.total_loss = None
        self.device = device

    def forward(self, prediction1, prediction2, gt1, gt2, weight):
        basic_loss = self.criterion(prediction1, gt1) + self.criterion(prediction2, gt2)
        weight_t = torch.transpose(deepcopy(weight), 1, 0).contiguous().to(self.device)
        additional_loss = self.lmbda * torch.det((torch.mm(weight, weight_t) - torch.eye(weight.size(0)).to(self.device)))
        count_loss = self.beta * self.count_criterion(torch.sum(prediction1) + torch.sum(prediction2), torch.sum(gt1) + torch.sum(gt2))
        self.total_loss = basic_loss + additional_loss + count_loss
        self.MSEloss.update(basic_loss, prediction1.size(0))
        self.additional_loss.update(additional_loss, prediction1.size(0))
        self.count_loss.update(count_loss, prediction1.size(0))
        return self.total_loss

    def reset(self):
        self.MSEloss.reset()
        self.additional_loss.reset()
        self.count_loss.reset()
