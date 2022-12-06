"""
Define the Smooth L1 Loss
smooth l1 loss { = 0.5x2    if |x| < 1
               { = |x|-0.5  otherwise

In this file, we define loss as follows:
beta = 0.11
smooth l1 loss = 1/(2*beta) * x2   if |x| < beta
               = |x|-beta/2        otherwise
"""
from torch.autograd import Variable
import torch
import torch.nn as nn


class Smooth_L1_Loss(nn.Module):
    def __init__(self, beta, reduction):
        super(Smooth_L1_Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = Variable(targets, requires_grad=False)
        flag = torch.abs(inputs - targets)

        loss = torch.where(flag.float() < self.beta, 0.5/self.beta*flag.float()**2, flag.float()-0.5*self.beta)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)

        return loss


if __name__ == "__main__":
    loss_function = Smooth_L1_Loss(beta=0.11, reduction="mean")
    inputs = torch.rand(8, 16)
    targets = torch.rand(8, 16)
    loss = loss_function(inputs=inputs, targets=targets)
    print(loss)
