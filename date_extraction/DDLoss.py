import torch
from torch.nn import L1Loss, CrossEntropyLoss
from torch.nn.modules.loss import _Loss


class DDLoss(_Loss):
    def __init__(self, device=torch.cpu):
        self.device = device
        super(DDLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = CrossEntropyLoss().to(self.device)
        input_day = torch.split(input, [31, 12], dim=1)[0]
        input_month = torch.split(input, [31, 12], dim=1)[1]
        target_day = torch.split(target, [31, 12], dim=1)[0]
        target_month = torch.split(target, [31, 12], dim=1)[1]
        loss_day = base_loss(input_day, target_day)
        loss_month = base_loss(input_month, target_month)
        weight_day = 12
        weight_month = 31
        loss = (weight_day * loss_day) + (weight_month * loss_month)
        return loss
