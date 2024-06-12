import torch
import torch.nn as nn


class MarginLoss(torch.nn.Module):
    m_pos: float
    m_neg: float
    lambda_: float

    def __init__(self, m_pos: float, m_neg: float, lambda_: float) -> None:
        super().__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(
        self, lengths: torch.Tensor, targets: torch.Tensor, size_average: bool = True
    ) -> torch.Tensor:
        t = torch.zeros_like(lengths, dtype=torch.int64, device=targets.device)

        targets = t.scatter_(1, targets.unsqueeze(-1), 1).type(
            torch.get_default_dtype()
        )

        losses = targets * torch.nn.functional.relu(self.m_pos - lengths) ** 2

        losses = (
            losses
            + self.lambda_
            * (1.0 - targets)
            * torch.nn.functional.relu(lengths - self.m_neg) ** 2
        )

        return losses.mean() if size_average else losses.sum()


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class MSECrossEntropyLoss(nn.Module):
    """
    David's MSE + CrossEntropy Loss
    """

    def __init__(
        self,
        mse_coeff: float = 0.5,
        kldiv_coeff: float = 1.0,
    ) -> torch.Tensor | None:
        super(MSECrossEntropyLoss, self).__init__()
        self.mse_coeff = mse_coeff
        self.kldiv_coeff = kldiv_coeff

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor | None:
        assert pred.ndim == 2, "pred should be 2D"

        # Convert label into one hot
        target_one_hot: torch.Tensor = torch.zeros(
            (
                target.shape[0],
                pred.shape[1],
            ),
            device=pred.device,
            dtype=pred.dtype,
        )
        target_one_hot.scatter_(
            1,
            target.to(pred.device).unsqueeze(1),
            torch.ones(
                (target.shape[0], 1),
                device=pred.device,
                dtype=pred.dtype,
            ),
        )

        loss: torch.Tensor = ((pred - target_one_hot) ** 2).mean(
            dim=[0, 1]
        ) * self.mse_coeff

        loss = (
            loss
            + (
                target_one_hot * torch.log((target_one_hot + 1e-20) / (pred + 1e-20))
            ).mean(dim=[0, 1])
            * self.kldiv_coeff
        )

        loss = loss / (abs(self.kldiv_coeff) + abs(self.mse_coeff))

        return loss
