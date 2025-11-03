from torch import nn
from model.utils import sequence_mask


class PredictNextLoss(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, input_, label, lens):
        mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)
        loss = self.loss_fn(input_, label)
        loss = loss * mask
        loss = loss.sum(dim=-1).sum(dim=-1).mean()
        return loss


# ============================================================
# 🩺 Dual-stream next-visit loss (Diagnosis + Procedure)
# ============================================================

class PredictNextLossDual(nn.Module):
    """
    BCE loss for both diagnosis and procedure prediction.
    """
    def __init__(self, max_len, weight_diag=1.0, weight_proc=1.0):
        super().__init__()
        self.max_len = max_len
        self.loss_fn = nn.BCELoss(reduction='none')
        self.weight_diag = weight_diag
        self.weight_proc = weight_proc

    def forward(self, input_diag, label_diag, input_proc, label_proc, lens):
        """
        Args:
            input_diag: [B, L, D1]
            label_diag: [B, L, D1]
            input_proc: [B, L, D2]
            label_proc: [B, L, D2]
            lens:       [B]
        Returns:
            total_loss (scalar)
        """
        mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)

        # Diagnosis loss
        loss_diag = self.loss_fn(input_diag, label_diag)
        loss_diag = (loss_diag * mask).sum(dim=-1).sum(dim=-1).mean()

        # Procedure loss
        loss_proc = self.loss_fn(input_proc, label_proc)
        loss_proc = (loss_proc * mask).sum(dim=-1).sum(dim=-1).mean()

        # Weighted total
        total_loss = self.weight_diag * loss_diag + self.weight_proc * loss_proc
        return total_loss, loss_diag.item(), loss_proc.item()
