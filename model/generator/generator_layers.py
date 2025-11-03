import torch
from torch import nn
from model.utils import MaskedAttention


# ============================================================
# 🧩 GRU cơ bản (Diagnosis-only)
# ============================================================
class GRU(nn.Module):
    def __init__(self, code_num, hidden_dim, max_len, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.device = device

        self.gru_cell = nn.GRUCell(input_size=code_num, hidden_size=hidden_dim)
        self.hidden2codes = nn.Sequential(
            nn.Linear(hidden_dim, code_num),
            nn.Sigmoid()
        )

    def step(self, x, h=None):
        h_n = self.gru_cell(x, h)
        codes = self.hidden2codes(h_n)
        return codes, h_n

    def forward(self, noise):
        codes = self.hidden2codes(noise)
        h = torch.zeros(len(codes), self.hidden_dim, device=self.device)
        samples, hiddens = [], []
        for _ in range(self.max_len):
            samples.append(codes)
            codes, h = self.step(codes, h)
            hiddens.append(h)
        samples = torch.stack(samples, dim=1)
        hiddens = torch.stack(hiddens, dim=1)
        return samples, hiddens


# ============================================================
# 🩺 Dual-Stream GRU (Diagnosis + Procedure)
# ============================================================
class DualGRUGenerator(nn.Module):
    """
    Generate diagnosis & procedure sequences simultaneously.
    Two independent GRU decoders (shared latent).
    """
    def __init__(self, code_num_diag, code_num_proc, hidden_dim, max_len, device=None):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Two GRU decoders
        self.gru_diag = GRU(code_num_diag, hidden_dim, max_len, device)
        self.gru_proc = GRU(code_num_proc, hidden_dim, max_len, device)

        # Optionally project same latent into two spaces
        self.fc_diag = nn.Linear(hidden_dim, hidden_dim)
        self.fc_proc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z):
        """
        z: latent noise [batch, hidden_dim]
        Return: (samples_diag, samples_proc)
        """
        z_diag = self.fc_diag(z)
        z_proc = self.fc_proc(z)

        samples_diag, h_diag = self.gru_diag(z_diag)
        samples_proc, h_proc = self.gru_proc(z_proc)
        return samples_diag, samples_proc, h_diag, h_proc


# ============================================================
# 🎯 Optional: smooth conditioning (fixed)
# ============================================================
class SmoothCondition(nn.Module):
    def __init__(self, code_num, attention_dim):
        super().__init__()
        self.attention = MaskedAttention(code_num, attention_dim)

    def forward(self, x, lens, target_codes):
        """
        x: [B, T, C]
        lens: [B]
        target_codes: 
          - [B] (chỉ số code),
          - hoặc [B, C] (one-hot / logits),
          - hoặc list[tensor], mỗi phần tử có thể là scalar hoặc vector.
        """
        device = x.device
        B, T, C = x.shape

        # 1) attention score [B, T]
        score = self.attention(x, lens)

        # 2) Chuẩn hoá target_codes -> tensor chỉ số [B] trên đúng device
        if isinstance(target_codes, torch.Tensor):
            # Nếu là phân phối/one-hot: chiều cuối == C -> argmax theo chiều cuối
            if target_codes.ndim >= 1 and target_codes.shape[-1] == C:
                target_codes = target_codes.argmax(dim=-1)
            # Còn lại: ép phẳng về (B,)
            if target_codes.ndim > 1:
                target_codes = target_codes.view(-1)
            target_codes = target_codes.to(device=device, dtype=torch.long)

        elif isinstance(target_codes, (list, tuple)):
            flat_idxs = []
            for t in target_codes:
                if isinstance(t, torch.Tensor):
                    tt = t.detach()
                    # nếu là phân phối/one-hot theo chiều cuối
                    if tt.ndim >= 1 and tt.shape[-1] == C:
                        idx = int(tt.argmax(dim=-1).view(-1)[0].item())
                    else:
                        # không chắc: lấy argmax của vector phẳng
                        if tt.numel() == 1:
                            idx = int(tt.item())
                        else:
                            idx = int(tt.view(-1).argmax().item())
                else:
                    idx = int(t)
                flat_idxs.append(idx)
            target_codes = torch.tensor(flat_idxs, dtype=torch.long, device=device)

        else:
            # scalar duy nhất
            target_codes = torch.tensor([int(target_codes)], dtype=torch.long, device=device)

        # đảm bảo đúng kích thước B (nếu cần, lặp lại)
        if target_codes.numel() != B:
            if target_codes.numel() == 1:
                target_codes = target_codes.expand(B)
            else:
                # fallback: cắt hoặc tile cho khớp B
                if target_codes.numel() > B:
                    target_codes = target_codes.view(-1)[:B]
                else:
                    reps = (B + target_codes.numel() - 1) // target_codes.numel()
                    target_codes = target_codes.repeat(reps)[:B]

        target_codes = target_codes.clamp_(0, C - 1)

        # 3) scatter score vào đúng cột code mục tiêu
        score_tensor = torch.zeros(B, T, C, device=device, dtype=x.dtype)
        idx = target_codes.view(B, 1, 1).expand(B, T, 1)      # [B,T,1]
        score_tensor.scatter_(2, idx, score.unsqueeze(-1))    # [B,T,1]

        # 4) cộng và clip
        x = torch.clamp(x + score_tensor, max=1)
        return x
