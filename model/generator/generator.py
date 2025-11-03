import torch
from torch import nn
from model.base_model import BaseModel
from model.utils import sequence_mask
from .generator_layers import GRU, SmoothCondition, DualGRUGenerator


# ============================================================
# 🧠 Single-stream Generator (Diagnosis only)
# ============================================================

class Generator(BaseModel):
    def __init__(self, code_num, hidden_dim, attention_dim, max_len, device=None):
        super().__init__(param_file_name='generator.pt')
        self.code_num = code_num
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.max_len = max_len
        self.device = device

        self.noise_dim = hidden_dim
        self.gru = GRU(code_num, hidden_dim, max_len, device)
        self.smooth_condition = SmoothCondition(code_num, attention_dim)

    def forward(self, target_codes, lens, noise):
        samples, hiddens = self.gru(noise)
        samples = self.smooth_condition(samples, lens, target_codes)
        return samples, hiddens

    def sample(self, target_codes, lens, noise=None, return_hiddens=False):
        if noise is None:
            noise = self.get_noise(len(lens))
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)
            prob, hiddens = self.forward(target_codes, lens, noise)
            samples = torch.bernoulli(prob).to(prob.dtype) * mask
            if return_hiddens:
                hiddens *= mask
                return samples, hiddens
            else:
                return samples

    def get_noise(self, batch_size):
        return torch.randn(batch_size, self.hidden_dim, device=self.device)

    def get_target_codes(self, batch_size):
        return torch.randint(0, self.code_num, (batch_size,))


# ============================================================
# 🩺 Dual-stream Generator (Diagnosis + Procedure)
# ============================================================

class GeneratorDual(BaseModel):
    """
    Dual-stream generator that jointly generates diagnosis and procedure codes.
    Each stream has its own GRU and smooth conditioning layer.
    """
    def __init__(self, code_num_diag, code_num_proc, hidden_dim, attention_dim, max_len, device=None):
        super().__init__(param_file_name='generator_dual.pt')
        self.code_num_diag = code_num_diag
        self.code_num_proc = code_num_proc
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.max_len = max_len
        self.device = device

        self.noise_dim = hidden_dim
        self.gru_dual = DualGRUGenerator(code_num_diag, code_num_proc, hidden_dim, max_len, device)

        # smooth-conditioning cho từng stream
        self.smooth_diag = SmoothCondition(code_num_diag, attention_dim)
        self.smooth_proc = SmoothCondition(code_num_proc, attention_dim)

    def forward(self, target_codes_diag, target_codes_proc, lens, noise):
        x_diag, x_proc, h_diag, h_proc = self.gru_dual(noise)
        x_diag = self.smooth_diag(x_diag, lens, target_codes_diag)
        x_proc = self.smooth_proc(x_proc, lens, target_codes_proc)
        return x_diag, x_proc, h_diag, h_proc

    def sample(self, target_codes_diag, target_codes_proc, lens, noise=None, return_hiddens=False):
        if noise is None:
            noise = self.get_noise(len(lens))
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)
            x_diag, x_proc, h_diag, h_proc = self.forward(target_codes_diag, target_codes_proc, lens, noise)
            samples_diag = torch.bernoulli(x_diag).to(x_diag.dtype) * mask
            samples_proc = torch.bernoulli(x_proc).to(x_proc.dtype) * mask

            if return_hiddens:
                h_diag *= mask
                h_proc *= mask
                return samples_diag, samples_proc, h_diag, h_proc
            else:
                return samples_diag, samples_proc

    def get_noise(self, batch_size):
        return torch.randn(batch_size, self.hidden_dim, device=self.device)

    def get_target_codes(self, batch_size):
        diag_codes = torch.randint(0, self.code_num_diag, (batch_size,))
        proc_codes = torch.randint(0, self.code_num_proc, (batch_size,))
        return diag_codes, proc_codes

    # ---------------------------------------------------------------------
    # 🆕 Thêm 2 hàm để trainer gọi riêng cho từng stream
    # ---------------------------------------------------------------------
    def get_target_codes_diag(self, batch_size):
        """Sinh mã chẩn đoán (diagnosis codes) ngẫu nhiên."""
        return torch.randint(
            0, self.code_num_diag, (batch_size,), device=self.device
        )

    def get_target_codes_proc(self, batch_size):
        """Sinh mã thủ thuật (procedure codes) ngẫu nhiên."""
        return torch.randint(
            0, self.code_num_proc, (batch_size,), device=self.device
        )