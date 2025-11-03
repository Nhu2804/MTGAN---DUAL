import torch
from torch import nn
from model.base_model import BaseModel
from model.utils import sequence_mask


class Critic(BaseModel):
    def __init__(self, code_num, hidden_dim, generator_hidden_dim, max_len):
        super().__init__(param_file_name='critic.pt')

        self.code_num = code_num
        self.hidden_dim = hidden_dim
        self.generator_hidden_dim = generator_hidden_dim
        self.max_len = max_len

        self.linear = nn.Sequential(
            nn.Linear(code_num + generator_hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, hiddens, lens):
        output = torch.cat([x, hiddens], dim=-1)
        output = self.linear(output).squeeze(dim=-1)
        mask = sequence_mask(lens, self.max_len)
        output = output * mask
        output = output.sum(dim=-1)
        output = output / lens
        return output


# ============================================================
# 🩺 Dual-stream Critic (Diagnosis + Procedure)
# ============================================================

class CriticDual(BaseModel):
    """
    Evaluate dual-stream (diagnosis + procedure) samples jointly.
    """
    def __init__(self, code_num_diag, code_num_proc, hidden_dim, generator_hidden_dim, max_len):
        super().__init__(param_file_name='critic_dual.pt')

        self.code_num_diag = code_num_diag
        self.code_num_proc = code_num_proc
        self.hidden_dim = hidden_dim
        self.generator_hidden_dim = generator_hidden_dim
        self.max_len = max_len

        # Total input size = diag + proc + hidden_diag + hidden_proc
        total_input_dim = code_num_diag + code_num_proc + 2 * generator_hidden_dim

        self.linear = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_diag, x_proc, h_diag, h_proc, lens):
        """
        Input:
            x_diag: [B, L, D1]
            x_proc: [B, L, D2]
            h_diag: [B, L, H1]
            h_proc: [B, L, H2]
            lens:   [B]
        """
        # Concatenate all features
        output = torch.cat([x_diag, x_proc, h_diag, h_proc], dim=-1)
        output = self.linear(output).squeeze(dim=-1)

        # Masking padded timesteps
        mask = sequence_mask(lens, self.max_len)
        output = output * mask
        output = output.sum(dim=-1)
        output = output / lens
        return output
