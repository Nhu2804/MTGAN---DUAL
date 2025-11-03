import torch
from torch import nn
from model.base_model import BaseModel
from model.utils import sequence_mask


# ============================================================
# 🧠 BaseGRU (gốc) — giữ nguyên
# ============================================================
class BaseGRU(BaseModel):
    def __init__(self, code_num, hidden_dim, max_len):
        super().__init__(param_file_name='base_gru.pt')
        self.gru = nn.GRU(input_size=code_num, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, code_num),
            nn.Sigmoid()
        )
        self.max_len = max_len

    def forward(self, x):
        outputs, _ = self.gru(x)
        output = self.linear(outputs)
        return output

    def calculate_hidden(self, x, lens):
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)
            outputs, _ = self.gru(x)
            output = outputs * mask
            return output


# ============================================================
# 🩺 BaseGRUDual — mở rộng cho Diagnosis + Procedure
# ============================================================
class BaseGRUDual(BaseModel):
    """
    Dual GRU pretraining for both diagnosis and procedure streams.
    """
    def __init__(self, code_num_diag, code_num_proc, hidden_dim, max_len):
        super().__init__(param_file_name='base_gru_dual.pt')

        # Hai GRU song song cho Diagnosis và Procedure
        self.gru_diag = nn.GRU(input_size=code_num_diag, hidden_size=hidden_dim, batch_first=True)
        self.gru_proc = nn.GRU(input_size=code_num_proc, hidden_size=hidden_dim, batch_first=True)

        # Dự đoán riêng từng stream
        self.linear_diag = nn.Sequential(
            nn.Linear(hidden_dim, code_num_diag),
            nn.Sigmoid()
        )
        self.linear_proc = nn.Sequential(
            nn.Linear(hidden_dim, code_num_proc),
            nn.Sigmoid()
        )

        self.max_len = max_len

    def forward(self, x_diag, x_proc):
        """
        Dự đoán mã bệnh và mã thủ thuật cho lượt khám kế tiếp.
        """
        out_diag, _ = self.gru_diag(x_diag)
        out_proc, _ = self.gru_proc(x_proc)

        y_pred_diag = self.linear_diag(out_diag)
        y_pred_proc = self.linear_proc(out_proc)
        return y_pred_diag, y_pred_proc

    def calculate_hidden(self, x_diag, x_proc, lens):
        """
        Tính hidden representation cho cả 2 luồng.
        """
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)
            h_diag, _ = self.gru_diag(x_diag)
            h_proc, _ = self.gru_proc(x_proc)

            h_diag = h_diag * mask
            h_proc = h_proc * mask
            return h_diag, h_proc
