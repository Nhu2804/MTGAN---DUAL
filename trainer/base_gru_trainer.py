import torch
from model import PredictNextLoss
from model.base_gru.loss import PredictNextLossDual


# ============================================================
# 🧠 BaseGRUTrainer — dùng cho Diagnosis (cũ)
# ============================================================
class BaseGRUTrainer:
    def __init__(self, args, base_gru, max_len, train_loader, params_path):
        self.base_gru = base_gru
        self.train_loader = train_loader
        self.params_path = params_path

        self.epochs = args.base_gru_epochs
        self.optimizer = torch.optim.Adam(base_gru.parameters(), lr=args.base_gru_lr)
        self.loss_fn = PredictNextLoss(max_len)

    def train(self):
        print('🧠 Pre-training BaseGRU (Diagnosis only)...')
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}/{self.epochs}')
            total_loss = 0.0
            total_num = 0
            steps = len(self.train_loader)

            for step, data in enumerate(self.train_loader, start=1):
                x, lens, y = data
                output = self.base_gru(x)
                loss = self.loss_fn(output, y, lens)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(x)
                total_num += len(x)
                print(f'\r    Step {step}/{steps}, loss: {total_loss / total_num:.4f}', end='')

            print(f'\r    Step {steps}/{steps}, loss: {total_loss / total_num:.4f}')
        self.base_gru.save(self.params_path)


# ============================================================
# 🩺 BaseGRUTrainerDual — mở rộng cho Diagnosis + Procedure
# ============================================================
class BaseGRUTrainerDual:
    def __init__(self, args, base_gru_dual, max_len, train_loader, params_path):
        self.model = base_gru_dual
        self.train_loader = train_loader
        self.params_path = params_path

        self.epochs = args.base_gru_epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.base_gru_lr)
        self.loss_fn = PredictNextLossDual(max_len)

    def train(self):
        print('🩺 Pre-training BaseGRUDual (Diagnosis + Procedure)...')
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}/{self.epochs}')
            total_loss, total_diag, total_proc, total_num = 0.0, 0.0, 0.0, 0
            steps = len(self.train_loader)

            for step, data in enumerate(self.train_loader, start=1):
                # unpack tuple from DatasetRealNextDual
                x_diag, x_proc, y_diag, y_proc, lens = data

                # forward pass
                y_pred_diag, y_pred_proc = self.model(x_diag, x_proc)
                loss, loss_d, loss_p = self.loss_fn(y_pred_diag, y_diag, y_pred_proc, y_proc, lens)

                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # metrics
                total_loss += loss.item() * len(x_diag)
                total_diag += loss_d * len(x_diag)
                total_proc += loss_p * len(x_diag)
                total_num += len(x_diag)

                avg_total = total_loss / total_num
                avg_diag = total_diag / total_num
                avg_proc = total_proc / total_num
                print(f'\r    Step {step}/{steps} | total: {avg_total:.4f} | diag: {avg_diag:.4f} | proc: {avg_proc:.4f}', end='')

            print(f'\r    Step {steps}/{steps} | total: {avg_total:.4f} | diag: {avg_diag:.4f} | proc: {avg_proc:.4f}')
        print('✅ Finished pretraining BaseGRUDual.')
        self.model.save(self.params_path)
