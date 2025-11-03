import torch
from model import WGANGPLoss
from model.critic.loss import WGANGPLossDual


# ============================================================
# ⚙️ CriticTrainer — gốc (Diagnosis only)
# ============================================================
class CriticTrainer:
    def __init__(self, critic, generator, base_gru, batch_size, train_num, lr, lambda_, betas, decay_step, decay_rate):
        self.critic = critic
        self.generator = generator
        self.base_gru = base_gru
        self.batch_size = batch_size
        self.train_num = train_num

        self.optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_step, gamma=decay_rate)
        self.loss_fn = WGANGPLoss(critic, lambda_=lambda_)

    def _step(self, real_data, real_lens, target_codes):
        real_hiddens = self.base_gru.calculate_hidden(real_data, real_lens)
        fake_data, fake_hiddens = self.generator.sample(target_codes, real_lens, return_hiddens=True)
        loss, wasserstein_distance = self.loss_fn(real_data, real_hiddens, fake_data, fake_hiddens, real_lens)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), wasserstein_distance.item()

    def step(self, real_data, real_lens, target_codes):
        self.critic.train()
        self.generator.eval()

        loss, w_distance = 0, 0
        for _ in range(self.train_num):
            loss_i, w_distance_i = self._step(real_data, real_lens, target_codes)
            loss += loss_i
            w_distance += w_distance_i
        loss /= self.train_num
        w_distance /= self.train_num
        self.scheduler.step()
        return loss, w_distance

    def evaluate(self, data_loader, device):
        self.critic.train()
        with torch.no_grad():
            loss = 0
            for data in data_loader:
                data, lens = data
                data, lens = data.to(device), lens.to(device)
                hiddens = self.base_gru.calculate_hidden(data, lens)
                loss += self.critic(data, hiddens, lens).mean().item()
            loss = -loss / len(data_loader)
            return loss


# ============================================================
# 🩺 CriticTrainerDual — mở rộng cho Diagnosis + Procedure
# ============================================================
class CriticTrainerDual:
    def __init__(self, critic, generator, base_gru_dual, batch_size, train_num,
                 lr, lambda_, betas, decay_step, decay_rate):
        self.critic = critic
        self.generator = generator
        self.base_gru_dual = base_gru_dual
        self.batch_size = batch_size
        self.train_num = train_num

        self.optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_step, gamma=decay_rate)
        self.loss_fn = WGANGPLossDual(critic, lambda_=lambda_)

    def _step(self, real_diag, real_proc, real_lens, target_diag, target_proc):
        # Hidden từ BaseGRUDual
        h_diag_real, h_proc_real = self.base_gru_dual.calculate_hidden(real_diag, real_proc, real_lens)

        # Sinh mẫu giả
        fake_diag, fake_proc, h_diag_fake, h_proc_fake = self.generator.sample(
            target_diag, target_proc, real_lens, return_hiddens=True
        )

        # Tính loss WGAN-GP
        loss, wasserstein_distance = self.loss_fn(
            real_diag, real_proc, h_diag_real, h_proc_real,
            fake_diag, fake_proc, h_diag_fake, h_proc_fake, real_lens
        )

        # Cập nhật Critic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), wasserstein_distance.item()

    def step(self, real_diag, real_proc, real_lens, target_diag, target_proc):
        self.critic.train()
        self.generator.eval()

        total_loss, total_wdist = 0.0, 0.0
        for _ in range(self.train_num):
            l, w = self._step(real_diag, real_proc, real_lens, target_diag, target_proc)
            total_loss += l
            total_wdist += w

        self.scheduler.step()
        return total_loss / self.train_num, total_wdist / self.train_num

    def evaluate(self, data_loader, device):
        self.critic.eval()
        with torch.no_grad():
            total_score = 0
            for x_diag, x_proc, lens in data_loader:
                x_diag, x_proc, lens = x_diag.to(device), x_proc.to(device), lens.to(device)
                h_diag, h_proc = self.base_gru_dual.calculate_hidden(x_diag, x_proc, lens)
                score = self.critic(x_diag, x_proc, h_diag, h_proc, lens).mean().item()
                total_score += score
            return -total_score / len(data_loader)
