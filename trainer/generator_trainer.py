import torch


# ============================================================
# ⚙️ GeneratorTrainer — gốc (Diagnosis only)
# ============================================================
class GeneratorTrainer:
    def __init__(self, generator, critic, batch_size, train_num, lr, betas, decay_step, decay_rate):
        self.generator = generator
        self.critic = critic
        self.batch_size = batch_size
        self.train_num = train_num

        self.code_num = self.generator.code_num
        self.optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_step, gamma=decay_rate)
        self.device = self.generator.device

    def _step(self, target_codes, lens):
        noise = self.generator.get_noise(len(lens))
        samples, hiddens = self.generator(target_codes, lens, noise)
        output = self.critic(samples, hiddens, lens)
        loss = -output.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def step(self, target_codes, lens):
        self.generator.train()
        self.critic.eval()

        loss = 0
        for _ in range(self.train_num):
            loss += self._step(target_codes, lens).item()
        loss /= self.train_num
        self.scheduler.step()

        return loss


# ============================================================
# 🩺 GeneratorTrainerDual — mở rộng cho Diagnosis + Procedure
# ============================================================
class GeneratorTrainerDual:
    """
    Dual-stream Generator training for WGAN-GP:
      generates both diagnosis and procedure sequences.
    """
    def __init__(self, generator, critic, batch_size, train_num, lr, betas, decay_step, decay_rate):
        self.generator = generator
        self.critic = critic
        self.batch_size = batch_size
        self.train_num = train_num

        self.optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_step, gamma=decay_rate)
        self.device = self.generator.device

    def _step(self, target_diag, target_proc, lens):
        noise = self.generator.get_noise(len(lens))
        fake_diag, fake_proc, h_diag, h_proc = self.generator(target_diag, target_proc, lens, noise)
        critic_out = self.critic(fake_diag, fake_proc, h_diag, h_proc, lens)
        loss = -critic_out.mean()  # maximize critic output → minimize -E[D(fake)]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def step(self, target_diag, target_proc, lens):
        self.generator.train()
        self.critic.eval()

        total_loss = 0
        for _ in range(self.train_num):
            total_loss += self._step(target_diag, target_proc, lens).item()
        total_loss /= self.train_num
        self.scheduler.step()
        return total_loss
