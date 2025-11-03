import torch
import numpy as np


# ============================================================
# 🧠 Generate EHR (support both single & dual mode)
# ============================================================
def generate_ehr(generator, number, len_dist, batch_size):
    """
    Generate synthetic EHR sequences.
    Works for both:
      - Generator  (single-stream, diagnosis-only)
      - GeneratorDual (dual-stream: diagnosis + procedure)
    """
    fake_x, fake_lens = [], []

    # Dual mode → lưu tách diag/proc
    fake_x_diag, fake_x_proc = [], []

    for i in range(0, number, batch_size):
        n = number - i if i + batch_size > number else batch_size
        lens = torch.multinomial(len_dist, num_samples=n, replacement=True) + 1

        # ✅ Nếu là Dual Generator
        if hasattr(generator, "code_num_proc"):
            target_diag, target_proc = generator.get_target_codes(n)
            x_diag, x_proc = generator.sample(target_diag, target_proc, lens)

            fake_x_diag.append(x_diag.cpu().numpy())
            fake_x_proc.append(x_proc.cpu().numpy())
            fake_lens.append(lens.cpu().numpy())

        # ✅ Nếu là Single Generator (MTGAN gốc)
        else:
            target_codes = generator.get_target_codes(n)
            x = generator.sample(target_codes, lens)
            fake_x.append(x.cpu().numpy())
            fake_lens.append(lens.cpu().numpy())

    fake_lens = np.concatenate(fake_lens, axis=-1)

    # Trả kết quả đúng định dạng
    if hasattr(generator, "code_num_proc"):
        fake_x_diag = np.concatenate(fake_x_diag, axis=0)
        fake_x_proc = np.concatenate(fake_x_proc, axis=0)
        return (fake_x_diag, fake_x_proc), fake_lens
    else:
        fake_x = np.concatenate(fake_x, axis=0)
        return fake_x, fake_lens


# ============================================================
# 🔍 Utility: check required number of samples to cover all codes
# ============================================================
def get_required_number(generator, len_dist, batch_size, upper_bound=1e7):
    """
    Estimate how many samples are needed to cover all possible codes.
    Supports both single and dual mode.
    """
    if hasattr(generator, "code_num_proc"):
        code_types_diag = torch.zeros(generator.code_num_diag, dtype=torch.bool, device=generator.device)
        code_types_proc = torch.zeros(generator.code_num_proc, dtype=torch.bool, device=generator.device)
    else:
        code_types = torch.zeros(generator.code_num, dtype=torch.bool, device=generator.device)

    rn = 0
    while True:
        n = np.random.randint(low=np.floor(0.5 * batch_size), high=np.floor(1.5 * batch_size))
        rn += n
        lens = torch.multinomial(len_dist, num_samples=n, replacement=True) + 1

        # ✅ Dual
        if hasattr(generator, "code_num_proc"):
            target_diag, target_proc = generator.get_target_codes(n)
            x_diag, x_proc = generator.sample(target_diag, target_proc, lens)

            code_types_diag = torch.logical_or(code_types_diag, x_diag.sum(dim=1).sum(dim=0) > 0)
            code_types_proc = torch.logical_or(code_types_proc, x_proc.sum(dim=1).sum(dim=0) > 0)

            total_diag = code_types_diag.sum().item()
            total_proc = code_types_proc.sum().item()
            print(f"[{rn}] diag covered: {total_diag}/{generator.code_num_diag}, proc covered: {total_proc}/{generator.code_num_proc}")

            if total_diag == generator.code_num_diag and total_proc == generator.code_num_proc:
                print(f"✅ required number to generate all codes (dual): {rn}")
                return

        # ✅ Single
        else:
            target_codes = generator.get_target_codes(n)
            x = generator.sample(target_codes, lens)
            code_types = torch.logical_or(code_types, x.sum(dim=1).sum(dim=0) > 0)
            total_code_types = code_types.sum()
            print(total_code_types.item(), rn)

            if total_code_types == generator.code_num:
                print('✅ required number to generate all diseases:', rn)
                return

        # Stop nếu quá upper bound
        if rn >= upper_bound:
            print(f"⚠️ Unable to generate all codes within {upper_bound} samples.")
            return
