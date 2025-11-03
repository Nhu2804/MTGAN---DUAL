import torch
import numpy as np

# ============================================================
# 🧩 Single-stream sampler (Diagnosis only)
# ============================================================

class CodeSampleIter:
    def __init__(self, code, samples, shuffle=True):
        self.code = code
        self.samples = samples
        self.current_index = 0
        self.length = len(samples)
        if shuffle:
            np.random.shuffle(self.samples)

    def __next__(self):
        sample = self.samples[self.current_index]
        self.current_index = (self.current_index + 1) % self.length
        return sample


class DataSampler:
    """Sampler cho diagnosis-only (MTGAN gốc)."""
    def __init__(self, ehr_data, lens, device=None):
        self.ehr_data = ehr_data
        self.lens = lens
        self.device = device
        self.size = len(ehr_data)
        self.code_samples = self._get_code_sample_map()

    def _get_code_sample_map(self):
        print('building ehr data sampler ...')
        code_sample_map = {}
        for i, (sample, len_i) in enumerate(zip(self.ehr_data, self.lens)):
            for t in range(len_i):
                codes = np.where(sample[t] > 0)[0]
                for code in codes:
                    code_sample_map.setdefault(code, set()).add(i)
        return {code: CodeSampleIter(code, list(samples))
                for code, samples in code_sample_map.items()}

    def sample(self, target_codes):
        valid_codes = [c for c in target_codes if c in self.code_samples]
        if not valid_codes:
            # fallback chọn ngẫu nhiên bệnh nhân
            rand_idx = np.random.randint(self.size)
            data = torch.from_numpy(self.ehr_data[[rand_idx]]).to(self.device, dtype=torch.float)
            lens = torch.from_numpy(self.lens[[rand_idx]]).to(self.device, torch.long)
            return data, lens
        lines = np.array([next(self.code_samples[c]) for c in valid_codes])
        data = torch.from_numpy(self.ehr_data[lines]).to(self.device, dtype=torch.float)
        lens = torch.from_numpy(self.lens[lines]).to(self.device, torch.long)
        return data, lens

    def __len__(self):
        return self.size


def get_train_sampler(train_loader, device):
    """Single-stream sampler."""
    return DataSampler(*train_loader.dataset.data, device)


# ============================================================
# 🩺 Dual-stream sampler (Diagnosis + Procedure)
# ============================================================

class DataSamplerDual:
    """
    Build a sampler for both diagnosis and procedure streams.
    Allows sampling based on either diagnosis or procedure codes.
    """
    def __init__(self, ehr_data_diag, ehr_data_proc, lens, device=None):
        self.ehr_data_diag = ehr_data_diag
        self.ehr_data_proc = ehr_data_proc
        self.lens = lens
        self.device = device
        self.size = len(ehr_data_diag)

        print('building dual-stream EHR data sampler ...')
        self.code_samples_diag = self._get_code_sample_map(ehr_data_diag, lens, name="diagnosis")
        self.code_samples_proc = self._get_code_sample_map(ehr_data_proc, lens, name="procedure")

    def _get_code_sample_map(self, ehr_data, lens, name=""):
        print(f'\tbuilding {name} code→patient map ...')
        code_sample_map = {}
        for i, (sample, len_i) in enumerate(zip(ehr_data, lens)):
            for t in range(len_i):
                codes = np.where(sample[t] > 0)[0]
                for code in codes:
                    code_sample_map.setdefault(code, set()).add(i)
        return {code: CodeSampleIter(code, list(samples))
                for code, samples in code_sample_map.items()}

    def sample(self, target_codes, mode="diag"):
        """
        Sample patients who have given codes.
        mode = 'diag' | 'proc'
        """
        assert mode in ["diag", "proc"], "mode must be 'diag' or 'proc'"
        print(f"[DataSamplerDual] Sampling mode = {mode}")

        # chọn map tương ứng
        code_samples = self.code_samples_diag if mode == "diag" else self.code_samples_proc

        valid_codes = [c for c in target_codes if c in code_samples]

        # 🩺 Nếu không tìm được mã hợp lệ — fallback ngẫu nhiên (KHÔNG raise lỗi)
        if not valid_codes:
            print(f"[Warning] No valid {mode} codes found → using random patient fallback.")
            rand_idx = np.random.randint(self.size)
            data_diag = torch.from_numpy(self.ehr_data_diag[[rand_idx]]).to(self.device, dtype=torch.float)
            data_proc = torch.from_numpy(self.ehr_data_proc[[rand_idx]]).to(self.device, dtype=torch.float)
            lens = torch.from_numpy(self.lens[[rand_idx]]).to(self.device, torch.long)
            return data_diag, data_proc, lens

        # nếu có mã hợp lệ thì sampling như thường
        lines = np.array([next(code_samples[c]) for c in valid_codes])
        data_diag = torch.from_numpy(self.ehr_data_diag[lines]).to(self.device, dtype=torch.float)
        data_proc = torch.from_numpy(self.ehr_data_proc[lines]).to(self.device, dtype=torch.float)
        lens = torch.from_numpy(self.lens[lines]).to(self.device, torch.long)
        return data_diag, data_proc, lens


    def __len__(self):
        return self.size


def get_train_sampler_dual(train_loader, device):
    print("🔄 Building dual-stream train sampler ...")
    return DataSamplerDual(*train_loader.dataset.data, device=device)
