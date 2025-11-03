import os
import torch
import numpy as np


# ============================================================
# 🧩 Base Dataset wrapper
# ============================================================

class Dataset:
    """Base wrapper that converts numpy → torch tensors."""
    def __init__(self, inputs, device):
        self.data = inputs
        self.device = device
        self.size = len(inputs[0])

    def __len__(self):
        return self.size

    def __getitem__(self, indices):
        data = [torch.tensor(x[indices], device=self.device) for x in self.data]
        return data


# ============================================================
# 📘 Diagnosis-only dataset (giữ nguyên)
# ============================================================

class DatasetReal:
    def __init__(self, path, device=None):
        self.path = path
        self.device = device
        print('loading real data ...')
        print('\tloading real training data ...')
        self.train_set = self._load('train.npz')
        print('\tloading real test data ...')
        self.test_set = self._load('test.npz')

    def _load(self, filename):
        data = np.load(os.path.join(self.path, filename))
        x = data['x'].astype(np.float32)
        lens = data['lens'].astype(np.int64)
        dataset = Dataset((x, lens), self.device)
        return dataset


class DatasetRealNext:
    def __init__(self, path, device=None):
        self.path = path
        self.device = device
        print('loading real next data ...')
        print('\tloading real next training data ...')
        self.train_set = self._load('train.npz')

    def _load(self, filename):
        data = np.load(os.path.join(self.path, filename))
        x = data['x'].astype(np.float32)
        lens = data['lens'].astype(np.int64)
        y = data['y'].astype(np.float32)
        dataset = Dataset((x, lens, y), self.device)
        return dataset


# ============================================================
# 🩺 Dual-Stream version (Diagnosis + Procedure)
# ============================================================

class DatasetDual(Dataset):
    """Dataset wrapper for (x_diag, x_proc, lens)."""
    def __getitem__(self, indices):
        x_diag = torch.tensor(self.data[0][indices], device=self.device)
        x_proc = torch.tensor(self.data[1][indices], device=self.device)
        lens = torch.tensor(self.data[2][indices], device=self.device)
        return x_diag, x_proc, lens


class DatasetRealDual:
    """Load real dual-stream (Diagnosis + Procedure) data."""
    def __init__(self, path, device=None):
        self.path = path
        self.device = device
        print('loading real dual-stream data ...')
        print('\tloading real training data ...')
        self.train_set = self._load('train_dual.npz')
        print('\tloading real test data ...')
        self.test_set = self._load('test_dual.npz')

    def _load(self, filename):
        file_path = os.path.join(self.path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found. Did you run preprocess with --use_proc?")
        data = np.load(file_path)
        x_diag = data['x_diag'].astype(np.float32)
        x_proc = data['x_proc'].astype(np.float32)
        lens = data['lens'].astype(np.int64)
        dataset = DatasetDual((x_diag, x_proc, lens), self.device)
        return dataset


# ============================================================
# ⚠️ (Tuỳ chọn) — Chưa dùng, chỉ bật khi có train_dual_next.npz
# ============================================================

class DatasetRealNextDual:
    """
    Dataset for dual-stream next-visit prediction.
    Expected file: train_dual.npz (contains 5 arrays)
        x_diag, x_proc, y_diag, y_proc, lens
    """
    def __init__(self, path, device=None):
        self.path = path
        self.device = device
        print('loading real next dual data ...')
        print('\tloading real next training data ...')
        self.train_set = self._load('train_dual.npz')

    def _load(self, filename):
        file_path = os.path.join(self.path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found. Did you run preprocess with --use_proc?")
        data = np.load(file_path)

        x_diag = torch.tensor(data['x_diag'], dtype=torch.float32, device=self.device)
        x_proc = torch.tensor(data['x_proc'], dtype=torch.float32, device=self.device)
        y_diag = torch.tensor(data['y_diag'], dtype=torch.float32, device=self.device)
        y_proc = torch.tensor(data['y_proc'], dtype=torch.float32, device=self.device)
        lens = torch.tensor(data['lens'], dtype=torch.long, device=self.device)

        dataset = self.DatasetDualNext(x_diag, x_proc, y_diag, y_proc, lens)
        return dataset

    class DatasetDualNext:
        def __init__(self, x_diag, x_proc, y_diag, y_proc, lens):
            self.data = (x_diag, x_proc, y_diag, y_proc, lens)

        def __len__(self):
            return len(self.data[0])

        def __getitem__(self, idx):
            x_diag, x_proc, y_diag, y_proc, lens = self.data
            return (
                x_diag[idx],
                x_proc[idx],
                y_diag[idx],
                y_proc[idx],
                lens[idx],
            )
