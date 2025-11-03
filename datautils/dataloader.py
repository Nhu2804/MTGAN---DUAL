import os
import pickle
import numpy as np
import pandas as pd

from .dataset import (
    DatasetReal,
    DatasetRealNext,
    DatasetRealDual,
    # DatasetRealNextDual   # chưa cần dùng
)

# ============================================================
# 🔁 Basic utilities
# ============================================================

def infinite_dataloader(dataloader):
    """Loop dataloader infinitely (useful for GAN training)."""
    while True:
        for x in dataloader:
            yield x


class DataLoader:
    """Simple manual dataloader (no torch DataLoader used)."""
    def __init__(self, dataset, shuffle=True, batch_size=32):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.size = len(dataset)
        self.idx = np.arange(self.size)
        self.n_batches = np.ceil(self.size / batch_size).astype(int)

        self.counter = 0
        if shuffle:
            np.random.shuffle(self.idx)

    def _get_item(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        index = self.idx[start:end]
        data = self.dataset[index]
        return data

    def __next__(self):
        if self.counter >= self.n_batches:
            self.counter = 0
            raise StopIteration
        data = self._get_item(self.counter)
        self.counter += 1
        return data

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches


# ============================================================
# 📘 Diagnosis-only loaders (giữ nguyên)
# ============================================================

def get_train_test_loader(dataset_path, batch_size, device):
    dataset = DatasetReal(os.path.join(dataset_path, 'standard', 'real_data'), device=device)
    train_loader = DataLoader(dataset.train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset.test_set, shuffle=False, batch_size=batch_size)
    max_len = dataset.train_set.data[0].shape[1]

    print('total code num in train:', dataset.train_set.data[0].sum())
    print('total code num in test:', dataset.test_set.data[0].sum())
    return train_loader, test_loader, max_len


def get_base_gru_train_loader(dataset_path, batch_size, device):
    dataset = DatasetRealNext(os.path.join(dataset_path, 'standard', 'real_next'), device=device)
    train_loader = DataLoader(dataset.train_set, shuffle=True, batch_size=batch_size)
    return train_loader


# ============================================================
# 🩺 Dual-Stream loaders (Diagnosis + Procedure)
# ============================================================
def get_train_test_loader_dual(dataset_path, batch_size, device):
    """
    Loaders for dual-stream data:
      - x_diag, x_proc, lens
    Expected files:
      {dataset_path}/standard/real_data/train_dual.npz
      {dataset_path}/standard/real_data/test_dual.npz
    """
    dataset = DatasetRealDual(os.path.join(dataset_path, 'standard', 'real_data'), device=device)
    train_loader = DataLoader(dataset.train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset.test_set, shuffle=False, batch_size=batch_size)

    x_diag_train = dataset.train_set.data[0]
    x_proc_train = dataset.train_set.data[1]
    max_len = x_diag_train.shape[1]
    code_num_diag = x_diag_train.shape[-1]
    code_num_proc = x_proc_train.shape[-1]

    print(f"Train set: diag codes={x_diag_train.sum():.0f}, proc codes={x_proc_train.sum():.0f}")
    print(f"Max visit length = {max_len}, diag_dim={code_num_diag}, proc_dim={code_num_proc}")

    # ✅ chỉ trả về 3 biến để tương thích với run_train.py
    return train_loader, test_loader, max_len



# ============================================================
# 🧮 Meta data utilities
# ============================================================

def load_meta_data(dataset_path):
    """
    Load metadata for both single- and dual-stream settings.
    Returns everything available (diag + proc + cross).
    """
    standard_path = os.path.join(dataset_path, 'standard')
    real_data_stat = np.load(os.path.join(standard_path, 'real_data_stat.npz'))
    len_dist = real_data_stat['admission_dist']
    code_visit_dist = real_data_stat['code_visit_dist']
    code_patient_dist = real_data_stat['code_patient_dist']

    # Diagnosis-level
    code_adj_diag = np.load(os.path.join(standard_path, 'code_adj_diag.npz'))['code_adj'] \
        if os.path.exists(os.path.join(standard_path, 'code_adj_diag.npz')) \
        else np.load(os.path.join(standard_path, 'code_adj.npz'))['code_adj']

    code_map_diag = pickle.load(open(os.path.join(dataset_path, 'encoded', 'code_map.pkl'), 'rb'))

    # Procedure-level (optional)
    proc_adj_path = os.path.join(standard_path, 'code_adj_proc.npz')
    proc_map_path = os.path.join(dataset_path, 'encoded', 'proc_map.pkl')
    cross_adj_path = os.path.join(standard_path, 'code_adj_cross.npz')

    code_adj_proc = np.load(proc_adj_path)['code_adj'] if os.path.exists(proc_adj_path) else None
    code_map_proc = pickle.load(open(proc_map_path, 'rb')) if os.path.exists(proc_map_path) else None
    code_adj_cross = np.load(cross_adj_path)['code_adj'] if os.path.exists(cross_adj_path) else None

    return {
        'len_dist': len_dist,
        'code_visit_dist': code_visit_dist,
        'code_patient_dist': code_patient_dist,
        'code_adj_diag': code_adj_diag,
        'code_map_diag': code_map_diag,
        'code_adj_proc': code_adj_proc,
        'code_map_proc': code_map_proc,
        'code_adj_cross': code_adj_cross
    }


def load_code_name_map(data_path):
    """Optional helper for mapping ICD9 → name."""
    names = pd.read_excel(os.path.join(data_path, 'map.xlsx'), engine='openpyxl')
    code_keys = names['DIAGNOSIS CODE'].tolist()
    name_vals = names['LONG DESCRIPTION'].tolist()
    code_name_map = {k: v for k, v in zip(code_keys, name_vals)}
    return code_name_map


# ============================================================
# 🩺 Dual-stream Base GRU dataloader
# ============================================================

def get_base_gru_train_loader_dual(dataset_path, batch_size, device):
    """
    Load data for pretraining BaseGRUDual (diagnosis + procedure).
    Expect file: {dataset_path}/standard/real_next/train_dual.npz
    """
    from datautils.dataset import DatasetRealNextDual
    real_next_path = os.path.join(dataset_path, 'standard', 'real_next')
    print("loading real next dual data ...")
    dataset = DatasetRealNextDual(real_next_path, device=device)
    train_loader = DataLoader(dataset.train_set, shuffle=True, batch_size=batch_size)
    return train_loader

