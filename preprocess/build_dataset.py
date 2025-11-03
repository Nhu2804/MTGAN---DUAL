import numpy as np
from preprocess.parse_csv import EHRParser


# ============================================================
# 🧩 Giữ nguyên các hàm gốc (Diagnosis only)
# ============================================================

def split_patients(patient_admission, admission_codes, code_map, train_num, seed=6669):
    np.random.seed(seed)
    common_pids = set()
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission[EHRParser.adm_id_col]]
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break
    print('\r\t100%')

    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    test_pids = remaining_pids[(train_num - len(common_pids)):]
    return train_pids, test_pids


def build_real_data(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num):
    n = len(pids)
    x = np.zeros((n, max_admission_num, code_num), dtype=bool)
    lens = np.zeros((n,), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions):
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            x[i, k, codes] = 1
        lens[i] = len(admissions)
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, lens


def build_code_xy(real_data, real_lens):
    x = np.zeros_like(real_data)
    y = np.zeros((real_data.shape[0], real_data.shape[-1]), dtype=real_data.dtype)
    lens = real_lens - 1
    for i, (real_data_i, len_i) in enumerate(zip(real_data, lens)):
        x[i][:len_i] = real_data_i[:len_i]
        y[i] = real_data_i[len_i]
    return x, y, lens


def build_visit_x(data, lens, code_num):
    n = np.sum(lens)
    x = np.zeros((n, code_num), dtype=bool)
    t = np.zeros((n,), dtype=int)
    i = 0
    for patient, len_ in zip(data, lens):
        for k in range(len_):
            x[i] = patient[k]
            t[i] = k
            i += 1
    return x, t


def build_real_next_xy(real_data, real_lens):
    x = np.zeros_like(real_data)
    y = np.zeros_like(real_data)
    lens = real_lens - 1
    for i, (real_data_i, len_i) in enumerate(zip(real_data, lens)):
        x[i][:len_i] = real_data_i[:len_i]
        y[i][:len_i] = real_data_i[1:(len_i + 1)]
    return x, y, lens


def build_heart_failure_y(hf_prefix, codes_y, code_map):
    hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
    hfs = np.zeros((len(code_map),), dtype=int)
    hfs[hf_list] = 1
    hf_exist = np.logical_and(codes_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(bool)
    return y


# ============================================================
# 🩺 Dual-Stream Version (Diagnosis + Procedure)
# ============================================================

def build_real_data_dual(pids, patient_admission,
                         diag_encoded, proc_encoded,
                         max_admission_num, code_num_diag, code_num_proc):
    """
    Build real tensors for both diagnosis and procedure per patient.
    Returns:
        x_diag: [N, T, |C_diag|]
        x_proc: [N, T, |C_proc|]
        lens:   [N]
    """
    n = len(pids)
    x_diag = np.zeros((n, max_admission_num, code_num_diag), dtype=bool)
    x_proc = np.zeros((n, max_admission_num, code_num_proc), dtype=bool)
    lens = np.zeros((n,), dtype=int)

    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, n), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions):
            adm_id = admission[EHRParser.adm_id_col]
            if adm_id in diag_encoded:
                codes_diag = diag_encoded[adm_id]
                x_diag[i, k, codes_diag] = 1
            if adm_id in proc_encoded:
                codes_proc = proc_encoded[adm_id]
                x_proc[i, k, codes_proc] = 1
        lens[i] = len(admissions)
    print('\r\t%d / %d' % (n, n))
    return x_diag, x_proc, lens


def build_real_next_xy_dual(x_diag, x_proc, lens):
    """
    Build next-visit pairs for both diagnosis and procedure streams.
    """
    x_diag_in = np.zeros_like(x_diag)
    x_proc_in = np.zeros_like(x_proc)
    y_diag = np.zeros_like(x_diag)
    y_proc = np.zeros_like(x_proc)
    lens_out = lens - 1

    for i, len_i in enumerate(lens_out):
        x_diag_in[i, :len_i] = x_diag[i, :len_i]
        x_proc_in[i, :len_i] = x_proc[i, :len_i]
        y_diag[i, :len_i] = x_diag[i, 1:len_i + 1]
        y_proc[i, :len_i] = x_proc[i, 1:len_i + 1]
    return x_diag_in, x_proc_in, y_diag, y_proc, lens_out
