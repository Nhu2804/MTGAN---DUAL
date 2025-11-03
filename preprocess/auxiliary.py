import numpy as np
from preprocess.parse_csv import EHRParser


# ============================================================
# 🧩 Hàm gốc (Diagnosis only)
# ============================================================

def generate_code_code_adjacent(pids, patient_admission, admission_codes_encoded, code_num):
    print('generating code–code adjacency matrix (diagnosis) ...')
    n = code_num
    adj = np.zeros((n, n), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i, len(pids)), end='')
        for admission in patient_admission[pid]:
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            for row in range(len(codes) - 1):
                for col in range(row + 1, len(codes)):
                    c_i = codes[row]
                    c_j = codes[col]
                    adj[c_i, c_j] = 1
                    adj[c_j, c_i] = 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return adj


# ============================================================
# 🩺 Dual-stream extensions
# ============================================================

def generate_code_code_adjacent_dual(pids, patient_admission,
                                     diag_encoded, proc_encoded,
                                     code_num_diag, code_num_proc):
    """
    Generate adjacency matrices for:
      (1) diag–diag
      (2) proc–proc
      (3) diag–proc (cross-dependency)
    """
    print('generating adjacency matrices (diag, proc, cross) ...')
    adj_diag = np.zeros((code_num_diag, code_num_diag), dtype=int)
    adj_proc = np.zeros((code_num_proc, code_num_proc), dtype=int)
    adj_cross = np.zeros((code_num_diag, code_num_proc), dtype=int)

    for i, pid in enumerate(pids):
        print(f'\r\t{i + 1} / {len(pids)}', end='')
        for admission in patient_admission[pid]:
            adm_id = admission[EHRParser.adm_id_col]

            # Diag–Diag
            if adm_id in diag_encoded:
                codes_diag = diag_encoded[adm_id]
                for r in range(len(codes_diag) - 1):
                    for c in range(r + 1, len(codes_diag)):
                        i1, i2 = codes_diag[r], codes_diag[c]
                        adj_diag[i1, i2] = 1
                        adj_diag[i2, i1] = 1

            # Proc–Proc
            if adm_id in proc_encoded:
                codes_proc = proc_encoded[adm_id]
                for r in range(len(codes_proc) - 1):
                    for c in range(r + 1, len(codes_proc)):
                        i1, i2 = codes_proc[r], codes_proc[c]
                        adj_proc[i1, i2] = 1
                        adj_proc[i2, i1] = 1

            # Cross Diag–Proc
            if adm_id in diag_encoded and adm_id in proc_encoded:
                for d in diag_encoded[adm_id]:
                    for p in proc_encoded[adm_id]:
                        adj_cross[d, p] = 1

    print(f'\r\t{len(pids)} / {len(pids)}')
    return adj_diag, adj_proc, adj_cross


# ============================================================
# 📊 Stats helpers
# ============================================================

def real_data_stat(real_data_x, lens):
    """Statistics for one modality (diag or proc)."""
    admission_num_count = {}
    max_admission_num = 0
    code_visit_count = {}
    code_patient_count = {}

    for patient, len_i in zip(real_data_x, lens):
        if max_admission_num < len_i:
            max_admission_num = len_i
        admission_num_count[len_i] = admission_num_count.get(len_i, 0) + 1

        codes_set = set()
        for i in range(len_i):
            admission = patient[i]
            codes = np.where(admission > 0)[0]
            codes_set.update(codes.tolist())
            for code in codes:
                code_visit_count[code] = code_visit_count.get(code, 0) + 1
        for code in codes_set:
            code_patient_count[code] = code_patient_count.get(code, 0) + 1

    # Admission-length distribution
    admission_dist = np.zeros((max_admission_num,), dtype=float)
    for num, count in admission_num_count.items():
        admission_dist[num - 1] = count
    admission_dist /= admission_dist.sum() if admission_dist.sum() > 0 else 1.0

    # Visit-level code distribution
    if len(code_visit_count) > 0:
        code_visit_dist = np.zeros(max(code_visit_count.keys()) + 1, dtype=float)
        for code, count in code_visit_count.items():
            code_visit_dist[code] = count
        code_visit_dist /= code_visit_dist.sum() if code_visit_dist.sum() > 0 else 1.0
    else:
        code_visit_dist = np.zeros((1,), dtype=float)

    # Patient-level code distribution
    if len(code_patient_count) > 0:
        code_patient_dist = np.zeros(max(code_patient_count.keys()) + 1, dtype=float)
        for code, count in code_patient_count.items():
            code_patient_dist[code] = count
        code_patient_dist /= code_patient_dist.sum() if code_patient_dist.sum() > 0 else 1.0
    else:
        code_patient_dist = np.zeros((1,), dtype=float)

    # ✅ MUST return
    return admission_dist, code_visit_dist, code_patient_dist


def real_data_stat_dual(real_diag, real_proc, lens):
    """
    Compute combined stats for diagnosis + procedure.
    Returns: dict with separate distributions and co-occurrence rate.
    """
    print('computing combined diag–proc stats ...')
    admission_dist_d, visit_d, pat_d = real_data_stat(real_diag, lens)
    admission_dist_p, visit_p, pat_p = real_data_stat(real_proc, lens)

    # Estimate co-occurrence rate per-visit
    cross_count = 0
    total_visits = 0
    for patient_d, patient_p, len_i in zip(real_diag, real_proc, lens):
        for i in range(len_i):
            d_codes = np.where(patient_d[i] > 0)[0]
            p_codes = np.where(patient_p[i] > 0)[0]
            if len(d_codes) > 0 and len(p_codes) > 0:
                cross_count += 1
            total_visits += 1
    cross_rate = cross_count / (total_visits + 1e-6)

    return {
        'admission_dist_diag': admission_dist_d,
        'admission_dist_proc': admission_dist_p,
        'code_visit_dist_diag': visit_d,
        'code_visit_dist_proc': visit_p,
        # (tuỳ ý) có thể thêm patient-level nếu cần ở downstream:
        # 'code_patient_dist_diag': pat_d,
        # 'code_patient_dist_proc': pat_p,
        'cross_diag_proc_rate': cross_rate
    }


# ============================================================
# 🧮 ICD9 utilities (giữ nguyên)
# ============================================================

def parse_icd9_range(range_: str):
    ranges = range_.lstrip().split('-')
    if ranges[0][0] == 'V':
        prefix = 'V'
        format_ = '%02d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    elif ranges[0][0] == 'E':
        prefix = 'E'
        format_ = '%03d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    else:
        prefix = ''
        format_ = '%03d'
        if len(ranges) == 1:
            start = int(ranges[0])
            end = start
        else:
            start, end = int(ranges[0]), int(ranges[1])
    return prefix, format_, start, end


def to_standard_icd9(code: str):
    code = str(code)
    if code == '':
        return code
    split_pos = 4 if code.startswith('E') else 3
    icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
    return icd9_code


def generate_code_levels(path, code_map: dict) -> np.ndarray:
    print('generating code levels ...')
    import os
    code_map = {to_standard_icd9(code): cid for code, cid in code_map.items()}
    three_level_code_set = set(code.split('.')[0] for code in code_map)
    icd9_path = os.path.join(path, 'icd9.txt')
    icd9_range = list(open(icd9_path, 'r', encoding='utf-8').readlines())
    three_level_dict = dict()
    level1, level2, level3 = (0, 0, 0)
    level1_can_add = False
    for range_ in icd9_range:
        range_ = range_.rstrip()
        if range_[0] == ' ':
            prefix, format_, start, end = parse_icd9_range(range_)
            level2_cannot_add = True
            for i in range(start, end + 1):
                code = prefix + format_ % i
                if code in three_level_code_set:
                    three_level_dict[code] = [level1, level2, level3]
                    level3 += 1
                    level1_can_add = True
                    level2_cannot_add = False
            if not level2_cannot_add:
                level2 += 1
        else:
            if level1_can_add:
                level1 += 1
                level1_can_add = False

    code_level = dict()
    for code, cid in code_map.items():
        three_level_code = code.split('.')[0]
        three_level = three_level_dict[three_level_code]
        code_level[code] = three_level + [cid]

    code_level_matrix = np.zeros((len(code_map), 4), dtype=int)
    for code, cid in code_map.items():
        code_level_matrix[cid] = code_level[code]

    return code_level_matrix
