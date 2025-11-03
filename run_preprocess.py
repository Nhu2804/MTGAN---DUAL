import os
import pickle
import numpy as np

from preprocess.parse_csv import Mimic3Parser, Mimic4Parser
from preprocess.encode import encode_concept, encode_dual_concepts
from preprocess.build_dataset import (
    split_patients,
    build_real_data,
    build_real_data_dual,
    build_real_next_xy,
    build_real_next_xy_dual,
    build_code_xy,
    build_visit_x,
    build_heart_failure_y,
)
from preprocess.auxiliary import (
    generate_code_code_adjacent,
    generate_code_code_adjacent_dual,
    real_data_stat,
    real_data_stat_dual,
    generate_code_levels
)
from config import get_preprocess_args


PARSERS = {
    'mimic3': Mimic3Parser,
    'mimic4': Mimic4Parser
}


if __name__ == '__main__':
    args = get_preprocess_args()

    data_path = args.data_path
    dataset = args.dataset  # mimic3 or mimic4

    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print(f'Please put the CSV files in data/{dataset}/raw/')
        exit()
    parsed_path = os.path.join(dataset_path, 'parsed')

    # ========================= PARSING =========================
    if args.from_saved:
        patient_admission = pickle.load(open(os.path.join(parsed_path, 'patient_admission.pkl'), 'rb'))
        admission_codes = pickle.load(open(os.path.join(parsed_path, 'admission_codes.pkl'), 'rb'))
        admission_procs = None
        if args.use_proc and os.path.exists(os.path.join(parsed_path, 'admission_procs.pkl')):
            admission_procs = pickle.load(open(os.path.join(parsed_path, 'admission_procs.pkl'), 'rb'))
    else:
        parser = PARSERS[dataset](raw_path, use_proc=args.use_proc)
        sample_num = args.sample_num if dataset == 'mimic4' else None
        if args.use_proc:
            patient_admission, admission_codes, admission_procs = parser.parse(sample_num)
        else:
            patient_admission, admission_codes = parser.parse(sample_num)
            admission_procs = None
        print('saving parsed data ...')
        os.makedirs(parsed_path, exist_ok=True)
        pickle.dump(patient_admission, open(os.path.join(parsed_path, 'patient_admission.pkl'), 'wb'))
        pickle.dump(admission_codes, open(os.path.join(parsed_path, 'admission_codes.pkl'), 'wb'))
        if args.use_proc:
            pickle.dump(admission_procs, open(os.path.join(parsed_path, 'admission_procs.pkl'), 'wb'))

    patient_num = len(patient_admission)
    print(f'patient num: {patient_num}')

    def stat(data):
        lens = [len(item) for item in data.values()]
        return max(lens), min(lens), sum(lens) / len(data)

    admission_stats = stat(patient_admission)
    visit_code_stats = stat(admission_codes)
    print(f'max, min, mean admission num: {admission_stats}')
    print(f'max, min, mean visit code num: {visit_code_stats}')

    max_admission_num = admission_stats[0]

    # ========================= ENCODING =========================
    print('encoding codes ...')
    if args.use_proc:
        diag_encoded, diag_map, proc_encoded, proc_map = encode_dual_concepts(
            patient_admission, admission_codes, admission_procs)
        code_num_diag = len(diag_map)
        code_num_proc = len(proc_map)
        print(f'Diagnosis codes: {code_num_diag}, Procedure codes: {code_num_proc}')
        pickle.dump(proc_map, open(os.path.join(parsed_path, 'proc_map.pkl'), 'wb'))
    else:
        diag_encoded, diag_map = encode_concept(patient_admission, admission_codes)
        code_num_diag = len(diag_map)
        code_num_proc = 0
        proc_encoded, proc_map = None, None
        print(f'Diagnosis codes: {code_num_diag}')

    code_levels = generate_code_levels(data_path, diag_map)
    pickle.dump({'code_levels': code_levels},
                open(os.path.join(parsed_path, 'code_levels.pkl'), 'wb'))

    # ========================= SPLIT TRAIN/TEST =========================
    print('splitting training, and test patients')
    train_pids, test_pids = split_patients(
        patient_admission=patient_admission,
        admission_codes=admission_codes,
        code_map=diag_map,
        train_num=args.train_num,
        seed=args.seed
    )
    print(f'Train: {len(train_pids)}, Test: {len(test_pids)}')

    # ========================= ADJACENCY =========================
    if args.use_proc:
        adj_diag, adj_proc, adj_cross = generate_code_code_adjacent_dual(
            train_pids, patient_admission,
            diag_encoded, proc_encoded,
            code_num_diag, code_num_proc
        )
    else:
        adj_diag = generate_code_code_adjacent(
            pids=train_pids,
            patient_admission=patient_admission,
            admission_codes_encoded=diag_encoded,
            code_num=code_num_diag
        )

    # ========================= BUILD REAL DATA =========================
    print('build real data ...')
    if args.use_proc:
        x_diag_train, x_proc_train, lens_train = build_real_data_dual(
            train_pids, patient_admission,
            diag_encoded, proc_encoded,
            max_admission_num, code_num_diag, code_num_proc
        )
        x_diag_test, x_proc_test, lens_test = build_real_data_dual(
            test_pids, patient_admission,
            diag_encoded, proc_encoded,
            max_admission_num, code_num_diag, code_num_proc
        )
        stats = real_data_stat_dual(x_diag_train, x_proc_train, lens_train)
    else:
        x_diag_train, lens_train = build_real_data(train_pids, patient_admission, diag_encoded,
                                                   max_admission_num, code_num_diag)
        x_diag_test, lens_test = build_real_data(test_pids, patient_admission, diag_encoded,
                                                 max_admission_num, code_num_diag)
        stats = real_data_stat(x_diag_train, lens_train)

    if args.use_proc:
        admission_dist = stats['admission_dist_diag']
        code_visit_dist = stats['code_visit_dist_diag']
        # dual-stream chưa có code_patient_dist riêng -> tạm dùng lại code_visit_dist
        code_patient_dist = stats['code_visit_dist_diag']
    else:
        admission_dist, code_visit_dist, code_patient_dist = stats



    # ========================= SAVE ENCODED =========================
    encoded_path = os.path.join(dataset_path, 'encoded')
    os.makedirs(encoded_path, exist_ok=True)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(diag_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump(diag_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    if args.use_proc:
        pickle.dump(proc_encoded, open(os.path.join(encoded_path, 'proc_encoded.pkl'), 'wb'))
        pickle.dump(proc_map, open(os.path.join(encoded_path, 'proc_map.pkl'), 'wb'))
    pickle.dump({'train_pids': train_pids, 'test_pids': test_pids},
                open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))

    # ========================= SAVE STANDARD DATA =========================
    standard_path = os.path.join(dataset_path, 'standard')
    os.makedirs(standard_path, exist_ok=True)

    print('saving real data ...')
    real_data_path = os.path.join(standard_path, 'real_data')
    os.makedirs(real_data_path, exist_ok=True)

    if args.use_proc:
        np.savez_compressed(os.path.join(real_data_path, 'train_dual.npz'),
                            x_diag=x_diag_train, x_proc=x_proc_train, lens=lens_train)
        np.savez_compressed(os.path.join(real_data_path, 'test_dual.npz'),
                            x_diag=x_diag_test, x_proc=x_proc_test, lens=lens_test)
    else:
        np.savez_compressed(os.path.join(real_data_path, 'train.npz'),
                            x=x_diag_train, lens=lens_train)
        np.savez_compressed(os.path.join(real_data_path, 'test.npz'),
                            x=x_diag_test, lens=lens_test)

    # ========================= SAVE META =========================
    np.savez_compressed(os.path.join(standard_path, 'real_data_stat.npz'),
                        admission_dist=admission_dist,
                        code_visit_dist=code_visit_dist,
                        code_patient_dist=code_patient_dist)
    if args.use_proc:
        np.savez_compressed(os.path.join(standard_path, 'code_adj_diag.npz'), code_adj=adj_diag)
        np.savez_compressed(os.path.join(standard_path, 'code_adj_proc.npz'), code_adj=adj_proc)
        np.savez_compressed(os.path.join(standard_path, 'code_adj_cross.npz'), code_adj=adj_cross)
    else:
        np.savez_compressed(os.path.join(standard_path, 'code_adj.npz'), code_adj=adj_diag)

    print('✅ Preprocess finished successfully!')

    # ========================= SAVE REAL NEXT DATA (Dual-stream) =========================
    real_next_path = os.path.join(standard_path, 'real_next')
    os.makedirs(real_next_path, exist_ok=True)

    print('saving real next data ...')
    if args.use_proc:
        print('\tsaving train real next dual data ...')
        (train_diag_next_x, train_proc_next_x,
         train_diag_next_y, train_proc_next_y,
         train_real_next_lens) = build_real_next_xy_dual(
             x_diag_train, x_proc_train, lens_train
         )

        np.savez_compressed(os.path.join(real_next_path, 'train_dual.npz'),
                            x_diag=train_diag_next_x,
                            x_proc=train_proc_next_x,
                            y_diag=train_diag_next_y,
                            y_proc=train_proc_next_y,
                            lens=train_real_next_lens)
    else:
        print('\tsaving train real next data ...')
        train_real_next_x, train_real_next_y, train_real_next_lens = build_real_next_xy(
            x_diag_train, lens_train
        )
        np.savez_compressed(os.path.join(real_next_path, 'train.npz'),
                            x=train_real_next_x,
                            y=train_real_next_y,
                            lens=train_real_next_lens)

    print('✅ All real_next data saved successfully!')
