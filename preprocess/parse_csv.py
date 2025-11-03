import os
from datetime import datetime
from collections import OrderedDict

import pandas

import pandas as pd
import numpy as np


class EHRParser:
    pid_col = 'pid'
    adm_id_col = 'adm_id'
    adm_time_col = 'adm_time'
    cid_col = 'cid'

    def __init__(self, path, use_proc=False):
        self.path = path
        self.use_proc = use_proc  # 🆕 cho phép bật/tắt đọc thủ thuật

        self.skip_pid_check = False

        # Các cấu trúc dữ liệu chính
        self.patient_admission = None
        self.admission_codes = None            # Diagnosis
        self.admission_procedures = None       # Procedure
        self.admission_medications = None

        self.parse_fn = {
            'd': self.set_diagnosis,
            'p': self.set_procedure,
            'm': self.set_medication
        }

    # ====== Các hàm con phải được định nghĩa ở subclass ======
    def set_admission(self):
        raise NotImplementedError

    def set_diagnosis(self):
        raise NotImplementedError

    def set_procedure(self):
        raise NotImplementedError

    def set_medication(self):
        raise NotImplementedError
    # ==========================================================

    def parse_admission(self):
        print('📂 Parsing the csv file of admission ...')
        filename, cols, converters = self.set_admission()
        admissions = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        admissions = self._after_read_admission(admissions, cols)

        all_patients = OrderedDict()
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print(f'\r\t{i + 1} / {len(admissions)} rows', end='')
            pid, adm_id, adm_time = row[cols[self.pid_col]], row[cols[self.adm_id_col]], row[cols[self.adm_time_col]]
            if pid not in all_patients:
                all_patients[pid] = []
            all_patients[pid].append({self.adm_id_col: adm_id, self.adm_time_col: adm_time})
        print(f'\r\t{len(admissions)} / {len(admissions)} rows')

        # Chỉ giữ bệnh nhân có ≥ 2 lượt khám
        patient_admission = OrderedDict()
        for pid, admissions in all_patients.items():
            if len(admissions) >= 2:
                patient_admission[pid] = sorted(admissions, key=lambda a: a[self.adm_time_col])
        self.patient_admission = patient_admission

    def _after_read_admission(self, admissions, cols):
        return admissions

    def _parse_concept(self, concept_type):
        assert concept_type in self.parse_fn.keys()
        filename, cols, converters = self.parse_fn[concept_type]()
        concepts = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        concepts = self._after_read_concepts(concepts, concept_type, cols)

        result = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print(f'\r\t{i + 1} / {len(concepts)} rows', end='')
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:
                adm_id, code = row[cols[self.adm_id_col]], row[cols[self.cid_col]]
                if code == '' or pd.isna(code):
                    continue
                result.setdefault(adm_id, []).append(code)
        print(f'\r\t{len(concepts)} / {len(concepts)} rows')
        return result

    def _after_read_concepts(self, concepts, concept_type, cols):
        return concepts

    # ====== Các hàm parse cụ thể ======
    def parse_diagnoses(self):
        print('📘 Parsing csv file of diagnosis ...')
        self.admission_codes = self._parse_concept('d')

    def parse_procedures(self):
        print('🩺 Parsing csv file of procedures ...')
        self.admission_procedures = self._parse_concept('p')

    def parse_medications(self):
        print('💊 Parsing csv file of medications ...')
        self.admission_medications = self._parse_concept('m')
    # ==================================

    def calibrate_patient_by_admission(self):
        """Chỉ giữ bệnh nhân có admission khớp với diagnosis (và nếu bật, procedure)."""
        print('🧩 Calibrating patients by admission ...')
        del_pids = []
        for pid, admissions in self.patient_admission.items():
            valid = True
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id not in self.admission_codes:
                    valid = False
                    break
                # Nếu dùng procedure, đảm bảo cũng tồn tại admission này
                if self.use_proc and adm_id not in self.admission_procedures:
                    valid = False
                    break
            if not valid:
                del_pids.append(pid)

        for pid in del_pids:
            admissions = self.patient_admission[pid]
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id in self.admission_codes:
                    del self.admission_codes[adm_id]
                if self.use_proc and adm_id in self.admission_procedures:
                    del self.admission_procedures[adm_id]
            del self.patient_admission[pid]

    def calibrate_admission_by_patient(self):
        """Đảm bảo mọi admission trong codes/procs đều thuộc về bệnh nhân còn lại."""
        print('🧮 Calibrating admission by patients ...')
        adm_id_set = set()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id_set.add(admission[self.adm_id_col])

        for attr in ['admission_codes', 'admission_procedures']:
            concepts = getattr(self, attr, None)
            if concepts is None:
                continue
            for adm_id in list(concepts.keys()):
                if adm_id not in adm_id_set:
                    del concepts[adm_id]

    def sample_patients(self, sample_num, seed):
        """Lấy mẫu ngẫu nhiên bệnh nhân cho training nhanh."""
        np.random.seed(seed)
        keys = list(self.patient_admission.keys())
        selected_pids = np.random.choice(keys, sample_num, False)
        self.patient_admission = {pid: self.patient_admission[pid] for pid in selected_pids}

        admission_codes, admission_procs = {}, {}
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                admission_codes[adm_id] = self.admission_codes.get(adm_id, [])
                if self.use_proc:
                    admission_procs[adm_id] = self.admission_procedures.get(adm_id, [])
        self.admission_codes = admission_codes
        if self.use_proc:
            self.admission_procedures = admission_procs

    def parse(self, sample_num=None, seed=6669):
        """Pipeline chính: đọc admission → diagnosis → (procedure) → lọc."""
        self.parse_admission()
        self.parse_diagnoses()
        if self.use_proc:
            self.parse_procedures()

        self.calibrate_patient_by_admission()
        self.calibrate_admission_by_patient()

        if sample_num is not None:
            self.sample_patients(sample_num, seed)

        # Trả dữ liệu tương ứng
        if self.use_proc:
            return self.patient_admission, self.admission_codes, self.admission_procedures
        else:
            return self.patient_admission, self.admission_codes


# ===========================================================
# MIMIC-III
# ===========================================================
class Mimic3Parser(EHRParser):
    def set_admission(self):
        filename = 'ADMISSIONS.csv'
        cols = {
            self.pid_col: 'SUBJECT_ID',
            self.adm_id_col: 'HADM_ID',
            self.adm_time_col: 'ADMITTIME'
        }
        converters = {
            'SUBJECT_ID': int,
            'HADM_ID': int,
            'ADMITTIME': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converters

    def set_diagnosis(self):
        filename = 'DIAGNOSES_ICD.csv'
        cols = {
            self.pid_col: 'SUBJECT_ID',
            self.adm_id_col: 'HADM_ID',
            self.cid_col: 'ICD9_CODE'
        }
        converters = {'SUBJECT_ID': int, 'HADM_ID': int, 'ICD9_CODE': str}
        return filename, cols, converters

    def set_procedure(self):
        filename = 'PROCEDURES_ICD.csv'
        cols = {
            self.pid_col: 'SUBJECT_ID',
            self.adm_id_col: 'HADM_ID',
            self.cid_col: 'ICD9_CODE'
        }
        converters = {'SUBJECT_ID': int, 'HADM_ID': int, 'ICD9_CODE': str}
        return filename, cols, converters

    def set_medication(self):
        filename = 'PRESCRIPTIONS.csv'
        cols = {
            self.pid_col: 'SUBJECT_ID',
            self.adm_id_col: 'HADM_ID',
            self.cid_col: 'NDC'
        }
        converters = {'SUBJECT_ID': int, 'HADM_ID': int, 'NDC': str}
        return filename, cols, converters



class Mimic4Parser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.icd_ver_col = 'icd_version'
        self.icd_map = self._load_icd_map()
        self.patient_year_map = self._load_patient()

    def _load_icd_map(self):
        print('loading ICD-10 to ICD-9 map ...')
        filename = 'icd10-icd9.csv'
        cols = ['ICD10', 'ICD9']
        converters = {'ICD10': str, 'ICD9': str}
        icd_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        icd_map = {row['ICD10']: row['ICD9'] for _, row in icd_csv.iterrows()}
        return icd_map

    def _load_patient(self):
        print('loading patients anchor year ...')
        filename = 'patients.csv'
        cols = ['subject_id', 'anchor_year', 'anchor_year_group']
        converters = {'subject_id': int, 'anchor_year': int, 'anchor_year_group': lambda cell: int(str(cell)[:4])}
        patient_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        patient_year_map = {row['subject_id']: row['anchor_year'] - row['anchor_year_group']
                            for i, row in patient_csv.iterrows()}
        return patient_year_map

    def set_admission(self):
        filename = 'admissions.csv'
        cols = {self.pid_col: 'subject_id', self.adm_id_col: 'hadm_id', self.adm_time_col: 'admittime'}
        converter = {
            'subject_id': int,
            'hadm_id': int,
            'admittime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnoses_icd.csv'
        cols = {
            self.pid_col: 'subject_id',
            self.adm_id_col: 'hadm_id',
            self.cid_col: 'icd_code',
            self.icd_ver_col: 'icd_version'
        }
        converter = {'subject_id': int, 'hadm_id': int, 'icd_code': str, 'icd_version': int}
        return filename, cols, converter

    def _after_read_admission(self, admissions, cols):
        print('\tselecting valid admission ...')
        valid_admissions = []
        n = len(admissions)
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t\t%d in %d rows' % (i + 1, n), end='')
            pid = row[cols[self.pid_col]]
            year = row[cols[self.adm_time_col]].year - self.patient_year_map[pid]
            if year > 2012:
                valid_admissions.append(i)
        print('\r\t\t%d in %d rows' % (n, n))
        print('\t\tremaining %d rows' % len(valid_admissions))
        return admissions.iloc[valid_admissions]

    def _after_read_concepts(self, concepts, concept_type, cols):
        print('\tmapping ICD-10 to ICD-9 ...')
        n = len(concepts)
        if concept_type == 'd':
            def _10to9(i, row):
                if i % 100 == 0:
                    print('\r\t\t%d in %d rows' % (i + 1, n), end='')
                cid = row[cid_col]
                if row[icd_ver_col] == 10:
                    if cid not in self.icd_map:
                        code = self.icd_map[cid + '1'] if cid + '1' in self.icd_map else ''
                    else:
                        code = self.icd_map[cid]
                    if code == 'NoDx':
                        code = ''
                else:
                    code = cid
                return code

            cid_col, icd_ver_col = cols[self.cid_col], self.icd_ver_col
            col = np.array([_10to9(i, row) for i, row in concepts.iterrows()])
            print('\r\t\t%d in %d rows' % (n, n))
            concepts[cid_col] = col
        return concepts
