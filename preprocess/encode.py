from collections import OrderedDict
from preprocess.parse_csv import EHRParser


# ============================================================
# 🧠 Encode một loại mã (diagnosis hoặc procedure)
# ============================================================
def encode_concept(patient_admission, admission_concepts):
    """
    Encode codes (diagnoses or procedures) into integer IDs.
    Returns:
        admission_concept_encoded: dict[adm_id] -> list[int]
        concept_map: dict[code_str] -> int
    """
    concept_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            adm_id = admission[EHRParser.adm_id_col]
            if adm_id in admission_concepts:
                concepts = admission_concepts[adm_id]
                for code in concepts:
                    if code not in concept_map:
                        concept_map[code] = len(concept_map)

    # ✅ fix bug: 'concept' → 'concepts'
    admission_concept_encoded = {
        adm_id: list(set(concept_map[code] for code in codes if code in concept_map))
        for adm_id, codes in admission_concepts.items()
    }

    return admission_concept_encoded, concept_map


# ============================================================
# 🩺 Encode song song Diagnosis + Procedure (dual-stream)
# ============================================================
def encode_dual_concepts(patient_admission, admission_diagnoses, admission_procedures):
    """
    Encode both diagnoses and procedures for dual-stream models.
    Each admission will have two encoded lists:
        - encoded_diagnosis[adm_id] -> list[int]
        - encoded_procedure[adm_id] -> list[int]
    Returns:
        admission_diag_encoded, diag_map, admission_proc_encoded, proc_map
    """
    # Diagnosis
    diag_encoded, diag_map = encode_concept(patient_admission, admission_diagnoses)
    # Procedure
    proc_encoded, proc_map = encode_concept(patient_admission, admission_procedures)
    return diag_encoded, diag_map, proc_encoded, proc_map
