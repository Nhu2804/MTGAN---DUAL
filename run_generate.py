import os
import random
import torch
import numpy as np

from config import get_generate_args, get_paths
from model import Generator, GeneratorDual
from datautils.dataloader import load_code_name_map, load_meta_data
from datautils.dataset import DatasetReal, DatasetRealDual
from generation.generate import generate_ehr
from generation.stat_ehr import (
    get_basic_statistics,
    get_top_k_codes,
    calc_distance
)

# ============================================================
# 🩺 Dual-stream helper
# ============================================================
def generate_ehr_dual(generator, number, len_dist, batch_size):
    fake_diag, fake_proc, fake_lens = [], [], []
    for i in range(0, number, batch_size):
        n = number - i if i + batch_size > number else batch_size
        target_diag, target_proc = generator.get_target_codes(n)
        lens = torch.multinomial(len_dist, num_samples=n, replacement=True) + 1
        x_diag, x_proc = generator.sample(target_diag, target_proc, lens)

        fake_diag.append(x_diag.cpu().numpy())
        fake_proc.append(x_proc.cpu().numpy())
        fake_lens.append(lens.cpu().numpy())

    fake_diag = np.concatenate(fake_diag, axis=0)
    fake_proc = np.concatenate(fake_proc, axis=0)
    fake_lens = np.concatenate(fake_lens, axis=-1)
    return fake_diag, fake_proc, fake_lens


# ============================================================
# 🚀 Main generation function
# ============================================================
def generate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path, _, params_path = get_paths(args)
    meta = load_meta_data(dataset_path)
    code_name_map = load_code_name_map(args.data_path)
    len_dist = torch.from_numpy(meta["len_dist"]).to(device)

    # ------------------------------------------------------------
    # 🩺 Dual-Stream Mode (Diagnosis + Procedure)
    # ------------------------------------------------------------
    if getattr(args, "use_proc", False):
        print("🩺 Generating Dual-Stream synthetic data...")

        dataset_real = DatasetRealDual(os.path.join(dataset_path, "standard", "real_data"))
        code_map_diag = meta["code_map_diag"]
        code_map_proc = meta["code_map_proc"]

        icode_map_diag = {v: k for k, v in code_map_diag.items()}
        icode_map_proc = {v: k for k, v in code_map_proc.items()}

        max_len = dataset_real.train_set.data[0].shape[1]
        if args.use_iteration == -1:
            param_file_name = "generator_dual.pt"
        else:
            param_file_name = f"generator_dual.{args.use_iteration}.pt"

        generator = GeneratorDual(
            code_num_diag=len(code_map_diag),
            code_num_proc=len(code_map_proc),
            hidden_dim=args.g_hidden_dim,
            attention_dim=args.g_attention_dim,
            max_len=max_len,
            device=device
        ).to(device)
        generator.load(params_path, param_file_name)

        fake_diag, fake_proc, fake_lens = generate_ehr_dual(generator, args.number, len_dist, args.batch_size)
        real_diag, real_proc, real_lens = dataset_real.train_set.data

        # --- Diagnosis stats ---
        print("\n[DIAG] Real data")
        n_types, n_codes, n_visits, avg_code_num, avg_visit_num = get_basic_statistics(real_diag, real_lens)
        print(f"{args.number} samples -- types: {n_types} -- codes: {n_codes} -- avg codes/visit: {avg_code_num:.3f}, avg visits/patient: {avg_visit_num:.3f}")

        print("\n[DIAG] Fake data")
        n_types, n_codes, n_visits, avg_code_num, avg_visit_num = get_basic_statistics(fake_diag, fake_lens)
        print(f"{args.number} samples -- types: {n_types} -- codes: {n_codes} -- avg codes/visit: {avg_code_num:.3f}, avg visits/patient: {avg_visit_num:.3f}")
        get_top_k_codes(fake_diag, fake_lens, icode_map_diag, code_name_map, top_k=10)

        # --- Procedure stats ---
        print("\n[PROC] Real data")
        n_types, n_codes, n_visits, avg_code_num, avg_visit_num = get_basic_statistics(real_proc, real_lens)
        print(f"{args.number} samples -- types: {n_types} -- codes: {n_codes} -- avg codes/visit: {avg_code_num:.3f}, avg visits/patient: {avg_visit_num:.3f}")

        print("\n[PROC] Fake data")
        n_types, n_codes, n_visits, avg_code_num, avg_visit_num = get_basic_statistics(fake_proc, fake_lens)
        print(f"{args.number} samples -- types: {n_types} -- codes: {n_codes} -- avg codes/visit: {avg_code_num:.3f}, avg visits/patient: {avg_visit_num:.3f}")

        # --- Distance metrics ---
        print("\n[DIAG] Distance (JSD, ND)")
        jsd_v, jsd_p, nd_v, nd_p = calc_distance(real_diag, real_lens, fake_diag, fake_lens, len(code_map_diag))
        print(f"JSD_v: {jsd_v:.4f}, JSD_p: {jsd_p:.4f}, ND_v: {nd_v:.4f}, ND_p: {nd_p:.4f}")

        print("\n[PROC] Distance (JSD, ND)")
        jsd_v, jsd_p, nd_v, nd_p = calc_distance(real_proc, real_lens, fake_proc, fake_lens, len(code_map_proc))
        print(f"JSD_v: {jsd_v:.4f}, JSD_p: {jsd_p:.4f}, ND_v: {nd_v:.4f}, ND_p: {nd_p:.4f}")

        synthetic_path = os.path.join(args.result_path, f"synthetic_dual_{args.dataset}.npz")
        np.savez_compressed(synthetic_path, x_diag=fake_diag, x_proc=fake_proc, lens=fake_lens)
        print(f"\n💾 Saved dual-stream data → {synthetic_path}\n✅ Done!")

    # ------------------------------------------------------------
    # 🧠 Single-Stream Mode (Diagnosis only)
    # ------------------------------------------------------------
    else:
        print("🧠 Generating Single-Stream synthetic data...")

        len_dist_np, _, _, _, code_map = meta.values()
        code_num = len(code_map)
        icode_map = {v: k for k, v in code_map.items()}

        dataset_real = DatasetReal(os.path.join(dataset_path, 'standard', 'real_data'))
        len_dist = torch.from_numpy(len_dist_np).to(device)
        max_len = dataset_real.train_set.data[0].shape[1]

        param_file_name = f'generator.{args.use_iteration}.pt' if args.use_iteration != -1 else 'generator.pt'
        generator = Generator(
            code_num=code_num,
            hidden_dim=args.g_hidden_dim,
            attention_dim=args.g_attention_dim,
            max_len=max_len,
            device=device
        ).to(device)
        generator.load(params_path, param_file_name)

        fake_x, fake_lens = generate_ehr(generator, args.number, len_dist, args.batch_size)
        real_x, real_lens = dataset_real.train_set.data

        print("\n[DIAG] Real data")
        n_types, n_codes, n_visits, avg_code_num, avg_visit_num = get_basic_statistics(real_x, real_lens)
        print(f"{args.number} samples -- types: {n_types} -- codes: {n_codes} -- avg codes/visit: {avg_code_num:.3f}, avg visits/patient: {avg_visit_num:.3f}")

        print("\n[DIAG] Fake data")
        n_types, n_codes, n_visits, avg_code_num, avg_visit_num = get_basic_statistics(fake_x, fake_lens)
        print(f"{args.number} samples -- types: {n_types} -- codes: {n_codes} -- avg codes/visit: {avg_code_num:.3f}, avg visits/patient: {avg_visit_num:.3f}")
        get_top_k_codes(fake_x, fake_lens, icode_map, code_name_map, top_k=10)

        jsd_v, jsd_p, nd_v, nd_p = calc_distance(real_x, real_lens, fake_x, fake_lens, code_num)
        print(f"JSD_v: {jsd_v:.4f}, JSD_p: {jsd_p:.4f}, ND_v: {nd_v:.4f}, ND_p: {nd_p:.4f}")

        synthetic_path = os.path.join(args.result_path, f"synthetic_{args.dataset}.npz")
        np.savez_compressed(synthetic_path, x=fake_x, lens=fake_lens)
        print(f"\n💾 Saved single-stream data → {synthetic_path}\n✅ Done!")


if __name__ == '__main__':
    args = get_generate_args()
    generate(args)
