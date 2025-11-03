import random
import torch
import numpy as np

from config import get_training_args, get_paths
from model import Generator, Critic, BaseGRU
from trainer import GANTrainer, BaseGRUTrainer
from datautils.dataloader import (
    load_code_name_map,
    load_meta_data,
    get_train_test_loader,
    get_base_gru_train_loader,
    get_train_test_loader_dual,
    get_base_gru_train_loader_dual
)

# ============================================================
# 🧮 Utilities
# ============================================================
def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
# 🚀 Main training function
# ============================================================
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path, records_path, params_path = get_paths(args)

    # ------------------------------------------------------------
    # Load meta (tự động tương thích 5 / 7 / 8 keys)
    # ------------------------------------------------------------
    meta = load_meta_data(dataset_path)
    len_dist          = meta['len_dist']
    code_visit_dist   = meta['code_visit_dist']
    code_patient_dist = meta['code_patient_dist']

    # ✅ Sửa lỗi ValueError (array truth value ambiguous)
    code_adj_diag = meta.get('code_adj_diag')
    code_adj      = code_adj_diag if code_adj_diag is not None else meta.get('code_adj')

    code_map_diag = meta.get('code_map_diag')
    code_map      = code_map_diag if code_map_diag is not None else meta.get('code_map')

    code_adj_proc  = meta.get('code_adj_proc')   # có thể None
    code_map_proc  = meta.get('code_map_proc')   # có thể None
    code_adj_cross = meta.get('code_adj_cross')  # có thể None

    code_name_map = load_code_name_map(args.data_path)

    # ------------------------------------------------------------
    # 🩺 Dual Mode (Diagnosis + Procedure)
    # ------------------------------------------------------------
    if getattr(args, "use_proc", False):
        from model import GeneratorDual, CriticDual, BaseGRUDual
        from trainer import GANTrainerDual, BaseGRUTrainerDual

        print("🩺 Running Dual-Stream MTGAN (Diagnosis + Procedure)")
        train_loader, test_loader, max_len = get_train_test_loader_dual(dataset_path, args.batch_size, device)
        base_gru_trainloader = get_base_gru_train_loader_dual(dataset_path, args.batch_size, device)

        diag_code_num = code_adj.shape[0]

        if code_adj_proc is not None:
            proc_code_num = code_adj_proc.shape[0]
        else:
            # fallback đọc file chuẩn hóa nếu meta chưa có proc
            proc_code_num = np.load(f"{dataset_path}/standard/code_adj_proc.npz")["code_adj"].shape[0]

        len_dist = torch.from_numpy(len_dist).to(device)

        # --------------------------------------------------------
        # Base GRU Dual
        # --------------------------------------------------------
        base_gru = BaseGRUDual(diag_code_num, proc_code_num, args.g_hidden_dim, max_len).to(device)
        try:
            base_gru.load(params_path)
        except FileNotFoundError:
            trainer_gru = BaseGRUTrainerDual(args, base_gru, max_len, base_gru_trainloader, params_path)
            trainer_gru.train()
        base_gru.eval()

        # --------------------------------------------------------
        # Generator & Critic Dual
        # --------------------------------------------------------
        generator = GeneratorDual(
            diag_code_num, proc_code_num,
            hidden_dim=args.g_hidden_dim,
            attention_dim=args.g_attention_dim,
            max_len=max_len, device=device
        ).to(device)

        critic = CriticDual(
            diag_code_num, proc_code_num,
            hidden_dim=args.c_hidden_dim,
            generator_hidden_dim=args.g_hidden_dim,
            max_len=max_len
        ).to(device)

        print("Param number:", count_model_params(generator) + count_model_params(critic))

        trainer = GANTrainerDual(
            args,
            generator=generator, critic=critic, base_gru_dual=base_gru,
            train_loader=train_loader, test_loader=test_loader,
            len_dist=len_dist,
            diag_map=code_map,
            proc_map=code_map_proc,
            code_name_map=code_name_map,
            records_path=records_path, params_path=params_path
        )
        trainer.train()

    # ------------------------------------------------------------
    # 🧠 Single Mode (Diagnosis only)
    # ------------------------------------------------------------
    else:
        print("🧠 Running Standard MTGAN (Diagnosis only)")
        train_loader, test_loader, max_len = get_train_test_loader(dataset_path, args.batch_size, device)
        len_dist = torch.from_numpy(len_dist).to(device)

        code_num = len(code_map)
        base_gru = BaseGRU(code_num=code_num, hidden_dim=args.g_hidden_dim, max_len=max_len).to(device)

        try:
            base_gru.load(params_path)
        except FileNotFoundError:
            base_gru_trainloader = get_base_gru_train_loader(dataset_path, args.batch_size, device)
            base_gru_trainer = BaseGRUTrainer(args, base_gru, max_len, base_gru_trainloader, params_path)
            base_gru_trainer.train()
        base_gru.eval()

        generator = Generator(
            code_num=code_num,
            hidden_dim=args.g_hidden_dim,
            attention_dim=args.g_attention_dim,
            max_len=max_len,
            device=device
        ).to(device)

        critic = Critic(
            code_num=code_num,
            hidden_dim=args.c_hidden_dim,
            generator_hidden_dim=args.g_hidden_dim,
            max_len=max_len
        ).to(device)

        print("Param number:", count_model_params(generator) + count_model_params(critic))

        trainer = GANTrainer(
            args,
            generator=generator, critic=critic, base_gru=base_gru,
            train_loader=train_loader, test_loader=test_loader,
            len_dist=len_dist, code_map=code_map, code_name_map=code_name_map,
            records_path=records_path, params_path=params_path
        )
        trainer.train()


# ============================================================
# 🏁 Entry point
# ============================================================
if __name__ == '__main__':
    args = get_training_args()
    train(args)
