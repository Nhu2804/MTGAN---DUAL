import os
import argparse


def _parse_base_setting(parser):
    group = parser.add_argument_group('base', 'base settings')
    group.add_argument('--seed', default=6669, type=int)
    group.add_argument('--data_path', default='data', type=str)
    group.add_argument('--dataset', default='mimic3', type=str, choices=['mimic3', 'mimic4'])
    group.add_argument('--result_path', default='result', type=str)


def _parse_preprocess_setting(parser):
    group = parser.add_argument_group('preprocess', 'preprocess settings')
    group.add_argument('--from_saved', action='store_true')
    group.add_argument('--train_num', default=6000, type=int)
    group.add_argument('--sample_num', default=10000, type=int, help='for mimic4')

    # 🆕 mới thêm: cho phép bật/tắt xử lý mã thủ thuật (procedure)
    group.add_argument('--use_proc', action='store_true',
                       help='If set, parse and encode procedure codes in addition to diagnosis codes.')
    group.add_argument('--max_proc_per_visit', default=None, type=int,
                       help='Optional cap on number of procedure codes per visit (None = no cap).')


def _parse_model_structure_setting(parser):
    group = parser.add_argument_group('model', 'model structure setting')
    group.add_argument('--g_hidden_dim', default=256, type=int)
    group.add_argument('--g_attention_dim', default=64, type=int)
    group.add_argument('--c_hidden_dim', default=64, type=int)

    # 🆕 thêm các tham số cấu trúc cho nhánh thủ thuật
    group.add_argument('--proc_embed_dim', default=64, type=int,
                       help='Embedding dimension for procedure conditioning vector.')
    group.add_argument('--proc_conditional', action='store_true',
                       help='If set, G_proc depends on both h_t and diagnosis x_diag_t.')
    group.add_argument('--proc_loss_weight', default=1.0, type=float,
                       help='Relative loss weight for procedure reconstruction/generation.')


def _parse_gan_training_setting(parser):
    group = parser.add_argument_group('gan_training', 'gan training setting')
    group.add_argument('--iteration', default=300000, type=int)
    group.add_argument('--batch_size', default=256, type=int)
    group.add_argument('--g_iter', default=1, type=int)
    group.add_argument('--g_lr', default=1e-4, type=float)
    group.add_argument('--c_iter', default=1, type=int)
    group.add_argument('--c_lr', default=1e-5, type=float)
    group.add_argument('--betas0', default=0.5, type=float)
    group.add_argument('--betas1', default=0.9, type=float)
    group.add_argument('--lambda_', default=10, type=float)
    group.add_argument('--decay_rate', default=0.1, type=float)
    group.add_argument('--decay_step', default=100000, type=int)

    # 🆕 nếu bạn muốn tách số iteration riêng cho critic-proc, có thể thêm ở đây sau này.


def _parse_base_gru_setting(parser):
    group = parser.add_argument_group('base_gru_training', 'base gru training setting')
    group.add_argument('--base_gru_epochs', default=200, type=int)
    group.add_argument('--base_gru_lr', default=1e-3, type=float)

    # 🆕 thêm cờ nếu base_gru cũng học từ proc
    group.add_argument('--base_gru_use_proc', action='store_true',
                       help='If set, base GRU input = concat([x_diag, x_proc])')


def _parse_log_setting(parser):
    group = parser.add_argument_group('log', 'log setting')
    group.add_argument('--test_freq', default=1000, type=int)
    group.add_argument('--save_freq', default=1000, type=int)
    group.add_argument('--save_batch_size', default=256, type=int)


def _parse_generate_setting(parser):
    group = parser.add_argument_group('generate', 'generate setting')
    group.add_argument('--batch_size', default=256, type=int)
    group.add_argument('--use_iteration', default=-1, type=int)
    group.add_argument('--number', default=1500, type=int)
    group.add_argument('--upper_bound', default=1e7, type=int)

    # 🆕 cờ sinh thêm thủ thuật
    group.add_argument('--generate_proc', action='store_true',
                       help='If set, generator outputs both diagnosis and procedure codes.')


def get_preprocess_args():
    parser = argparse.ArgumentParser(description='Parameters for Data Preprocess')
    _parse_base_setting(parser)
    _parse_preprocess_setting(parser)
    args = parser.parse_args()
    return args


def get_training_args():
    parser = argparse.ArgumentParser(description='Parameters for training MTGAN')
    _parse_base_setting(parser)
    _parse_model_structure_setting(parser)
    _parse_base_gru_setting(parser)
    _parse_gan_training_setting(parser)
    _parse_log_setting(parser)
     # 🩺 Cho phép bật dual-stream (Diagnosis + Procedure) khi training
    parser.add_argument('--use_proc', action='store_true',
                        help='Enable dual-stream mode with procedure codes during training.')
    args = parser.parse_args()
    return args


def get_generate_args():
    parser = argparse.ArgumentParser(description='Parameters for Generation')
    _parse_base_setting(parser)
    _parse_model_structure_setting(parser)
    _parse_generate_setting(parser)
    # 🩺 thêm dòng này để cho phép chế độ dual-stream khi generate
    parser.add_argument('--use_proc', action='store_true',
                        help='Enable dual-stream generation (Diagnosis + Procedure).')
    args = parser.parse_args()
    return args


def get_paths(args):
    dataset_path = os.path.join(args.data_path, args.dataset)
    result_path = os.path.join(args.result_path, args.dataset)

    # Tạo thư mục output nếu chưa có
    os.makedirs(result_path, exist_ok=True)
    records_path = os.path.join(result_path, 'records')
    os.makedirs(records_path, exist_ok=True)
    params_path = os.path.join(result_path, 'params')
    os.makedirs(params_path, exist_ok=True)

    return dataset_path, records_path, params_path
