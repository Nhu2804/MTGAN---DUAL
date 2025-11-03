import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from generation.generate import generate_ehr
from generation.stat_ehr import get_basic_statistics, get_top_k_codes


class Logger:
    def __init__(self, plot_path, generator, code_map, code_name_map, len_dist,
                 save_number, save_batch_size,
                 proc_map=None):
        self.plot_path = plot_path
        self.generator = generator
        self.save_number = save_number
        self.save_batch_size = save_batch_size
        self.device = generator.device

        self.plots = {
            'train': {
                'd_loss': {'data': [], 'title': 'Discriminator Loss'},
                'g_loss': {'data': [], 'title': 'Generator Loss'},
                'w_distance': {'data': [], 'title': 'Wasserstein Distance'}
            },
            'test': {
                'test_d_loss': {'data': [], 'title': 'Test Discriminator Loss'}
            },
            'generate': {
                'gen_code_type': {'data': [], 'title': 'Generated Code Type'},
                'gen_code_num': {'data': [], 'title': 'Generated Code Number'},
                'gen_avg_code_num': {'data': [], 'title': 'Generated Avg Code Number'}
            }
        }

        self.logfile = open(os.path.join(plot_path, 'output.log'), 'w', encoding='utf-8')

        self.code_name_map = code_name_map
        self.icode_map = {v: k for k, v in code_map.items()}
        self.proc_map = proc_map
        self.icode_map_proc = {v: k for k, v in proc_map.items()} if proc_map else None
        self.len_dist = len_dist

    # ============================================================
    #  Utility helpers
    # ============================================================
    def append_point(self, key, loss_type, loss):
        self.plots[key][loss_type]['data'].append(loss)

    def add_train_point(self, d_loss, g_loss, w_distance):
        self.append_point('train', 'd_loss', d_loss)
        self.append_point('train', 'g_loss', g_loss)
        self.append_point('train', 'w_distance', w_distance)

    def add_test_point(self, test_d_loss):
        self.append_point('test', 'test_d_loss', test_d_loss)

    def add_gen_point(self, gen_code_type, gen_code_num, gen_avg_code_num):
        self.append_point('generate', 'gen_code_type', gen_code_type)
        self.append_point('generate', 'gen_code_num', gen_code_num)
        self.append_point('generate', 'gen_avg_code_num', gen_avg_code_num)

    # ============================================================
    #  Plot helpers
    # ============================================================
    def plot_dict(self, key, x):
        for item in self.plots[key].values():
            y, title = item['data'], item['title']
            plt.clf()
            plt.plot(x, y)
            plt.xlabel('Iteration')
            plt.ylabel(title)
            plt.savefig(os.path.join(self.plot_path, title.replace(' ', '_') + '.png'))

    def plot_train(self):
        x = np.arange(1, len(self.plots['train']['d_loss']['data']) + 1)
        self.plot_dict('train', x)

    def plot_test(self):
        train_points = len(self.plots['train']['d_loss']['data'])
        test_points = len(self.plots['test']['test_d_loss']['data'])
        step = max(1, train_points // max(1, test_points))
        x = np.arange(1, test_points + 1) * step
        self.plot_dict('test', x)

    def plot_gen(self):
        train_points = len(self.plots['train']['d_loss']['data'])
        gen_points = len(self.plots['generate']['gen_code_type']['data'])
        step = max(1, train_points // max(1, gen_points))
        x = np.arange(1, gen_points + 1) * step
        self.plot_dict('generate', x)

    # ============================================================
    #  Core: evaluate and log generated samples
    # ============================================================
    def stat_generation(self):
        # Sinh dữ liệu
        gen_result = generate_ehr(self.generator, self.save_number,
                                  self.len_dist, self.save_batch_size)

        # ---------------- Dual mode ----------------
        if isinstance(gen_result, tuple) and len(gen_result) == 2:
            (fake_x_diag, fake_x_proc), fake_lens = gen_result
            # Diagnosis stats
            n_types_d, n_codes_d, n_visits_d, avg_code_num_d, avg_visit_num_d = \
                get_basic_statistics(fake_x_diag, fake_lens)
            log_d = (f"[DIAG] Generated {self.save_number} samples — types: {n_types_d} "
                     f"— codes: {n_codes_d} — avg codes/visit: {avg_code_num_d:.3f} "
                     f"— avg visits/patient: {avg_visit_num_d:.3f}")
            print(log_d); self.add_log(log_d)

            # Procedure stats
            n_types_p, n_codes_p, n_visits_p, avg_code_num_p, avg_visit_num_p = \
                get_basic_statistics(fake_x_proc, fake_lens)
            log_p = (f"[PROC] Generated {self.save_number} samples — types: {n_types_p} "
                     f"— codes: {n_codes_p} — avg codes/visit: {avg_code_num_p:.3f} "
                     f"— avg visits/patient: {avg_visit_num_p:.3f}")
            print(log_p); self.add_log(log_p)

            # Top-K
            get_top_k_codes(fake_x_diag, fake_lens, self.icode_map,
                            self.code_name_map, top_k=10, file=self.logfile, prefix="Diagnosis")
            if self.proc_map:
                get_top_k_codes(fake_x_proc, fake_lens, self.icode_map_proc,
                                self.code_name_map, top_k=10, file=self.logfile, prefix="Procedure")
            self.add_log('\n')

            # Summary for generator plot (use Diagnosis as main)
            self.add_gen_point(n_types_d, n_codes_d, avg_code_num_d)
            self.plot_gen()

        # ---------------- Single mode ----------------
        else:
            fake_x, fake_lens = gen_result
            n_types, n_codes, n_visits, avg_code_num, avg_visit_num = \
                get_basic_statistics(fake_x, fake_lens)
            log = (f"Generated {self.save_number} samples — code types: {n_types} "
                   f"— code num: {n_codes} — avg code num: {avg_code_num:.4f}, "
                   f"avg visit len: {avg_visit_num:.4f}")
            print(log); self.add_log(log)

            get_top_k_codes(fake_x, fake_lens, self.icode_map,
                            self.code_name_map, top_k=10, file=self.logfile, prefix="Diagnosis")
            self.add_log('\n')

            self.add_gen_point(n_types, n_codes, avg_code_num)
            self.plot_gen()

        self.logfile.flush()

    # ============================================================
    #  Logging helpers
    # ============================================================
    def add_log(self, line):
        if isinstance(line, str):
            self.logfile.write(line + '\n')
        elif isinstance(line, list):
            self.logfile.writelines([l + '\n' for l in line])
        self.logfile.flush()

    def save(self):
        pickle.dump(self.plots, open(os.path.join(self.plot_path, 'history.log'), 'wb'))
