from .generator_trainer import GeneratorTrainer
from .critic_trainer import CriticTrainer, CriticTrainerDual
from datautils.data_sampler import get_train_sampler, get_train_sampler_dual
from logger import Logger


# ============================================================
# 🧠 GANTrainer — gốc (Diagnosis only)
# ============================================================
class GANTrainer:
    def __init__(self, args,
                 generator, critic, base_gru,
                 train_loader, test_loader,
                 len_dist, code_map, code_name_map,
                 records_path, params_path):

        self.generator = generator
        self.critic = critic
        self.base_gru = base_gru
        self.params_path = params_path
        self.test_loader = test_loader

        # Trainers
        self.g_trainer = GeneratorTrainer(generator, critic,
                                          batch_size=args.batch_size, train_num=args.g_iter,
                                          lr=args.g_lr, betas=(args.betas0, args.betas1),
                                          decay_step=args.decay_step, decay_rate=args.decay_rate)
        self.d_trainer = CriticTrainer(critic, generator, base_gru,
                                       batch_size=args.batch_size, train_num=args.c_iter,
                                       lr=args.c_lr, lambda_=args.lambda_, betas=(args.betas0, args.betas1),
                                       decay_step=args.decay_step, decay_rate=args.decay_rate)

        self.logger = Logger(records_path, generator, code_map, code_name_map,
                             len_dist, train_loader.size, args.save_batch_size)

        self.test_freq = args.test_freq
        self.save_freq = args.save_freq
        self.device = generator.device
        self.iters = args.iteration
        self.train_sampler = get_train_sampler(train_loader, self.device)
        self.batch_size = train_loader.batch_size

    def train(self):
        for i in range(1, self.iters + 1):
            target_codes = self.generator.get_target_codes(self.batch_size)
            real_data, real_lens = self.train_sampler.sample(target_codes)

            d_loss, w_distance = self.d_trainer.step(real_data, real_lens, target_codes)
            g_loss = self.g_trainer.step(target_codes, real_lens)

            self.logger.add_train_point(d_loss, g_loss, w_distance)
            if i % self.test_freq == 0:
                test_d_loss = self.d_trainer.evaluate(self.test_loader, self.device)
                self.logger.add_test_point(test_d_loss)
                line = (f'{i} / {self.iters} iterations: '
                        f'D_Loss -- {d_loss:.6f} -- G_Loss -- {g_loss:.6f} -- '
                        f'W_dist -- {w_distance:.6f} -- Test_D_Loss -- {test_d_loss:.6f}')
                print('\r' + line)
                self.logger.add_log(line)
                self.logger.plot_train()
                self.logger.plot_test()
                self.logger.stat_generation()
            else:
                print(f'\r{i} / {self.iters} iterations: '
                      f'D_Loss -- {d_loss:.6f} -- G_Loss -- {g_loss:.6f} -- W_dist -- {w_distance:.6f}', end='')

            if i % self.save_freq == 0:
                self.generator.save(self.params_path, f'generator.{i}.pt')
                self.critic.save(self.params_path, f'critic.{i}.pt')

        self.generator.save(self.params_path)
        self.critic.save(self.params_path)
        self.logger.save()


# ============================================================
# 🩺 GANTrainerDual — mở rộng cho Diagnosis + Procedure
# ============================================================
class GANTrainerDual:
    def __init__(self, args,
                 generator, critic, base_gru_dual,
                 train_loader, test_loader,
                 len_dist, diag_map, proc_map, code_name_map,
                 records_path, params_path):

        self.generator = generator
        self.critic = critic
        self.base_gru_dual = base_gru_dual
        self.params_path = params_path
        self.test_loader = test_loader

        # Trainers
        from .generator_trainer import GeneratorTrainerDual
        self.g_trainer = GeneratorTrainerDual(generator, critic,
                                              batch_size=args.batch_size, train_num=args.g_iter,
                                              lr=args.g_lr, betas=(args.betas0, args.betas1),
                                              decay_step=args.decay_step, decay_rate=args.decay_rate)
        self.d_trainer = CriticTrainerDual(critic, generator, base_gru_dual,
                                           batch_size=args.batch_size, train_num=args.c_iter,
                                           lr=args.c_lr, lambda_=args.lambda_, betas=(args.betas0, args.betas1),
                                           decay_step=args.decay_step, decay_rate=args.decay_rate)

        # Logger
        self.logger = Logger(records_path, generator, diag_map, code_name_map,
                             len_dist, train_loader.size, args.save_batch_size)

        self.test_freq = args.test_freq
        self.save_freq = args.save_freq
        self.device = generator.device
        self.iters = args.iteration
        self.train_sampler = get_train_sampler_dual(train_loader, self.device)
        self.batch_size = train_loader.batch_size

    def train(self):
        for i in range(1, self.iters + 1):
            target_diag = self.generator.get_target_codes(self.batch_size)
            target_proc = self.generator.get_target_codes_proc(self.batch_size)
            real_diag, real_proc, real_lens = self.train_sampler.sample(target_diag, mode="diag")

            # update Critic and Generator
            d_loss, w_distance = self.d_trainer.step(real_diag, real_proc, real_lens, target_diag, target_proc)
            g_loss = self.g_trainer.step(target_diag, target_proc, real_lens)

            # log
            self.logger.add_train_point(d_loss, g_loss, w_distance)
            if i % self.test_freq == 0:
                test_d_loss = self.d_trainer.evaluate(self.test_loader, self.device)
                self.logger.add_test_point(test_d_loss)
                line = (f'{i} / {self.iters} iterations: '
                        f'D_Loss -- {d_loss:.6f} -- G_Loss -- {g_loss:.6f} -- '
                        f'W_dist -- {w_distance:.6f} -- Test_D_Loss -- {test_d_loss:.6f}')
                print('\r' + line)
                self.logger.add_log(line)
                self.logger.plot_train()
                self.logger.plot_test()
                self.logger.stat_generation()
            else:
                print(f'\r{i} / {self.iters} iterations: '
                      f'D_Loss -- {d_loss:.6f} -- G_Loss -- {g_loss:.6f} -- W_dist -- {w_distance:.6f}', end='')

            # save
            if i % self.save_freq == 0:
                self.generator.save(self.params_path, f'generator.{i}.pt')
                self.critic.save(self.params_path, f'critic.{i}.pt')

        self.generator.save(self.params_path)
        self.critic.save(self.params_path)
        self.logger.save()
