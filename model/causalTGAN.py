import json
import os
import pickle
import torch
import numpy as np
from torch import autograd
from helper.utils import print_progress, load_options
from model.module.generator import causal_generator
from model.module.discriminator import causalGAN_discriminator
import matplotlib.pyplot as plt


def compute_distance_matrix(X):
    return torch.cdist(X, X, p=2)


def center_distance_matrix(A):
    row_mean = A.mean(dim=1, keepdim=True)
    col_mean = A.mean(dim=0, keepdim=True)
    total_mean = A.mean()
    A_centered = A - row_mean - col_mean + total_mean
    return A_centered


def distance_correlation_matrix(X, Y):
    # Step 1: Compute the distance matrices
    A = compute_distance_matrix(X)
    B = compute_distance_matrix(Y)

    # Step 2: Center the distance matrices
    A_centered = center_distance_matrix(A)
    B_centered = center_distance_matrix(B)

    # Step 3: Compute distance covariance
    dCov = (A_centered * B_centered).sum()

    # Compute distance variances
    dVarX = (A_centered * A_centered).sum()
    dVarY = (B_centered * B_centered).sum()

    # Step 4: Compute distance correlation
    dCor = dCov / torch.sqrt(dVarX * dVarY)
    return dCor


def load_model(model_path, device, feature_info, transfomer, label):
    this_run_folder = model_path
    options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
    train_options, controller_config = load_options(options_file)
    check_point_folder = os.path.join(this_run_folder, 'checkpoints')

    model = CausalTGAN(device, controller_config, feature_info, transfomer, label)
    if model.causal_controller is not None:
        checkpoint = torch.load(os.path.join(check_point_folder, 'causal-TGAN.pyt'), map_location='cpu')
        model.load_checkpoint(checkpoint)

    return model, train_options.experiment_name


class CausalTGAN(object):
    def __init__(self, device, config, feature_info, transformer, label):
        self.config = config
        self.device = device
        self.feature_info = feature_info
        self.transformer = transformer
        self._init_model(label)

    def _init_model(self, label):
        self.causal_controller = causal_generator(self.device, self.config, self.feature_info)
        data_dim = sum(self.feature_info.dim_info.values())
        data_dim = data_dim * self.config.pac_num
        self.discriminator = causalGAN_discriminator(data_dim).to(self.device)
        self.generators_params = self.causal_controller.get_causal_mechanisms_params()
        self.gen_optimizer = torch.optim.Adam(
            self.generators_params, lr=2e-5, betas=(0.5, 0.9),
            weight_decay=1e-6)
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=2e-5,
            betas=(0.5, 0.9), weight_decay=1e-6)
        self.gy = [{"params": self.causal_controller.nodes[label].causal_mechanism.parameters()}]
        self.gy_optimizer = torch.optim.Adam(
            self.gy, lr=2e-5, betas=(0.5, 0.9),
            weight_decay=1e-6)
        self.label = label

    def load_gy_state_dict(self):
        state_dict = {}
        for p in self.causal_controller.nodes[self.label].causal_mechanism.parameters():
            state_dict[p] = self.gen_optimizer.state[p]
        self.gy_optimizer.state = state_dict


    def fit(self, train_data, train_options, verbose=True):
        self._fit_causalGAN(train_data, train_options, verbose=verbose)

    def _fit_causalGAN(self, train_data, train_options, verbose):
        distances = []
        g_loss = []
        d_loss = []
        g_total = []
        for i in range(train_options.number_of_epochs):
            losses = self.train_one_epoch(
                train_data,
                edges=train_options.edges,
                lamb=train_options.lamb,
                label=train_options.label,
                sa=train_options.sensitive_attribute
            )
            distances.append(losses.get("IDC").astype(float))
            g_loss.append(losses.get("G_cost").astype(float))
            d_loss.append(losses.get("D_cost").astype(float))
            g_total.append(losses.get("G_total").astype(float))
            if verbose:
                print('Epoch {}/{}'.format(i, train_options.number_of_epochs))
                print_progress(losses)
                print('-' * 40)

        with open(os.path.join(train_options.runs_folder, 'featureInfo.pickle'), 'wb') as f:
            pickle.dump(self.feature_info, f)
        dic = {
            "G_loss": g_loss,
            "IDC": distances,
            "D_loss": d_loss
        }
        with open(os.path.join(train_options.runs_folder, "losses.json"), "w") as f:
            json.dump(dic, f)

        checkpoint = self.fetch_checkpoint()
        self._save_checkpoint(checkpoint, os.path.join(train_options.runs_folder, 'checkpoints'), 'causal-TGAN.pyt')

    def _save_checkpoint(self, checkpoint, checkpoint_folder, checkpoint_filename):
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
        torch.save(checkpoint, checkpoint_filename)
        print('Saving checkpoint done.')

    def train_one_epoch(self, train_data, sa, label, lamb=0, edges={}):
        G_losses = []
        D_losses = []
        distances = []
        G_total = []
        for steps, data in enumerate(train_data):
            batch_size = data.size(0)
            z = torch.Tensor(np.random.normal(size=(data.shape[1], batch_size, self.config.z_dim))).to(self.device)
            if batch_size % self.config.pac_num != 0:
                continue

            real_data = data.to(self.device)

            D_real = self.discriminator(real_data)
            D_real = D_real.mean()

            fake_data = self.causal_controller.sample(batch_size, z).contiguous()
            D_fake = self.discriminator(fake_data)
            D_fake = D_fake.mean()

            # train with gradient penalty
            gradient_penalty = self.calc_gradient_penalty(real_data, fake_data, self.config.pac_num)

            D_cost = D_fake.mean() - D_real.mean()

            self.discriminator.zero_grad()
            gradient_penalty.backward(retain_graph=True)
            D_cost.backward()
            self.disc_optimizer.step()
            D_losses.append(D_cost.data.cpu().numpy())

            label_pos = self.causal_controller.feature_info.get_position_by_name(label)
            group_pos = self.causal_controller.feature_info.get_position_by_name(sa)
            if (steps + 1) % self.config.D_iter == 0:
                z = torch.normal(0, 1, (len(self.causal_controller.nodes), batch_size, self.config.z_dim),
                                 requires_grad=True).to(self.device)
                fake_data = self.causal_controller.sample(batch_size, z).contiguous()
                G = self.discriminator(fake_data)
                G = G.mean()
                if lamb == 0:
                    # Phase 1
                    d = 0
                    distances.append(torch.tensor(d))
                    G_cost = -G
                    self.gen_optimizer.zero_grad()
                    G_cost.backward()
                    self.gen_optimizer.step()
                else:
                    # Phase 2
                    fake_remove_edges_data, remove_dict = (self.causal_controller.
                                                           remove_edges_sample(batch_size, z, edges))
                    fake_remove_edges_data = fake_remove_edges_data.contiguous()
                    group_diff = (fake_data[:, group_pos] - remove_dict[sa])
                    label_diff = (fake_data[:, label_pos] - fake_remove_edges_data[:, label_pos])
                    d = distance_correlation_matrix(group_diff, label_diff)
                    G = self.discriminator(fake_data)
                    G = G.mean()
                    G_cost = -1.0 * G * (1 - lamb) + lamb * d
                    self.gy_optimizer.zero_grad()
                    G_cost.backward()
                    self.gy_optimizer.step()
                    distances.append(d.detach().cpu().numpy())
                G_losses.append((-G).data.cpu().numpy())
                G_total.append(G_cost.data.cpu().numpy())

        losses = {
            'G_cost': np.mean(G_losses),
            'D_cost': np.mean(D_losses),
            'IDC': np.mean(distances),
            'G_total': np.mean(G_total)
        }
        return losses

    def calc_gradient_penalty(self, real_data, fake_data, pac_num=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac_num, 1, 1, device=self.device)
        alpha = alpha.repeat(1, pac_num, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
                                    gradients.view(-1, pac_num * real_data.size(1)).norm(2, dim=1) - 1
                            ) ** 2).mean() * lambda_

        return gradient_penalty

    def fetch_checkpoint(self):
        if self.causal_controller is not None:
            # fetch the checkpoint of causal mechanisms
            checkpoints = self.causal_controller.fetch_checkpoints()
            checkpoints['discriminator'] = self.discriminator.state_dict()
            checkpoints['gen_optim'] = self.gen_optimizer.state_dict()
            checkpoints['dis_optim'] = self.disc_optimizer.state_dict()

            return checkpoints

    def load_checkpoint(self, checkpoints):
        self.gen_optimizer.load_state_dict(checkpoints['gen_optim'])
        self.disc_optimizer.load_state_dict(checkpoints['dis_optim'])
        self.discriminator.load_state_dict(checkpoints['discriminator'])
        self.causal_controller.restore_from_checkpoints(checkpoints)

    def to_stirng(self):
        return '{}\n{}'.format(str(list(self.causal_controller.nodes.values())[0].causal_mechanism),
                               str(self.discriminator))

    def sample(self, batch_size, z):
        return self.causal_controller.sample(batch_size, z)
