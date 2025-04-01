import functools

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class base_continuous_generator(nn.Module):
    def __init__(self, parent_dim, z_dim, feature_dim):
        super(base_continuous_generator, self).__init__()
        self.parent_dim = parent_dim
        self.z_dim = z_dim
        self.feature_dim = feature_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.parent_dim + self.z_dim, 64, normalize=True),
            *block(64, 128),
            *block(128, 128),
            *block(128, 128),
            nn.Linear(128, self.feature_dim)
        )

    def forward(self, noise, parents):
        x = torch.cat([parents, noise], dim=-1) if parents is not None else noise
        x = self.model(x)
        x = torch.tanh(x)
        return x


class base_catogory_generator(nn.Module):
    def __init__(self, parent_dim, z_dim, feature_dim):
        super(base_catogory_generator, self).__init__()
        self.parent_dim = parent_dim
        self.z_dim = z_dim

        self.feature_dim = feature_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.parent_dim + self.z_dim, 64, normalize=True),
            *block(64, 128),
            # *block(128, 128),
            *block(128, 128),
            nn.Linear(128, self.feature_dim)
        )

    def forward(self, noise, parents):
        x = torch.cat([parents, noise], dim=-1) if parents is not None else noise
        x = self.model(x)
        if self.feature_dim == 1:
            x = torch.relu(x)
        else:
            x = nn.functional.gumbel_softmax(x, tau=0.2, hard=False, eps=1e-10, dim=-1)
        return x


def get_continuous_generator(parent_dim, z_dim, feature_dim):
    return base_continuous_generator(parent_dim, z_dim, feature_dim)


def get_catogory_generator(parent_dim, z_dim, feature_dim):
    return base_catogory_generator(parent_dim, z_dim, feature_dim)


class CausalNode(object):
    """
    A node in causal graph.
    Fields in each node: parents node info; causal mechanism (nn.Module)
    """

    def __init__(self, device, z_dim, name, parents, feature_info):
        """
        :param parents: a list of names of parents nodes
        :param z_dim: dim_exogenous + dim_confounder
        """
        self.feature_dim = feature_info.dim_info[name]
        self.parents = parents
        self.parent_dim = sum([feature_info.dim_info[item] for item in parents]) if parents != [] else 0
        self.z_dim = z_dim
        self.device = device
        if feature_info.type_info[name] == 'continuous':
            self.causal_mechanism = get_continuous_generator(self.parent_dim, self.z_dim, self.feature_dim).to(
                self.device)
        else:
            self.causal_mechanism = get_catogory_generator(self.parent_dim, self.z_dim, self.feature_dim).to(
                self.device)
        self.val = None

    def cal_val(self, noises, parents):
        """
        calculate the value of the nodes given its parents
        :param parents: list: parents values. This var is different from self.parents
        :return: the value of this node given its parents
        """
        self.val = self.causal_mechanism(noises, parents)

        return self.val

    def load_checkpoint(self, checkpoint):
        self.causal_mechanism.load_state_dict(checkpoint, strict=False)

    def fetch_checkpoint(self):
        return self.causal_mechanism.state_dict()


class causal_generator(object):
    def __init__(self, device, config, feature_info):

        self.config = config
        self.device = device
        self.causal_graph = config.causal_graph
        self.keys = [self.causal_graph[i][0] for i in range(len(self.causal_graph))]
        self.feature_info = feature_info
        self.init_nodes()

    def init_nodes(self):
        self.nodes = {}
        for node_name, parent_list in self.causal_graph:
            z_dim = self.config.z_dim  # exogenous dim
            self.nodes[node_name] = CausalNode(self.device, z_dim, node_name, parent_list, self.feature_info)
        # topology sorting
        self.name2idx = self.node_order()
        self.idx2name = dict((v, k) for k, v in self.name2idx.items())

    def remove_edges_sample(self, batch_size, z, removed_edges: dict):
        fake_sample = torch.zeros((batch_size, sum(self.feature_info.dim_info.values()))).to(self.device)
        removed_dict = {}
        for idx in range(len(self.nodes)):
            exogenous_var = z[idx]
            current_node = self.nodes[self.idx2name[idx]]
            removed_parents = []
            for pa in current_node.parents:
                if pa in removed_edges.keys() and (
                        isinstance(removed_edges[pa], list) and (self.idx2name[idx] in removed_edges[pa]) or
                        removed_edges[pa] == self.idx2name[idx]):
                    removed_parents.append(pa)
            parents_name = current_node.parents

            def sort_parents(p1, p2):
                pos1 = self.feature_info.get_position_by_name(p1)
                pos2 = self.feature_info.get_position_by_name(p2)
                if pos1[0] < pos2[0]:
                    return -1
                return 1

            parents_name.sort(key=functools.cmp_to_key(sort_parents))
            parents_val = None
            for pa in parents_name:
                pa_val = fake_sample[:, self.feature_info.get_position_by_name(pa)]
                if pa in removed_parents:
                    if pa not in removed_dict.keys():
                        randidx = torch.randperm(batch_size)
                        removed_dict[pa] = pa_val.clone().detach()[randidx, :]
                    pa_val = removed_dict[pa]
                if parents_val is None:
                    parents_val = pa_val.clone()
                else:
                    parents_val = torch.cat((parents_val, pa_val), dim=1)
            val_position = self.feature_info.get_position_by_name(self.idx2name[idx])
            fake_sample[:, val_position] = current_node.cal_val(exogenous_var, parents_val)

        return fake_sample, removed_dict

    def sample(self, batch_size, z):
        fake_sample = torch.zeros((batch_size, sum(self.feature_info.dim_info.values()))).to(self.device)
        for idx in range(len(self.nodes)):
            # exogenous_var = torch.Tensor(np.random.normal(size=(batch_size, self.config.z_dim))).to(self.device)
            exogenous_var = z[idx]
            current_node = self.nodes[self.idx2name[idx]]
            parents_name = current_node.parents
            parents_idx = self.feature_info.get_position_by_name(
                parents_name)  # get feature position (column index) in dataset
            parents_val = fake_sample[:, parents_idx] if parents_idx != [] else None
            val_position = self.feature_info.get_position_by_name(self.idx2name[idx])

            fake_sample[:, val_position] = current_node.cal_val(exogenous_var, parents_val)

        return fake_sample

    def node_order(self):
        check_list = []
        graph = self.causal_graph.copy()
        for node in graph:
            # nodes with no parents
            if node[1] == []:
                check_list.append(node[0])
        while (len(graph) != 0):
            if (graph[0][1] == []):
                graph.remove(graph[0])
                continue
            flag = 1
            for b in graph[0][1]:
                if b not in check_list:
                    flag = 0
            # all parents of the current node is in check_list
            if flag == 1:
                check_list.append(graph[0][0])
                graph.remove(graph[0])
            else:
                tem = graph[0]
                graph.remove(graph[0])
                graph.append(tem)

        name2idx = {}
        for idx, item in enumerate(check_list):
            name2idx[item] = idx
        return name2idx

    def restore_from_checkpoints(self, checkpoints):
        for k in self.nodes.keys():
            self.nodes[k].load_checkpoint(checkpoints[k])

    def fetch_checkpoints(self):
        checkpoints = {}
        for k in self.nodes.keys():
            checkpoints[k] = self.nodes[k].fetch_checkpoint()
        return checkpoints

    def get_causal_mechanisms(self):
        return [node.causal_mechanism for node in self.nodes]

    def get_causal_mechanisms_params(self):
        return [{'params': self.nodes[k].causal_mechanism.parameters()} for k in self.nodes.keys()]

    def set_causal_mechanisms_train(self):
        for k in self.nodes.keys():
            self.nodes[k].causal_mechanism.train()

    def set_causal_mechanisms_eval(self):
        for k in self.nodes.keys():
            self.nodes[k].causal_mechanism.eval()

    def set_causal_mechanisms_zero_grad(self):
        for k in self.nodes.keys():
            self.nodes[k].causal_mechanism.zero_grad()


class Residual(nn.Module):

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class condGAN_generator(nn.Module):
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(condGAN_generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data
