import torch
from dataset import NumpyDataset
from model.causalTGAN import CausalTGAN
from torch.utils.data import DataLoader


def train_model(train_options, transform_data, trainer: CausalTGAN):
    dataset = NumpyDataset(transform_data)
    train_data = torch.utils.data.DataLoader(dataset, batch_size=train_options.batch_size, shuffle=True)
    trainer.fit(train_data, train_options, verbose=True)