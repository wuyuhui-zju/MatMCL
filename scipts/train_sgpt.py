import pickle
import numpy as np
import random
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import sys
sys.path.append('..')
from src.models import SGPT
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.pretrain_trainer import Trainer
from src.utils import set_random_seed
from src.data.multimodal_dataset import MultiModalDataset
from src.model_config import config_dict
from src.trainer.loss import GMCLoss

import warnings
warnings.filterwarnings("ignore")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for Structure-Guided Pre-Training")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--n_epochs", type=int, default=200)

    parser.add_argument("--config_sgpt", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_threads", type=int, default=8)
    args = parser.parse_args()
    return args


def main(args):
    config_sgpt = config_dict[args.config_sgpt]
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open("../datasets/table/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    train_dataset = MultiModalDataset(root_path=args.data_path, dataset_type="train", split_name=config_sgpt['split'], image_size=config_sgpt['image_size'], scaler=scaler)
    test_dataset = MultiModalDataset(root_path=args.data_path, dataset_type="test", split_name=config_sgpt['split'], image_size=config_sgpt['image_size'], scaler=scaler)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False)

    # Model Initialization
    model = SGPT(
        cond_dim=config_sgpt["cond_dim"],
        hidden_dim=config_sgpt["hidden_dim"],
        common_dim=config_sgpt["common_dim"],
        latent_dim=config_sgpt["latent_dim"],
        dropout=config_sgpt["dropout"],
        backbone=config_sgpt["backbone"]
    ).to(device)
    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))

    optimizer = Adam(model.parameters(), lr=config_sgpt["lr"], weight_decay=config_sgpt["weight_decay"])
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.n_epochs * len(train_dataset) // 32 // 10, tot_updates=args.n_epochs * len(train_dataset) // 32, lr=config_sgpt["lr"], end_lr=1e-9, power=1)
    loss_fn = GMCLoss()
    summary_writer = None
    trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, summary_writer, device=device, model_name='SGPT')

    train_result, test_result = trainer.fit(model, train_loader, test_loader)
    print(f"Train loss: {train_result}\tTest loss: {test_result}")


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
