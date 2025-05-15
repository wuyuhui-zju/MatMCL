import argparse
import pickle
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

import sys
sys.path.append('..')
from src.data.table_dataset import TableDataset
from src.model_config import config_dict
from src.models.sgpt import SGPT
from src.models.mlp import MechModel
from src.trainer.evaluator import Evaluator
from src.trainer.finetune_trainer import Trainer
from src.trainer.result_tracker import Result_Tracker
from src.trainer.scheduler import PolynomialDecayLR
from src.utils import set_random_seed


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--n_epochs", type=int, default=100)

    parser.add_argument("--config_sgpt", type=str, required=True)
    parser.add_argument("--config_mech", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)

    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)

    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_threads", type=int, default=2)
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open("../datasets/table/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    train_set = TableDataset(split="mech", dataset_type="train", root_path=args.data_path, scaler=scaler)
    val_set = TableDataset(split="mech", dataset_type="val", root_path=args.data_path, scaler=scaler)
    test_set = TableDataset(split="mech", dataset_type="test", root_path=args.data_path, scaler=scaler)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # Model Initialization
    config_sgpt = config_dict[args.config_sgpt]
    config_mech = config_dict[args.config_mech]
    sgpt = SGPT(
        cond_dim=config_sgpt["cond_dim"],
        hidden_dim=config_sgpt["hidden_dim"],
        common_dim=config_sgpt["common_dim"],
        latent_dim=config_sgpt["latent_dim"],
        dropout=config_sgpt["dropout"],
        backbone=config_sgpt["backbone"]
    ).to(device)
    sgpt.load_state_dict(torch.load(args.model_path))

    model = MechModel(
        sgpt,
        latent_dim=config_mech["latent_dim"],
        hidden_dim=config_mech["hidden_dim"],
        out_dim=train_set.n_tasks,
        num_layers=config_mech["num_layers"],
        dropout=args.dropout
    ).to(device)
    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.n_epochs * len(train_set) // 32 // 10, tot_updates=args.n_epochs * len(train_set) // 32, lr=args.lr, end_lr=1e-9, power=1)

    loss_fn = MSELoss()
    evaluator = Evaluator("MechModel", args.metric, train_set.n_tasks, mean=train_set.mean.numpy(), std=train_set.std.numpy())
    final_evaluator_1 = Evaluator("MechModel", "rmse_split", train_set.n_tasks, mean=train_set.mean.numpy(), std=train_set.std.numpy())
    final_evaluator_2 = Evaluator("MechModel", "r2", train_set.n_tasks, mean=train_set.mean.numpy(), std=train_set.std.numpy())
    result_tracker = Result_Tracker(args.metric)
    summary_writer = None

    trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator, final_evaluator_1, final_evaluator_2, result_tracker,
                      summary_writer, device=device, model_name='MechModel',
                      label_mean=train_set.mean.to(device) if train_set.mean is not None else None,
                      label_std=train_set.std.to(device) if train_set.std is not None else None)
    best_train, best_val, best_test, test_final = trainer.fit(model, train_loader, val_loader, test_loader)
    print(f"train: {best_train:.3f}, val: {best_val:.3f}, test: {best_test:.3f}")

    for i in range(len(test_final[0])):
        print(f"test rmse: {test_final[0][i]:.3f}\ttest r2: {test_final[1][i]:.3f}")


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
