import argparse
import pickle
import os
import torch
from torch.utils.data import DataLoader
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior, DiffusionPriorTrainer
from tqdm import tqdm

import sys
sys.path.append('..')
from src.data.gen_dataset import MultiModalDataset
from src.model_config import config_dict
from src.models import SGPT
from src.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--n_epochs", type=int, default=100)

    parser.add_argument("--config_sgpt", type=str, required=True)
    parser.add_argument("--config_prior", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)

    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_threads", type=int, default=1)
    args = parser.parse_args()
    return args


def get_model(config_sgpt, config_prior, device):
    # Model Initialization
    sgpt = SGPT(
        cond_dim=config_sgpt["cond_dim"],
        hidden_dim=config_sgpt["hidden_dim"],
        common_dim=config_sgpt["common_dim"],
        latent_dim=config_sgpt["latent_dim"],
        dropout=config_sgpt["dropout"],
        backbone=config_sgpt["backbone"]
    ).to(device)
    sgpt.load_state_dict(torch.load(args.model_path))
    sgpt.eval()

    prior_network = DiffusionPriorNetwork(
        dim=config_prior["dim"],
        depth=config_prior["depth"],
        dim_head=config_prior["dim_head"],
        heads=config_prior["heads"]
    ).to(device)

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        image_embed_dim=config_prior["image_embed_dim"],
        timesteps=config_prior["timesteps"],
        cond_drop_prob=config_prior["cond_drop_prob"],
        condition_on_text_encodings=False
    ).to(device)

    return sgpt, diffusion_prior


def main(args):
    config_sgpt = config_dict[args.config_sgpt]
    config_prior = config_dict[args.config_prior]
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open("../datasets/table/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    dataset = MultiModalDataset(root_path=args.data_path, dataset=args.dataset, scaler=scaler)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=args.n_threads, drop_last=True)
    sgpt, diffusion_prior = get_model(config_sgpt, config_prior, device)
    diffusion_prior_trainer = DiffusionPriorTrainer(
        diffusion_prior,
        lr=args.lr,
        wd=args.weight_decay,
        ema_beta=0.99,
        ema_update_after_step=1000,
        ema_update_every=10,
    ).to(device)

    step = 0
    batch_loss_accumulator = []  # To track losses for every 10 steps

    for epoch in range(args.n_epochs):
        for batched_table, batched_images in tqdm(loader, total=len(loader), desc=f"Epoch: {epoch}"):
            batched_table = batched_table.to(device)
            batched_images = batched_images.to(device)

            table_emb = sgpt.encode_table_repr(batched_table, batched_table).detach()
            images_emb = sgpt.encode_image_repr(batched_images, batched_images).detach()

            loss = diffusion_prior_trainer(
                text_embed=table_emb,
                image_embed=images_emb,
            )
            diffusion_prior_trainer.update()

            batch_loss_accumulator.append(loss)
            step += 1
            if step % 10 == 0:
                avg_loss = sum(batch_loss_accumulator) / len(batch_loss_accumulator)
                print(f"Step {step}: Average Loss = {avg_loss:.4f}")
                batch_loss_accumulator = []

    if args.save:
        os.makedirs(args.save_path, exist_ok=True)
        diffusion_prior_trainer.save(os.path.join(args.save_path, f"{args.config_prior}.pth"))


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
