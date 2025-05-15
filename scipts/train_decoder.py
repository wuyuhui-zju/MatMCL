import argparse
import torch
import os
from torch.utils.data import DataLoader
from dalle2_pytorch import Unet, Decoder, DecoderTrainer
from torchvision.utils import save_image
from tqdm import tqdm

import sys
sys.path.append('..')
from src.data.gen_dataset import ImageDataset
from src.model_config import config_dict
from src.models import SGPT


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n_epochs", type=int, default=50)

    parser.add_argument("--config_sgpt", type=str, required=True)
    parser.add_argument("--config_decoder", type=str, required=True)
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


def get_model(config_sgpt, config_decoder, device):
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

    unet = Unet(
        dim=config_decoder['dim'],
        image_embed_dim=config_decoder['image_embed_dim'],
        cond_dim=config_decoder['cond_dim'],
        channels=config_decoder['channels'],
        dim_mults=config_decoder['dim_mults']
    ).to(device)

    decoder = Decoder(
        unet=unet,
        image_size=config_decoder['image_size'],
        timesteps=config_decoder['timesteps'],
        image_cond_drop_prob=config_decoder['image_cond_drop_prob'],
        text_cond_drop_prob=config_decoder['text_cond_drop_prob'],
        learned_variance=False
    ).to(device)

    return sgpt, decoder


def main(args):
    config_sgpt = config_dict[args.config_sgpt]
    config_decoder = config_dict[args.config_decoder]
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(root_path=args.data_path, dataset=args.dataset)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=args.n_threads, drop_last=True)
    gmc, decoder = get_model(config_sgpt, config_decoder, device)
    decoder_trainer = DecoderTrainer(
        decoder,
        lr=args.lr,
        wd=args.weight_decay,
        ema_beta=0.99,
        ema_update_after_step=1000,
        ema_update_every=10,
    ).cuda()

    decoder.train()
    step = 0
    batch_loss_accumulator = []  # To track losses for every 100 steps
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(args.n_epochs):
        for batched_images in tqdm(loader, total=len(loader), desc=f"Epoch: {epoch}"):
            batched_images = batched_images.to(device)
            images_emb = gmc.encode_image_repr(batched_images, batched_images).detach()

            loss = decoder_trainer(
                batched_images,
                image_embed=images_emb,
            )
            decoder_trainer.update(1)

            batch_loss_accumulator.append(loss)
            step += 1
            if step % 100 == 0:
                avg_loss = sum(batch_loss_accumulator) / len(batch_loss_accumulator)
                print(f"Step {step}: Average Loss = {avg_loss:.4f}")
                batch_loss_accumulator = []

            if step % 1000 == 0:
                sample = decoder_trainer.sample(image_embed=images_emb[0].unsqueeze(0))
                save_image(sample, os.path.join(log_dir, f'{epoch}_{step}.png'))

    if args.save:
        os.makedirs(args.save_path, exist_ok=True)
        decoder_trainer.save(os.path.join(args.save_path, f"{args.config_decoder}.pth"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
