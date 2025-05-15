import copy
import os
import pickle

import pandas as pd
from PIL import Image
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPriorTrainer, Decoder, Unet, DecoderTrainer

import sys
sys.path.append('..')
from src.models import SGPT
from src.models.mlp import MechModel
from src.model_config import config_dict
from src.models.diffusion_prior import DiffusionPriorEmbed
from src.data.table_dataset import TableDataset


class MatMCL(nn.Module):
    def __init__(
            self,
            config_sgpt,
            config_mech=None,
            config_prior=None,
            config_decoder=None,
            weight_paths=None,
            save_path="../generated/",
            device="cuda"
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load scaler
        with open("../datasets/table/scaler.pkl", "rb") as file:
            self.scaler = pickle.load(file)

        # Backbone
        self.sgpt = None
        if config_sgpt and weight_paths and "sgpt" in weight_paths:
            self.sgpt = SGPT(
                cond_dim=config_sgpt["cond_dim"],
                hidden_dim=config_sgpt["hidden_dim"],
                common_dim=config_sgpt["common_dim"],
                latent_dim=config_sgpt["latent_dim"],
                dropout=config_sgpt["dropout"],
                backbone=config_sgpt["backbone"]
            ).to(self.device)

            self.sgpt.load_state_dict(torch.load(weight_paths["sgpt"]))
            self.sgpt.eval()

        # Mech model (property predictor)
        self.mech_model = None
        if config_mech and weight_paths and "mech" in weight_paths:
            self.mech_model = MechModel(
                copy.deepcopy(self.sgpt),
                latent_dim=config_mech["latent_dim"],
                hidden_dim=config_mech["hidden_dim"],
                out_dim=4,
                num_layers=config_mech["num_layers"],
                dropout=0.2
            ).to(self.device)
            self.mech_model.load_state_dict(torch.load(weight_paths["mech"]))
            self.mech_model.eval()

        # Diffusion prior
        self.prior = None
        if config_prior and weight_paths and "prior" in weight_paths:
            prior_network = DiffusionPriorNetwork(
                dim=config_prior["dim"],
                depth=config_prior["depth"],
                dim_head=config_prior["dim_head"],
                heads=config_prior["heads"]
            ).to(device)

            diffusion_prior = DiffusionPriorEmbed(
                net=prior_network,
                image_embed_dim=config_prior["image_embed_dim"],
                timesteps=config_prior["timesteps"],
                cond_drop_prob=config_prior["cond_drop_prob"],
                condition_on_text_encodings=False
            ).to(device)

            self.prior = DiffusionPriorTrainer(
                diffusion_prior,
                lr=1e-4,
                wd=1e-2,
                ema_beta=0.99,
                ema_update_after_step=1000,
                ema_update_every=10,
            ).to(device)
            self.prior.load(weight_paths["prior"])

        # Decoder
        self.decoder = None
        if config_decoder and weight_paths and "decoder" in weight_paths:
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

            self.decoder = DecoderTrainer(
                decoder,
                lr=1e-4,
                wd=1e-2,
                ema_beta=0.99,
                ema_update_after_step=1000,
                ema_update_every=10,
            ).to(device)

            self.decoder.load(weight_paths["decoder"])

        train_set = TableDataset(split="mech", dataset_type="train", root_path="../datasets", scaler=self.scaler)
        self.mean = train_set.mean.numpy()
        self.std = train_set.std.numpy()

        self.transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.save_path = save_path

    def preprocess_params(self, proc_params: list[float], dir: int = 1) -> torch.tensor:
        """
        Normalize input parameters and append an integer direction flag.

        Args:
            param_list (list[float]): List of raw structure parameters (before scaling).
            dir (int): Direction indicator, typically 1 or 0.

        Returns:
            torch.tensor: Normalized parameter tensor with direction appended, shape [1, N+1]
        """
        proc_params = np.array(proc_params)[np.newaxis, :].astype(np.float32)
        proc_params = self.scaler.transform(proc_params)
        dir = np.array([[dir]]).astype(np.float32)
        proc_params = np.concatenate([proc_params, dir], axis=1)
        proc_params = torch.tensor(proc_params).to(self.device)
        return proc_params

    def predict_property(self, proc_params: list):
        """conditions -> property"""
        if self.mech_model is None:
            raise ValueError("MechModel not initialized.")

        results = []
        for dir in [0, 1]:  # Transverse/Longitudinal
            proc_params_processed = self.preprocess_params(proc_params, dir=dir)
            pred = self.mech_model(proc_params_processed).cpu().detach().numpy()
            pred = pred * self.std + self.mean
            results.append(pred.squeeze().tolist())
        df = pd.DataFrame(results, columns=['Fracture(MPa)', 'Elongation(%)', 'Elastic modulus(MPa)', 'Tangent modulus(MPa)'])
        print(f"Predicted mechanical properties:\n{df}")
        return df

    def retrieve_structure_by_params(
            self,
            proc_params: list,
            gallery_images: dict[str, PIL.Image.Image],
            topk: int = 6
    ):
        """conditions -> structure"""

        if self.sgpt is None:
            raise ValueError("SGPT is not initialized.")

        proc_params = self.preprocess_params(proc_params)  # [1, 7]
        query_repr = self.sgpt.encode_table_repr(proc_params, proc_params)  # [1, dim]
        sim_dict = {}
        for fn, img in gallery_images.items():
            if not isinstance(img, torch.Tensor):
                img = self.transformer(img).unsqueeze(0)  # transform PIL.Image to Tensor
            img_tensor = img.to(self.device)

            structure_repr = self.sgpt.encode_image_repr(img_tensor, img_tensor)
            sim = F.cosine_similarity(query_repr, structure_repr).item()
            sim_dict[fn] = sim

        return dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)[:topk])

    def retrieve_params_by_structure(
            self,
            query_img: PIL.Image.Image,
            gallery_params: list[list],
            topk: int = 6
    ):
        """Structure -> conditions"""

        query_img_tensor = self.transformer(query_img).unsqueeze(0).to(self.device)
        query_repr = self.sgpt.encode_image_repr(query_img_tensor, query_img_tensor)
        sim_dict = {}
        for idx, proc_params in enumerate(gallery_params):
            proc_params = self.preprocess_params(proc_params)  # [1, 7]
            cond_repr = self.sgpt.encode_table_repr(proc_params, proc_params)  # [1, dim]
            sim = F.cosine_similarity(query_repr, cond_repr).item()
            sim_dict[idx] = sim

        return dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)[:topk])

    def generate_structure(self, proc_params: list, n_num: int = 3):
        """Conditions -> Structure"""
        if self.prior is None or self.decoder is None:
            raise ValueError("Prior or Decoder not initialized.")

        print(f"\nGenerating Structure")
        proc_params = self.preprocess_params(proc_params)  # [1, 7]
        cond_embed = self.sgpt.encode_table_repr(proc_params, proc_params).detach()  # [1, dim]

        os.makedirs(self.save_path, exist_ok=True)
        for i in range(n_num):
            structure_emb_pred = self.prior.sample(cond_embed)
            structure = self.decoder.sample(image_embed=structure_emb_pred)
            save_image(structure, os.path.join(self.save_path, f"gen_{i}.png"))


if __name__ == "__main__":
    matmcl = MatMCL(
        config_sgpt=config_dict['sgpt'],
        config_mech=config_dict['mech'],
        config_prior=config_dict['prior'],
        config_decoder=config_dict['decoder'],
        weight_paths=config_dict['weight_paths'],
        device="cuda:0"
    )

    params = {
        "flow_rate": 0.4,
        "concentration": 16,
        "voltage": 16,
        "rotation_spped": 1600,
        "temperature": 22,
        "humidity": 41
    }

    matmcl.predict_property(list(params.values()))
    matmcl.generate_structure(list(params.values()))
