import pickle
import numpy as np
import os
import pandas as pd
from PIL import Image
import argparse
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F

import sys
sys.path.append('..')
from src.models import SGPT
from src.model_config import config_dict

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for retrieval")
    parser.add_argument("--config_sgpt", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gallery_path", type=str, required=True)

    parser.add_argument("--mode", type=str, choices=["retrieve_struct", "retrieve_cond"], required=True, help="Choose the mode: retrieve_struct or retrieve_cond")
    parser.add_argument("--params", type=float, nargs='+', help="List of parameters (for retrieve_struct mode)")
    parser.add_argument("--filename", type=str, help="Filename of image (for retrieve_cond mode)")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results to return")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_threads", type=int, default=8)
    args = parser.parse_args()
    return args


def retrieve_img_given_params(args, params_lst, k=5):
    config_sgpt = config_dict[args.config_sgpt]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open("../datasets/table/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

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

    # preprocess proc_params
    proc_params = np.array(params_lst)[np.newaxis, :].astype(np.float32)
    proc_params = scaler.transform(proc_params)
    dir = np.array([[1]]).astype(np.float32)
    proc_params = np.concatenate([proc_params, dir], axis=1)
    proc_params = torch.tensor(proc_params).to(device)

    # preprocess imgs
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Load and transform the image
    root_path = os.path.join(args.data_path, 'images', args.gallery_path)
    fns = os.listdir(root_path)
    sim_dict = {}
    for fn in fns:
        img = Image.open(os.path.join(root_path, fn))
        img = transformer(img).unsqueeze(0).to(device)
        representations = sgpt.forward_unsupervised(proc_params, img)
        cos_sim = F.cosine_similarity(representations[0], representations[1]).item()
        sim_dict[fn] = cos_sim

    top_ranked_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)[:k])
    return top_ranked_dict


def retrieve_params_given_img(args, fn, k=5):
    config_sgpt = config_dict[args.config_sgpt]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # get scaler
    with open("../datasets/table/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

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

    # preprocess params
    df = pd.read_csv(os.path.join(args.data_path, 'table', args.gallery_path, f'{args.gallery_path}.csv'))
    tabular_data = df.values[:, 1:8][::2]
    id_lst = df.values[:, 0][::2]

    tabular_data_cont = tabular_data[:, :-1]
    tabular_data_cont = scaler.transform(tabular_data_cont)
    tabular_data = np.concatenate([tabular_data_cont, tabular_data[:, -1][:, np.newaxis]], axis=1)
    params = torch.tensor(tabular_data, dtype=torch.float32).to(device)

    # preprocess img
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_path = os.path.join(args.data_path, 'images', args.gallery_path, fn)
    img = Image.open(img_path)
    img = transformer(img).unsqueeze(0).to(device)

    sim_dict = {}
    for id, param in zip(id_lst, params):
        representations = sgpt.forward_unsupervised(param.unsqueeze(0), img)
        cos_sim = F.cosine_similarity(representations[0], representations[1]).item()
        sim_dict[int(id)] = cos_sim

    top_ranked_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)[:k])
    return top_ranked_dict


if __name__ == '__main__':
    args = parse_args()
    print(f"Running mode: {args.mode}, device: {args.device}")

    if args.mode == "retrieve_struct":
        if args.params is None:
            raise ValueError("You must provide --params for retrieve_struct mode.")
        results = retrieve_img_given_params(args, args.params, k=args.topk)  # [0.2,22,18,800,25,36]

    elif args.mode == "retrieve_cond":
        if args.filename is None:
            raise ValueError("You must provide --filename for retrieve_cond mode.")
        results = retrieve_params_given_img(args, args.filename, k=args.topk)  # "78_0.jpg"

    for k, v in results.items():
        print(f"{k}: similarity = {v:.4f}")
