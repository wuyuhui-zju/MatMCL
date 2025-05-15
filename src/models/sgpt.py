from torch import nn
import torch.nn.functional as F

from src.models import MLP, CNN, MultiModalTransformer, MultimodalCNN, TableTransformerWrapper
from src.models.vit import ViT


class Encoder(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super(Encoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.encode = nn.Linear(common_dim, latent_dim)

    def forward(self, x):
        return F.normalize(self.encode(x), dim=-1)


class SGPT(nn.Module):
    def __init__(self, cond_dim, hidden_dim, common_dim, latent_dim, dropout=0, backbone="cnn"):
        super(SGPT, self).__init__()
        if backbone == "cnn":
            self.table_processor = MLP(in_dim=cond_dim, hidden_dim=hidden_dim, out_dim=common_dim)
            self.vision_processor = CNN(hidden_dim=hidden_dim, out_dim=common_dim)
            self.joint_processor = MultimodalCNN(in_dim=cond_dim, hidden_dim=hidden_dim, out_dim=common_dim)

        elif backbone == "transformer":
            self.table_processor = TableTransformerWrapper(in_dim=cond_dim, out_dim=common_dim, dropout=dropout)
            self.vision_processor = ViT(out_dim=common_dim)
            self.joint_processor = MultiModalTransformer(in_dim=cond_dim, out_dim=common_dim, dropout=dropout)

        else:
            raise ValueError("Backbone must be in ['cnn', 'transformer']")

        self.processors = [
            self.table_processor,
            self.vision_processor,
            self.joint_processor,
        ]

        self.encoder = Encoder(common_dim=common_dim, latent_dim=latent_dim)

        # Classifier
        self.proj1 = nn.Linear(latent_dim, latent_dim)
        self.proj2 = nn.Linear(latent_dim, latent_dim)
        self.regressor = nn.Linear(latent_dim, 2)

    def forward_unsupervised(self, x_table, x_img):
        batch_representations = []
        for processor_idx in range(len(self.processors) - 1):
            mod_representations = self.encoder(
                self.processors[processor_idx](x_table, x_img)
            )
            batch_representations.append(mod_representations)

        joint_representation = self.encoder(self.processors[-1](x_table, x_img))
        batch_representations.append(joint_representation)

        return batch_representations

    def encode_table_repr(self, x_table, _):
        table_representation = self.encoder(self.processors[0](x_table, _))
        return table_representation

    def encode_image_repr(self, _, x_img):
        image_representation = self.encoder(self.vision_processor(_, x_img))
        return image_representation

    def encode_fusion_repr(self, x_table, x_img):
        fusion_representation = self.encoder(self.processors[2](x_table, x_img))
        return fusion_representation


if __name__ == "__main__":
    pass