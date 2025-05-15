import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from src.models.gmc_module import TransformerEncoder


def get_affect_network(layers=5):
    embed_dim, attn_dropout = 30, 0.0
    return TransformerEncoder(embed_dim=embed_dim,
                              num_heads=5,
                              layers=min(5, layers),
                              attn_dropout=attn_dropout,
                              relu_dropout=0,
                              res_dropout=0,
                              embed_dropout=0,
                              attn_mask=True)


class JointProcessor(torch.nn.Module):
    def __init__(self, common_dim):
        super(JointProcessor, self).__init__()

        self.common_dim = common_dim

        # Table
        self.proj_t = nn.Conv1d(768, 30, kernel_size=1, padding=0, bias=False)
        self.trans_t_with_i = get_affect_network()
        self.trans_t_mem = get_affect_network()

        # Image
        self.proj_i = nn.Conv1d(768, 30, kernel_size=1, padding=0, bias=False)
        self.trans_i_with_t = get_affect_network()
        self.trans_i_mem = get_affect_network()

        # Projector
        self.proj1 = nn.Linear(30*2, 30*2)
        self.proj2 = nn.Linear(30*2, 30*2)
        self.projector = nn.Linear(30*2, common_dim)

    def forward(self, x_t, x_i):
        """
        table and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_t = x_t.transpose(1, 2)
        x_i = x_i.transpose(1, 2)

        # Project the table/image features
        proj_x_t = self.proj_t(x_t)
        proj_x_i = self.proj_i(x_i)
        proj_x_t = proj_x_t.permute(2, 0, 1)
        proj_x_i = proj_x_i.permute(2, 0, 1)

        # Cross attention: Table with Image
        h_t_with_i = self.trans_t_with_i(proj_x_t, proj_x_i, proj_x_i)  # Dimension (L, N, d_l)
        h_t = h_t_with_i
        h_t = self.trans_t_mem(h_t)
        if type(h_t) == tuple:
            h_t = h_t[0]
        last_h_t = h_t[-1]  # Take the last output for prediction

        # Cross attention: Image with Table
        h_i_with_t = self.trans_i_with_t(proj_x_i, proj_x_t, proj_x_t)
        h_i = h_i_with_t
        h_i = self.trans_i_mem(h_i)
        if type(h_i) == tuple:
            h_i = h_i[0]
        last_h_i = h_i[-1]

        # Concatenate
        last_hs = torch.cat([last_h_t, last_h_i], dim=1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=0.0, training=self.training))
        last_hs_proj += last_hs

        # Project
        return self.projector(last_hs_proj)


class AffectGRUEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, timestep, batch_first=False):
        super(AffectGRUEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          batch_first=batch_first)
        self.projector = nn.Linear(self.hidden_dim*timestep, latent_dim)

        self.ts = timestep

    def forward(self, x):
        batch = len(x)
        input = x.reshape(batch, self.ts, self.input_dim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        return self.projector(output.flatten(start_dim=1))


class AffectEncoder(LightningModule):

    def __init__(self, common_dim, latent_dim):
        super(AffectEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.encode = nn.Linear(common_dim, latent_dim)

    def forward(self, x):
        return F.normalize(self.encode(x), dim=-1)