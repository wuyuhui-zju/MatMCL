import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, num_layers=5):
        super(MLP, self).__init__()

        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2 (input and output layers).")

        if num_layers == 1:
            predictor = nn.Linear(in_dim, out_dim)
        else:
            predictor = nn.ModuleList()
            predictor.append(nn.Linear(in_dim, hidden_dim))
            predictor.append(nn.ReLU())
            for _ in range(num_layers - 2):
                predictor.append(nn.Linear(hidden_dim, hidden_dim))
                predictor.append(nn.ReLU())
            predictor.append(nn.Linear(hidden_dim, out_dim))
        self.tabular_mlp = nn.Sequential(*predictor)

    def forward(self, x_tabular, _):
        output = self.tabular_mlp(x_tabular)
        return output


class MechModel(nn.Module):
    def __init__(self, sgpt, latent_dim, hidden_dim, out_dim, num_layers, dropout=0):
        """
        Initialize the MLP model.

        Args:
            input_dim (int): The size of the input features.
            hidden_dim (int): The size of the hidden layer features.
            output_dim (int): The size of the output features.
            num_layers (int): The total number of layers (including input and output layers).
        """
        super(MechModel, self).__init__()
        self.sgpt = sgpt

        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2 (input and output layers).")

        if num_layers == 1:
            predictor = nn.Linear(latent_dim, out_dim)
        else:
            predictor = nn.ModuleList()
            predictor.append(nn.Linear(latent_dim, hidden_dim))
            predictor.append(nn.Dropout(dropout))
            predictor.append(nn.ReLU())
            for _ in range(num_layers - 2):
                predictor.append(nn.Linear(hidden_dim, hidden_dim))
                predictor.append(nn.Dropout(dropout))
                predictor.append(nn.ReLU())
            predictor.append(nn.Linear(hidden_dim, out_dim))
        self.predictor = nn.Sequential(*predictor)

    def forward(self, x):
        """
        Forward pass for the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        feat = self.sgpt.encode_table_repr(x, x).detach()
        return self.predictor(feat)
