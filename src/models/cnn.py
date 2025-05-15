import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class CNN(nn.Module):
    def __init__(self, hidden_dim, out_dim=1):
        super(CNN, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(2048, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, _, x_img):
        image_features = self.resnet(x_img)
        output = self.head(image_features)

        return output
