import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class MultimodalCNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super(MultimodalCNN, self).__init__()

        # ResNet for image processing
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(2048, hidden_dim)

        # MLP for tabular data processing
        self.tabular_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Fusion and final head
        self.fusion_head = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x_tabular, x_img):
        image_features = self.resnet(x_img)
        tabular_features = self.tabular_mlp(x_tabular)
        combined_features = torch.cat((image_features, tabular_features), dim=1)

        output = self.fusion_head(combined_features)

        return output


if __name__ == "__main__":
    pass
