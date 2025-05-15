from torch import nn
from torchvision.models import vit_b_32, ViT_B_32_Weights
import torch


class ViT(nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()
        self.vit = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        self.vit.heads = nn.Linear(768, out_dim)

    def forward(self, _, x_img):
        return self.vit(x_img)


class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        self.vit_model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)

    def forward(self, x):
        x = self.vit_model._process_input(x)
        n = x.shape[0]
        cls_token = self.vit_model.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit_model.encoder(x)
        return x[:, 1:]  # 去掉第一个cls token，保留patch特征


if __name__ == "__main__":
    # model = ViT()
    # print(model(None, torch.randn([16, 3, 224, 224])).size())
    model = ViTFeatureExtractor()
    print(model(torch.randn([16, 3, 224, 224])).size())
