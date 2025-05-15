import torch
import torch.nn as nn
from src.models.ft_transformer import FTTransformer
from src.models.gmc_networks import JointProcessor
from src.models.vit import ViTFeatureExtractor


class MultiModalTransformer(nn.Module):
    def __init__(self, in_dim=7, out_dim=2, dropout=0):
        super(MultiModalTransformer, self).__init__()
        self.table_feat_extractor = FTTransformer(
            n_cont_features=in_dim-1,
            cat_cardinalities=[2],
            d_out=None,
            n_blocks=3,
            d_block=768,
            attention_n_heads=8,
            attention_dropout=0,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=dropout,
            residual_dropout=dropout,
            )
        self.img_feat_extractor = ViTFeatureExtractor()
        self.pad_layer = nn.ConstantPad1d((0, 42), 0)
        self.joint_processor = JointProcessor(common_dim=out_dim)

    def forward(self, x_table, x_img):
        # Extract table feature
        x_cont = x_table[:, :-1]
        x_cat = x_table[:, -1].unsqueeze(-1).long()
        x_t = self.table_feat_extractor(x_cont, x_cat)[:, 1:]

        x_t = x_t.permute(0, 2, 1)  # [batch_size, common_dim, seq_len_t=8]
        x_t = self.pad_layer(x_t)  # [batch_size, common_dim, seq_len=49]
        x_t = x_t.permute(0, 2, 1)  # [batch_size, seq_len=49, common_dim]

        # Extract image feature
        x_i = self.img_feat_extractor(x_img)

        return self.joint_processor(x_t, x_i)


if __name__ == "__main__":
    x_table = torch.rand([16, 7])
    x_img = torch.randn([16, 3, 224, 224])
    model = MultiModalTransformer()
    output = model(x_table, x_img)
    print(output.size())