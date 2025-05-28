import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torchvision import models  # type: ignore
from torchvision import transforms  # type: ignore
import torch.nn.functional as F  # type: ignore
import numpy as np  # type: ignore


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)


class TAGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=64, num_layers=3):
        super(TAGANDiscriminator, self).__init__()
        self.eps = 1e-7

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.GAP_1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.GAP_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.GAP_3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # text feature
        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

        self.gen_filter = nn.ModuleList(
            [nn.Linear(512, 256 + 1), nn.Linear(512, 512 + 1), nn.Linear(512, 512 + 1)]
        )
        self.gen_weight = nn.Sequential(nn.Linear(512, 3), nn.Softmax(-1))

        self.classifier = nn.Conv2d(512, 1, 4)

        self.apply(init_weights)

    def forward(self, stylized_img, style_img, negative=False):
        stylized_img_feat_1 = self.encoder_1(stylized_img)
        stylized_img_feat_2 = self.encoder_2(stylized_img_feat_1)
        stylized_img_feat_3 = self.encoder_3(stylized_img_feat_2)
        stylized_img_feats = [
            self.GAP_1(stylized_img_feat_1),
            self.GAP_2(stylized_img_feat_2),
            self.GAP_3(stylized_img_feat_3),
        ]
        D = self.classifier(stylized_img_feat_3).squeeze()

        style_img_feat_1 = self.encoder_1(style_img)
        style_img_feat_2 = self.encoder_2(style_img_feat_1)
        style_img_feat_3 = self.encoder_3(style_img_feat_2)
        style_img_feats = [
            self.GAP_1(style_img_feat_1),
            self.GAP_2(style_img_feat_2),
            self.GAP_3(style_img_feat_3),
        ]
        # D_style = self.classifier(style_img_feat_3).squeeze()

        sim = 0
        sim_n = 0
        idx = np.arange(0, stylized_img.size(0))
        idx_n = torch.tensor(np.roll(idx, 1), dtype=torch.long, device="cuda")

        h, w = 32, 32
        for i in range(3):

            bs, dim, _, _ = stylized_img_feats[i].size()

            stylized_img_feat = stylized_img_feats[i]
            style_img_feat = style_img_feats[i]

            stylized_img_feat = torch.nn.functional.interpolate(
                stylized_img_feat, size=(h, w)
            )
            style_img_feat = torch.nn.functional.interpolate(
                style_img_feat, size=(h, w)
            )

            stylized_img_feat = stylized_img_feat.view(bs, dim, -1)
            style_img_feat = style_img_feat.view(bs, dim, -1)

            if negative:
                stylized_img_feat = stylized_img_feat[idx_n]
                sim += torch.sigmoid(
                    torch.bmm(stylized_img_feat.transpose(1, 2), style_img_feat).mean(
                        -1
                    )
                )
            else:
                sim += torch.sigmoid(
                    torch.bmm(stylized_img_feat.transpose(1, 2), style_img_feat).mean(
                        -1
                    )
                )

        sim = sim / 3
        return sim


# Example usage
if __name__ == "__main__":
    # Instantiate the model
    discriminator = TAGANDiscriminator().cuda()

    # Print model architecture
    print(discriminator)

    # Test forward pass
    input_tensor = torch.randn(2, 3, 256, 256).cuda()  # Batch size 1, RGB image 256x256
    output = discriminator(
        torch.randn(2, 3, 256, 256).cuda(), torch.randn(2, 3, 256, 256).cuda()
    )
    print(f"Output shape: {output.shape}")
