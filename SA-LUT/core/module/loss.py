import kornia.color as color  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import torchvision  # type: ignore
from torchvision import transforms  # type: ignore


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(weights="VGG16_Weights.DEFAULT")
        features = vgg16.features

        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        return h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG16()
        self.style_weight = 1e5
        self.content_weight = 1
        self.tv_weight = 1e-7
        self.mse_loss = nn.MSELoss()
        # self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (c * h * w)
        return G

    def compute_style_loss(self, input_features, target_features):
        style_loss = 0
        for inp_f, tar_f in zip(input_features, target_features):
            inp_gram = self.gram(inp_f)
            tar_gram = self.gram(tar_f)
            style_loss += self.mse_loss(inp_gram, tar_gram)
        return style_loss

    def compute_content_loss(self, input_features, target_features):
        content_loss = self.mse_loss(input_features, target_features)
        return content_loss

    def tv_loss(self, x):
        b, c, h, w = x.size()
        tv_loss = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(
            torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        )
        return tv_loss / (b * c * h * w)

    def forward(self, content_img, style_img, input_img):
        content_img = self.preprocess(content_img)
        style_img = self.preprocess(style_img)
        input_img = self.preprocess(input_img)

        features_content = self.vgg(content_img)[1]  # relu2_2 for content loss
        features_style = self.vgg(style_img)
        features_input = self.vgg(input_img)

        content_loss = self.compute_content_loss(features_input[1], features_content)
        style_loss = self.compute_style_loss(features_input, features_style)
        # tv_loss = self.tv_loss(input_img)
        # self.tv_weight * tv_loss

        return content_loss, self.style_weight * style_loss


"""
##### Copyright 2021 Mahmoud Afifi.

 If you find this code useful, please cite our paper:

 Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN:
 Controlling Colors of GAN-Generated and Real Images via Color Histograms."
 In CVPR, 2021.

 @inproceedings{afifi2021histogan,
  title={Histo{GAN}: Controlling Colors of {GAN}-Generated and Real Images via
  Color Histograms},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={CVPR},
  year={2021}
}
####
"""

EPS = 1e-6


class RGBuvHistBlock(nn.Module):
    def __init__(
        self,
        h=64,
        insz=128,
        resizing="interpolation",
        method="inverse-quadratic",
        sigma=0.02,
        intensity_scale=True,
        hist_boundary=None,
        green_only=False,
        device="cuda",
    ):
        """Computes the RGB-uv histogram feature of a given image.
        Args:
          h: histogram dimension size (scalar). The default value is 64.
          insz: maximum size of the input image; if it is larger than this size, the
            image will be resized (scalar). Default value is 150 (i.e., 150 x 150
            pixels).
          resizing: resizing method if applicable. Options are: 'interpolation' or
            'sampling'. Default is 'interpolation'.
          method: the method used to count the number of pixels for each bin in the
            histogram feature. Options are: 'thresholding', 'RBF' (radial basis
            function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
          sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
            the sigma parameter of the kernel function. The default value is 0.02.
          intensity_scale: boolean variable to use the intensity scale (I_y in
            Equation 2). Default value is True.
          hist_boundary: a list of histogram boundary values. Default is [-3, 3].
          green_only: boolean variable to use only the log(g/r), log(g/b) channels.
            Default is False.

        Methods:
          forward: accepts input image and returns its histogram feature. Note that
            unless the method is 'thresholding', this is a differentiable function
            and can be easily integrated with the loss function. As mentioned in the
             paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
             training.
        """
        super(RGBuvHistBlock, self).__init__()
        self.h = h
        self.insz = insz
        self.device = device
        self.resizing = resizing
        self.method = method
        self.intensity_scale = intensity_scale
        self.green_only = green_only
        if hist_boundary is None:
            hist_boundary = [-3, 3]
        hist_boundary.sort()
        self.hist_boundary = hist_boundary
        if self.method == "thresholding":
            self.eps = (abs(hist_boundary[0]) + abs(hist_boundary[1])) / h
        else:
            self.sigma = sigma

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            if self.resizing == "interpolation":
                x_sampled = F.interpolate(
                    x, size=(self.insz, self.insz), mode="bilinear", align_corners=False
                )
            elif self.resizing == "sampling":
                inds_1 = torch.LongTensor(
                    np.linspace(0, x.shape[2], self.h, endpoint=False)
                ).to(device=self.device)
                inds_2 = torch.LongTensor(
                    np.linspace(0, x.shape[3], self.h, endpoint=False)
                ).to(device=self.device)
                x_sampled = x.index_select(2, inds_1)
                x_sampled = x_sampled.index_select(3, inds_2)
            else:
                raise Exception(
                    f"Wrong resizing method. It should be: interpolation or sampling. "
                    f"But the given value is {self.resizing}."
                )
        else:
            x_sampled = x

        L = x_sampled.shape[0]  # size of mini-batch
        if x_sampled.shape[1] > 3:
            x_sampled = x_sampled[:, :3, :, :]
        X = torch.unbind(x_sampled, dim=0)
        hists = torch.zeros(
            (x_sampled.shape[0], 1 + int(not self.green_only) * 2, self.h, self.h)
        ).to(device=self.device)
        for l in range(L):
            I = torch.t(torch.reshape(X[l], (3, -1)))
            II = torch.pow(I, 2)
            if self.intensity_scale:
                Iy = torch.unsqueeze(
                    torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS), dim=1
                )
            else:
                Iy = 1
            if not self.green_only:
                Iu0 = torch.unsqueeze(
                    torch.log(I[:, 0] + EPS) - torch.log(I[:, 1] + EPS), dim=1
                )
                Iv0 = torch.unsqueeze(
                    torch.log(I[:, 0] + EPS) - torch.log(I[:, 2] + EPS), dim=1
                )
                diff_u0 = abs(
                    Iu0
                    - torch.unsqueeze(
                        torch.tensor(
                            np.linspace(
                                self.hist_boundary[0], self.hist_boundary[1], num=self.h
                            )
                        ),
                        dim=0,
                    ).to(self.device)
                )
                diff_v0 = abs(
                    Iv0
                    - torch.unsqueeze(
                        torch.tensor(
                            np.linspace(
                                self.hist_boundary[0], self.hist_boundary[1], num=self.h
                            )
                        ),
                        dim=0,
                    ).to(self.device)
                )
                if self.method == "thresholding":
                    diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
                    diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
                elif self.method == "RBF":
                    diff_u0 = (
                        torch.pow(torch.reshape(diff_u0, (-1, self.h)), 2)
                        / self.sigma**2
                    )
                    diff_v0 = (
                        torch.pow(torch.reshape(diff_v0, (-1, self.h)), 2)
                        / self.sigma**2
                    )
                    diff_u0 = torch.exp(-diff_u0)  # Radial basis function
                    diff_v0 = torch.exp(-diff_v0)
                elif self.method == "inverse-quadratic":
                    diff_u0 = (
                        torch.pow(torch.reshape(diff_u0, (-1, self.h)), 2)
                        / self.sigma**2
                    )
                    diff_v0 = (
                        torch.pow(torch.reshape(diff_v0, (-1, self.h)), 2)
                        / self.sigma**2
                    )
                    diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
                    diff_v0 = 1 / (1 + diff_v0)
                else:
                    raise Exception(
                        f"Wrong kernel method. It should be either thresholding, RBF,"
                        f" inverse-quadratic. But the given value is {self.method}."
                    )
                diff_u0 = diff_u0.type(torch.float32)
                diff_v0 = diff_v0.type(torch.float32)
                a = torch.t(Iy * diff_u0)
                hists[l, 0, :, :] = torch.mm(a, diff_v0)

            Iu1 = torch.unsqueeze(
                torch.log(I[:, 1] + EPS) - torch.log(I[:, 0] + EPS), dim=1
            )
            Iv1 = torch.unsqueeze(
                torch.log(I[:, 1] + EPS) - torch.log(I[:, 2] + EPS), dim=1
            )
            diff_u1 = abs(
                Iu1
                - torch.unsqueeze(
                    torch.tensor(
                        np.linspace(
                            self.hist_boundary[0], self.hist_boundary[1], num=self.h
                        )
                    ),
                    dim=0,
                ).to(self.device)
            )
            diff_v1 = abs(
                Iv1
                - torch.unsqueeze(
                    torch.tensor(
                        np.linspace(
                            self.hist_boundary[0], self.hist_boundary[1], num=self.h
                        )
                    ),
                    dim=0,
                ).to(self.device)
            )

            if self.method == "thresholding":
                diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
                diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
            elif self.method == "RBF":
                diff_u1 = (
                    torch.pow(torch.reshape(diff_u1, (-1, self.h)), 2) / self.sigma**2
                )
                diff_v1 = (
                    torch.pow(torch.reshape(diff_v1, (-1, self.h)), 2) / self.sigma**2
                )
                diff_u1 = torch.exp(-diff_u1)  # Gaussian
                diff_v1 = torch.exp(-diff_v1)
            elif self.method == "inverse-quadratic":
                diff_u1 = (
                    torch.pow(torch.reshape(diff_u1, (-1, self.h)), 2) / self.sigma**2
                )
                diff_v1 = (
                    torch.pow(torch.reshape(diff_v1, (-1, self.h)), 2) / self.sigma**2
                )
                diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
                diff_v1 = 1 / (1 + diff_v1)

            diff_u1 = diff_u1.type(torch.float32)
            diff_v1 = diff_v1.type(torch.float32)
            a = torch.t(Iy * diff_u1)
            if not self.green_only:
                hists[l, 1, :, :] = torch.mm(a, diff_v1)
            else:
                hists[l, 0, :, :] = torch.mm(a, diff_v1)

            if not self.green_only:
                Iu2 = torch.unsqueeze(
                    torch.log(I[:, 2] + EPS) - torch.log(I[:, 0] + EPS), dim=1
                )
                Iv2 = torch.unsqueeze(
                    torch.log(I[:, 2] + EPS) - torch.log(I[:, 1] + EPS), dim=1
                )
                diff_u2 = abs(
                    Iu2
                    - torch.unsqueeze(
                        torch.tensor(
                            np.linspace(
                                self.hist_boundary[0], self.hist_boundary[1], num=self.h
                            )
                        ),
                        dim=0,
                    ).to(self.device)
                )
                diff_v2 = abs(
                    Iv2
                    - torch.unsqueeze(
                        torch.tensor(
                            np.linspace(
                                self.hist_boundary[0], self.hist_boundary[1], num=self.h
                            )
                        ),
                        dim=0,
                    ).to(self.device)
                )
                if self.method == "thresholding":
                    diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
                    diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
                elif self.method == "RBF":
                    diff_u2 = (
                        torch.pow(torch.reshape(diff_u2, (-1, self.h)), 2)
                        / self.sigma**2
                    )
                    diff_v2 = (
                        torch.pow(torch.reshape(diff_v2, (-1, self.h)), 2)
                        / self.sigma**2
                    )
                    diff_u2 = torch.exp(-diff_u2)  # Gaussian
                    diff_v2 = torch.exp(-diff_v2)
                elif self.method == "inverse-quadratic":
                    diff_u2 = (
                        torch.pow(torch.reshape(diff_u2, (-1, self.h)), 2)
                        / self.sigma**2
                    )
                    diff_v2 = (
                        torch.pow(torch.reshape(diff_v2, (-1, self.h)), 2)
                        / self.sigma**2
                    )
                    diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
                    diff_v2 = 1 / (1 + diff_v2)
                diff_u2 = diff_u2.type(torch.float32)
                diff_v2 = diff_v2.type(torch.float32)
                a = torch.t(Iy * diff_u2)
                hists[l, 2, :, :] = torch.mm(a, diff_v2)

        # normalization
        hists_normalized = hists / (
            ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS
        )

        return hists_normalized


class HistLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_block = RGBuvHistBlock()

    def forward(self, img_pred, gt):
        hist_pred = self.hist_block(img_pred)
        hist_gt = self.hist_block(gt)
        loss_hist = (
            1
            / np.sqrt(2.0)
            * (
                torch.sqrt(
                    torch.sum(torch.pow(torch.sqrt(hist_gt) - torch.sqrt(hist_pred), 2))
                )
            )
            / hist_pred.shape[0]
        )
        return loss_hist


class LuminanceLoss(nn.Module):
    def __init__(self):
        super(LuminanceLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, gt):
        # Assume pred and gt are in shape (batch_size, channels, height, width) and are RGB
        # Convert RGB to luminance using the formula
        pred_luminance = (
            0.2989 * pred[:, 0, :, :]
            + 0.5870 * pred[:, 1, :, :]
            + 0.1140 * pred[:, 2, :, :]
        )
        gt_luminance = (
            0.2989 * gt[:, 0, :, :] + 0.5870 * gt[:, 1, :, :] + 0.1140 * gt[:, 2, :, :]
        )

        # Compute the loss using MSE between luminance images
        loss = self.mse_loss(pred_luminance, gt_luminance)
        return loss


class LABLoss(nn.Module):
    def __init__(self, l_weight=1.0, ab_weight=1.0):
        super(LABLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l_weight = l_weight
        self.ab_weight = ab_weight

    def forward(self, pred, gt):
        # Convert RGB to Lab using Kornia
        pred_lab = color.rgb_to_lab(pred)
        gt_lab = color.rgb_to_lab(gt)

        # Extract L, a, and b channels from both pred and gt
        pred_l = pred_lab[:, 0, :, :]  # L channel
        pred_a = pred_lab[:, 1, :, :]  # a channel
        pred_b = pred_lab[:, 2, :, :]  # b channel

        gt_l = gt_lab[:, 0, :, :]  # L channel
        gt_a = gt_lab[:, 1, :, :]  # a channel
        gt_b = gt_lab[:, 2, :, :]  # b channel

        # Standardize L, a, b to [0, 1]
        pred_l = pred_l / 100.0  # L in [0, 1]
        pred_a = (pred_a + 128) / 255.0  # a in [0, 1]
        pred_b = (pred_b + 128) / 255.0  # b in [0, 1]

        gt_l = gt_l / 100.0  # L in [0, 1]
        gt_a = (gt_a + 128) / 255.0  # a in [0, 1]
        gt_b = (gt_b + 128) / 255.0  # b in [0, 1]

        # Compute the loss for each channel
        loss_l = self.mse_loss(pred_l, gt_l)  # Luminance loss
        loss_a = self.mse_loss(pred_a, gt_a)  # Chromatic a loss
        loss_b = self.mse_loss(pred_b, gt_b)  # Chromatic b loss

        # Combine losses using specified weights
        total_loss = self.l_weight * loss_l + self.ab_weight * (loss_a + loss_b) / 2.0

        return total_loss


if __name__ == "__main__":
    # Example usage
    pred = torch.randn(
        4, 3, 256, 256
    )  # Batch of predicted images (batch_size, channels, height, width)
    gt = torch.randn(4, 3, 256, 256)  # Batch of ground truth images

    luminance_loss = LuminanceLoss()
    loss_value = luminance_loss(pred, gt)
    print("Luminance Loss:", loss_value.item())

    lab_loss = LABLoss(l_weight=1.0, ab_weight=0.5)
    loss_value = lab_loss(pred, gt)
    print("Weighted Lab Loss:", loss_value.item())
