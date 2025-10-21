import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torchvision import transforms  # type: ignore
import numpy as np  # type: ignore
import quadrilinear4d  # type: ignore

from core.module import net
from core.module.clut4d import CLUT4D, TV_4D, identity4d_tensor
from core.dataset.utils import read_3dlut_from_file
from core.module.interpolation import TrilinearInterpolation

##########################################################################
# Helper function for Adaptive Instance Normalization (AdaIN)
##########################################################################


def adaptive_instance_normalization(content_feat, style_feat, eps=1e-5):
    """
    Adjust the content feature’s channel-wise mean and variance to match that of the style.

    Args:
        content_feat: Tensor of shape [B, C, H, W]
        style_feat:   Tensor of shape [B, C, H, W]
    Returns:
        Fused feature of the same shape as content_feat.
    """
    B, C, H, W = content_feat.size()
    # Compute per-channel means and stds for content and style
    content_mean = content_feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
    content_std = content_feat.view(B, C, -1).std(dim=2).view(B, C, 1, 1) + eps
    style_mean = style_feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
    style_std = style_feat.view(B, C, -1).std(dim=2).view(B, C, 1, 1)
    # Normalize the content features, then re-scale and shift using style statistics
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


##########################################################################
# Basic building blocks
##########################################################################


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        nn.init.normal_(self.conv2d.weight, mean=0, std=0.5)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class SplattingBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SplattingBlock2, self).__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, 3, 1)
        self.conv2 = ConvLayer(in_channels, out_channels, 3, 1)

    def forward(self, s):
        s1 = torch.tanh(self.conv1(s))
        s = torch.tanh(self.conv2(s1 + s))
        return s


##########################################################################
# Cross-Attention Context Generator
##########################################################################


class CrossAttentionContextGenerator(nn.Module):
    """
    Enhanced context map generator with improved feature extraction and attention mechanism.
    Maintains the same input/output signature while improving style transfer quality.
    """

    def __init__(
        self, in_channels=3, base_channels=32, attn_channels=64, target_resolution=256
    ):
        super(CrossAttentionContextGenerator, self).__init__()
        self.target_resolution = target_resolution

        # --- Feature Encoders with Instance Norm and Residual Connections ---
        self.content_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.style_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --- Enhanced Feature Projections for Attention ---
        self.proj_query = nn.Sequential(
            nn.Conv2d(base_channels, attn_channels, kernel_size=1),
            nn.InstanceNorm2d(attn_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.proj_key = nn.Sequential(
            nn.Conv2d(base_channels, attn_channels, kernel_size=1),
            nn.InstanceNorm2d(attn_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.proj_value = nn.Sequential(
            nn.Conv2d(base_channels, attn_channels, kernel_size=1),
            nn.InstanceNorm2d(attn_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --- Attention Temperature Control ---
        self.attn_temperature = nn.Parameter(torch.tensor(1.0))

        # --- Channel Attention Module ---
        self.channel_attention = ChannelAttention(attn_channels)

        # --- Enhanced Fusion with Residual Connections ---
        self.modulation_conv = nn.Sequential(
            nn.Conv2d(
                base_channels + attn_channels, base_channels, kernel_size=3, padding=1
            ),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --- Adaptive Feature Modulation ---
        self.fixed_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --- Output Refinement ---
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # --- Improved Upsampling ---
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # --- Initialize weights properly ---
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, content, style):
        orig_size = content.shape[-2:]

        # Resize inputs if needed
        if self.target_resolution is not None and orig_size != self.target_resolution:
            content_small = F.interpolate(
                content,
                size=self.target_resolution,
                mode="bilinear",
                align_corners=True,
            )
            style_small = F.interpolate(
                style, size=self.target_resolution, mode="bilinear", align_corners=True
            )
        else:
            content_small = content
            style_small = style

        # Extract features with enhanced encoders
        feat_content = self.content_encoder(content_small)
        feat_style = self.style_encoder(style_small)

        # Enhanced projections
        Q = self.proj_query(feat_content)
        K = self.proj_key(feat_style)
        V = self.proj_value(feat_style)

        # Reshape for attention
        B, C_attn, H_small, W_small = Q.shape
        num_tokens = H_small * W_small

        Q_flat = Q.view(B, C_attn, num_tokens).permute(0, 2, 1)
        K_flat = K.view(B, C_attn, num_tokens)
        V_flat = V.view(B, C_attn, num_tokens).permute(0, 2, 1)

        # Temperature-controlled attention
        attn_scores = torch.bmm(Q_flat, K_flat)
        attn_scores = attn_scores / (C_attn**0.5) * self.attn_temperature
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention
        aggregated = torch.bmm(attn_weights, V_flat)
        aggregated = aggregated.permute(0, 2, 1).view(B, C_attn, H_small, W_small)

        # Apply channel attention for feature refinement
        aggregated = self.channel_attention(aggregated)

        # Upsample and combine features
        aggregated_full = self.upsample(aggregated)
        feat_content_full = self.upsample(feat_content)

        # Feature fusion with modulation
        combined = torch.cat([feat_content_full, aggregated_full], dim=1)
        modulation = self.modulation_conv(combined)

        # Adaptive feature modulation
        conv_out = self.fixed_conv(feat_content_full)
        dynamic_out = conv_out * modulation + feat_content_full

        # Generate context map
        context_map = self.out_conv(dynamic_out)

        # Resize to original dimensions if needed
        if self.target_resolution is not None and orig_size != self.target_resolution:
            context_map = F.interpolate(
                context_map, size=orig_size, mode="bilinear", align_corners=True
            )

        return context_map


class ResidualBlock(nn.Module):
    """Lightweight residual block with instance normalization"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    """Efficient channel attention module"""

    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale


##########################################################################
# Quadrilinear Interpolation
##########################################################################


class QuadrilinearInterpolation_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()
        output = x.new(x.size()[0], 3, x.size()[2], x.size()[3])
        dim = lut.size()[-1]
        shift = 2 * dim**3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)
        assert 1 == quadrilinear4d.forward(
            lut, x, output, dim, shift, binsize, W, H, batch
        )
        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        ctx.save_for_backward(*variables)
        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        x_grad = x_grad.contiguous()
        output_grad = x_grad.new(
            x_grad.size()[0], 4, x_grad.size()[2], x_grad.size()[3]
        ).fill_(0)
        output_grad[:, 1:, :, :] = x_grad
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
        assert 1 == quadrilinear4d.backward(
            x, output_grad, lut, lut_grad, dim, shift, binsize, W, H, batch
        )
        return lut_grad, output_grad


class QuadrilinearInterpolation_4D(torch.nn.Module):
    def __init__(self):
        super(QuadrilinearInterpolation_4D, self).__init__()

    def forward(self, lut, x):
        return QuadrilinearInterpolation_Function.apply(lut, x)


##########################################################################
# Generator4DLUT_identity
##########################################################################


class Generator4DLUT_identity(nn.Module):
    def __init__(self, dim=17):
        super(Generator4DLUT_identity, self).__init__()
        if dim == 17:
            file = open("SA-LUT/core/module/Identity4DLUT17.txt", "r")
        elif dim == 33:
            file = open("SA-LUT/core/module/Identity4DLUT33.txt", "r")
        lines = file.readlines()
        buffer = np.zeros((3, 2, dim, dim, dim), dtype=np.float32)
        for p in range(0, 2):
            for i in range(0, dim):
                for j in range(0, dim):
                    for k in range(0, dim):
                        n = p * dim * dim * dim + i * dim * dim + j * dim + k
                        x = lines[n].split()
                        buffer[0, p, i, j, k] = float(x[0])
                        buffer[1, p, i, j, k] = float(x[1])
                        buffer[2, p, i, j, k] = float(x[2])
        self.LUT_en = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.QuadrilinearInterpolation_4D = QuadrilinearInterpolation_4D()

    def forward(self, x):
        _, output = self.QuadrilinearInterpolation_4D(self.LUT_en, x)
        return output


##########################################################################
# Enhanced Main StyleTransferNet4D with AdaIN-based Feature Fusion
##########################################################################


class VLog2StyleNet4D(nn.Module):
    def __init__(self, dim=17, num_basis=64):
        super().__init__()
        self.dim = dim
        self.num_basis = num_basis
        print(f"num_basis this time: {num_basis}")

        # 1) Pre-trained VGG for multi-scale feature extraction.
        vgg = net.vgg
        vgg.load_state_dict(torch.load("ckpts/vgg_normalised.pth", weights_only=False))
        self.encoder = net.Net(vgg)

        import os

        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        standard_cube_path = os.path.join(repo_root, "core", "assets", "Standard.cube")
        if not os.path.exists(standard_cube_path):
            # fallback to legacy relative path
            standard_cube_path = os.path.join(repo_root, "assets", "Standard.cube")
        standard_lut = read_3dlut_from_file(standard_cube_path)
        self.register_buffer("standard_lut", standard_lut, persistent=False)

        # 2) Splatting Blocks to further process (fused) features.
        #    Here we assume features at four scales (from shallower to deeper).
        self.SB2 = SplattingBlock2(64, 256)  # for features from layer -4
        self.SB3 = SplattingBlock2(128, 256)  # for features from layer -3
        self.SB4 = SplattingBlock2(256, 256)  # for features from layer -2
        self.SB5 = SplattingBlock2(512, 256)  # for features from layer -1

        # Pooling layers to bring each feature map to a fixed 3×3 size.
        self.pg2 = nn.AdaptiveAvgPool2d(3)
        self.pg3 = nn.AdaptiveAvgPool2d(3)
        self.pg4 = nn.AdaptiveAvgPool2d(3)
        self.pg5 = nn.AdaptiveAvgPool2d(3)

        # Context extractor to optionally guide the 4D LUT fusion.
        self.context_extractor = CrossAttentionContextGenerator(
            target_resolution=(512, 512)
        )

        # 3) Classifier for weight prediction:
        #    The classifier takes the concatenated pooled features and predicts weights
        #    for a linear combination of basis LUTs.
        last_channel = 256 * 4  # four pooled feature maps concatenated
        self.classifier = nn.Sequential(
            nn.Conv2d(last_channel, 512, 3, 2),
            nn.Tanh(),
            nn.Conv2d(512, 512 * 2, 1, 1),
            nn.Tanh(),
            nn.Conv2d(512 * 2, 512, 1, 1),
            nn.Tanh(),
            nn.Conv2d(512, num_basis, 1, 1),
        )

        # 4) Identity 4D LUT and modules for LUT reconstruction.
        id_lut = identity4d_tensor(dim)
        self.register_buffer("id_lut", id_lut)
        self.CLUTs = CLUT4D(num=num_basis, dim=dim)
        self.tvmn = TV_4D(dim=dim)

        # Interpolation modules.
        self.quadrilinear_interpolation = QuadrilinearInterpolation_4D()
        self.trilinear_interpolation = TrilinearInterpolation()

    def forward(self, style, content):
        # === Step 1: Feature Extraction ===
        # Extract multi-scale features for both style and content images.
        style_feats = self.encoder.encode_with_intermediate(style)
        content_feats = self.encoder.encode_with_intermediate(content)

        # === Step 2: Feature Fusion using AdaIN ===
        # Fuse features from four scales (using indices -4, -3, -2, -1).
        fused2 = adaptive_instance_normalization(content_feats[-4], style_feats[-4])
        fused3 = adaptive_instance_normalization(content_feats[-3], style_feats[-3])
        fused4 = adaptive_instance_normalization(content_feats[-2], style_feats[-2])
        fused5 = adaptive_instance_normalization(content_feats[-1], style_feats[-1])

        # Process the fused features through the splatting blocks.
        fused2 = self.SB2(fused2)
        fused3 = self.SB3(fused3)
        fused4 = self.SB4(fused4)
        fused5 = self.SB5(fused5)

        # === Step 3: Weight Prediction ===
        # Apply adaptive pooling to get a consistent dimension from each scale.
        pooled2 = self.pg2(fused2)
        pooled3 = self.pg3(fused3)
        pooled4 = self.pg4(fused4)
        pooled5 = self.pg5(fused5)

        # Concatenate pooled features along the channel dimension.
        combined_feature = torch.cat([pooled2, pooled3, pooled4, pooled5], dim=1)
        # Classify to predict weights for the basis LUTs.
        weight = self.classifier(combined_feature)[:, :, 0, 0]  # shape: [B, num_basis]
        weight = torch.softmax(weight, dim=1)

        # Optionally, apply the standard LUT to the content image (using trilinear interpolation).
        content_lut_applied = self.trilinear_interpolation(
            self.standard_lut.unsqueeze(0), content
        )

        # Use the context extractor (which internally resizes to 256×256) to produce a context map.
        context_map = (
            self.context_extractor(content, style).clamp(0, 1)
            if content is not None
            else None
        )

        # === Step 4: LUT Reconstruction ===
        # Generate the final 4D LUT as a linear combination of learned basis LUTs.
        lut_res, tvmn = self.CLUTs(weight, self.id_lut, self.tvmn)
        fused_lut = torch.clamp(lut_res, min=0, max=1)

        # Apply the reconstructed LUT to the (possibly context-enhanced) content image.
        if self.training:
            if content is not None:
                combined_input = torch.cat([context_map, content], dim=1)
                output = torch.zeros_like(content)
                for b in range(content.size(0)):
                    _, out_b = self.quadrilinear_interpolation(
                        fused_lut[b], combined_input[b : b + 1]
                    )
                    output[b : b + 1] = out_b
                return output, fused_lut, combined_feature, tvmn, context_map
            return None, fused_lut, combined_feature, tvmn, None
        else:
            if content is not None:
                combined_input = torch.cat([context_map, content], dim=1)
                output = torch.zeros_like(content)
                for b in range(content.size(0)):
                    _, out_b = self.quadrilinear_interpolation(
                        fused_lut[b], combined_input[b : b + 1]
                    )
                    output[b : b + 1] = out_b
                return output, fused_lut, context_map
            return None, fused_lut, None


if __name__ == "__main__":
    # Example usage with dummy inputs.
    model_4d = VLog2StyleNet4D(dim=17, num_basis=3).cuda()
    model_4d.train()

    # Create dummy content and style images (e.g. 224×224).
    input_tensor_content = torch.rand((2, 3, 224, 224)).cuda()
    input_tensor_style = torch.rand((2, 3, 224, 224)).cuda()
    outputs = model_4d(input_tensor_style, input_tensor_content)
    output_img, fused_lut, feature, tvmn, context_map = outputs
    tv, mn = tvmn
    print("Output image shape:", output_img.shape)
    print("Fused LUT shape:", fused_lut.shape)
    print("Feature shape:", feature.shape)
    print("TV loss:", tv, "MN loss:", mn)
