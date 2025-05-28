import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from transformers import ViTModel  # type: ignore

from core.module.lut import identity3d_tensor
from core.module.lut import CLUT, TVMN


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class Style2VLogImage2ImageNet(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        width=32,
        middle_blk_num=6,
        enc_blk_nums=[1, 1, 2, 4],
        dec_blk_nums=[1, 1, 1, 1],
    ):
        super().__init__()
        self.intro = nn.Conv2d(
            in_channels=in_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=out_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, style, style_2=None):

        style = (style - 0.5) * 2
        if style_2 is not None:
            style_2 = (style_2 - 0.5) * 2

        B, C, H, W = style.shape
        style = self.check_image_size(style)

        x = self.intro(
            torch.cat((style, style_2), dim=1) if style_2 is not None else style
        )

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + style
        x = torch.clamp(x, min=-1, max=1)
        x = x / 2 + 0.5
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class Style2VLogNet(nn.Module):
    def __init__(
        self,
        num_luts: int = 256,
        finetune: bool = False,
        # id_lut: str = "id",
        num_layers: int = 2,
        # use_vlog: str = None,
        input_two_styles: bool = False,
        # deploy: bool = False,
    ):
        super().__init__()
        self.finetune = finetune
        # self.use_vlog = use_vlog
        # self.deploy = deploy
        # self.contrastive = contrastive
        self.input_two_styles = input_two_styles

        # Load the ViT model and processor from transformers
        model_name = "google/vit-base-patch16-224-in21k"
        self.vit_model = ViTModel.from_pretrained(model_name)
        # self.image_processor = ViTImageProcessor.from_pretrained(model_name)

        if not finetune:
            # Freeze ViT model parameters
            for param in self.vit_model.parameters():
                param.requires_grad = False

        vit_hidden_size = self.vit_model.config.hidden_size

        # if use_vlog is None:
        #     pass
        # elif use_vlog == "direct":
        #     pass
        # elif use_vlog == "predict":
        #     # feature predictor
        #     self.mlp_projector = nn.Sequential(
        #         nn.Linear(vit_hidden_size, vit_hidden_size),
        #         nn.ReLU(),
        #         nn.Linear(vit_hidden_size, vit_hidden_size),
        #     )
        # else:
        #     raise NotImplementedError

        # Define a classifier
        input_size = 2 * vit_hidden_size if input_two_styles else vit_hidden_size
        hidden_size = vit_hidden_size  # previous: 512
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_luts))
        self.classifier = nn.Sequential(*layers)

        # if contrastive:
        #     dim = 2048
        #     pred_dim = 512
        #     self.projector = nn.Sequential(
        #         nn.Linear(hidden_size, hidden_size, bias=False),
        #         nn.BatchNorm1d(hidden_size),
        #         nn.ReLU(inplace=True), # first layer
        #         nn.Linear(hidden_size, hidden_size, bias=False),
        #         nn.BatchNorm1d(hidden_size),
        #         nn.ReLU(inplace=True), # second layer
        #         nn.Linear(hidden_size, dim),
        #         nn.BatchNorm1d(dim, affine=False)) # output layer
        #     self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
        #         nn.BatchNorm1d(pred_dim),
        #         nn.ReLU(inplace=True), # hidden layer
        #         nn.Linear(pred_dim, dim)) # output layer

        self.CLUTs = CLUT(num_luts)
        self.tvmn = TVMN()
        # self.apply_lut = TrilinearInterpolation()

        # if id_lut == "id":
        #     id_lut = identity3d_tensor(33)
        # elif id_lut == "709":
        #     id_lut = read_3dlut_from_file("assets/Standard.cube")
        # else:
        #     raise NotImplementedError
        id_lut = identity3d_tensor(33)
        self.register_buffer("id_lut", id_lut, persistent=False)

        # # load ckpt
        # if "projector" in load_ckpt:
        #     self.mlp_projector.load_state_dict(torch.load(load_ckpt["projector"])["state_dict"])
        #     print(f"Loaded projector weights from the checkpoint at {load_ckpt["projector"]}")
        # if "classifier" in load_ckpt:
        #     self.classifier.load_state_dict(torch.load(load_ckpt["classifier"])["state_dict"])
        #     print(f"Loaded classifier weights from the checkpoint at {load_ckpt["classifier"]}")
        # if "decoder" in load_ckpt:
        #     self.CLUTs.load_state_dict(torch.load(load_ckpt["decoder"])["state_dict"])
        #     print(f"Loaded decoder weights from the checkpoint at {load_ckpt["decoder"]}")

    def _extract_vit_feature(self, images):
        if not self.finetune:
            self.vit_model.eval()

        outputs = self.vit_model(pixel_values=images)
        features = outputs.last_hidden_state
        cls_token = features[:, 0, :]  # batch, dim
        return cls_token

    def forward(self, style, style_2=None):

        feature = self._extract_vit_feature(style)
        if self.input_two_styles:
            style_2_feature = self._extract_vit_feature(style_2)
            feature = torch.cat([feature, style_2_feature], dim=-1)

        weight = self.classifier(feature)

        lut_res, tvmn = self.CLUTs(weight, self.tvmn if self.training else None)
        lut_out = torch.clamp(lut_res + self.id_lut, min=0, max=1)

        if self.training:
            # img_out = self.apply_lut(lut_out, content)
            # return lut_out, img_out, tvmn, feat_pred, feat_gt, contrastive_results if self.contrastive else None
            return lut_out, tvmn
        else:
            return lut_out


if __name__ == "__main__":

    model = Style2VLogImage2ImageNet(
        in_channel=3,
        out_channel=3,
        width=32,
        middle_blk_num=4,
        enc_blk_nums=[1, 1, 1, 2],
        dec_blk_nums=[1, 1, 1, 1],
    )

    input_tensor = torch.rand((4, 3, 224, 224))
    outputs = model(input_tensor)
    print([x.shape for x in outputs])
