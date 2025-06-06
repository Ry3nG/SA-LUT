import torch.nn as nn  # type: ignore
import torch  # type: ignore


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
)


class Net(nn.Module):
    def __init__(self, encoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        # for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
        #     for param in getattr(self, name).parameters():
        #         param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, "enc_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, "enc_{:d}".format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False

        bs, ch = input.size()[:2]
        input = input.view(bs, ch, -1)
        target = target.view(bs, ch, -1)
        input_mean, input_std, input_p3 = feature_moments_caculation(input)
        target_mean, target_std, target_p3 = feature_moments_caculation(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(
            input_std, target_std
        )  # + \
        # self.mse_loss(input_p3, target_p3)

    def forward(self, content_images, style_images, stylized_images):
        style_feats = self.encode_with_intermediate(
            style_images
        )  # style_images[2, 3, 256, 256];4
        content_feat = self.encode(content_images)  # content_feat[2, 512, 32, 32]
        stylized_feats = self.encode_with_intermediate(stylized_images)

        loss_c = self.calc_content_loss(stylized_feats[-1], content_feat)
        loss_s = self.calc_style_loss(stylized_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_feats[i], style_feats[i])
        return loss_c, loss_s

    # def forward(self, style_image, output_image):
    #     style_feats = self.encode_with_intermediate(style_image)
    #     output_feats = self.encode_with_intermediate(output_image)

    #     loss = self.calc_style_loss(output_feats[0], style_feats[0])
    #     for i in range(1, 4):
    #         loss += self.calc_style_loss(output_feats[i], style_feats[i])
    #     return loss


def feature_moments_caculation(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 3
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    # the first order
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)

    # the second order
    feat_size = 2
    feat_p2 = torch.abs(feat - feat_mean).pow(feat_size).view(N, C, -1)
    N, C, L = feat_p2.shape
    feat_p2 = feat_p2.sum(dim=2) / L
    feat_p2 = feat_p2.pow(1 / feat_size).view(N, C, 1)
    # the third order
    feat_size = 3
    feat_p3 = torch.abs(feat - feat_mean).pow(feat_size).view(N, C, -1)
    feat_p3 = feat_p3.sum(dim=2) / L
    feat_p3 = feat_p3.pow(1 / feat_size).view(N, C, 1)

    return feat_mean.view(N, C), feat_p2.view(N, C), feat_p3.view(N, C)
