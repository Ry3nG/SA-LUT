import numpy as np  # type: ignore
import torch  # type: ignore
import trilinear  # type: ignore
from torch import nn  # type: ignore


def identity3d_tensor(dim):  # 3,d,d,d
    step = np.arange(0, dim) / (dim - 1)
    rgb = torch.tensor(step, dtype=torch.float32)
    LUT = torch.empty(3, dim, dim, dim)
    LUT[0] = rgb.unsqueeze(0).unsqueeze(0).expand(dim, dim, dim)  # r
    LUT[1] = rgb.unsqueeze(-1).unsqueeze(0).expand(dim, dim, dim)  # g
    LUT[2] = rgb.unsqueeze(-1).unsqueeze(-1).expand(dim, dim, dim)  # b
    return LUT


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim**3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        if batch == 1:
            assert 1 == trilinear.forward(
                lut, x, output, dim, shift, binsize, W, H, batch
            )
        elif batch > 1:
            output = output.permute(1, 0, 2, 3).contiguous()
            assert 1 == trilinear.forward(
                lut,
                x.permute(1, 0, 2, 3).contiguous(),
                output,
                dim,
                shift,
                binsize,
                W,
                H,
                batch,
            )
            output = output.permute(1, 0, 2, 3).contiguous()

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        if batch == 1:
            assert 1 == trilinear.backward(
                x, x_grad, lut_grad, dim, shift, binsize, W, H, batch
            )
        elif batch > 1:
            assert 1 == trilinear.backward(
                x.permute(1, 0, 2, 3).contiguous(),
                x_grad.permute(1, 0, 2, 3).contiguous(),
                lut_grad,
                dim,
                shift,
                binsize,
                W,
                H,
                batch,
            )
        return lut_grad, x_grad


# trilinear_need: imgs=nchw, lut=3ddd or 13ddd
class TrilinearInterpolation(torch.nn.Module):
    def __init__(self, mo=False, clip=False):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        if lut.shape[0] > 1:
            if lut.shape[0] == x.shape[0]:  # n, c, H, W
                res = torch.empty_like(x)
                for i in range(lut.shape[0]):
                    res[i : i + 1] = TrilinearInterpolationFunction.apply(
                        lut[i : i + 1], x[i : i + 1]
                    )[1]
            else:
                n, c, h, w = x.shape
                res = torch.empty(n, lut.shape[0], c, h, w).cuda()
                for i in range(lut.shape[0]):
                    res[:, i] = TrilinearInterpolationFunction.apply(lut[i : i + 1], x)[
                        1
                    ]
        else:  # n, c, H, W
            res = TrilinearInterpolationFunction.apply(lut, x)[1]
        return res


class TVMN(nn.Module):  # ([n,]3,d,d,d) or ([n,]3,d)
    def __init__(self, dim=33):
        super(TVMN, self).__init__()
        self.dim = dim
        self.relu = torch.nn.ReLU()
        weight_r = torch.ones(1, 1, dim, dim, dim - 1, dtype=torch.float)
        weight_r[..., (0, dim - 2)] *= 2.0
        weight_g = torch.ones(1, 1, dim, dim - 1, dim, dtype=torch.float)
        weight_g[..., (0, dim - 2), :] *= 2.0
        weight_b = torch.ones(1, 1, dim - 1, dim, dim, dtype=torch.float)
        weight_b[..., (0, dim - 2), :, :] *= 2.0
        self.register_buffer("weight_r", weight_r, persistent=False)
        self.register_buffer("weight_g", weight_g, persistent=False)
        self.register_buffer("weight_b", weight_b, persistent=False)
        self.register_buffer("tvmn_shape", torch.empty(3), persistent=False)

    def forward(self, LUT):
        dim = self.dim
        tvmn = 0 + self.tvmn_shape
        # tvmn[0]: smooth, tvmn[1]: monotonicity
        if len(LUT.shape) > 3:  # (n, 3, d, d, d) or (3, d, d, d)
            dif_r = LUT[..., :-1] - LUT[..., 1:]
            dif_g = LUT[..., :-1, :] - LUT[..., 1:, :]
            dif_b = LUT[..., :-1, :, :] - LUT[..., 1:, :, :]
            # Total Variation
            tvmn[0] = (
                torch.mean(dif_r**2 * self.weight_r[:, 0])
                + torch.mean(dif_g**2 * self.weight_g[:, 0])
                + torch.mean(dif_b**2 * self.weight_b[:, 0])
            )
            # Monotonicity
            tvmn[1] = (
                torch.mean(self.relu(dif_r * self.weight_r[:, 0]) ** 2)
                + torch.mean(self.relu(dif_g * self.weight_g[:, 0]) ** 2)
                + torch.mean(self.relu(dif_b * self.weight_b[:, 0]) ** 2)
            )
            tvmn[2] = 0
        else:  # (n,3,d) or (3,d)
            dif = LUT[..., :-1] - LUT[..., 1:]
            tvmn[1] = torch.mean(self.relu(dif))
            dif = dif**2
            dif[..., (0, dim - 2)] *= 2.0
            tvmn[0] = torch.mean(dif)
            tvmn[2] = 0
        return tvmn


class CLUT(nn.Module):
    def __init__(self, num, dim=33, s=32, w=32, *args, **kwargs):
        super(CLUT, self).__init__()
        self.num = num
        self.dim = dim
        self.s, self.w = s, w = eval(str(s)), eval(str(w))
        # +: compressed;  -: uncompressed
        if s == -1 and w == -1:  # standard 3DLUT
            self.mode = "--"
            self.LUTs = nn.Parameter(torch.zeros(num, 3, dim, dim, dim))
        elif s != -1 and w == -1:
            self.mode = "+-"
            self.s_Layers = nn.Parameter(torch.rand(dim, s) / 5 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(s, num * 3 * dim * dim))
        elif s == -1 and w != -1:
            self.mode = "-+"
            self.w_Layers = nn.Parameter(torch.rand(w, dim * dim) / 5 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(num * 3 * dim, w))
        else:  # full-version CLUT
            self.mode = "++"
            self.s_Layers = nn.Parameter(torch.rand(dim, s) / 5 - 0.1)
            self.w_Layers = nn.Parameter(torch.rand(w, dim * dim) / 5 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(s * num * 3, w))
        print("n=%d s=%d w=%d" % (num, s, w), self.mode)
        # self.tvmn = TVMN(dim)

    def _cube_to_lut(self, cube):  # (n,)3,d,d,d
        if len(cube.shape) == 5:
            to_shape = [
                [0, 2, 3, 1],
                [0, 2, 1, 3],
            ]
        else:
            to_shape = [
                [1, 2, 0],
                [1, 0, 2],
            ]
        if isinstance(cube, torch.Tensor):
            lut = torch.empty_like(cube)
            lut[..., 0, :, :, :] = cube[..., 0, :, :, :].permute(*to_shape[0])
            lut[..., 1, :, :, :] = cube[..., 1, :, :, :].permute(*to_shape[1])
            lut[..., 2, :, :, :] = cube[..., 2, :, :, :]
        else:
            lut = np.empty_like(cube)
            lut[..., 0, :, :, :] = cube[..., 0, :, :, :].transpose(*to_shape[0])
            lut[..., 1, :, :, :] = cube[..., 1, :, :, :].transpose(*to_shape[1])
            lut[..., 2, :, :, :] = cube[..., 2, :, :, :]
        return lut

    def _reconstruct_luts(self):
        if self.mode == "--":
            D3LUTs = self.LUTs
        else:
            if self.mode == "+-":
                # d, s x s, num * 3dd -> d, num * 3dd -> d, num * 3,dd -> num, 3, d, dd -> num, -1
                CUBEs = (
                    self.s_Layers.mm(self.LUTs)
                    .reshape(self.dim, self.num * 3, self.dim * self.dim)
                    .permute(1, 0, 2)
                    .reshape(self.num, 3, self.dim, self.dim, self.dim)
                )
            if self.mode == "-+":
                # num * 3d, w x w, dd -> num * 3d, dd -> num, 3ddd
                CUBEs = self.LUTs.mm(self.w_Layers).reshape(
                    self.num, 3, self.dim, self.dim, self.dim
                )
            if self.mode == "++":
                # s * num * 3, w x w, dd -> s * num * 3,dd -> s, num * 3 * dd -> d, num * 3 * dd -> num, -1
                CUBEs = (
                    self.s_Layers.mm(
                        self.LUTs.mm(self.w_Layers).reshape(
                            -1, self.num * 3 * self.dim * self.dim
                        )
                    )
                    .reshape(self.dim, self.num * 3, self.dim**2)
                    .permute(1, 0, 2)
                    .reshape(self.num, 3, self.dim, self.dim, self.dim)
                )
            D3LUTs = self._cube_to_lut(CUBEs)

        return D3LUTs

    def combine(self, weight, TVMN):  # n, num
        D3LUTs = self._reconstruct_luts()
        if TVMN is None:
            tvmn = 0
        else:
            tvmn = TVMN(D3LUTs)
        # if TVMN is not None:
        #     tvmn = TVMN(D3LUTs)
        # elif self.tvmn is not None:
        #     tvmn = self.tvmn(D3LUTs)
        # else:
        #     tvmn = 0
        D3LUT = weight.mm(D3LUTs.reshape(self.num, -1)).reshape(
            -1, 3, self.dim, self.dim, self.dim
        )
        return D3LUT, tvmn

    def forward(self, weight, TVMN=None):
        lut, tvmn = self.combine(weight, TVMN)
        return lut, tvmn
