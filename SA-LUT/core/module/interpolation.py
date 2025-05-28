import numpy as np  # type: ignore
import trilinear  # type: ignore
import torch  # type: ignore
import quadrilinear4d  # type: ignore


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
        if len(lut.shape) == 4:  # [3,dim,dim,dim]
            lut = lut.unsqueeze(0)  # Make it [1,3,dim,dim,dim]

        if lut.shape[0] == x.shape[0]:  # Batch-specific LUTs
            res = torch.empty_like(x)
            for i in range(lut.shape[0]):
                res[i : i + 1] = TrilinearInterpolationFunction.apply(
                    lut[i : i + 1], x[i : i + 1]
                )[1]
        else:  # Same LUT for whole batch
            res = TrilinearInterpolationFunction.apply(lut[0:1], x)[1]

        return res
