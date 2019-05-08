import torch
import maskrcnn_utils
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

## NMS test
a = torch.tensor([[0,0,2,2],[1,1,3,3]]).float()
b = torch.tensor([0.9, 0.2])
thr = 0.1

print(maskrcnn_utils.nms(a,b,thr))

#a = a.type(torch.cuda.FloatTensor)
#b = b.type(torch.cuda.FloatTensor)
a = a.cuda()
b = b.cuda()

print(maskrcnn_utils.nms(a,b,thr))

## ROIAlign test
class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = maskrcnn_utils.roi_align_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = maskrcnn_utils.roi_align_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr

output_size = (7, 7, 1)
spatial_scale = 0.25
sampling_ratio = 2
RA = ROIAlign(output_size, spatial_scale, sampling_ratio).cuda()
inputx = torch.randn((1,1,14,14)).cuda()
roi = torch.ones(1).float().cuda()
output = RA(inputx, roi)
print(inputx)
print(output)
