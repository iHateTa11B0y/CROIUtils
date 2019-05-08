from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

setup(
    name='_C',
    ext_modules=[
        CUDAExtension('_C', 
            [
                'vision.cpp',
                'cpu/ROIAlign_cpu.cpp',
                'cpu/nms_cpu.cpp',
                'cuda/ROIAlign_cuda.cu',
                'cuda/ROIPool_cuda.cu',
                'cuda/SigmoidFocalLoss_cuda.cu',
                'cuda/nms.cu',
                ],
            extra_compile_args=['-DWITH_CUDA'],
            )
        ],
    cmdclass={'build_ext': BuildExtension})
