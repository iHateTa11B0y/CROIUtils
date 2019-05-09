from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

ext_modules=[
        CUDAExtension('maskrcnn_utils._C',
            [
                'csrc/vision.cpp',
                'csrc/cpu/ROIAlign_cpu.cpp',
                'csrc/cpu/nms_cpu.cpp',
                'csrc/cuda/ROIAlign_cuda.cu',
                'csrc/cuda/ROIPool_cuda.cu',
                'csrc/cuda/SigmoidFocalLoss_cuda.cu',
                'csrc/cuda/nms.cu',
                ],
            extra_compile_args=['-DWITH_CUDA', '-O2'],
            )
        ]

setup(
        name='maskrcnn_utils',
        version='0.1.5',
        description='this is a tool package for maskrcnn',
        author='Kuuuurt',
        author_email='niu1187203155@gmail.com',
        url='https://github.com/iHateTa11B0y/CROIUtils',
        packages=['maskrcnn_utils'],
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
)
