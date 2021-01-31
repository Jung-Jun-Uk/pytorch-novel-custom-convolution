from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='DYD_CONV_CUDA',
    ext_modules=[
        CUDAExtension('dyd_C',
                      ['dydconv_optim.cpp', 'dydconv_optim_kernel.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)