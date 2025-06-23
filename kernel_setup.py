from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_t5',
    ext_modules=[
        CUDAExtension(
            'custom_t5_cpp',
            [
                'kernels/t5_bindings.cpp',
                'kernels/t5_kernels.cu',
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O3', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)