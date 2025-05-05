from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_extensions():
    srcs = ["sam2/csrc/connected_components.cu"]
    compile_args = {
        "cxx": [],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    }
    ext_modules = [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    return ext_modules


if __name__ == "__main__":
    # Setup configuration
    setup(
        name="sam2",
        version="0.1.0",
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    )
    
    

