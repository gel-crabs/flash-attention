import sys
import warnings
import os
import glob
import shutil
from packaging.version import parse, Version

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME, IS_HIP_EXTENSION
from setuptools import setup, find_packages
import subprocess


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.2"):
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

cmdclass = {}
ext_modules = []


def rename_cpp_to_hip(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".hip")

# Defining a function to validate the GPU architectures and update them if necessary
def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx1100", "gfx1101"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by Flash-Attention"

def build_for_rocm():
    """build for ROCm platform"""

    archs = os.getenv("GPU_ARCHS", "native").split(";")
    validate_and_update_archs(archs)
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

    if int(os.environ.get("FLASH_ATTENTION_INTERNAL_USE_RTN", 0)):
        print("RTN IS USED")
        cc_flag.append("-DUSE_RTN_BF16_CONVERT")
    else:
        print("RTZ IS USED")

    fa_sources = ["ft_attention.cpp", "decoder_masked_multihead_attention.cpp"] #+ glob.glob("src/*.cpp")

    rename_cpp_to_hip(fa_sources)

    ext_modules.append(
        CUDAExtension(
            'ft_attention', [
                "ft_attention.hip",
                "decoder_masked_multihead_attention.hip"
            ], #+ glob.glob("src/*.hip"),
            extra_compile_args={"cxx": ["-O3", "-std=c++17", "-DNDEBUG"],
                                'nvcc': [
                                    "-O3",
                                    "-std=c++17",
                                    "-DNDEBUG",
                                    "-U__HIP_NO_HALF_OPERATORS__",
                                    "-U__HIP_NO_HALF_CONVERSIONS__",
                                    "-U__HIP_NO_BFLOAT16_OPERATORS__",
                                    "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
                                    "-U__HIP_NO_BFLOAT162_OPERATORS__",
                                    "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
                                    "-U__HIPCC_RTC__",
                                    "-I/opt/rocm/include/hip/amd_detail",
                                ] + cc_flag
                               }
        )
    )

build_for_rocm()

setup(
    name='ft_attention',
    version="0.1",
    description="Attention for single query from FasterTransformer",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
)
