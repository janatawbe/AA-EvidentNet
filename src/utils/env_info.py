"""Environment information collection for reproducibility logging."""

import platform
import sys
from typing import Any, Dict


def collect_environment_info() -> Dict[str, Any]:
    """Collect Python, OS, and (if available) PyTorch/CUDA environment info.

    Intended to be dumped alongside experiment results so runs can be traced
    back to the exact software environment that produced them.
    """
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        )
        info["num_gpus"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        info["gpu_names"] = (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False
        info["cuda_version"] = None
        info["cudnn_version"] = None
        info["num_gpus"] = 0
        info["gpu_names"] = []

    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except ImportError:
        info["numpy_version"] = None

    return info
