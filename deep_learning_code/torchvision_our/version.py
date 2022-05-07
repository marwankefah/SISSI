__version__ = '0.12.0'
git_version = '9b5a3fecc72434dbd65148723efe54b28b9728c9'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
