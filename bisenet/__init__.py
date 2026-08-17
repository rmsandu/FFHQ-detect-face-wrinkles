"""Vendored BiSeNet face-parsing model, from zllrunning/face-parsing.PyTorch (MIT).

Only model.py and resnet.py are vendored (inference architecture only) --
training/eval/dataset utilities from the upstream repo are not needed here.
See LICENSE in this directory for the original project's license.
"""

from .model import BiSeNet  # noqa: F401
