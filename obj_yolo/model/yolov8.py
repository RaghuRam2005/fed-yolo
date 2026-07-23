"""
Full YOLOv8 model: CSPDarknet backbone + PAN-FPN neck + anchor-free Detect
head, wired to mirror `yolo_config/*_yolo8.yaml` layer-for-layer (each
submodule's comment cites the yaml's own `[from, repeats, module, args]`
line index).

Channel/repeat scaling formulas verified against Ultralytics'
`ultralytics/nn/tasks.py::parse_model` (`make_divisible`, `round(n * depth)`).
Bias init verified against `ultralytics/nn/modules/head.py::Detect.bias_init`.
"""
import math
from typing import Literal

import torch
import torch.nn as nn

from obj_yolo.model.blocks import Conv, C2f, SPPF
from obj_yolo.model.head import Detect

Scale = Literal["n", "s", "m", "l", "x"]

# scale -> (depth_multiple, width_multiple, max_channels), from yolo_config/*_yolo8.yaml
SCALES: dict[Scale, tuple[float, float, int]] = {
    "n": (0.33, 0.25, 1024),
    "s": (0.33, 0.50, 1024),
    "m": (0.67, 0.75, 768),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
}


def make_divisible(x: float, divisor: int = 8) -> int:
    """Round `x` up to the nearest multiple of `divisor` (ultralytics.utils.make_divisible)."""
    return math.ceil(x / divisor) * divisor


class YOLOv8(nn.Module):
    """YOLOv8 detection model. `scale` selects the n/s/m/l/x compound-scaling variant."""

    def __init__(self, nc: int, scale: Scale = "n", ch: int = 3) -> None:
        super().__init__()
        self.nc = nc
        self.scale = scale
        depth, width, max_channels = SCALES[scale]

        def c(base: int) -> int:
            """Scale a yaml base channel count (width_multiple + max_channels cap)."""
            return make_divisible(min(base, max_channels) * width, 8)

        def rep(base: int) -> int:
            """Scale a yaml base repeat count (depth_multiple), min 1."""
            return max(round(base * depth), 1) if base > 1 else base

        # ---- backbone (yaml `backbone:`, indices 0-9) ----
        self.b0 = Conv(ch, c(64), 3, 2)                     # 0: P1/2
        self.b1 = Conv(c(64), c(128), 3, 2)                  # 1: P2/4
        self.b2 = C2f(c(128), c(128), rep(3), True)           # 2
        self.b3 = Conv(c(128), c(256), 3, 2)                    # 3: P3/8
        self.b4 = C2f(c(256), c(256), rep(6), True)              # 4  -> neck (P3)
        self.b5 = Conv(c(256), c(512), 3, 2)                        # 5: P4/16
        self.b6 = C2f(c(512), c(512), rep(6), True)                  # 6  -> neck (P4)
        self.b7 = Conv(c(512), c(1024), 3, 2)                           # 7: P5/32
        self.b8 = C2f(c(1024), c(1024), rep(3), True)                    # 8
        self.b9 = SPPF(c(1024), c(1024), 5)                                # 9  -> neck (P5)

        # ---- neck / head (yaml `head:`, indices 10-21) ----
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.n12 = C2f(c(1024) + c(512), c(512), rep(3), False)               # 10-12: cat(up(9), 6)
        self.n15 = C2f(c(512) + c(256), c(256), rep(3), False)                  # 13-15: cat(up(12), 4) -> P3/8-small
        self.n16 = Conv(c(256), c(256), 3, 2)                                      # 16: downsample(15)
        self.n18 = C2f(c(256) + c(512), c(512), rep(3), False)                       # 17-18: cat(16, 12) -> P4/16-medium
        self.n19 = Conv(c(512), c(512), 3, 2)                                          # 19: downsample(18)
        self.n21 = C2f(c(512) + c(1024), c(1024), rep(3), False)                         # 20-21: cat(19, 9) -> P5/32-large

        # ---- detect head (yaml index 22) ----
        self.detect = Detect(nc, (c(256), c(512), c(1024)))
        self._init_strides()
        self._init_biases()

    def _forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)  # P3 backbone feature (feeds neck)
        x5 = self.b5(x4)
        x6 = self.b6(x5)  # P4 backbone feature (feeds neck)
        x7 = self.b7(x6)
        x8 = self.b8(x7)
        x9 = self.b9(x8)  # P5 backbone feature (feeds neck)

        p12 = self.n12(torch.cat((self.up(x9), x6), 1))
        p15 = self.n15(torch.cat((self.up(p12), x4), 1))     # P3/8-small
        p18 = self.n18(torch.cat((self.n16(p15), p12), 1))    # P4/16-medium
        p21 = self.n21(torch.cat((self.n19(p18), x9), 1))      # P5/32-large
        return [p15, p18, p21]

    def _init_strides(self, imgsz: int = 256) -> None:
        """Probe the network once with a dummy input to derive each level's stride
        (standard ultralytics `DetectionModel.__init__` trick)."""
        with torch.no_grad():
            feats = self._forward_features(torch.zeros(1, 3, imgsz, imgsz))
        self.detect.stride = torch.tensor([imgsz / f.shape[-2] for f in feats])

    def _init_biases(self) -> None:
        """Detect head bias init (helps early-training stability); see
        `ultralytics/nn/modules/head.py::Detect.bias_init`."""
        for cv2, cv3, s in zip(self.detect.cv2, self.detect.cv3, self.detect.stride):
            cv2[-1].bias.data[:] = 1.0
            cv3[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s.item()) ** 2)

    def forward(self, x: torch.Tensor):
        feats = self._forward_features(x)
        return self.detect(feats)
