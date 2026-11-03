"""The photometric augmentation of training: light and colour, blur, noise and
codec artefacts, applied to a batch of patches turned to the track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg, encode_jpeg

# RGB to YIQ; hue turns the I-Q plane.
_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


@dataclass(frozen=True)
class Augment:
    """Each patch draws its own values, uniformly within each range. ``g_px`` is
    a gauge in patch pixels; motion blur runs along the track, the patch width."""

    g_px: float
    brightness: tuple[float, float] = (0.6, 1.4)
    contrast: tuple[float, float] = (0.7, 1.3)
    gamma: tuple[float, float] = (0.7, 1.4)
    white_balance: float = 0.12
    """Largest change of each channel's gain."""
    hue_deg: float = 15.0
    grayscale: float = 0.1
    """Share of patches turned gray."""
    blur: float = 0.3
    blur_sigma_px: tuple[float, float] = (0.3, 1.2)
    motion: float = 0.3
    motion_g: float = 1.0
    """Longest motion blur, in gauges."""
    noise_sigma: tuple[float, float] = (0.0, 0.04)
    jpeg: float = 0.5
    jpeg_quality: tuple[int, int] = (25, 90)

    def __call__(self, x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Augment ``x``, B × RGB × height × width in [0, 1]."""
        b, device = len(x), x.device

        def uniform(r: tuple[float, float], *shape: int) -> torch.Tensor:
            return torch.from_numpy(rng.uniform(*r, (b, *shape))).float().to(device)

        def chosen(p: float) -> torch.Tensor:
            return torch.from_numpy(rng.random(b) < p).to(device)

        wb = self.white_balance
        x = x * uniform(self.brightness, 1, 1, 1) * uniform((1 - wb, 1 + wb), 3, 1, 1)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mean) * uniform(self.contrast, 1, 1, 1) + mean).clamp(0, 1)
        x = x ** uniform(self.gamma, 1, 1, 1)
        x = self._hue(x, uniform((-self.hue_deg, self.hue_deg)) * torch.pi / 180)
        gray = x.mul(_YIQ[0].to(device)[:, None, None]).sum(1, keepdim=True).expand_as(x)
        x = torch.where(chosen(self.grayscale)[:, None, None, None], gray, x).clamp(0, 1)
        x = self._blur(x, chosen(self.blur), rng)
        x = self._motion(x, chosen(self.motion), rng)
        x = x + torch.randn_like(x) * uniform(self.noise_sigma, 1, 1, 1)
        return self._jpeg(x.clamp(0, 1), rng)

    def _hue(self, x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        yiq = _YIQ.to(x.device)
        c, s = torch.cos(angle), torch.sin(angle)
        turn = torch.zeros(len(x), 3, 3, device=x.device)
        turn[:, 0, 0] = 1
        turn[:, 1, 1], turn[:, 1, 2], turn[:, 2, 1], turn[:, 2, 2] = c, -s, s, c
        m = torch.linalg.inv(yiq) @ turn @ yiq
        return torch.einsum("bij,bjhw->bihw", m, x)

    def _blur(self, x: torch.Tensor, which: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        sigma = torch.from_numpy(rng.uniform(*self.blur_sigma_px, len(x))).float().to(x.device)
        taps = torch.arange(-3, 4, device=x.device).float()
        k = torch.exp(-0.5 * (taps / sigma[:, None]) ** 2)
        k = torch.where(which[:, None], k, (taps == 0).float())
        k = k / k.sum(1, keepdim=True)
        x = _conv(x, k[:, None, :])
        return _conv(x, k[:, :, None])

    def _motion(
        self, x: torch.Tensor, which: torch.Tensor, rng: np.random.Generator
    ) -> torch.Tensor:
        reach = round(self.motion_g * self.g_px / 2)
        length = rng.uniform(1, 2 * reach + 1, len(x))
        taps = torch.arange(-reach, reach + 1, device=x.device).float()
        half = torch.from_numpy(length / 2).float().to(x.device)[:, None]
        k = (taps.abs() <= half).float()
        k = torch.where(which[:, None], k, (taps == 0).float())
        return _conv(x, (k / k.sum(1, keepdim=True))[:, None, :])

    def _jpeg(self, x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        which = np.flatnonzero(rng.random(len(x)) < self.jpeg)
        if len(which) == 0:
            return x
        lo, hi = self.jpeg_quality
        pixels = (x[which] * 255).round().to(torch.uint8).cpu()
        coded = [
            cast(torch.Tensor, decode_jpeg(encode_jpeg(p, quality=int(q))))
            for p, q in zip(pixels, rng.integers(lo, hi + 1, len(which)), strict=True)
        ]
        x = x.clone()
        x[torch.from_numpy(which).to(x.device)] = torch.stack(coded).to(x.device).float() / 255
        return x


def _conv(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Each patch of ``x`` convolved with its own kernel of ``k``, B × kh × kw."""
    b, c, h, w = x.shape
    kernel = k.repeat_interleave(c, 0)[:, None]
    kh, kw = k.shape[1:]
    out = F.conv2d(
        F.pad(x.reshape(1, b * c, h, w), (kw // 2, kw // 2, kh // 2, kh // 2), mode="replicate"),
        kernel,
        groups=b * c,
    )
    return out.reshape(b, c, h, w)
