"""
CTransPath patch encoder (Wang et al., Med. Image Anal. 2022).

Weights: https://github.com/Xiyue-Wang/TransPath  (download `ctranspath.pth`).
License: GPLv3, non-commercial academic only.

Architecture: Swin-Tiny with a 3-block ConvStem replacing the standard patch
embedding. Output feature dim = 768.

Requires `timm==0.5.4` (CTransPath relies on the `embed_layer` kwarg of the
older Swin Transformer API; newer timm dropped it):
    pip install timm==0.5.4
"""
from __future__ import annotations

import torch
import torch.nn as nn


CTRANSPATH_FEATURE_DIM = 768


class ConvStem(nn.Module):
    """3-block ConvStem from CTransPath (replaces standard patch embed)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer=None,
        flatten: bool = True,
    ) -> None:
        super().__init__()
        assert patch_size == 4, "CTransPath ConvStem requires patch_size=4"
        assert embed_dim % 8 == 0, "embed_dim must be divisible by 8"

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem: list[nn.Module] = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


def build_ctranspath_encoder(weights_path: str) -> nn.Module:
    """Build CTransPath encoder (Swin-T + ConvStem) and load pretrained weights.

    Returns a model whose forward(x) -> (B, 768) feature vector.
    """
    import timm
    encoder = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=0,
    )

    # CTransPath replaces the standard single-Conv patch embedding with a
    # 3-block ConvStem. Newer timm versions silently ignore the `embed_layer`
    # kwarg passed at create_model time, so swap it explicitly here.
    encoder.patch_embed = ConvStem(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        norm_layer=nn.LayerNorm,
    )

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    msg = encoder.load_state_dict(state, strict=False)
    print(
        f"CTransPath weights loaded from {weights_path} "
        f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})"
    )
    if msg.missing_keys:
        print(f"  missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        critical = [k for k in msg.unexpected_keys if not k.startswith('head.')]
        print(f"  unexpected keys (non-head): {critical}")
    return encoder


class CTransPathClassifier(nn.Module):
    """CTransPath encoder + linear classification head.

    When `freeze_backbone=True` the encoder is in eval() mode permanently
    (BN stats frozen) and runs under torch.no_grad(); only the linear head
    receives gradients.
    """

    def __init__(self, encoder: nn.Module, num_classes: int, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(CTRANSPATH_FEATURE_DIM, num_classes)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        return self.head(feats)


def build_ctranspath_classifier(
    num_classes: int,
    weights_path: str,
    freeze_backbone: bool = True,
) -> CTransPathClassifier:
    """Convenience builder: encoder + linear head, encoder frozen by default."""
    encoder = build_ctranspath_encoder(weights_path)
    return CTransPathClassifier(encoder=encoder, num_classes=num_classes, freeze_backbone=freeze_backbone)
