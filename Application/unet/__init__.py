"""UNet package exports.

This module exposes the UNet model and its basic building Block for easy import:

from Application.unet import UNet, Block

You can also use build_unet(...) as a tiny factory.
"""

from .unet import UNet, Block

__all__ = ["UNet", "Block", "build_unet"]


def build_unet(in_channels: int = 3, out_channels: int = 1, init_features: int = 32) -> UNet:
    """Create a UNet instance with common defaults.

    Args:
        in_channels: 输入通道数（RGB 通常为 3）
        out_channels: 输出通道数（二分类分割通常为 1）
        init_features: 初始特征通道数（会按倍数逐层扩大）

    Returns:
        A configured UNet model.
    """
    return UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
