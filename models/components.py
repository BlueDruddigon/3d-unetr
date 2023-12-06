from collections.abc import Callable
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


class MLP(nn.Sequential):
    def __init__(
      self,
      in_features: int,
      hidden_features: Optional[int] = None,
      out_features: Optional[int] = None,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      dropout: float = 0.
    ) -> None:
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        dropout = nn.Dropout(dropout)
        layers = nn.ModuleList([
          nn.Linear(in_features, hidden_features),
          act_layer(),
          dropout,
          nn.Linear(hidden_features, out_features),
          dropout,
        ])
        super().__init__(*layers)


class MSA(nn.Module):
    def __init__(
      self,
      dim: int,
      num_heads: int = 8,
      qkv_bias: bool = False,
      qk_scale: Optional[None] = None,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      save_attn: bool = False
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.save_attn = save_attn
        self.attn_mat = torch.Tensor()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_token, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_token, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        if self.save_attn:
            self.attn_mat = attn.detach()
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        x = self.proj_drop(self.proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(
      self,
      dim: int,
      num_heads: int,
      mlp_ratio: float = 4.,
      qkv_bias: bool = False,
      qk_scale: Optional[None] = None,
      drop_path_rate: float = 0.,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      save_attn: bool = False
    ) -> None:
        super().__init__()
        
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = MSA(
          dim,
          num_heads,
          qkv_bias=qkv_bias,
          qk_scale=qk_scale,
          attn_drop=attn_drop,
          proj_drop=proj_drop,
          save_attn=save_attn
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.ffn_norm = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, dropout=proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.attn_norm(x)))
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(
      self,
      in_channels: int,
      img_size: Sequence[int],
      patch_size: Sequence[int],
      embed_dim: int = 768,
      dropout_rate: float = 0.
    ) -> None:
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = tuple([i // p for i, p in zip(self.img_size, self.patch_size)])
        self.num_patches = np.prod([i // p for i, p in zip(self.img_size, self.patch_size)])
        
        # embeddings
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)
        
        self.apply(self._init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(-1, -2)  # patch embeddings
        x += self.pos_embed  # including positional encoding to the embedding vector
        return self.pos_drop(x)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)


class ConvBlock(nn.Sequential):
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.BatchNorm3d
    ) -> None:
        layers = nn.ModuleList([
          nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
          norm_layer(out_channels),
          act_layer(),
          nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
          norm_layer(out_channels),
          act_layer()
        ])
        super().__init__(*layers)


class DeconvBlock(nn.Sequential):
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      upsample_kernel_size: Union[int, Sequence[int]] = 2,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.BatchNorm3d
    ) -> None:
        layers = nn.ModuleList([
          nn.ConvTranspose3d(in_channels, out_channels, kernel_size=upsample_kernel_size, stride=upsample_kernel_size),
          nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
          norm_layer(out_channels),
          act_layer(),
        ])
        super().__init__(*layers)


class EncoderBlock(nn.Module):
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      num_layers: int,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      upsample_kernel_size: Union[int, Sequence[int]] = 2,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.BatchNorm3d
    ) -> None:
        super().__init__()
        
        self.blocks = nn.ModuleList([
          DeconvBlock(
            in_channels=in_channels if i == num_layers - 1 else out_channels * 2 ** (i + 1),
            out_channels=out_channels * 2 ** i,
            kernel_size=kernel_size,
            stride=stride,
            upsample_kernel_size=upsample_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer
          ) for i in reversed(range(num_layers))
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      upsample_kernel_size: Union[int, Sequence[int]] = 2,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.BatchNorm3d
    ) -> None:
        super().__init__()
        
        self.deconv = nn.ConvTranspose3d(
          in_channels, out_channels, kernel_size=upsample_kernel_size, stride=upsample_kernel_size
        )
        self.conv = ConvBlock(
          out_channels * 2,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          act_layer=act_layer,
          norm_layer=norm_layer
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x
