from collections.abc import Callable
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from timm.models.layers import to_3tuple

from .components import ConvBlock, DecoderBlock, EncoderBlock
from .vit import VisionTransformer


class UNETR(nn.Module):
    def __init__(
      self,
      in_channels: int,
      num_classes: int,
      img_size: Tuple[int, int, int],
      patch_size: Tuple[int, int, int] = (16, 16, 16),
      embed_dim: int = 768,
      feature_size: int = 64,
      depths: int = 4,
      mlp_ratio: float = 4.,
      num_layers: int = 12,
      num_heads: int = 12,
      qkv_bias: bool = False,
      qk_scale: Optional[None] = None,
      drop_path_rate: float = 0.,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.BatchNorm3d,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      upsample_kernel_size: Union[int, Sequence[int]] = 2,
      post_activation: str = 'tanh',
      save_attn: bool = False,
    ) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.depths = depths
        
        self.vit = VisionTransformer(
          in_channels=in_channels,
          img_size=to_3tuple(img_size),
          patch_size=to_3tuple(patch_size),
          num_classes=num_classes,
          embed_dim=embed_dim,
          num_layers=num_layers,
          num_heads=num_heads,
          classification=False,
          mlp_ratio=mlp_ratio,
          qkv_bias=qkv_bias,
          qk_scale=qk_scale,
          drop_path_rate=drop_path_rate,
          attn_drop=attn_drop,
          proj_drop=proj_drop,
          act_layer=act_layer,
          post_activation=post_activation,
          save_attn=save_attn
        )
        self.feat_size = self.vit.input_resolution
        
        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        for depth in range(depths):
            encoder = ConvBlock(
              in_channels=in_channels,
              out_channels=feature_size,
              kernel_size=kernel_size,
              stride=stride,
              act_layer=act_layer,
              norm_layer=norm_layer
            ) if depth == 0 else EncoderBlock(
              in_channels=embed_dim,
              out_channels=feature_size * 2 ** depth,
              num_layers=depths - depth,
              kernel_size=kernel_size,
              stride=stride,
              upsample_kernel_size=upsample_kernel_size,
              act_layer=act_layer,
              norm_layer=norm_layer
            )
            
            self.encoders.append(encoder)
            self.decoders.append(
              DecoderBlock(
                in_channels=embed_dim if depth == depths - 1 else feature_size * 2 ** (depth + 1),
                out_channels=feature_size * 2 ** depth,
                kernel_size=kernel_size,
                stride=stride,
                act_layer=act_layer,
                norm_layer=norm_layer
              )
            )
        
        self.out_proj = nn.Conv3d(in_channels=feature_size, out_channels=num_classes, kernel_size=1, stride=1)
    
    def proj_feat(self, x):
        x = x.view(x.size(0), self.feat_size[0], self.feat_size[1], self.feat_size[2], self.embed_dim)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    
    def forward(self, x_in: torch.Tensor) -> None:
        # feature extraction
        x, hidden_states = self.vit(x_in)
        
        # encode
        encoders = []
        for depth in range(self.depths):
            x_depth = x_in if depth == 0 else self.proj_feat(hidden_states[3 * depth])
            enc = self.encoders[depth](x_depth)
            encoders.append(enc)
        
        # decode
        for depth in reversed(range(self.depths)):
            skip = encoders[depth]
            if depth == self.depths - 1:
                x_depth = self.proj_feat(x)
            x_depth = self.decoders[depth](x_depth, skip)
        
        # output proj
        logits = self.out_proj(x_depth)
        return logits
