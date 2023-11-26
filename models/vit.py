from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .components import PatchEmbed, TransformerBlock


class VisionTransformer(nn.Module):
    def __init__(
      self,
      in_channels: int,
      img_size: Sequence[int],
      patch_size: Sequence[int],
      num_classes: int,
      embed_dim: int = 768,
      num_layers: int = 12,
      num_heads: int = 12,
      classification: bool = False,
      mlp_ratio: float = 4.,
      qkv_bias: bool = False,
      qk_scale: Optional[None] = None,
      drop_path_rate: float = .1,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      post_activation: str = 'tanh',
      save_attn: bool = False,
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(in_channels, img_size, patch_size, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.input_resolution = self.patch_embed.patches_resolution
        
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList([
          TransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path_rate=dpr[i],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            act_layer=act_layer,
            save_attn=save_attn
          ) for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.classification = classification
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=.02)
            self.head = nn.Linear(embed_dim, num_classes)
            if post_activation == 'tanh':
                self.head = nn.Sequential(nn.Linear(embed_dim, num_classes), nn.Tanh())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.patch_embed(x)
        # class_token adding
        if hasattr(self, 'cls_token'):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        hidden_state_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_state_out.append(x)
        
        x = self.norm(x)
        if hasattr(self, 'head'):
            x = self.head(x[:, 0])  # classifier "token" as used by standard language architectures.
        return x, hidden_state_out
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
