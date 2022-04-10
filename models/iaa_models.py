import torch
import torch.nn as nn
from models.vit_editors import ViT_Editor_Channels
from models.vit_components import Mlp, Attention, Block, PatchEmbed

class IAAModel(ViT_Editor_Channels):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super(IAAModel, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, qk_scale, representation_size, distilled,
                 drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer, act_layer, weight_init)
        self.aes_token = torch.nn.Parameter( torch.ones(1, 1, self.embed_dim) )
        num_patches = self.pos_embed.shape[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.aes_head = nn.Linear(self.num_features, 10)    # project to 10-dim aesthetic score
        self.softmax = nn.Softmax(dim=1)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
    
        x = torch.cat((x, self.aes_token.expand(B, -1, -1)), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        x_aes = x[:, -1:]
        x_aes = self.aes_head(x_aes).squeeze(dim=1)
        
        x_aes = self.softmax(x_aes)
        
        return x_aes
