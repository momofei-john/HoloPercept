import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from model.common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(self, *, transformer_dim, transformer,
                 num_multimask_outputs: int = 1,
                 activation: Type[nn.Module] = nn.GELU,
                 ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(self.num_mask_tokens)]
        )

    def forward(
        self,
        memory_embeddings: torch.Tensor,   # b 1 112 56 56
        x: torch.Tensor,                   # b 112 56 56
        sparse_prompt_embeddings: torch.Tensor = None,
        multimask_output: bool = False,
    ):
        masks, fused_feats, out_56 = self.predict_masks(
            memory_embeddings=memory_embeddings,
            x=x,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )

        if multimask_output:
            mask_slice = slice(0, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :]
        out_56 = out_56[:, mask_slice, :, :]
        return masks, fused_feats, out_56

    def predict_masks(
        self,
        memory_embeddings: torch.Tensor,
        x: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor = None,
    ):
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        if sparse_prompt_embeddings is not None:
            output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        else:
            tokens = output_tokens.unsqueeze(0)

        mem = memory_embeddings.squeeze(1)   # b 112 56 56
        if mem.shape[0] != x.shape[0]:
            raise ValueError(f"Batch size mismatch: mem {mem.shape}, x {x.shape}")
        src = mem + x   # b 112 56 56
        b, c, h, w = src.shape

        # Transformer
        hs, src = self.transformer(src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        out_56 = src
        upscaled_embedding = self.output_upscaling(src)

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks, upscaled_embedding, out_56


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
