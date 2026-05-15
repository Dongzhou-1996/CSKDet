import torch
import torch.nn as nn
import math


class SinPositionEncoding1D(nn.Module):
    def __init__(self, max_sequence_length, d_model, base=10000):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.base = base

    def forward(self):
        pe = torch.zeros(self.max_sequence_length, self.d_model, dtype=torch.float)
        exp_1 = torch.arange(self.d_model // 2, dtype=torch.float)
        exp_value = exp_1 / (self.d_model / 2)

        alpha = 1 / (self.base ** exp_value)
        out = torch.arange(self.max_sequence_length, dtype=torch.float)[:, None] @ alpha[None, :]
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)

        pe[:, 0::2] = embedding_sin
        pe[:, 1::2] = embedding_cos
        return pe


class SinePositionalEncoding2D(nn.Module):
    def __init__(self,
                 num_feats,
                 temperature = 10000,
                 normalize = False,
                 scale = 2 * math.pi,
                 eps = 1e-6,
                 offset = 0.,):
        super(SinePositionalEncoding2D, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        mask = mask.to(torch.int)
        not_mask = 1 - mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward_coordinates(self, coord):
        """
        Forward funtion for normalized coordinates input with the shape of [bs, kpt, 2]
        return:
            pos (Tensor): position embedding with the shape of [bs, kpt, num_feats*2]
        """
        x_embed, y_embed = coord[:,:,0], coord[:,:,1] # [bs, kpt]
        x_embed = x_embed * self.scale # [bs, kpt]
        y_embed = y_embed * self.scale

        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=coord.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, None] / dim_t   # [bs, kpt, num_feats]
        pos_y = y_embed[:, :, None] / dim_t   # [bs, kpt, num_feats]
        bs, kpt, _ = pos_x.shape

        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
            dim=3).view(bs, kpt, -1) # [bs, kpt, num_feats]
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
            dim=3).view(bs, kpt, -1) # [bs, kpt, num_feats]
        pos = torch.cat((pos_y, pos_x), dim=2) # [bs, kpt, num_feats * 2]

        return pos

if __name__ == '__main__':
    pass