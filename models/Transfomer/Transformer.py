import torch
import torch.nn as nn
import copy
from models.utils.heatmap import KeypointToHeatMap
from typing import Optional
from torch import Tensor
from models.utils.SinePositionEncoding import SinePositionalEncoding2D

import torch.nn.functional as F
import math

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Transformer(nn.Module):
    def __init__(self,
                 detach,
                 num_feats: int = 128,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dropout: float = 0.1,
                 n_head: int = 8,
                 dim_embed: int = 2048,

                 ):
        super(Transformer,self).__init__()
        self.encoder = Encoder(num_feats = num_feats, num_layers = num_encoder_layers, dropout = dropout, dim_embed = dim_embed, num_heads = n_head)
        self.decoder = Decoder(num_layers = num_decoder_layers, num_feats = num_feats, dim_embed = dim_embed, num_heads = n_head, dropout = dropout, detach = detach)
        self.pos_encod = SinePositionalEncoding2D(num_feats=num_feats, normalize=True)

    def forward(self, feature_q, feature_s, kp_s, mask_s, coordinates, ratio_heatmap, ratio_feature):

        """
        :param feature_q: feature map of query image [bs, 2048, h, w]
        :param feature_s: feature map of support image [bs, nshot, 2048, h, w]
        :param kp_s: keypoints of support image [bs, nshot, num_kp, 2] (//ratio_heatmap)
        :param mask_s: mask of support kp [bs, num_kp, 1] :"1" means efficient while "0" means not efficient
        :param coordinates: coordinates of kps [bs, num_kp, 2] (//ratio_heatmap)
        :param ratio_heatmap: 4
        :param ratio_feature: H // h
        :return:
        """

        feature_q, feature_s, pos = self.encoder(feature_q = feature_q,
                                                 feature_s = feature_s,
                                                 kp_s = kp_s,
                                                 mask_s = mask_s,
                                                 pos_encoding = self.pos_encod,
                                                 ratio_heatmap = ratio_heatmap,
                                                 ratio_feature = ratio_feature)


        coordinates_list = self.decoder(feature_s = feature_s,
                                        feature_q = feature_q,
                                        mask_s = mask_s,
                                        coordinates = coordinates,
                                        pos_encoding = self.pos_encod,
                                        pos = pos,)

        return coordinates_list


class EncoderLayer(nn.Module):
    def __init__(self,
                 num_feats: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 dim_embed: int = 2048,
                 ):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim= num_feats * 2 ,num_heads= num_heads, dropout = dropout)

        self.fc1 = nn.Linear(num_feats * 2, dim_embed)
        self.fc2 = nn.Linear(dim_embed, num_feats * 2)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_feats * 2)
        self.norm2 = nn.LayerNorm(num_feats * 2)

    def forward(self,
                feature_fused,
                mask_q,
                mask_s,
                pos: Optional[Tensor] = None):

        """
        :param feature_fused: concate feature_q and feature_s [n+hw, bs, c]
        :param mask_q: mask of query [bs, h, w]
        :param mask_s: mask of support keypoint [bs, 1, num_kp]
        :param pos: concate pos_q and identifier(Is) [hw + n, bs, c]
        :return:feature_fused
        """

        # add pos to feature
        if pos is not None:
            feature_fused2 = feature_fused + pos
        else:
            feature_fused2 = feature_fused
            # print('encoder lack of pos')

        # Self-Attention
        q = k = feature_fused2

        mask_q = mask_q.reshape(mask_q.size(0), -1)
        mask_s = mask_s.flatten(1)
        mask_fused = torch.cat((mask_q, mask_s),dim = 1)

        attn_output, _ = self.attn(q,
                                   k,
                                   value = feature_fused2,
                                   attn_mask = None,
                                   key_padding_mask = mask_fused,)

        # FFN
        out = feature_fused2 + self.dropout1(attn_output)
        out = self.norm1(out)
        out1 = self.fc2(self.dropout(F.relu(self.fc1(out))))
        out = out + self.dropout2(out1)
        out = self.norm2(out)

        return out

class Encoder(nn.Module):

    def __init__(self,
                 in_c: int = 2048,
                 out_c: int = 256,
                 num_layers: int = 3,
                 num_feats: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 dim_embed: int = 2048,
                 ):
        super(Encoder, self).__init__()
        encoder_layer = EncoderLayer(num_feats = num_feats, num_heads = num_heads, dropout = dropout, dim_embed = dim_embed)
        self.out_c = out_c
        self.in_conv = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.pos_encod = SinePositionalEncoding2D(num_feats=num_feats, normalize=True)
        self.norm = nn.LayerNorm(num_feats * 2)

    def forward(self, feature_s, feature_q, kp_s, mask_s, pos_encoding, ratio_heatmap, ratio_feature):

        """
        :param feature_s: feature of support image [bs, nshot, c, h, w]
        :param feature_q: feature of query image [bs, c, h, w]
        :param kp_s: coordinates of keypoints [bs, nshot, num_keypoints, 2] (ratio_heatmap)
        :param mask_s: mask of support keypoints feature [bs, num_keypoints, 1]
        :param pos_encod: instantiation positional encoding
        :param ratio_heatmap: 4
        :param ratio_feature: H // h
        :return:
        """
        bs, nshot, c, h, w = feature_s.shape
        H = W = h * ratio_feature
        h_heatmap = w_heatmap = H // ratio_heatmap

        _, _, K, _ = kp_s.shape

        feature_s = feature_s.view(bs*nshot, c, h, w)
        feature_s = self.in_conv(feature_s)
        feature_s = feature_s.view(bs, nshot, self.out_c, h, w)

        feature_q = self.in_conv(feature_q)
        feature_q = feature_q.flatten(2).permute(2, 0, 1)

        # generate feature of kp
        ################################################
        sup_k_embed_list = []
        transform = KeypointToHeatMap(gaussian_sigma=2)
        images = torch.zeros((nshot, c, h_heatmap, w_heatmap))
        device = feature_s.device
        for i in range(bs):
            visible = mask_s[0].repeat(nshot, 1, 1).flatten(1)
            # torch.Size([10, 16])
            targets = {
                "keypoints": kp_s[i].clone().detach(),
                "visible": visible.clone().detach().to(torch.float32)

            }
            _, target = transform(images, targets)
            # feature_s[i]: torch.Size([10, 256, 7, 7])
            feature_extend_s = F.interpolate(feature_s[i], size=(h_heatmap, w_heatmap), mode='bilinear', align_corners=False)
            heatmap_numpy = target["heatmap"].numpy()
            heatmap = torch.as_tensor(heatmap_numpy, dtype=torch.float32).to(device)
            # heapmap: torrch.Size([10, 16, 56, 56])
            heatmap = heatmap / (heatmap.sum(dim=-1).sum(dim=-1)[:, :, None, None] + 1e-8)
            feature_extend_s = heatmap.flatten(2)@ feature_extend_s.flatten(2).permute(0, 2, 1) # (nshot, num_kp, dim_c)
            # feature_extend_s: torch.Size([10, 16, 256])
            feature_extend_s = feature_extend_s.mean(dim=0)
            sup_k_embed_list.append(feature_extend_s)

            # feature_s = Soft_ROI_Pooling(feature_s, kp_s)
        ################################################
        feature_s = torch.stack(sup_k_embed_list, dim=0)

        feature_s = feature_s * mask_s
        feature_s = feature_s.permute(1, 0, 2)

        feature_fused = torch.cat((feature_q, feature_s), dim = 0)

        # generate positional encoding
        mask_q = feature_q.new_zeros(bs, h, w).to(torch.bool)
        pos_q = pos_encoding(mask_q).flatten(2).permute(2, 0, 1)  # [n, bs, embed]

        # pos_s: support keypoint identifier
        masks = feature_s.new_zeros(bs, 1, K).to(torch.bool)
        pos_s = pos_encoding(masks).flatten(2).permute(2, 0, 1)  # [K, bs, embed]

        pos = torch.cat((pos_q, pos_s))
        mask_s = (~mask_s.to(torch.bool))
        # input feature_fused into encoder
        for layer in self.layers:
            feature_fused = layer(feature_fused, mask_q, mask_s, None)
            # feature_fused = layer(feature_fused, mask_q, mask_s, pos)

        feature_fused = self.norm(feature_fused)

        # split feature_fused to feature_q and feature_s
        feature_s = feature_fused[h * w:, :, :]
        feature_q = feature_fused[:h * w, :, :]

        return feature_q, feature_s, pos

class DecoderLayer(nn.Module):
    def __init__(self,
                 num_feats: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_mlp_layers: int = 1,
                 dim_embed: int = 2048,):

        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim= num_feats * 2 ,num_heads= num_heads, dropout = dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim= num_feats * 2, num_heads= num_heads, dropout = dropout)
        self.mlp = MLP(input_dim= num_feats * 2, hidden_dim= num_feats * 2, output_dim = 2, num_layers = num_mlp_layers)

        # FFN1
        self.fc11 = nn.Linear(num_feats * 2, dim_embed)
        self.dropout1 = nn.Dropout(dropout)
        self.fc12 = nn.Linear(dim_embed, num_feats * 2)
        self.norm11 = nn.LayerNorm(num_feats * 2)
        self.norm12 = nn.LayerNorm(num_feats * 2)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)

        # FFN2
        self.fc21 = nn.Linear(num_feats * 2, dim_embed)
        self.dropout2 = nn.Dropout(dropout)
        self.fc22 = nn.Linear(dim_embed, num_feats * 2)
        self.norm21 = nn.LayerNorm(num_feats * 2)
        self.norm22 = nn.LayerNorm(num_feats * 2)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)


    def forward(self,feature_q, feature_s, coordinates, pos_encoding, pos, mask_s):

        """
        :param feature_q: feature of query image [hw, bs, c]
        :param feature_s: feature of support keypoints [num_kp, bs, c]
        :param coordinates: predicted coordinates of keypoints [num_kp, bs, 2]
        :param pos_encoding: class positional encoding
        :param pos: concate positional encoding of support keypoints and positional encoding of query [hw + num_kp, bs, c]
        :param mask_s: mask of support keypoints [bs, 1, num_kp]
        :return: fixed coordinates and refined feature_s
        """

        hw, bs, c = feature_q.shape
        h = w = int(math.sqrt(hw))

        # generate pos for self-attention
        coordinates_embed = pos_encoding.forward_coordinates(coordinates).permute(1, 0, 2)
        # [bs, kpt, num_feats * 2]
        self_embed = coordinates_embed + pos[hw:]

        mask_s = mask_s.flatten(1)
        # add pos to Fs
        self_q = self_k = self_embed + feature_s
        output, _ = self.self_attn(self_q,
                                   self_k,
                                   value = feature_s,
                                   attn_mask = None,
                                   key_padding_mask = mask_s)

        # FFN
        out = feature_s + self.dropout1(output)
        feature_s = self.norm11(out)

        #generate pos for cross-attention
        pos_s = coordinates_embed + pos[hw:]
        pos_q = pos[:hw]

        mask_q = feature_q.new_zeros((bs, h, w)).to(torch.bool)
        mask_q = mask_q.flatten(1)

        cross_q = feature_s + pos_s
        cross_k = feature_q + pos_q
        output, _ = self.cross_attn(cross_q,
                                    cross_k,
                                    feature_q,
                                    attn_mask = None,
                                    key_padding_mask = mask_q)

        # FFN
        out = feature_s + self.dropout2(output)
        feature_s = self.norm21(out)
        out = self.fc22(self.dropout21(F.relu(self.fc21(feature_s))))
        feature_s = feature_s + self.dropout22(out)
        feature_s = self.norm22(feature_s)

        # MLP
        offset = self.mlp(feature_s.transpose(0, 1))
        coordinates = coordinates + offset

        return coordinates, feature_s

class Decoder(nn.Module):
    def __init__(self,
                 detach,
                 num_feats = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dim_embed: int = 2048,
                 dropout: float = 0.1,):

        super(Decoder, self).__init__()

        decoder_layer = DecoderLayer(num_feats = num_feats,
                                     num_heads = num_heads,
                                     dropout = dropout,
                                     dim_embed = dim_embed,
                                     )
        self.layers = _get_clones(decoder_layer, num_layers)
        self.detach = detach
    def forward(self,feature_q, feature_s, coordinates, pos_encoding, pos, mask_s):

        mask_s = (~mask_s.to(torch.bool))
        if self.detach:
            coordinates = coordinates.detach()  

        coordinates_list = [coordinates]
        for idx, layer in enumerate(self.layers):
            coordinates, feature_s = layer(feature_q = feature_q,
                                           feature_s = feature_s,
                                           coordinates = coordinates,
                                           pos_encoding = pos_encoding,
                                           pos = pos,
                                           mask_s = mask_s)
            coordinates_list.append(coordinates)
        return coordinates_list

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

if __name__ == '__main__':
    feature_q = torch.rand(16, 2048, 7, 7)
    feature_s = torch.rand(1, 2048, 7, 7)
    kp_s = torch.randint(0, 7, (1, 13, 2))
    mask_s = torch.rand(1, 13)
    coordinates = torch.randint(0, 7, (1, 13, 2))
    model = Transformer(num_feats = 128,)
    out = model(feature_q, feature_s,  kp_s, mask_s, coordinates)