import torch
import torch.nn as nn
from .backbone import Backbone, ResNet50
import torch.nn.functional as F
from .Transfomer import Transformer

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class CSKDet(nn.Module):
    def __init__(self,
                 detach,
                 num_classes: int =  16,
                 num_feats: int = 128,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dropout: float = 0.1,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 resnet: bool = True,

                 ):
        super(CSKDet, self).__init__()
        if resnet:
            self.backbone = ResNet50()
        else:
            self.backbone = Backbone()

        self.up1 = Up(2048 + 1024, 1024)
        self.up2 = Up(1024 + 512, 512)
        self.up3 = Up(512 + 256, 64)
        self.up4 = Up(64 + 64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.transformer = Transformer(num_feats = num_feats,
                                       num_encoder_layers = num_encoder_layers,
                                       num_decoder_layers = num_decoder_layers,
                                       dropout = dropout,
                                       n_head = nhead,
                                       dim_embed = dim_feedforward,
                                       detach = detach
                                       )

    def forward(self, img_s, img_q, kp_s, mask_s):
        bs, nshot, c, H, W = img_s.shape

        img_s_reshape = img_s.view(bs * nshot, c, H, W)
        with torch.no_grad():
            _, _, _, _, feature_s = self.backbone(img_s_reshape)
        feature_s = feature_s.view(bs, nshot, *feature_s.shape[1:])


        x1, x2, x3, x4, feature_q = self.backbone(img_q)
        ratio_heatmap = 4
        ratio_feature = H // feature_q.shape[2]

        similarity = self.up1(feature_q, x4)
        similarity = self.up2(similarity, x3)
        similarity = self.up3(similarity, x2)
        similarity = self.up4(similarity, x1)
        similarity = self.out_conv(similarity)

        init_proposal, _ = self.get_max_preds(similarity)

        kp_s = kp_s // ratio_heatmap
        coordinates_list = self.transformer(
            feature_q=feature_q,
            feature_s=feature_s,
            kp_s=kp_s,
            mask_s=mask_s,
            coordinates=init_proposal,
            ratio_feature=ratio_feature,
            ratio_heatmap=ratio_heatmap,
        )

        return similarity, coordinates_list

    @staticmethod
    def get_max_preds(batch_heatmaps):
        assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
        assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

        batch_size, num_joints, h, w = batch_heatmaps.shape
        heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
        maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

        maxvals = maxvals.unsqueeze(dim=-1)
        idx = idx.float()

        preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps.device)

        preds[:, :, 0] = idx % w
        preds[:, :, 1] = torch.floor(idx / w)

        pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)  # float会将bool张量转化为1.0和0.0

        preds *= pred_mask
        return preds, maxvals



