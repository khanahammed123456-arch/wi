import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool,build_resnet_fpn_backbone 
from detectron2.layers import ShapeSpec
import fvcore.nn.weight_init as weight_init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .channel import Channel
from .HFF_HFS import Pyramid_U_encoder,Pyramid_U_decoder
from .MCE_MCD import MIMO_encoder, MIMO_decoder
class SoM_MIMO_bacbone(Backbone):
    def __init__(self,cfg,source):
        super(SoM_MIMO_bacbone, self).__init__()
        self.source = source
        self.cfg=cfg
        self.C=cfg.MODEL.MIMO.C
        self.Nt=cfg.MODEL.MIMO.Nt
        self.Nr=cfg.MODEL.MIMO.Nr
        self.activate=nn.Tanh()
        if cfg.MODEL.MIMO.HFF:
            self.encoder= Pyramid_U_encoder(cfg)
            self.decoder= Pyramid_U_decoder(cfg)         
        if cfg.MODEL.MIMO.CHANNEL:
            self.channel= Channel(cfg)
        if cfg.MODEL.MIMO.MCE:
            self.diversity_encoder=MIMO_encoder(cfg)
            self.diversity_decoder=MIMO_decoder(cfg)
        if cfg.MODEL.MIMO.HFF:
            self.compress=nn.Conv2d(in_channels=256, out_channels=self.C, kernel_size=1)
            self.decompress=nn.Conv2d(in_channels=self.C, out_channels=256, kernel_size=1)  
        else:
            self.compress=nn.Identity()
            self.decompress=nn.Identity()

    @property
    def size_divisibility(self):
        return self.source._size_divisibility
    @property
    def padding_constraints(self):
        return {"square_size": self.source._square_pad}

    def before_channel(self,x):
        x=self.compress(x)
        [B,C,H,W]=x.shape
        x=x.contiguous().view(B,self.Nt,-1)
        if self.cfg.MODEL.MIMO.TANH: #nonlinear activation
            x=self.activate(x)
        return x,[B,C,H,W]

    def after_channel(self,x,shape):
        [B,C,H,W]=shape
        x=x.reshape(B,C,H,W)
        x=self.decompress(x)
        return x
    def forward(self, x):
        pyramid_tx = self.source(x)
        if self.cfg.MODEL.MIMO.CHANNEL: #Pass Channel
            [h,noise_std]= self.channel.channel_generator(x.get_device(), x.size(0))
            if self.cfg.MODEL.MIMO.HFF: 
                p_tx=self.encoder(pyramid_tx)
                if self.cfg.MODEL.MIMO.MCE:
                    tx_feature= self.diversity_encoder(p_tx,h,noise_std)
                else:
                    tx_feature=p_tx
                tx_feature,shape=self.before_channel(tx_feature)
                rx_feature=self.channel(tx_feature,h,noise_std)
                rx_feature=self.after_channel(rx_feature,shape)
                if self.cfg.MODEL.MIMO.MCE:
                    p_rx =self.diversity_decoder(rx_feature,h,noise_std)
                else:
                    p_rx= rx_feature
                pyramid_rx=self.decoder(p_rx)
                return pyramid_rx,pyramid_rx,pyramid_tx
            else: # Channel Only
                tx_feature,shape=self.channel.source_to_channel(pyramid_tx)
                rx_feature=self.channel(tx_feature,h,noise_std)
                pyramid_rx= self.channel.channel_to_source(rx_feature,shape)
                return pyramid_rx, pyramid_rx, pyramid_tx
        else: #Without Channel 
            if self.cfg.MODEL.MIMO.MCE: # Compression Only
                p_tx=self.encoder(pyramid_tx)
                p_tx=self.compress(p_tx)
                p_tx=self.activate(p_tx)
                p_tx=self.decompress(p_tx)
                pyramid_rx=self.decoder(p_tx)
                return pyramid_rx, pyramid_rx, pyramid_tx
            else: # Direct forward
                return pyramid_tx, pyramid_tx, pyramid_tx

    def output_shape(self):
        return self.source.output_shape()
        

@BACKBONE_REGISTRY.register()
def build_som_mimo_backbone_CITY(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    fpn_source = build_resnet_fpn_backbone(cfg, input_shape)
    backbone = SoM_MIMO_bacbone(cfg,source=fpn_source)
    return backbone