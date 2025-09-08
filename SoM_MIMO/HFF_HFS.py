import torch.nn as nn
import numpy as np
import os
import torch
import time
import math
import torch.nn.functional as F


class Pyramid_U_encoder(nn.Module):
    def __init__(self,cfg,source_name=["p2","p3","p4","p5"],source_channels=256):
        super(Pyramid_U_encoder,self).__init__()
        self.cfg=cfg
        self.C=cfg.MODEL.MIMO.C
    
        self.source_name=source_name
        self.source_channels=source_channels
        
        self.sampler=nn.ModuleDict()
        self.fusion = nn.ModuleDict()
        self.SE=nn.ModuleDict()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        ## P2 & P3 
        self.sampler["p2"]= nn.Conv2d(in_channels=self.source_channels,out_channels=self.source_channels,kernel_size=2, stride=2)
        self.fusion["p2+p3"]=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(512),
        )
        self.SE["p2+p3"]=nn.Sequential(
            nn.Linear(512,32),
            nn.GELU(),
            nn.Linear(32,512),
            nn.Sigmoid(),
        )
        self.fusion["p2_3"]=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1)

        ## P2_3 & P4 
        self.sampler["p2_3"]= nn.Conv2d(in_channels=self.source_channels,out_channels=self.source_channels,kernel_size=2, stride=2)
        self.fusion["p2_3+p4"]=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(512),
        )
        self.SE["p2_3+p4"]=nn.Sequential(
            nn.Linear(512,32),
            nn.GELU(),
            nn.Linear(32,512),
            nn.Sigmoid(),
        )
        self.fusion["p3_4"]=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1)

        ## P3_4 & P5 
        self.sampler["p5"]= nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True)
        self.fusion["p3_4+p5"]=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(512),
        )
        self.SE["p3_4+p5"]=nn.Sequential(
            nn.Linear(512,32),
            nn.GELU(),
            nn.Linear(32,512),
            nn.Sigmoid(),
        )
        self.fusion["p4_5"]=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1)

     
    def forward(self,x):
        
        p2=x["p2"]
        p3=x["p3"]
        p4=x["p4"]
        p5=x["p5"]
        B,_,_,_=p2.shape
        p2_s=self.sampler["p2"](p2)
        p2_3=self.fusion["p2+p3"](torch.cat((p2_s,p3),dim=1))
        pool=self.global_avg_pool(p2_3).reshape(B,512)
        attn=self.SE["p2+p3"](pool).unsqueeze(2).unsqueeze(3)
        p2_3=self.fusion["p2_3"](attn*p2_3)

        p2_3_s=self.sampler["p2_3"](p2_3)
        p3_4=self.fusion["p2_3+p4"](torch.cat((p2_3_s,p4),dim=1))
        pool=self.global_avg_pool(p3_4).reshape(B,512)
        attn=self.SE["p2_3+p4"](pool).unsqueeze(2).unsqueeze(3)
        p3_4=self.fusion["p3_4"](attn*p3_4)

        p5_s=self.sampler["p5"](p5)
        p4_5=self.fusion["p3_4+p5"](torch.cat((p3_4,p5_s),dim=1))
        pool=self.global_avg_pool(p4_5).reshape(B,512)
        attn=self.SE["p3_4+p5"](pool).unsqueeze(2).unsqueeze(3)
        p4_5=self.fusion["p4_5"](attn*p4_5)        

        p_c=p4_5
        return p_c

class Pyramid_U_decoder(nn.Module):
    def __init__(self,cfg,source_name=["p2","p3","p4","p5","p6"],source_channels=256):
        super(Pyramid_U_decoder,self).__init__()
        self.cfg=cfg
        self.C=cfg.MODEL.MIMO.C
    
        self.source_name=source_name
        self.source_channels=source_channels
        

        self.sampler=nn.ModuleDict()
        self.seperate = nn.ModuleDict()
        self.SE=nn.ModuleDict()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        ## P3_4 & P5 
        self.seperate["p4_5"]=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1)
        self.SE["p3_4+p5"]=nn.Sequential(
            nn.Linear(512,32),
            nn.GELU(),
            nn.Linear(32,512),
            nn.Sigmoid(),
        )
        self.seperate["p3_4+p5"]=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(512),
        )
        self.sampler["p5"]= nn.MaxPool2d(2,stride=2)
        self.sampler["p6"]= nn.MaxPool2d(1,stride=2)

        ## P2_3 & P4 
        self.seperate["p3_4"]=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1)
        self.SE["p2_3+p4"]=nn.Sequential(
            nn.Linear(512,32),
            nn.GELU(),
            nn.Linear(32,512),
            nn.Sigmoid(),
        )
        self.seperate["p2_3+p4"]=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(512),
        )

        ## P2 & P3
        self.seperate["p2_3"]=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1)
        self.SE["p2+p3"]=nn.Sequential(
            nn.Linear(512,32),
            nn.GELU(),
            nn.Linear(32,512),
            nn.Sigmoid(),
        )
        self.seperate["p2+p3"]=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(512),
        )
        self.sampler["p3"]= nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True)
        self.sampler["p2"]= nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True)
    

    def forward(self,x):
        output={}
        B,_,_,_=x.shape
        p=x
        p4_5=self.seperate["p4_5"](p)
        pool=self.global_avg_pool(p4_5).reshape(B,512)
        attn=self.SE["p3_4+p5"](pool).unsqueeze(2).unsqueeze(3)
        p3_4_p5=self.seperate["p3_4+p5"](attn*p4_5)

        output["p5"]=self.sampler["p5"](p3_4_p5[:,256:,:,:])
        output["p6"]=self.sampler["p6"](output["p5"])
        

        p3_4=self.seperate["p3_4"](p3_4_p5[:,0:256,:,:])
        pool=self.global_avg_pool(p3_4).reshape(B,512)
        attn=self.SE["p2_3+p4"](pool).unsqueeze(2).unsqueeze(3)
        p2_3_p4=self.seperate["p2_3+p4"](attn*p3_4)

        output["p4"]=p2_3_p4[:,256:,:,:]
        
        p2_3=self.seperate["p2_3"](self.sampler["p3"](p2_3_p4[:,0:256,:,:]))
        pool=self.global_avg_pool(p2_3).reshape(B,512)
        attn=self.SE["p2+p3"](pool).unsqueeze(2).unsqueeze(3)
        p2_p3=self.seperate["p2+p3"](attn*p2_3)
        
        output["p3"]=p2_p3[:,256:,:,:]
        output["p2"]=self.sampler["p2"](p2_p3[:,0:256,:,:])
     
        return output