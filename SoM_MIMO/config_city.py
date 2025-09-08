# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN
from datetime import datetime
import os
def add_mimo_config_city(cfg):
    current_datetime = datetime.now().strftime("%m%d_%H%M%S")
    cfg.OUTPUT_DIR=os.path.join(cfg.OUTPUT_DIR, current_datetime)
    cfg.MODEL.MIMO = CN()
    cfg.SEED=3407


    cfg.MODEL.WEIGHTS="model/SoM_MIMO_C48.pth"

    cfg.MODEL.MIMO.C=48 # CR= (H/16 * W/16 * C)/(H * W * 3)= C/768
    cfg.MODEL.MIMO.HFF=1
    cfg.MODEL.MIMO.MCE=1
    cfg.MODEL.MIMO.MCE_DEPTH=6

    cfg.MODEL.MIMO.CHANNEL=1
    cfg.MODEL.MIMO.TANH=1
    cfg.MODEL.MIMO.BITS=4
    cfg.MODEL.MIMO.Nt=2
    cfg.MODEL.MIMO.Nr=2
    cfg.MODEL.MIMO.INFER_SNR=15

    cfg.SOLVER.EVAL_REPEAT=50

 

    



