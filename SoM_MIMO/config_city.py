from detectron2.config import CfgNode as CN
from datetime import datetime
import os


def add_mimo_config_city(cfg):
    current_datetime = datetime.now().strftime("%m%d_%H%M%S")
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, current_datetime)

    if not hasattr(cfg.MODEL, "MIMO"):
        cfg.MODEL.MIMO = CN()

    cfg.SEED = 3407
    cfg.MODEL.WEIGHTS = "model/SoM_MIMO_C48.pth"

    # =========================
    # Base SoM-MIMO
    # =========================
    cfg.MODEL.MIMO.C = 48
    cfg.MODEL.MIMO.HFF = 1
    cfg.MODEL.MIMO.MCE = 1
    cfg.MODEL.MIMO.MCE_DEPTH = 6
    cfg.MODEL.MIMO.CHANNEL = 1
    cfg.MODEL.MIMO.TANH = 1
    cfg.MODEL.MIMO.BITS = 4

    cfg.MODEL.MIMO.Nt = 2
    cfg.MODEL.MIMO.Nr = 2
    cfg.MODEL.MIMO.INFER_SNR = 15

    cfg.SOLVER.EVAL_REPEAT = 50

    # =========================
    # WiFo branch
    # =========================
    cfg.MODEL.MIMO.USE_WIFO = True
    cfg.MODEL.MIMO.WIFO_CKPT = "model/wifo_base.pkl"
    cfg.MODEL.MIMO.FREEZE_WIFO = False

    cfg.MODEL.MIMO.T_HIST = 16
    cfg.MODEL.MIMO.T_PRED = 8

    cfg.MODEL.MIMO.CSI_MODE = "pred"
    cfg.MODEL.MIMO.CSI_LAG = 16

    cfg.MODEL.MIMO.USE_DESCRIPTOR = False

    cfg.MODEL.MIMO.WIFO_H = 16
    cfg.MODEL.MIMO.WIFO_W = 16
    cfg.MODEL.MIMO.CSI_DIR = "data/csi"

    cfg.MODEL.MIMO.DOPPLER_FREQ = 120.0
    cfg.MODEL.MIMO.TRAIN_DOPPLER_RANGE = [0.0, 480.0]
    cfg.MODEL.MIMO.TRAIN_SNR_RANGE = [-5.0, 5.0]

    # =========================
    # Adapter
    # =========================
    cfg.MODEL.MIMO.USE_WIFO_ADAPTER = True
    cfg.MODEL.MIMO.WIFO_ADAPTER_HIDDEN = 128
    cfg.MODEL.MIMO.WIFO_ADAPTER_POOL = 4

    # =========================
    # Semantic-aware CSI reliability gate
    # =========================
    cfg.MODEL.MIMO.USE_SEMANTIC_CSI_GATE = True
    cfg.MODEL.MIMO.SEMANTIC_CSI_GATE_HIDDEN = 128
    cfg.MODEL.MIMO.SEMANTIC_CSI_GATE_INIT_BIAS = -1.3862944
    cfg.MODEL.MIMO.SEMANTIC_CSI_GATE_TAU = 0.05
    cfg.MODEL.MIMO.SEMANTIC_CSI_GATE_LOSS_WEIGHT = 0.05

    # =========================
    # CSI auxiliary loss
    # =========================
    cfg.MODEL.MIMO.WIFO_AUX_LOSS_WEIGHT = 0.3

    # =========================
    # LR multipliers
    # =========================
    cfg.MODEL.MIMO.WIFO_LR_MULT = 5.0
    cfg.MODEL.MIMO.WIFO_ADAPTER_LR_MULT = 10.0
    cfg.MODEL.MIMO.BACKBONE_LR_MULT = 1.0

    # =========================
    # Legacy residual gate flag, kept disabled.
    # =========================
    cfg.MODEL.MIMO.USE_RESIDUAL_GATE = False
    cfg.MODEL.MIMO.RESIDUAL_GATE_HIDDEN = 64
    cfg.MODEL.MIMO.ENABLE_CSI_DUMP = False
    cfg.MODEL.MIMO.CSI_DUMP_DIR = "ana/csi_dump"
    cfg.MODEL.MIMO.CSI_DUMP_MAX_SAMPLES = 500
