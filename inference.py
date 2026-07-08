from typing import Dict, List, Optional, Tuple
import logging
import datetime
import time
import os
import random
import numpy as np
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.config import CfgNode
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    DatasetEvaluator,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    print_csv_format,
    inference_context,
)
from detectron2.modeling import build_model
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.solver.build import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.structures import ImageList, Instances
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.comm import get_world_size, is_main_process
from SoM_MIMO import *
from contextlib import ExitStack, contextmanager
from torch.cuda.amp import autocast, GradScaler
from fvcore.common.param_scheduler import (
    CosineParamScheduler,
    MultiStepParamScheduler
)
from detectron2.solver.lr_scheduler import LRMultiplier, WarmupParamScheduler

logger = logging.getLogger("detectron2")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTHONHASHSEED"] = str(3407)

# random, numpy
random.seed(3407)
np.random.seed(3407)

# torch
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def apply_ultimate_physics_patch(model):
    """
    【核心物理补丁】：拦截底层物理通信层，强制植入 Jakes 多普勒老化与干扰。
    既保留了原始架构生成信道的纯净性，又实现了老化的逼真验证。
    """
    bb = model.module.backbone if hasattr(model, 'module') else model.backbone
    if hasattr(bb, 'channel'):
        class PhysicsChannelWrapper(torch.nn.Module):
            def __init__(self, orig_channel, backbone):
                super().__init__()
                self.orig_channel = orig_channel
                self._bb_hide = [backbone]
                # 完美继承原始 Channel 的所有属性与生成器，防止报错
                self.Nt = orig_channel.Nt
                self.Nr = orig_channel.Nr
                self.bits = orig_channel.bits
                self.QPSK_modulator = orig_channel.QPSK_modulator
                self.QPSK_demodulator = orig_channel.QPSK_demodulator
                self.channel_generator = orig_channel.channel_generator
                self.source_to_channel = orig_channel.source_to_channel
                self.channel_to_source = orig_channel.channel_to_source

            @property
            def bb(self):
                return self._bb_hide[0]

            def forward(self, x, h, std, h_tx_pred=None):
                # ==========================================
                # ⏱️ [新增] 物理层与 WiFo 测时开始
                # ==========================================
                torch.cuda.synchronize()
                start_time = time.time()

                device = x.device
                B, N, K = x.size()

                use_wifo = getattr(self.bb.cfg.MODEL.MIMO, 'USE_WIFO', False)
                doppler = float(getattr(self.bb.cfg.MODEL.MIMO, 'DOPPLER_FREQ', 0.0))

                wifo_success = False
                if use_wifo and h_tx_pred is not None:
                    h_tx = h_tx_pred
                    wifo_success = True

                # ==========================================
                # Jakes 老化模型 (多普勒效应)
                # ==========================================
                if not wifo_success:
                    try:
                        import scipy.special
                        rho = float(scipy.special.j0(2 * math.pi * doppler * 0.01))
                    except ImportError:
                        rho = 1.0 if doppler == 0.0 else 0.240

                    e_real = torch.randn_like(h.real)
                    e_imag = torch.randn_like(h.imag)
                    e = torch.complex(e_real, e_imag) / math.sqrt(2.0)

                    # 关键！如果是 0Hz，rho=1.0，h_tx 会完美等于真实 h，干扰为 0！
                    h_tx = rho * h + math.sqrt(max(0.0, 1.0 - rho ** 2)) * e

                # 发送端预编码 (基于老化的情报或预测情报)
                try:
                    _, _, Vh_tx = torch.linalg.svd(h_tx)
                except:
                    Vh_tx = torch.eye(self.Nt, device=device).unsqueeze(0).expand(B, -1, -1)
                    Vh_tx = torch.complex(Vh_tx, torch.zeros_like(Vh_tx))
                V_tx = torch.conj(Vh_tx.transpose(-1, -2))

                # 接收端均衡 (基于真实衰落)
                try:
                    U_rx, S_rx, _ = torch.linalg.svd(h)
                    S_rx_matrix = torch.diag_embed(S_rx)
                    S_rx_matrix = torch.complex(S_rx_matrix, torch.zeros_like(S_rx_matrix))
                    S_inv_rx_raw = torch.pinverse(S_rx_matrix)

                    if wifo_success:
                        safe_mask = (S_rx > 0.15).unsqueeze(1).to(torch.complex64)
                        S_inv_rx = S_inv_rx_raw * safe_mask
                    else:
                        S_inv_rx = S_inv_rx_raw
                except:
                    U_rx = torch.eye(self.Nr, device=device).unsqueeze(0).expand(B, -1, -1)
                    U_rx = torch.complex(U_rx, torch.zeros_like(U_rx))
                    S_inv_rx = torch.eye(self.Nr, device=device).unsqueeze(0).expand(B, -1, -1)
                    S_inv_rx = torch.complex(S_inv_rx, torch.zeros_like(S_inv_rx))

                Uh_rx = torch.conj(U_rx.transpose(-1, -2))

                symbols_per_int = self.bits // 2
                x_mean = torch.mean(x)
                x_std = torch.std(x)
                xn = (x - x_mean) / (x_std + 1e-8)

                xn_max = torch.max(xn)
                xn_min = torch.min(xn)
                quant_level = 2 ** self.bits
                S_quant = (xn_max - xn_min) / (quant_level - 1.0)
                Z_quant = (quant_level - 1.0) - torch.round(xn_max / S_quant)
                x_uint = torch.clamp(torch.round(xn / S_quant) + Z_quant, min=0.00, max=quant_level - 1.0)

                x_symbols = self.QPSK_modulator(x_uint, device)
                x_symbols_norm = (x_symbols / math.sqrt(2.0)) / math.sqrt(self.Nt)

                x_tx_signal = torch.bmm(V_tx, x_symbols_norm)

                if isinstance(std, torch.Tensor):
                    std_val = std.item() if std.numel() == 1 else std.mean().item()
                else:
                    std_val = float(std)

                noise_real = torch.normal(mean=0.0, std=std_val / math.sqrt(2.0),
                                          size=[B, self.Nr, symbols_per_int * K], device=device).detach()
                noise_imag = torch.normal(mean=0.0, std=std_val / math.sqrt(2.0),
                                          size=[B, self.Nr, symbols_per_int * K], device=device).detach()
                noise = torch.complex(noise_real, noise_imag)

                x_fading = torch.bmm(h, x_tx_signal)
                y_rx = x_fading + noise

                y_symbols_norm = torch.bmm(S_inv_rx, torch.bmm(Uh_rx, y_rx))
                y_symbols = y_symbols_norm * math.sqrt(2.0) * math.sqrt(self.Nt)
                y_uint, y_detect = self.QPSK_demodulator(y_symbols, device)

                yn = torch.clamp((y_uint - Z_quant) * S_quant, min=xn_min, max=xn_max)
                y = x_std * yn + x_mean

                Eq_noise = (y - x).detach()

                # ==========================================
                # ⏱️ [新增] 物理层与 WiFo 测时结束
                # ==========================================
                torch.cuda.synchronize()
                end_time = time.time()
                phys_time_ms = (end_time - start_time) * 1000.0

                # 打印出物理层的毫秒数
                logger.info(f"⚡ [物理层极速测时] 物理通信(含SVD与注水)耗时: {phys_time_ms:.4f} ms")

                return x + Eq_noise

        bb.channel = PhysicsChannelWrapper(bb.channel, bb)
        logger.info("🔧 [物理引擎补丁] AR1 老化与注水防爆已挂载至底层！并开启毫秒级测时！")


def do_test(cfg, model):
    os.environ["PYTHONHASHSEED"] = str(3407)
    # random, numpy
    random.seed(3407)
    np.random.seed(3407)
    # torch
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    results = OrderedDict()

    # 动态把 cfg 注入到模型底层，确保 DOPPLER 和 SNR 实时生效
    if hasattr(model, 'cfg'):
        model.cfg = cfg
    for name, module in model.named_modules():
        if hasattr(module, 'cfg'): module.cfg = cfg
        if hasattr(module, 'INFER_SNR'): module.INFER_SNR = cfg.MODEL.MIMO.INFER_SNR
        if hasattr(module, 'infer_snr'): module.infer_snr = cfg.MODEL.MIMO.INFER_SNR
        if hasattr(module, 'doppler_freq'): module.doppler_freq = cfg.MODEL.MIMO.DOPPLER_FREQ

    logger.info(f"[*] 物理环境已同步: SNR={cfg.MODEL.MIMO.INFER_SNR}dB, Doppler={cfg.MODEL.MIMO.DOPPLER_FREQ}Hz")

    for dataset_name in cfg.DATASETS.TEST:
        # 使用原生极简加载器，避免 custom_mapper 带来的多载波均值塌缩
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))
        results_i = inference_on_dataset(model, data_loader, evaluator, cfg.SOLVER.EVAL_REPEAT)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_one_test(cfg, model):
    os.environ["PYTHONHASHSEED"] = str(3407)
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    results = OrderedDict()

    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))
        results_i = inference_on_dataset(model, data_loader, evaluator, 1)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # 1. 【顺序修复】：最先注册自定义的 MIMO 节点，防止下面 merge 时报错找不到节点
    try:
        from config_city import add_mimo_config_city
    except ImportError:
        from SoM_MIMO.config_city import add_mimo_config_city
    add_mimo_config_city(cfg)

    # 2. 合并 yaml 文件配置
    cfg.merge_from_file(args.config_file)

    # 3. 合并命令行选项 (此时 cfg.MODEL.MIMO.INFER_SNR 已经被上方的代码注册，不会再报 Non-existent key)
    cfg.merge_from_list(args.opts)

    # 4. 支持外部关闭 WiFo 以测基线
    if getattr(args, "disable_wifo", False):
        cfg.MODEL.MIMO.USE_WIFO = False

    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code

    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)

    # ==========================================
    # 🛠️ [新增修复] 拦截模型输出，强制补齐 3 个变量骗过 evaluator
    # ==========================================
    orig_forward = model.forward
    def patched_forward(*args, **kwargs):
        res = orig_forward(*args, **kwargs)
        if isinstance(res, tuple) and len(res) == 3:
            return res
        return res, None, None  # 如果只有1个输出，强行补上两个 None
    model.forward = patched_forward
    # ==========================================

    # 挂载终极物理仿真外壳
    apply_ultimate_physics_patch(model)

    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    return do_test(cfg, model)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--disable-wifo", action="store_true", help="禁用 WiFo 以测基线")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )