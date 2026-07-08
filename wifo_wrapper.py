import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

# 让 SoM_MIMO/wifo_core 可被直接 import
current_dir = os.path.dirname(os.path.abspath(__file__))
wifo_core_path = os.path.join(current_dir, "wifo_core")
if wifo_core_path not in sys.path:
    sys.path.insert(0, wifo_core_path)

from wifo_core.model import WiFo


class WiFoPredictor(nn.Module):
    """
    真实 WiFo 前向桥接器

    输入:
        h_history_wifo: [B, T_hist, H, W, 2]

    输出:
        h_pred: [B, T_pred, H, W, 2]

    注意:
    1. WiFo core 真正吃的是 [B, 2, T, H, W]
    2. model.py 的 unpatchify() 返回 [B, T, H, W] complex
    3. temporal masking 是合法的 mask strategy
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.t_hist = int(cfg.MODEL.MIMO.T_HIST)
        self.t_pred = int(cfg.MODEL.MIMO.T_PRED)
        self.freeze_wifo = bool(getattr(cfg.MODEL.MIMO, "FREEZE_WIFO", True))
        self.weights_path = getattr(cfg.MODEL.MIMO, "WIFO_CKPT", "")

        # 与 wifo_core/model.py 对齐
        self.args = SimpleNamespace(
            size="base",
            t_patch_size=4,
            patch_size=4,
            pos_emb="SinCos",
            no_qkv_bias=0,
            mask_ratio=0.0,
            mask_strategy="temporal",
            dataset="Proxy_Data",
        )

        self.model = WiFo(
            patch_size=self.args.patch_size,
            t_patch_size=self.args.t_patch_size,
            in_chans=2,
            embed_dim=512,
            depth=6,
            decoder_embed_dim=512,
            decoder_depth=4,
            decoder_num_heads=8,
            num_heads=8,
            pos_emb=self.args.pos_emb,
            args=self.args,
        )

        if not self.weights_path or not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"WIFO_CKPT not found: {self.weights_path}")

        self._load_weights(self.weights_path)

        if self.freeze_wifo:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        self._printed_debug = False

    def _load_weights(self, weights_path: str):
        ckpt = torch.load(weights_path, map_location="cpu")

        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        print("[WiFoPredictor] load_state_dict:", msg)

    def _to_wifo_layout(self, h_history_wifo: torch.Tensor) -> torch.Tensor:
        """
        [B, T, H, W, 2] -> [B, 2, T, H, W]
        """
        if h_history_wifo.dim() != 5 or h_history_wifo.shape[-1] != 2:
            raise ValueError(
                f"Expected h_history_wifo shape [B, T, H, W, 2], "
                f"got {tuple(h_history_wifo.shape)}"
            )
        return h_history_wifo.permute(0, 4, 1, 2, 3).contiguous()

    def _check_patch_constraints(self, x_input: torch.Tensor):
        """
        WiFo core 要求:
            T % t_patch_size == 0
            H % patch_size == 0
            W % patch_size == 0
        """
        _, _, T, H, W = x_input.shape
        tp = int(self.args.t_patch_size)
        pp = int(self.args.patch_size)

        if T % tp != 0 or H % pp != 0 or W % pp != 0:
            raise ValueError(
                f"WiFo input shape [B,2,T,H,W]={tuple(x_input.shape)} "
                f"does not satisfy patch constraints: "
                f"T % {tp} == 0, H % {pp} == 0, W % {pp} == 0"
            )

    def forward(self, h_history_wifo: torch.Tensor) -> torch.Tensor:
        """
        h_history_wifo: [B, T_hist, H, W, 2]
        return:
            h_pred: [B, T_pred, H, W, 2]
        """
        if h_history_wifo is None:
            return None

        device = h_history_wifo.device
        self.model = self.model.to(device)

        B, T, H, W, C = h_history_wifo.shape
        if T != self.t_hist:
            raise ValueError(f"Expected T_hist={self.t_hist}, got input T={T}")

        tp = int(self.args.t_patch_size)
        if self.t_hist % tp != 0 or self.t_pred % tp != 0:
            raise ValueError(
                f"T_HIST and T_PRED must both be divisible by WiFo t_patch_size={tp}; "
                f"got T_HIST={self.t_hist}, T_PRED={self.t_pred}"
            )

        # Append masked future slots. WiFo keeps historical temporal patches
        # and reconstructs the future horizon from its mask tokens.
        future_placeholder = torch.zeros(
            B,
            self.t_pred,
            H,
            W,
            C,
            device=device,
            dtype=h_history_wifo.dtype,
        )
        h_input_full = torch.cat([h_history_wifo, future_placeholder], dim=1)

        # [B, T_hist + T_pred, H, W, 2] -> [B, 2, T, H, W]
        x_input = self._to_wifo_layout(h_input_full)

        # patch 尺寸硬约束检查
        self._check_patch_constraints(x_input)

        if not self._printed_debug:
            print("[WiFoPredictor] REAL WiFo branch")
            print("[WiFoPredictor] x_input shape:", tuple(x_input.shape))
            print("[WiFoPredictor] prediction horizon:", self.t_pred)
            print("[WiFoPredictor] x_input dtype before list:", x_input.dtype)
            self._printed_debug = True

        # model.py 的 forward() 第一行是:
        # imgs = torch.stack(imgs).squeeze(1)
        # 所以这里传 list，每个元素 shape = [1, 2, T, H, W]
        x_input_list = [x.unsqueeze(0) for x in x_input]

        ratio = float(self.t_pred) / float(self.t_hist + self.t_pred)

        with torch.no_grad():
            # 关键修复：
            # 1) 显式转 float32，避免 AMP 把它变成 half
            x_input_list = [x.float() for x in x_input_list]

            # 2) 只在 WiFo 这段关闭 autocast
            with torch.cuda.amp.autocast(enabled=False):
                _, _, pred_complex_tokens, _, _ = self.model(
                    x_input_list,
                    mask_ratio=ratio,
                    mask_strategy="temporal",
                )

                # model.py 的 unpatchify() 输出: [B, T, H, W] complex
                h_rec = self.model.unpatchify(pred_complex_tokens)

                # 显式保证 complex64，避免 ComplexHalf warning 后续扩散
                if torch.is_complex(h_rec):
                    h_rec = h_rec.to(torch.complex64)

        if not torch.is_complex(h_rec):
            raise ValueError(
                f"Expected complex output from unpatchify(), "
                f"got dtype={h_rec.dtype}, shape={tuple(h_rec.shape)}"
            )

        # 取最后一个时间步作为未来预测
        # [B, H, W] complex
        # Reconstructed future CSI horizon: [B, T_pred, H, W] complex.
        h_pred_complex = h_rec[:, self.t_hist:self.t_hist + self.t_pred, :, :]

        # -> [B, T_pred, H, W, 2]
        h_pred = torch.stack(
            [h_pred_complex.real, h_pred_complex.imag],
            dim=-1
        ).to(torch.float32)

        # 如果 T_pred > 1，先简单重复
        return h_pred
