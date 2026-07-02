import math
import os
import torch
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone

from .channel import Channel
from .HFF_HFS import Pyramid_U_encoder, Pyramid_U_decoder
from .MCE_MCD import MIMO_encoder, MIMO_decoder
from wifo_wrapper import WiFoPredictor

class SoM_MIMO_bacbone(Backbone):
    def __init__(self, cfg, source):
        super(SoM_MIMO_bacbone, self).__init__()
        self.source = source
        self.cfg = cfg

        self.C = cfg.MODEL.MIMO.C
        self.Nt = cfg.MODEL.MIMO.Nt
        self.Nr = cfg.MODEL.MIMO.Nr
        self.activate = nn.Tanh()

        if cfg.MODEL.MIMO.HFF:
            self.encoder = Pyramid_U_encoder(cfg)
            self.decoder = Pyramid_U_decoder(cfg)

        if cfg.MODEL.MIMO.CHANNEL:
            self.channel = Channel(cfg)

        if cfg.MODEL.MIMO.MCE:
            self.diversity_encoder = MIMO_encoder(cfg)
            self.diversity_decoder = MIMO_decoder(cfg)

        if cfg.MODEL.MIMO.HFF:
            self.compress = nn.Conv2d(in_channels=256, out_channels=self.C, kernel_size=1)
            self.decompress = nn.Conv2d(in_channels=self.C, out_channels=256, kernel_size=1)
        else:
            self.compress = nn.Identity()
            self.decompress = nn.Identity()

        self.use_wifo = bool(getattr(cfg.MODEL.MIMO, "USE_WIFO", False))
        if self.use_wifo:
            self.wifo = WiFoPredictor(cfg)
        else:
            self.wifo = None

        self.use_wifo_adapter = bool(getattr(cfg.MODEL.MIMO, "USE_WIFO_ADAPTER", True))
        if self.use_wifo_adapter:
            hidden_dim = int(getattr(cfg.MODEL.MIMO, "WIFO_ADAPTER_HIDDEN", 128))
            pool_hw = int(getattr(cfg.MODEL.MIMO, "WIFO_ADAPTER_POOL", 4))
            adapter_in_dim = 2 * pool_hw * pool_hw + self.Nr * self.Nt * 2

            self.wifo_adapter_pool = nn.AdaptiveAvgPool2d((pool_hw, pool_hw))
            self.wifo_adapter_out = nn.Linear(hidden_dim, self.Nr * self.Nt * 2)
            nn.init.zeros_(self.wifo_adapter_out.weight)
            nn.init.zeros_(self.wifo_adapter_out.bias)
            self.wifo_adapter = nn.Sequential(
                nn.Flatten(),
                nn.Linear(adapter_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                self.wifo_adapter_out,
            )

        # ===== 璋冭瘯璁℃暟 =====
        self._mce_select_debug_count = 0
        self._forward_csi_debug_count = 0
        self._wifo_adapter_debug_count = 0
        self._csi_compare_debug_count = 0
        self._residual_debug_count = 0
        self.csi_log_period = int(getattr(cfg.MODEL.MIMO, "CSI_LOG_PERIOD", 100))

        # ===== eval鏃跺鍑篊SI鏍锋湰 =====
        self.enable_csi_dump = bool(getattr(cfg.MODEL.MIMO, "ENABLE_CSI_DUMP", False))
        self.csi_dump_dir = str(getattr(cfg.MODEL.MIMO, "CSI_DUMP_DIR", "tmp/csi_dump"))
        self.csi_dump_max_samples = int(getattr(cfg.MODEL.MIMO, "CSI_DUMP_MAX_SAMPLES", 500))
        self._csi_dump_count = 0

        if self.enable_csi_dump:
            os.makedirs(self.csi_dump_dir, exist_ok=True)

    @property
    def size_divisibility(self):
        return self.source._size_divisibility

    @property
    def padding_constraints(self):
        return {"square_size": self.source._square_pad}

    def before_channel(self, x):
        x = self.compress(x)
        B, C, H, W = x.shape
        x = x.contiguous().view(B, self.Nt, -1)
        if self.cfg.MODEL.MIMO.TANH:
            x = self.activate(x)
        return x, [B, C, H, W]

    def after_channel(self, x, shape):
        B, C, H, W = shape
        x = x.reshape(B, C, H, W)
        x = self.decompress(x)
        return x

    def _to_complex(self, h):
        if h is None:
            return None
        if torch.is_complex(h):
            return h
        if h.size(-1) != 2:
            raise ValueError(f"Expected last dim == 2 for real/imag tensor, got shape {h.shape}")
        return torch.complex(h[..., 0], h[..., 1])

    def _build_noise_std(self, snr_db, device, batch_size):
        if snr_db is None:
            snr = torch.full(
                (batch_size,),
                float(self.cfg.MODEL.MIMO.INFER_SNR),
                device=device,
                dtype=torch.float32,
            )
        else:
            snr = torch.as_tensor(snr_db, device=device, dtype=torch.float32).view(-1)
            if snr.numel() == 1 and batch_size > 1:
                snr = snr.expand(batch_size)
            if snr.numel() != batch_size:
                raise ValueError(f"Expected {batch_size} SNR values, got shape {tuple(snr.shape)}")

        pn = 1.0 / (10 ** (snr / 10.0))
        return torch.sqrt(pn / self.Nt).view(batch_size, 1)

    def _build_true_channel(self, h_gt_som, device, batch_size, snr_db=None):
        """
        鏋勯€犵湡瀹炰紶鎾俊閬?h_true銆?
        h_true 鐢ㄤ簬 y = h_true x + n
        """
        if h_gt_som is None:
            if self.cfg.MODEL.MIMO.CHANNEL:
                h_true, noise_std = self.channel.channel_generator(device, batch_size)
                if snr_db is not None:
                    noise_std = self._build_noise_std(snr_db, device, batch_size)
                return h_true, noise_std
            return None, None

        h_gt_complex = self._to_complex(h_gt_som)

        if h_gt_complex.dim() == 4:
            h_true = h_gt_complex[:, 0]
        elif h_gt_complex.dim() == 3:
            h_true = h_gt_complex
        else:
            raise ValueError(f"Unexpected h_gt_som shape: {h_gt_som.shape}")

        noise_std = self._build_noise_std(snr_db, device, batch_size)

        if h_true.is_complex():
            h_true = h_true.to(torch.complex64)

        return h_true, noise_std

    def _get_outdated_channel(self, h_history_som):
        """
        鍙栧巻鍙查噷鏈€鏂板彲鐢ㄧ殑 outdated CSI锛屼綔涓烘畫宸娴嬪熀搴曘€?
        """
        if h_history_som is None:
            return None

        h_hist_complex = self._to_complex(h_history_som)
        if h_hist_complex.dim() != 4:
            raise ValueError(f"Unexpected h_history_som shape: {h_history_som.shape}")

        lag = int(getattr(self.cfg.MODEL.MIMO, "CSI_LAG", 1))
        lag = max(1, lag)
        lag = min(lag, h_hist_complex.shape[1])

        h_outdated = h_hist_complex[:, -lag].to(torch.complex64)
        return h_outdated

    def _project_wifo_to_residual(self, h_pred_wifo, h_outdated=None):
        """
        WiFo 杈撳嚭 -> adapter -> 娈嬪樊 delta_h
        """
        if h_pred_wifo is None:
            return None

        if h_pred_wifo.dim() != 5 or h_pred_wifo.shape[-1] != 2:
            raise ValueError(f"Unexpected h_pred_wifo shape: {h_pred_wifo.shape}")

        x = h_pred_wifo[:, 0].to(torch.float32)

        raw_complex = torch.complex(x[..., 0], x[..., 1]).to(torch.complex64)
        if self._wifo_adapter_debug_count < 20:
            print(f"[WIFO RAW] h_pred_wifo_frame_mean={raw_complex.abs().mean().item():.6f}")

        if self.use_wifo_adapter:
            x = x.permute(0, 3, 1, 2).contiguous().to(torch.float32)
            x = self.wifo_adapter_pool(x)
            x = torch.flatten(x, start_dim=1)
            if h_outdated is None:
                outdated_feat = torch.zeros(
                    x.shape[0],
                    self.Nr * self.Nt * 2,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                h_outdated = h_outdated.to(torch.complex64)
                outdated_feat = torch.stack(
                    [h_outdated.real, h_outdated.imag],
                    dim=-1,
                ).reshape(h_outdated.shape[0], -1).to(dtype=x.dtype, device=x.device)
            x = torch.cat([x, outdated_feat], dim=1)
            x = self.wifo_adapter(x)
            x = x.view(x.shape[0], self.Nr, self.Nt, 2)
            delta_h = torch.complex(x[..., 0], x[..., 1]).to(torch.complex64)
        else:
            B, H, W = raw_complex.shape
            if H < self.Nr or W < self.Nt:
                raise ValueError(
                    f"WiFo prediction spatial size {(H, W)} is smaller than SoM target {(self.Nr, self.Nt)}"
                )
            delta_h = raw_complex[:, :self.Nr, :self.Nt]

        if self._wifo_adapter_debug_count < 20:
            print(f"[WIFO DELTA] delta_h_mean={delta_h.abs().mean().item():.6f}")
            self._wifo_adapter_debug_count += 1

        return delta_h

    def _build_pred_channel_from_residual(self, h_outdated, delta_h):
        """
        娈嬪樊棰勬祴锛?
            h_pred_som = h_outdated + delta_h
        """
        if h_outdated is None and delta_h is None:
            return None
        if h_outdated is None:
            return delta_h
        if delta_h is None:
            return h_outdated

        h_pred_som = h_outdated + delta_h

        if self._residual_debug_count < 20:
            print(
                f"[RESIDUAL PRED] "
                f"h_outdated_mean={h_outdated.abs().mean().item():.6f} "
                f"delta_h_mean={delta_h.abs().mean().item():.6f} "
                f"h_pred_som_mean={h_pred_som.abs().mean().item():.6f}"
            )
            self._residual_debug_count += 1

        return h_pred_som

    def _select_mce_channel(self, h_true, h_history_som=None, h_pred_som=None):
        """
        閫夋嫨绯荤粺鍐呴儴鈥滆涓衡€濈殑 CSI锛屽嵆 h_mce
        """
        mode = str(getattr(self.cfg.MODEL.MIMO, "CSI_MODE", "pred")).lower()

        if mode == "true":
            h_mce = h_true
            lag_used = -1

        elif mode == "outdated":
            if h_history_som is None:
                raise ValueError("CSI_MODE='outdated' requires h_history_som")
            h_mce = self._get_outdated_channel(h_history_som)
            lag_used = int(getattr(self.cfg.MODEL.MIMO, "CSI_LAG", 1))

        elif mode == "pred":
            if h_pred_som is not None:
                h_mce = h_pred_som.to(torch.complex64)
            else:
                h_mce = h_true
            lag_used = -1

        else:
            raise ValueError(f"Unknown CSI_MODE: {mode}")

        if self._mce_select_debug_count < 20:
            mismatch = (h_mce - h_true).abs().mean().item()

            if h_history_som is not None:
                h_hist_complex = self._to_complex(h_history_som)
                hist_oldest_mean = h_hist_complex[:, 0].abs().mean().item()
                hist_latest_mean = h_hist_complex[:, -1].abs().mean().item()
            else:
                hist_oldest_mean = -1.0
                hist_latest_mean = -1.0

            true_mean = h_true.abs().mean().item()
            mce_mean = h_mce.abs().mean().item()

            print(
                f"[MCE SELECT] mode={mode} "
                f"lag={lag_used} "
                f"h_true_mean={true_mean:.6f} "
                f"h_mce_mean={mce_mean:.6f} "
                f"hist_oldest_mean={hist_oldest_mean:.6f} "
                f"hist_latest_mean={hist_latest_mean:.6f} "
                f"mismatch_mean={mismatch:.6f}"
            )
            self._mce_select_debug_count += 1

        return h_mce

    def _compute_nmse_per_sample(self, h_est, h_true, eps=1e-8):
        if h_est is None or h_true is None:
            return None
        num = torch.sum(torch.abs(h_est - h_true) ** 2, dim=(-2, -1))
        den = torch.sum(torch.abs(h_true) ** 2, dim=(-2, -1)) + eps
        return num / den

    def _compute_cosine_per_sample(self, h_est, h_true, eps=1e-8):
        if h_est is None or h_true is None:
            return None
        est_ri = torch.stack([h_est.real, h_est.imag], dim=-1).reshape(h_est.shape[0], -1)
        true_ri = torch.stack([h_true.real, h_true.imag], dim=-1).reshape(h_true.shape[0], -1)
        est_norm = torch.norm(est_ri, dim=1) + eps
        true_norm = torch.norm(true_ri, dim=1) + eps
        return torch.sum(est_ri * true_ri, dim=1) / (est_norm * true_norm)

    def _compute_snr_proxy_per_sample(self, h, noise_std, eps=1e-8):
        if h is None or noise_std is None:
            return None
        signal_power = torch.sum(torch.abs(h) ** 2, dim=(-2, -1))
        if isinstance(noise_std, torch.Tensor):
            noise = noise_std.to(device=h.device, dtype=signal_power.dtype).view(h.shape[0], -1).mean(dim=1)
            noise_power = torch.clamp(noise ** 2, min=eps)
        else:
            noise_power = max(float(noise_std) ** 2, eps)
        return signal_power / noise_power

    def _compute_snr_proxy_error_per_sample(self, h_est, h_true, noise_std, eps=1e-8):
        snr_est = self._compute_snr_proxy_per_sample(h_est, noise_std, eps)
        snr_true = self._compute_snr_proxy_per_sample(h_true, noise_std, eps)
        if snr_est is None or snr_true is None:
            return None
        return torch.abs(snr_est - snr_true)

    def _dump_csi_sample(self, h_true, h_outdated, h_pred_som, delta_h, noise_std, snr_db=None):
        if (not self.enable_csi_dump) or self.training:
            return
        if self._csi_dump_count >= self.csi_dump_max_samples:
            return
        if h_true is None or h_outdated is None or h_pred_som is None:
            return

        B = h_true.shape[0]
        for b in range(B):
            if self._csi_dump_count >= self.csi_dump_max_samples:
                break

            save_path = os.path.join(self.csi_dump_dir, f"sample_{self._csi_dump_count:05d}.pt")
            torch.save(
                {
                    "h_true": h_true[b].detach().cpu(),
                    "h_outdated": h_outdated[b].detach().cpu(),
                    "h_pred": h_pred_som[b].detach().cpu(),
                    "delta_h": None if delta_h is None else delta_h[b].detach().cpu(),
                    "noise_std": (
                        None
                        if noise_std is None
                        else float(noise_std[b].detach().cpu().mean())
                        if isinstance(noise_std, torch.Tensor) and noise_std.numel() > 1
                        else float(noise_std)
                    ),
                    "doppler": float(getattr(self.cfg.MODEL.MIMO, "DOPPLER_FREQ", -1.0)),
                    "snr_db": (
                        float(getattr(self.cfg.MODEL.MIMO, "INFER_SNR", -1.0))
                        if snr_db is None
                        else float(torch.as_tensor(snr_db).view(-1)[b].detach().cpu())
                    ),
                    "csi_lag": int(getattr(self.cfg.MODEL.MIMO, "CSI_LAG", 1)),
                },
                save_path,
            )
            self._csi_dump_count += 1

    def _debug_compare_csi(self, h_true, noise_std=None, h_history_som=None, h_pred_som=None, delta_h=None, snr_db=None):
        outdated_err = -1.0
        pred_err = -1.0
        h_pred_mean = -1.0

        nmse_outdated_mean = -1.0
        nmse_pred_mean = -1.0
        cos_outdated_mean = -1.0
        cos_pred_mean = -1.0
        snr_proxy_err_outdated_mean = -1.0
        snr_proxy_err_pred_mean = -1.0

        if h_history_som is not None:
            h_outdated = self._get_outdated_channel(h_history_som)
            outdated_err = (h_outdated - h_true).abs().mean().item()

            nmse_outdated = self._compute_nmse_per_sample(h_outdated, h_true)
            cos_outdated = self._compute_cosine_per_sample(h_outdated, h_true)
            snr_err_outdated = self._compute_snr_proxy_error_per_sample(h_outdated, h_true, noise_std)

            nmse_outdated_mean = nmse_outdated.mean().item()
            cos_outdated_mean = cos_outdated.mean().item()
            snr_proxy_err_outdated_mean = snr_err_outdated.mean().item()
        else:
            h_outdated = None

        if h_pred_som is not None:
            pred_err = (h_pred_som - h_true).abs().mean().item()
            h_pred_mean = h_pred_som.abs().mean().item()

            nmse_pred = self._compute_nmse_per_sample(h_pred_som, h_true)
            cos_pred = self._compute_cosine_per_sample(h_pred_som, h_true)
            snr_err_pred = self._compute_snr_proxy_error_per_sample(h_pred_som, h_true, noise_std)

            nmse_pred_mean = nmse_pred.mean().item()
            cos_pred_mean = cos_pred.mean().item()
            snr_proxy_err_pred_mean = snr_err_pred.mean().item()

        self._log_csi_metrics(
            outdated_err=outdated_err,
            pred_err=pred_err,
            nmse_outdated_mean=nmse_outdated_mean,
            nmse_pred_mean=nmse_pred_mean,
            cos_outdated_mean=cos_outdated_mean,
            cos_pred_mean=cos_pred_mean,
            snr_proxy_err_outdated_mean=snr_proxy_err_outdated_mean,
            snr_proxy_err_pred_mean=snr_proxy_err_pred_mean,
        )

        # 1) 鏃ュ織鍙墦鍗板墠20涓?
        if self._csi_compare_debug_count < 20:
            print(
                f"[CSI COMPARE] "
                f"h_true_mean={h_true.abs().mean().item():.6f} "
                f"h_pred_mean={h_pred_mean:.6f} "
                f"outdated_err={outdated_err:.6f} "
                f"pred_err={pred_err:.6f} "
                f"nmse_outdated_mean={nmse_outdated_mean:.6f} "
                f"nmse_pred_mean={nmse_pred_mean:.6f} "
                f"cos_outdated_mean={cos_outdated_mean:.6f} "
                f"cos_pred_mean={cos_pred_mean:.6f} "
                f"snr_proxy_err_outdated={snr_proxy_err_outdated_mean:.6f} "
                f"snr_proxy_err_pred={snr_proxy_err_pred_mean:.6f}"
            )
            self._csi_compare_debug_count += 1
        elif self.csi_log_period > 0:
            try:
                from detectron2.utils.events import get_event_storage

                storage = get_event_storage()
                if storage.iter % self.csi_log_period == 0:
                    print(
                        f"[CSI METRIC] iter={storage.iter} "
                        f"outdated_err={outdated_err:.6f} "
                        f"pred_err={pred_err:.6f} "
                        f"nmse_outdated={nmse_outdated_mean:.6f} "
                        f"nmse_pred={nmse_pred_mean:.6f} "
                        f"cos_outdated={cos_outdated_mean:.6f} "
                        f"cos_pred={cos_pred_mean:.6f}"
                    )
            except Exception:
                pass

        # 2) dump 涓嶅彈鍓?0涓檺鍒?
        self._dump_csi_sample(
            h_true=h_true,
            h_outdated=h_outdated,
            h_pred_som=h_pred_som,
            delta_h=delta_h,
            noise_std=noise_std,
            snr_db=snr_db,
        )

    def _log_csi_metrics(
        self,
        outdated_err,
        pred_err,
        nmse_outdated_mean,
        nmse_pred_mean,
        cos_outdated_mean,
        cos_pred_mean,
        snr_proxy_err_outdated_mean,
        snr_proxy_err_pred_mean,
    ):
        try:
            from detectron2.utils.events import get_event_storage

            storage = get_event_storage()
            storage.put_scalar("csi/outdated_err", outdated_err)
            storage.put_scalar("csi/pred_err", pred_err)
            storage.put_scalar("csi/nmse_outdated", nmse_outdated_mean)
            storage.put_scalar("csi/nmse_pred", nmse_pred_mean)
            storage.put_scalar("csi/cos_outdated", cos_outdated_mean)
            storage.put_scalar("csi/cos_pred", cos_pred_mean)
            storage.put_scalar("csi/snr_proxy_err_outdated", snr_proxy_err_outdated_mean)
            storage.put_scalar("csi/snr_proxy_err_pred", snr_proxy_err_pred_mean)
        except Exception:
            pass

    def _build_mixed_csi_loss(self, h_pred_som, h_true):
        eps = 1e-6

        pred_scale = h_pred_som.abs().mean(dim=(-2, -1), keepdim=True)
        true_scale = h_true.abs().mean(dim=(-2, -1), keepdim=True)

        h_pred_norm = h_pred_som / (pred_scale + eps)
        h_true_norm = h_true / (true_scale + eps)

        loss_shape = torch.mean(torch.abs(h_pred_norm - h_true_norm))
        loss_scale = torch.mean(torch.abs(h_pred_som - h_true))

        return loss_shape + 0.3 * loss_scale

    def forward(self, x, h_history_wifo=None, h_history_som=None, h_gt_som=None, snr_db=None):
        pyramid_tx = self.source(x)

        # 1) 鐪熷疄浼犳挱淇￠亾
        h_true, noise_std = self._build_true_channel(
            h_gt_som=h_gt_som,
            device=x.device,
            batch_size=x.size(0),
            snr_db=snr_db,
        )

        # 2) outdated CSI
        h_outdated = self._get_outdated_channel(h_history_som)

        # 3) WiFo 棰勬祴 residual
        delta_h = None
        h_pred_som = None
        if self.use_wifo and (h_history_wifo is not None):
            h_pred_wifo = self.wifo(h_history_wifo)
            delta_h = self._project_wifo_to_residual(h_pred_wifo, h_outdated=h_outdated)
            h_pred_som = self._build_pred_channel_from_residual(h_outdated, delta_h)

        # 4) debug compare + dump
        self._debug_compare_csi(
            h_true=h_true,
            noise_std=noise_std,
            h_history_som=h_history_som,
            h_pred_som=h_pred_som,
            delta_h=delta_h,
            snr_db=snr_db,
        )

        # 5) CSI 杈呭姪鎹熷け
        aux_losses = {}
        if self.use_wifo and (h_pred_som is not None):
            loss_wifo_csi = self._build_mixed_csi_loss(h_pred_som, h_true)
            aux_weight = float(getattr(self.cfg.MODEL.MIMO, "WIFO_AUX_LOSS_WEIGHT", 1.0))
            aux_losses["loss_wifo_csi"] = aux_weight * loss_wifo_csi

        # 6) 閫夋嫨绯荤粺璁や负鐨?CSI
        h_mce = self._select_mce_channel(
            h_true=h_true,
            h_history_som=h_history_som,
            h_pred_som=h_pred_som,
        )

        if self._forward_csi_debug_count < 20:
            print(
                f"[FORWARD DEBUG] CSI_MODE={self.cfg.MODEL.MIMO.CSI_MODE} "
                f"h_mce_mean={h_mce.abs().mean().item():.6f} "
                f"h_true_mean={h_true.abs().mean().item():.6f} "
                f"mismatch_mean={(h_mce - h_true).abs().mean().item():.6f}"
            )
            self._forward_csi_debug_count += 1

        # 7) 閫氫俊閾?
        if self.cfg.MODEL.MIMO.CHANNEL:
            if self.cfg.MODEL.MIMO.HFF:
                p_tx = self.encoder(pyramid_tx)

                if self.cfg.MODEL.MIMO.MCE:
                    tx_feature = self.diversity_encoder(p_tx, h_mce, noise_std)
                else:
                    tx_feature = p_tx

                tx_feature, shape = self.before_channel(tx_feature)
                rx_feature = self.channel(tx_feature, h_mce, h_true, noise_std)
                rx_feature = self.after_channel(rx_feature, shape)

                if self.cfg.MODEL.MIMO.MCE:
                    p_rx = self.diversity_decoder(rx_feature, h_mce, noise_std)
                else:
                    p_rx = rx_feature

                pyramid_rx = self.decoder(p_rx)
                return pyramid_rx, pyramid_rx, pyramid_tx, aux_losses

            else:
                tx_feature, shape = self.channel.source_to_channel(pyramid_tx)
                rx_feature = self.channel(tx_feature, h_mce, h_true, noise_std)
                pyramid_rx = self.channel.channel_to_source(rx_feature, shape)
                return pyramid_rx, pyramid_rx, pyramid_tx, aux_losses

        else:
            if self.cfg.MODEL.MIMO.MCE:
                p_tx = self.encoder(pyramid_tx)
                p_tx = self.compress(p_tx)
                p_tx = self.activate(p_tx)
                p_tx = self.decompress(p_tx)
                pyramid_rx = self.decoder(p_tx)
                return pyramid_rx, pyramid_rx, pyramid_tx, aux_losses
            else:
                return pyramid_tx, pyramid_tx, pyramid_tx, aux_losses

    def output_shape(self):
        return self.source.output_shape()

@BACKBONE_REGISTRY.register()
def build_som_mimo_backbone_CITY(cfg, input_shape: ShapeSpec):
    fpn_source = build_resnet_fpn_backbone(cfg, input_shape)
    backbone = SoM_MIMO_bacbone(cfg, source=fpn_source)
    return backbone
