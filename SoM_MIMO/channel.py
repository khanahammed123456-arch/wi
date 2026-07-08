import math
import torch
import torch.nn as nn


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Channel(nn.Module):
    """
    关键改动：
    1. SVD / 预编码 / 均衡 使用 h_mce（系统认为的 CSI）
    2. 真实传播 y = h_true x + n 使用 h_true（真实 CSI）

    这样就形成：
        h_mce != h_true
    时的物理层 mismatch。
    """

    def __init__(self, cfg):
        super(Channel, self).__init__()
        self.Nt = cfg.MODEL.MIMO.Nt
        self.Nr = cfg.MODEL.MIMO.Nr
        self.INFER_SNR = cfg.MODEL.MIMO.INFER_SNR
        self.bits = cfg.MODEL.MIMO.BITS
        self.cfg = cfg
        self.test_counter = 0
        self._debug_print_count = 0

    def channel_generator(self, device, batch_size):
        """
        回退用随机信道生成器。
        正常情况下，如果 backbone 已经传入 h_true / h_mce，这个函数通常不走。
        """
        B = batch_size
        Nt = self.Nt
        Nr = self.Nr

        SNR = getattr(self.cfg.MODEL.MIMO, "INFER_SNR", self.INFER_SNR)
        Pn = 1 / (10 ** (SNR / 10))
        std = math.sqrt(Pn / Nt)

        h_real = torch.normal(
            mean=0.0,
            std=1 / torch.sqrt(torch.tensor(2.0, device=device)),
            size=[B, Nr, Nt],
            device=device,
        )
        h_imag = torch.normal(
            mean=0.0,
            std=1 / torch.sqrt(torch.tensor(2.0, device=device)),
            size=[B, Nr, Nt],
            device=device,
        )
        h = h_real + 1j * h_imag
        h = h / math.sqrt(Nt)
        return [h, std]

    def QPSK_modulator(self, x_uint, device):
        symbols_per_int = self.bits // 2
        B, N, K = x_uint.size()

        qpsk_values = torch.tensor(
            [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j],
            device=device
        )

        x_uint_expanded = x_uint.unsqueeze(-1).expand(B, N, K, symbols_per_int).to(torch.uint8)
        shifts = torch.tensor([0, 2, 4, 6], device=device)[:symbols_per_int]
        two_low_bits = (x_uint_expanded >> shifts) & 0b11
        x_symbols = torch.take(qpsk_values, two_low_bits).view(B, N, symbols_per_int * K)
        return x_symbols

    def QPSK_demodulator(self, y_symbols, device):
        symbols_per_int = self.bits // 2
        B, N, K = y_symbols.size()
        K //= symbols_per_int

        y_symbols_expanded = y_symbols.view(B, N, K, symbols_per_int)
        y_real_detect = torch.sign(y_symbols_expanded.real)
        y_imag_detect = torch.sign(y_symbols_expanded.imag)
        y_detect = (y_real_detect + 1j * y_imag_detect).view(B, N, -1)

        y_real_map = ((1 - y_real_detect) / 2).to(torch.uint8)
        y_imag_map = ((1 - y_imag_detect) / 2).to(torch.uint8)
        y_mapped = y_real_map + (y_imag_map << 1)

        shifts = torch.tensor([0, 2, 4, 6], device=device)[:symbols_per_int]
        y_unit = torch.sum(y_mapped << shifts, dim=-1)
        return y_unit, y_detect

    def source_to_channel(self, source_feature):
        shape = {}
        reshaped_tensor = []
        for key, value in source_feature.items():
            shape[key] = value.shape
            reshape_temp = value.view(value.size(0), self.Nt, -1)
            reshaped_tensor.append(reshape_temp)
        reshaped_tensor = torch.cat(reshaped_tensor, dim=-1)
        return reshaped_tensor, shape

    def channel_to_source(self, noise_feature, shape):
        start_index = 0
        reshape_feature = {}
        for key, value in shape.items():
            temp = value[1] * value[2] * value[3] // self.Nt
            reshape_feature[key] = noise_feature[:, :, start_index:start_index + temp].reshape(value)
            start_index += temp
        return reshape_feature

    def _compute_svd_from_h_mce(self, h_mce: torch.Tensor):
        """
        用系统认为的信道 h_mce 做 SVD。
        """
        U, S, Vh = torch.linalg.svd(h_mce)

        S_mat = torch.diag_embed(S)
        S_mat = torch.complex(S_mat, torch.zeros_like(S_mat))

        V = torch.conj(Vh.transpose(-1, -2))
        Uh = torch.conj(U.transpose(-1, -2))
        S_inv = torch.pinverse(S_mat)
        return U, S, Vh, V, Uh, S_inv

    def forward(self, x, h_mce, h_true, std):
        """
        输入:
            x:      [B, Nt, K]            发送特征
            h_mce:  [B, Nr, Nt] complex   系统认为的 CSI（滞后/预测/真实）
            h_true: [B, Nr, Nt] complex   真实传播 CSI
            std:    标量或张量            噪声标准差

        核心逻辑:
            - SVD / 预编码 / 均衡 用 h_mce
            - 真正传播 y = h_true x + n
        """
        device = x.device
        B, N, K = x.size()
        symbols_per_int = self.bits // 2

        # ===== 1) 用 h_mce 做 SVD，得到系统内部认为的预编码/均衡矩阵 =====
        U, S, Vh, V, Uh, S_inv = self._compute_svd_from_h_mce(h_mce)

        # ===== 2) Digital Tx =====
        x_mean = torch.mean(x)
        x_std = torch.std(x)
        xn = (x - x_mean) / (x_std + 1e-8)

        xn_max = torch.max(xn)
        xn_min = torch.min(xn)
        quant_level = 2 ** self.bits

        S_quant = (xn_max - xn_min) / (quant_level - 1.0 + 1e-8)
        Z_quant = (quant_level - 1.0) - torch.round(xn_max / (S_quant + 1e-8))

        x_uint = torch.clamp(
            RoundSTE.apply(xn / (S_quant + 1e-8)) + Z_quant,
            min=0.0,
            max=quant_level - 1.0,
        )

        x_symbols = self.QPSK_modulator(x_uint, device)
        x_symbols_norm = (x_symbols / math.sqrt(2.0)) / math.sqrt(self.Nt)

        # 用 h_mce 对应的 V 做预编码
        x_tx = torch.bmm(V, x_symbols_norm)

        # ===== 3) Noise =====
        if isinstance(std, torch.Tensor):
            std_per_sample = std.to(device=device, dtype=torch.float32).view(B, -1).mean(dim=1)
        else:
            std_per_sample = torch.full((B,), float(std), device=device, dtype=torch.float32)
        std_val = std_per_sample.mean().item()
        noise_scale = std_per_sample.view(B, 1, 1) / math.sqrt(2.0)

        noise_real = (torch.randn(B, N, symbols_per_int * K, device=device) * noise_scale).detach()
        noise_imag = (torch.randn(B, N, symbols_per_int * K, device=device) * noise_scale).detach()
        noise = noise_real + 1j * noise_imag

        # ===== 4) 真正传播：用真实信道 h_true =====
        x_fading = torch.bmm(h_true, x_tx)
        y_rx = x_fading + noise

        # ===== 5) 接收端均衡：仍按系统认为的 h_mce 去恢复 =====
        y_symbols_norm = torch.bmm(S_inv, torch.bmm(Uh, y_rx))
        y_symbols = y_symbols_norm * math.sqrt(2.0) * math.sqrt(self.Nt)

        y_uint, y_detect = self.QPSK_demodulator(y_symbols, device)
        yn = torch.clamp((y_uint - Z_quant) * S_quant, min=xn_min, max=xn_max)

        y = x_std * yn + x_mean

        Eq_noise = (y - x).detach()
        y = x + Eq_noise

        # ===== debug =====
        if self._debug_print_count < 20:
            mismatch = (h_mce - h_true).abs().mean().item()
            hm = h_mce.abs().mean().item()
            ht = h_true.abs().mean().item()
            print(
                f"[CHANNEL USE] "
                f"h_mce_mean={hm:.6f} "
                f"h_true_mean={ht:.6f} "
                f"mismatch_mean={mismatch:.6f} "
                f"std={std_val:.6f}"
            )
            self._debug_print_count += 1

        return y

    def forward_with_external_h(self, tx_feature, h_mce, h_true, noise_std):
        """
        可选别名。
        """
        return self.forward(tx_feature, h_mce, h_true, noise_std)
