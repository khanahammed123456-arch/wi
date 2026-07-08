import math
import numpy as np
import torch


class WiFoChannelGenerator:
    """
    在线生成时变复数 Rayleigh 信道序列。

    默认输出:
        [B, T, Nr, Nt, 2]
    最后一维 2 表示 [real, imag]

    说明:
    使用稳定的时间相关模型:
        h_t = rho * h_{t-1} + sqrt(1-rho^2) * eps_t

    其中:
        fd 越大 -> rho 越小 -> 时间变化越快
    """

    def __init__(self, T=17, Nr=16, Nt=16, fd=60.0, T_s=0.01):
        self.T = int(T)
        self.Nr = int(Nr)
        self.Nt = int(Nt)
        self.fd = float(fd)
        self.T_s = float(T_s)

        self._debug_print_count = 0

    def _fd_to_rho(self, fd: float) -> float:
        """
        更温和的 fd -> rho 映射。
        之前映射太激进，会导致非零 fd 基本全崩。
        """
        alpha = 0.02
        rho = math.exp(-alpha * float(fd) * self.T_s)
        rho = min(max(rho, 0.0), 0.999999)
        return rho

    def _randn_complex(self, rng, shape):
        real = rng.normal(0.0, 1.0 / math.sqrt(2.0), size=shape).astype(np.float32)
        imag = rng.normal(0.0, 1.0 / math.sqrt(2.0), size=shape).astype(np.float32)
        return (real + 1j * imag).astype(np.complex64)

    def _build_one_batch(self, batch_size, fd, rng):
        B = int(batch_size)
        T = self.T
        Nr = self.Nr
        Nt = self.Nt

        rho = self._fd_to_rho(fd)
        noise_scale = math.sqrt(max(1.0 - rho * rho, 0.0))

        h_prev = self._randn_complex(rng, (B, Nr, Nt))

        h_seq = np.zeros((B, T, Nr, Nt), dtype=np.complex64)
        h_seq[:, 0] = h_prev

        for t in range(1, T):
            eps_t = self._randn_complex(rng, (B, Nr, Nt))
            h_t = rho * h_prev + noise_scale * eps_t
            h_seq[:, t] = h_t
            h_prev = h_t

        return h_seq

    def get_batch(self, batch_size=1, fd=None, seed=None, as_wifo_layout=False):
        fd = self.fd if fd is None else float(fd)

        rng = np.random.default_rng(seed)
        h_complex = self._build_one_batch(batch_size=batch_size, fd=fd, rng=rng)

        h_real = torch.from_numpy(h_complex.real).float()
        h_imag = torch.from_numpy(h_complex.imag).float()

        if self._debug_print_count < 10:
            if self.T > 1:
                temporal_delta = np.mean(np.abs(h_complex[:, 1:] - h_complex[:, :-1]))
            else:
                temporal_delta = 0.0

            if self.T > 1:
                end_to_end_delta = np.mean(np.abs(h_complex[:, -1] - h_complex[:, 0]))
            else:
                end_to_end_delta = 0.0

            print(
                f"[CHANNEL DEBUG] fd={fd:.4f} "
                f"T={self.T} Nr={self.Nr} Nt={self.Nt} "
                f"rho={self._fd_to_rho(fd):.6f} "
                f"|h|_mean={np.mean(np.abs(h_complex)):.6f} "
                f"temporal_delta={float(temporal_delta):.6f} "
                f"end_to_end_delta={float(end_to_end_delta):.6f}"
            )
            self._debug_print_count += 1

        if as_wifo_layout:
            # [B, 2, T, Nr, Nt]
            return torch.stack([h_real, h_imag], dim=1)

        # [B, T, Nr, Nt, 2]
        return torch.stack([h_real, h_imag], dim=-1)