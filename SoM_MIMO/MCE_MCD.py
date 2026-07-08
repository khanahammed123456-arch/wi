import torch
import torch.nn as nn
import math

class Mlp(nn.Module):
    """Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MIMO_encoder(nn.Module):
    def __init__(self, cfg):
        super(MIMO_encoder, self).__init__()
        self.cfg = cfg
        self.Nt = cfg.MODEL.MIMO.Nt
        self.channel_embedding = Mlp(1, 128 // int(self.Nt), 256 // int(self.Nt))
        self.depth = cfg.MODEL.MIMO.MCE_DEPTH
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.channel_conv_pre = nn.ModuleList()
        self.SE = nn.ModuleList()
        self.channel_conv_post = nn.ModuleList()

        for _ in range(self.depth):
            self.channel_conv_pre.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(256),
                )
            )
            self.SE.append(
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.Sigmoid(),
                )
            )
            self.channel_conv_post.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.BatchNorm2d(256),
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(256),
                )
            )

    def _build_log_snr(self, x, h=None, std=None):
        """
        维持原 GitHub 接口逻辑:
            h -> SVD -> Eq_SNR -> Log_SNR
        但加上鲁棒性保护。
        """
        B = x.shape[0]

        if h is None:
            # 回退到固定 SNR
            log_snr = torch.ones(B, self.Nt, 1, device=x.device) * self.cfg.MODEL.MIMO.INFER_SNR
            return log_snr

        try:
            _, S, _ = torch.linalg.svd(h)
        except RuntimeError:
            # SVD 不收敛时回退到零奇异值
            S = torch.zeros(B, min(h.shape[-2], h.shape[-1]), device=x.device)

        if std is None:
            std = math.sqrt(1.0 / (10 ** (float(self.cfg.MODEL.MIMO.INFER_SNR) / 10.0)) / self.Nt)

        if isinstance(std, torch.Tensor):
            std = std.to(device=x.device, dtype=S.dtype).view(B, -1).mean(dim=1, keepdim=True)

        eq_snr = (torch.square(S) / (std ** 2)) / self.Nt
        log_snr = 10.0 * torch.log10(eq_snr + 1e-8)
        log_snr = log_snr.unsqueeze(1).permute(0, 2, 1)  # [B, Nt, 1]
        return log_snr

    def forward(self, x, h, std):
        B, C, H, W = x.shape

        log_snr = self._build_log_snr(x, h=h, std=std)
        S_vector = self.channel_embedding(log_snr).reshape(B, 256)

        for i in range(self.depth):
            shortcut = x
            x = self.channel_conv_pre[i](x)
            pool = self.global_avg_pool(x).reshape(B, 256)
            attn = self.SE[i](torch.cat((pool, S_vector), dim=1)).unsqueeze(2).unsqueeze(3)
            x = self.channel_conv_post[i](attn * x) + shortcut

        return x


class MIMO_decoder(nn.Module):
    def __init__(self, cfg):
        super(MIMO_decoder, self).__init__()
        self.cfg = cfg
        self.Nt = cfg.MODEL.MIMO.Nt
        self.channel_embedding = Mlp(1, 128 // int(self.Nt), 256 // int(self.Nt))
        self.depth = cfg.MODEL.MIMO.MCE_DEPTH
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.channel_conv_pre = nn.ModuleList()
        self.SE = nn.ModuleList()
        self.channel_conv_post = nn.ModuleList()

        for _ in range(self.depth):
            self.channel_conv_pre.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(256),
                )
            )
            self.SE.append(
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.Sigmoid(),
                )
            )
            self.channel_conv_post.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.BatchNorm2d(256),
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(256),
                )
            )

    def _build_log_snr(self, x, h=None, std=None):
        B = x.shape[0]

        if h is None:
            log_snr = torch.ones(B, self.Nt, 1, device=x.device) * self.cfg.MODEL.MIMO.INFER_SNR
            return log_snr

        try:
            _, S, _ = torch.linalg.svd(h)
        except RuntimeError:
            S = torch.zeros(B, min(h.shape[-2], h.shape[-1]), device=x.device)

        if std is None:
            std = math.sqrt(1.0 / (10 ** (float(self.cfg.MODEL.MIMO.INFER_SNR) / 10.0)) / self.Nt)

        if isinstance(std, torch.Tensor):
            std = std.to(device=x.device, dtype=S.dtype).view(B, -1).mean(dim=1, keepdim=True)

        eq_snr = (torch.square(S) / (std ** 2)) / self.Nt
        log_snr = 10.0 * torch.log10(eq_snr + 1e-8)
        log_snr = log_snr.unsqueeze(1).permute(0, 2, 1)  # [B, Nt, 1]
        return log_snr

    def forward(self, x, h, std):
        B, _, H, W = x.shape

        log_snr = self._build_log_snr(x, h=h, std=std)
        S_vector = self.channel_embedding(log_snr).reshape(B, 256)

        for i in range(self.depth):
            shortcut = x
            x = self.channel_conv_pre[i](x)
            pool = self.global_avg_pool(x).reshape(B, 256)
            attn = self.SE[i](torch.cat((pool, S_vector), dim=1)).unsqueeze(2).unsqueeze(3)
            x = self.channel_conv_post[i](attn * x) + shortcut

        return x
