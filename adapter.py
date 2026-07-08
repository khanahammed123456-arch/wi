import torch


def h_to_snr_adapter(h_pred_complex, noise_std, cfg):
    """
    将 WiFo 预测的复数信道矩阵转换为 MCE 模块所需的 SNR (dB)

    参数:
        h_pred_complex: WiFo 预测输出 [Batch, Freq, Space]
        noise_std: 当前的物理层噪声标准差
        cfg: 项目的全局配置项

    输出:
        snr_db: 预测出的等效 SNR [Batch, 1, Nt]
    """

    # ========================================================
    # 【终极修复：AMP 精度防爆装甲】
    # 强制将实验性的 ComplexHalf 转回标准的 ComplexFloat (complex64)
    # 彻底解决 RuntimeError: "mean_cuda" not implemented for 'ComplexHalf'
    # ========================================================
    if h_pred_complex.is_complex():
        h_pred_complex = h_pred_complex.to(torch.complex64)

    B, F, Space = h_pred_complex.shape
    Nr = cfg.MODEL.MIMO.Nr
    Nt = cfg.MODEL.MIMO.Nt

    # 1. 空间维度对齐 (将 WiFo 预训练的 Space 维度映射为 MIMO 的 Nr x Nt)
    try:
        h_matrix = h_pred_complex.view(B, F, Nr, Nt)
    except RuntimeError:
        # 如果维度不直接匹配，先展平，再截取我们需要的前 Nr*Nt 个通道
        h_matrix = h_pred_complex.view(B, F, -1)[:, :, :Nr * Nt].view(B, F, Nr, Nt)

    # 2. 频域平均
    # 此时 h_matrix 是 standard complex64，mean_cuda 绝对不会再报错了！
    h_flat = torch.mean(h_matrix, dim=1)  # [B, Nr, Nt]

    # 3. 奇异值分解 (SVD)
    try:
        # 只需要提取奇异值 S，不需要 U 和 V 矩阵
        S = torch.linalg.svd(h_flat)[1]
    except RuntimeError:
        # 保护机制：如果 GPU 上 SVD 偶尔不收敛，返回极小的奇异值避免系统崩溃
        S = torch.zeros(B, min(Nr, Nt), device=h_pred_complex.device)

    # 4. 计算等效线性 SNR
    snr_linear = (S ** 2) / (Nt * (noise_std ** 2))

    # 5. 转换为对数域 (dB)
    snr_db = 10 * torch.log10(snr_linear + 1e-8)

    # 增加一个维度以适配 MCE 内部的处理 (输出形状为 [B, 1, Nt])
    return snr_db.unsqueeze(1)