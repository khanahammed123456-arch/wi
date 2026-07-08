import os
import csv
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


def nmse(h_est, h_true, eps=1e-8):
    num = torch.sum(torch.abs(h_est - h_true) ** 2).item()
    den = torch.sum(torch.abs(h_true) ** 2).item() + eps
    return num / den


def cosine_sim(h_est, h_true, eps=1e-8):
    est = torch.stack([h_est.real, h_est.imag], dim=-1).reshape(-1).float()
    tru = torch.stack([h_true.real, h_true.imag], dim=-1).reshape(-1).float()
    return torch.dot(est, tru).item() / ((torch.norm(est).item() + eps) * (torch.norm(tru).item() + eps))


def snr_proxy(h, noise_std, eps=1e-8):
    p = torch.sum(torch.abs(h) ** 2).item()
    n = max(float(noise_std) ** 2, eps)
    return p / n


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_heatmaps(h_true, h_out, h_pred, save_path, title=""):
    true_mag = h_true.abs().cpu().numpy()
    out_mag = h_out.abs().cpu().numpy()
    pred_mag = h_pred.abs().cpu().numpy()
    out_err = (h_out - h_true).abs().cpu().numpy()
    pred_err = (h_pred - h_true).abs().cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    items = [
        (true_mag, "|H_true|"),
        (out_mag, "|H_outdated|"),
        (pred_mag, "|H_pred|"),
        (out_err, "|H_outdated - H_true|"),
        (pred_err, "|H_pred - H_true|"),
    ]

    for ax, (img, name) in zip(axes, items):
        im = ax.imshow(img, aspect="auto")
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_complex_scatter(h_true, h_out, h_pred, save_path, title=""):
    t = h_true.reshape(-1)
    o = h_out.reshape(-1)
    p = h_pred.reshape(-1)

    plt.figure(figsize=(6, 6))
    plt.scatter(t.real.numpy(), t.imag.numpy(), label="true", marker="o", alpha=0.8)
    plt.scatter(o.real.numpy(), o.imag.numpy(), label="outdated", marker="x", alpha=0.8)
    plt.scatter(p.real.numpy(), p.imag.numpy(), label="pred", marker="^", alpha=0.8)
    plt.axhline(0.0)
    plt.axvline(0.0)
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_hist(values, save_path, title, xlabel):
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_sorted(values, save_path, title, ylabel):
    vals = np.sort(np.array(values))
    plt.figure(figsize=(7, 4))
    plt.plot(vals)
    plt.title(title)
    plt.xlabel("Sorted sample index")
    plt.ylabel(ylabel)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_global_h_mean_heatmap(h_true_list, h_out_list, h_pred_list, save_path):
    true_mean = torch.stack([x.abs() for x in h_true_list], dim=0).mean(dim=0).cpu().numpy()
    out_mean = torch.stack([x.abs() for x in h_out_list], dim=0).mean(dim=0).cpu().numpy()
    pred_mean = torch.stack([x.abs() for x in h_pred_list], dim=0).mean(dim=0).cpu().numpy()
    out_err_mean = torch.stack([(o - t).abs() for o, t in zip(h_out_list, h_true_list)], dim=0).mean(dim=0).cpu().numpy()
    pred_err_mean = torch.stack([(p - t).abs() for p, t in zip(h_pred_list, h_true_list)], dim=0).mean(dim=0).cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    items = [
        (true_mean, "mean(|H_true|)"),
        (out_mean, "mean(|H_outdated|)"),
        (pred_mean, "mean(|H_pred|)"),
        (out_err_mean, "mean(|H_outdated-H_true|)"),
        (pred_err_mean, "mean(|H_pred-H_true|)"),
    ]

    for ax, (img, name) in zip(axes, items):
        im = ax.imshow(img, aspect="auto")
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Global mean H heatmaps")
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_global_h_complex_scatter(h_true_list, h_out_list, h_pred_list, save_path, max_points=5000):
    t = torch.cat([x.reshape(-1) for x in h_true_list], dim=0)
    o = torch.cat([x.reshape(-1) for x in h_out_list], dim=0)
    p = torch.cat([x.reshape(-1) for x in h_pred_list], dim=0)

    # 点太多就随机采样
    n = t.numel()
    if n > max_points:
        idx = torch.randperm(n)[:max_points]
        t = t[idx]
        o = o[idx]
        p = p[idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(t.real.numpy(), t.imag.numpy(), label="true", marker="o", alpha=0.4, s=10)
    plt.scatter(o.real.numpy(), o.imag.numpy(), label="outdated", marker="x", alpha=0.4, s=10)
    plt.scatter(p.real.numpy(), p.imag.numpy(), label="pred", marker="^", alpha=0.4, s=10)
    plt.axhline(0.0)
    plt.axvline(0.0)
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.legend()
    plt.title("Global H complex scatter")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    vis_dir = os.path.join(args.out_dir, "sample_vis")
    ensure_dir(vis_dir)

    files = sorted([f for f in os.listdir(args.dump_dir) if f.endswith(".pt")])
    if not files:
        raise RuntimeError(f"No .pt files found in {args.dump_dir}")

    rows = []
    h_true_list = []
    h_out_list = []
    h_pred_list = []

    for fname in files:
        obj = torch.load(os.path.join(args.dump_dir, fname), map_location="cpu")

        h_true = obj["h_true"]
        h_out = obj["h_outdated"]
        h_pred = obj["h_pred"]
        noise_std = obj["noise_std"]

        h_true_list.append(h_true)
        h_out_list.append(h_out)
        h_pred_list.append(h_pred)

        nmse_out = nmse(h_out, h_true)
        nmse_pred = nmse(h_pred, h_true)

        cos_out = cosine_sim(h_out, h_true)
        cos_pred = cosine_sim(h_pred, h_true)

        snr_true = snr_proxy(h_true, noise_std)
        snr_out = snr_proxy(h_out, noise_std)
        snr_pred = snr_proxy(h_pred, noise_std)

        snr_err_out = abs(snr_out - snr_true)
        snr_err_pred = abs(snr_pred - snr_true)

        rows.append({
            "file": fname,
            "nmse_out": nmse_out,
            "nmse_pred": nmse_pred,
            "nmse_gap": nmse_pred - nmse_out,
            "cos_out": cos_out,
            "cos_pred": cos_pred,
            "cos_gap": cos_pred - cos_out,
            "snr_true": snr_true,
            "snr_out": snr_out,
            "snr_pred": snr_pred,
            "snr_err_out": snr_err_out,
            "snr_err_pred": snr_err_pred,
            "snr_gap": snr_err_pred - snr_err_out,
            "doppler": obj.get("doppler", None),
            "snr_db": obj.get("snr_db", None),
            "csi_lag": obj.get("csi_lag", None),
        })

    csv_path = os.path.join(args.out_dir, "per_sample_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    nmse_gap = [r["nmse_gap"] for r in rows]
    cos_gap = [r["cos_gap"] for r in rows]
    snr_gap = [r["snr_gap"] for r in rows]

    summary = {
        "num_samples": len(rows),
        "nmse_out_mean": float(np.mean([r["nmse_out"] for r in rows])),
        "nmse_pred_mean": float(np.mean([r["nmse_pred"] for r in rows])),
        "cos_out_mean": float(np.mean([r["cos_out"] for r in rows])),
        "cos_pred_mean": float(np.mean([r["cos_pred"] for r in rows])),
        "snr_err_out_mean": float(np.mean([r["snr_err_out"] for r in rows])),
        "snr_err_pred_mean": float(np.mean([r["snr_err_pred"] for r in rows])),
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_hist(
        nmse_gap,
        os.path.join(args.out_dir, "nmse_gap_hist.png"),
        "NMSE gap histogram",
        "nmse_pred - nmse_outdated",
    )
    plot_hist(
        cos_gap,
        os.path.join(args.out_dir, "cos_gap_hist.png"),
        "Cosine gap histogram",
        "cos_pred - cos_outdated",
    )
    plot_hist(
        snr_gap,
        os.path.join(args.out_dir, "snr_gap_hist.png"),
        "SNR proxy error gap histogram",
        "snr_err_pred - snr_err_outdated",
    )

    plot_sorted(
        nmse_gap,
        os.path.join(args.out_dir, "nmse_gap_sorted.png"),
        "Sorted NMSE gap",
        "nmse_pred - nmse_outdated",
    )
    plot_sorted(
        cos_gap,
        os.path.join(args.out_dir, "cos_gap_sorted.png"),
        "Sorted Cosine gap",
        "cos_pred - cos_outdated",
    )
    plot_sorted(
        snr_gap,
        os.path.join(args.out_dir, "snr_gap_sorted.png"),
        "Sorted SNR proxy error gap",
        "snr_err_pred - snr_err_outdated",
    )

    # ===== 新增：整体 H 图 =====
    plot_global_h_mean_heatmap(
        h_true_list, h_out_list, h_pred_list,
        os.path.join(args.out_dir, "global_h_mean_heatmap.png")
    )
    plot_global_h_complex_scatter(
        h_true_list, h_out_list, h_pred_list,
        os.path.join(args.out_dir, "global_h_complex_scatter.png")
    )

    # 代表样本：NMSE最好/最差
    rows_best = sorted(rows, key=lambda x: x["nmse_gap"])[:args.topk]
    rows_worst = sorted(rows, key=lambda x: x["nmse_gap"], reverse=True)[:args.topk]

    selected = [("best", r) for r in rows_best] + [("worst", r) for r in rows_worst]

    for prefix, rec in selected:
        obj = torch.load(os.path.join(args.dump_dir, rec["file"]), map_location="cpu")
        h_true = obj["h_true"]
        h_out = obj["h_outdated"]
        h_pred = obj["h_pred"]

        base = f"{prefix}_{os.path.splitext(rec['file'])[0]}"
        title = (
            f"{base}\n"
            f"nmse_out={rec['nmse_out']:.4f}, nmse_pred={rec['nmse_pred']:.4f}, "
            f"cos_out={rec['cos_out']:.4f}, cos_pred={rec['cos_pred']:.4f}, "
            f"snr_err_out={rec['snr_err_out']:.4f}, snr_err_pred={rec['snr_err_pred']:.4f}"
        )

        plot_heatmaps(
            h_true, h_out, h_pred,
            os.path.join(vis_dir, f"{base}_heatmap.png"),
            title
        )
        plot_complex_scatter(
            h_true, h_out, h_pred,
            os.path.join(vis_dir, f"{base}_complex.png"),
            title
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()