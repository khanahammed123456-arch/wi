import argparse
import csv
import math
import os
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


CSI_METRIC_RE = re.compile(
    r"\[CSI METRIC\]\s+iter=(?P<iter>\d+)\s+"
    r"outdated_err=(?P<outdated_err>[-+0-9.eE]+)\s+"
    r"pred_err=(?P<pred_err>[-+0-9.eE]+)\s+"
    r"nmse_outdated=(?P<nmse_outdated>[-+0-9.eE]+)\s+"
    r"nmse_pred=(?P<nmse_pred>[-+0-9.eE]+)\s+"
    r"cos_outdated=(?P<cos_outdated>[-+0-9.eE]+)\s+"
    r"cos_pred=(?P<cos_pred>[-+0-9.eE]+)"
)

CSI_COMPARE_RE = re.compile(
    r"\[CSI COMPARE\]\s+"
    r"h_true_mean=(?P<h_true_mean>[-+0-9.eE]+)\s+"
    r"h_pred_mean=(?P<h_pred_mean>[-+0-9.eE]+)\s+"
    r"outdated_err=(?P<outdated_err>[-+0-9.eE]+)\s+"
    r"pred_err=(?P<pred_err>[-+0-9.eE]+)\s+"
    r"nmse_outdated_mean=(?P<nmse_outdated>[-+0-9.eE]+)\s+"
    r"nmse_pred_mean=(?P<nmse_pred>[-+0-9.eE]+)\s+"
    r"cos_outdated_mean=(?P<cos_outdated>[-+0-9.eE]+)\s+"
    r"cos_pred_mean=(?P<cos_pred>[-+0-9.eE]+)\s+"
    r"snr_proxy_err_outdated=(?P<snr_proxy_err_outdated>[-+0-9.eE]+)\s+"
    r"snr_proxy_err_pred=(?P<snr_proxy_err_pred>[-+0-9.eE]+)"
)

CSI_DEBUG_RE = re.compile(
    r"\[CSI DEBUG\]\s+is_train=(?P<is_train>\w+)\s+"
    r"doppler=(?P<doppler>[-+0-9.eE]+)\s+"
    r"snr=(?P<snr>[-+0-9.eE]+)\s+"
    r"hist_last_mean=(?P<hist_last_mean>[-+0-9.eE]+)\s+"
    r"gt_mean=(?P<gt_mean>[-+0-9.eE]+)\s+"
    r"delta_mean=(?P<delta_mean>[-+0-9.eE]+)\s+"
    r"hist_temporal_delta=(?P<hist_temporal_delta>[-+0-9.eE]+)"
)

COPYPASTE_RE = re.compile(
    r"copypaste:\s+(?P<ap>[-+0-9.eE]+)\s*,\s*(?P<ap50>[-+0-9.eE]+)"
)

CITYSCAPES_AVG_RE = re.compile(
    r"average\s*:\s*(?P<ap>[-+0-9.eE]+)\s+(?P<ap50>[-+0-9.eE]+)"
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_float_dict(match):
    out = {}
    for key, value in match.groupdict().items():
        if key in {"iter"}:
            out[key] = int(value)
        elif key == "is_train":
            out[key] = value
        else:
            out[key] = float(value)
    return out


def parse_run(label, path):
    if str(path).lower().endswith(".csv"):
        return parse_metric_csv(label, path)

    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    metric_rows = []
    for idx, match in enumerate(CSI_METRIC_RE.finditer(text)):
        row = to_float_dict(match)
        row["sample_idx"] = idx
        row["run"] = label
        row["source"] = str(path)
        row["metric_source"] = "CSI_METRIC"
        row.setdefault("snr_proxy_err_outdated", None)
        row.setdefault("snr_proxy_err_pred", None)
        row.setdefault("h_true_mean", None)
        row.setdefault("h_pred_mean", None)
        metric_rows.append(row)

    offset = len(metric_rows)
    for idx, match in enumerate(CSI_COMPARE_RE.finditer(text)):
        row = to_float_dict(match)
        row["iter"] = None
        row["sample_idx"] = offset + idx
        row["run"] = label
        row["source"] = str(path)
        row["metric_source"] = "CSI_COMPARE"
        metric_rows.append(row)

    debug_rows = []
    for idx, match in enumerate(CSI_DEBUG_RE.finditer(text)):
        row = to_float_dict(match)
        row["sample_idx"] = idx
        row["run"] = label
        row["source"] = str(path)
        debug_rows.append(row)

    ap = None
    ap50 = None
    for match in CITYSCAPES_AVG_RE.finditer(text):
        ap = float(match.group("ap")) * 100.0
        ap50 = float(match.group("ap50")) * 100.0
    for match in COPYPASTE_RE.finditer(text):
        ap = float(match.group("ap"))
        ap50 = float(match.group("ap50"))

    return metric_rows, debug_rows, ap, ap50


def parse_metric_csv(label, path):
    rows = []
    debug_rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader):
            row = {
                "iter": int(float(raw.get("iter", -1))),
                "sample_idx": idx,
                "run": label,
                "source": str(path),
                "metric_source": "CSI_METRIC_CSV",
            }
            for key in [
                "outdated_err",
                "pred_err",
                "nmse_outdated",
                "nmse_pred",
                "cos_outdated",
                "cos_pred",
                "snr_proxy_err_outdated",
                "snr_proxy_err_pred",
                "h_true_mean",
                "h_pred_mean",
            ]:
                value = raw.get(key, "")
                row[key] = float(value) if value not in {"", None} else float("nan")
            rows.append(row)

            debug = {
                "sample_idx": idx,
                "run": label,
                "source": str(path),
                "is_train": raw.get("is_train", ""),
            }
            for key in [
                "doppler",
                "snr_db",
                "delta_mean",
                "h_outdated_mean",
                "h_true_mean",
            ]:
                value = raw.get(key, "")
                debug[key] = float(value) if value not in {"", None} else float("nan")
            debug["snr"] = debug.get("snr_db", float("nan"))
            debug["hist_last_mean"] = debug.get("h_outdated_mean", float("nan"))
            debug["gt_mean"] = debug.get("h_true_mean", float("nan"))
            debug["hist_temporal_delta"] = float("nan")
            debug_rows.append(debug)
    return rows, debug_rows, None, None


def mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def std(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.std(values)) if values else float("nan")


def pct(numer, denom):
    return 100.0 * numer / denom if denom else float("nan")


def summarize_run(label, metric_rows, debug_rows, ap, ap50):
    n = len(metric_rows)
    out_better = sum(r["pred_err"] < r["outdated_err"] for r in metric_rows)
    nmse_better = sum(r["nmse_pred"] < r["nmse_outdated"] for r in metric_rows)
    cos_better = sum(r["cos_pred"] > r["cos_outdated"] for r in metric_rows)

    pred_err_mean = mean([r["pred_err"] for r in metric_rows])
    outdated_err_mean = mean([r["outdated_err"] for r in metric_rows])
    nmse_pred_mean = mean([r["nmse_pred"] for r in metric_rows])
    nmse_outdated_mean = mean([r["nmse_outdated"] for r in metric_rows])
    cos_pred_mean = mean([r["cos_pred"] for r in metric_rows])
    cos_outdated_mean = mean([r["cos_outdated"] for r in metric_rows])

    return {
        "run": label,
        "num_csi_metric": n,
        "num_csi_debug": len(debug_rows),
        "AP": ap,
        "AP50": ap50,
        "outdated_err_mean": outdated_err_mean,
        "pred_err_mean": pred_err_mean,
        "err_improvement_abs": outdated_err_mean - pred_err_mean,
        "err_improvement_pct": pct(outdated_err_mean - pred_err_mean, outdated_err_mean),
        "pred_err_better_rate": pct(out_better, n),
        "nmse_outdated_mean": nmse_outdated_mean,
        "nmse_pred_mean": nmse_pred_mean,
        "nmse_improvement_abs": nmse_outdated_mean - nmse_pred_mean,
        "nmse_improvement_pct": pct(nmse_outdated_mean - nmse_pred_mean, nmse_outdated_mean),
        "nmse_better_rate": pct(nmse_better, n),
        "cos_outdated_mean": cos_outdated_mean,
        "cos_pred_mean": cos_pred_mean,
        "cos_improvement_abs": cos_pred_mean - cos_outdated_mean,
        "cos_better_rate": pct(cos_better, n),
        "delta_mean": mean([r["delta_mean"] for r in debug_rows]),
        "hist_temporal_delta_mean": mean([r["hist_temporal_delta"] for r in debug_rows]),
        "hist_last_mean": mean([r["hist_last_mean"] for r in debug_rows]),
        "gt_mean": mean([r["gt_mean"] for r in debug_rows]),
        "doppler_mean": mean([r["doppler"] for r in debug_rows]),
        "snr_mean": mean([r["snr"] for r in debug_rows]),
        "pred_err_std": std([r["pred_err"] for r in metric_rows]),
        "nmse_pred_std": std([r["nmse_pred"] for r in metric_rows]),
        "cos_pred_std": std([r["cos_pred"] for r in metric_rows]),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def write_markdown(path, summaries):
    lines = [
        "# Evaluation Metric Summary",
        "",
        "This report summarizes CSI prediction quality and final Cityscapes instance AP from evaluation logs.",
        "",
        "## Main Table",
        "",
        "| Run | AP | AP50 | pred err | outdated err | err impr. | NMSE pred | NMSE outdated | NMSE impr. | cos pred | cos outdated | cos gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for s in summaries:
        lines.append(
            "| {run} | {AP} | {AP50} | {pred_err} | {out_err} | {err_imp}% | "
            "{nmse_pred} | {nmse_out} | {nmse_imp}% | {cos_pred} | {cos_out} | {cos_gap} |".format(
                run=s["run"],
                AP=fmt(s["AP"], 2),
                AP50=fmt(s["AP50"], 2),
                pred_err=fmt(s["pred_err_mean"]),
                out_err=fmt(s["outdated_err_mean"]),
                err_imp=fmt(s["err_improvement_pct"], 2),
                nmse_pred=fmt(s["nmse_pred_mean"]),
                nmse_out=fmt(s["nmse_outdated_mean"]),
                nmse_imp=fmt(s["nmse_improvement_pct"], 2),
                cos_pred=fmt(s["cos_pred_mean"]),
                cos_out=fmt(s["cos_outdated_mean"]),
                cos_gap=fmt(s["cos_improvement_abs"]),
            )
        )

    lines.extend(
        [
            "",
            "## How To Read These Metrics",
            "",
            "- `pred err` and `NMSE pred` measure whether predicted CSI is numerically closer to the future true CSI.",
            "- `cos pred` measures directional consistency of the complex CSI vector.",
            "- `AP` is the final semantic task metric and should be treated as the primary criterion.",
            "- If CSI metrics improve while AP drops, this supports the argument that CSI numerical accuracy alone is insufficient for semantic communication.",
            "",
            "## Paper-Ready Observations To Check",
            "",
            "- Does WiFo residual reduce `pred err` or `NMSE` compared with outdated CSI?",
            "- Does higher CSI quality always lead to higher AP?",
            "- Which method has the best trade-off between CSI improvement and final AP?",
            "- Is cosine consistency degraded even when L1/NMSE improves?",
            "",
        ]
    )

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def autolabel(ax, bars, digits=2):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{digits}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_csi_means(out_dir, summaries):
    if plt is None:
        return
    labels = [s["run"] for s in summaries]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    pairs = [
        ("CSI L1 error", "outdated_err_mean", "pred_err_mean", "lower is better"),
        ("CSI NMSE", "nmse_outdated_mean", "nmse_pred_mean", "lower is better"),
        ("CSI cosine", "cos_outdated_mean", "cos_pred_mean", "higher is better"),
    ]

    for ax, (title, out_key, pred_key, ylabel) in zip(axes, pairs):
        out_vals = [s[out_key] for s in summaries]
        pred_vals = [s[pred_key] for s in summaries]
        b1 = ax.bar(x - width / 2, out_vals, width, label="outdated")
        b2 = ax.bar(x + width / 2, pred_vals, width, label="pred")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        autolabel(ax, b1, 3)
        autolabel(ax, b2, 3)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "figure_csi_quality_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ap_vs_csi(out_dir, summaries):
    if plt is None:
        return
    valid = [s for s in summaries if s["AP"] is not None and not math.isnan(s["AP"])]
    if not valid:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    specs = [
        ("err_improvement_pct", "CSI L1 improvement (%)"),
        ("nmse_improvement_pct", "CSI NMSE improvement (%)"),
        ("cos_improvement_abs", "CSI cosine gap"),
    ]

    for ax, (key, xlabel) in zip(axes, specs):
        xs = [s[key] for s in valid]
        ys = [s["AP"] for s in valid]
        ax.scatter(xs, ys, s=55)
        for s, x, y in zip(valid, xs, ys):
            ax.annotate(s["run"], (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("AP")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "figure_ap_vs_csi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_rates(out_dir, summaries):
    if plt is None:
        return
    labels = [s["run"] for s in summaries]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    keys = [
        ("pred_err_better_rate", "L1 better"),
        ("nmse_better_rate", "NMSE better"),
        ("cos_better_rate", "Cos better"),
    ]
    for i, (key, name) in enumerate(keys):
        vals = [s[key] for s in summaries]
        ax.bar(x + (i - 1) * width, vals, width, label=name)
    ax.set_ylabel("Sample ratio (%)")
    ax.set_title("Per-sample predicted CSI better than outdated CSI")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "figure_better_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_distributions(out_dir, metric_rows):
    if plt is None:
        return
    if not metric_rows:
        return
    by_run = {}
    for row in metric_rows:
        by_run.setdefault(row["run"], []).append(row)

    specs = [
        ("pred_err", "outdated_err", "L1 error gap", "pred_err - outdated_err", "figure_dist_l1_gap.png"),
        ("nmse_pred", "nmse_outdated", "NMSE gap", "nmse_pred - nmse_outdated", "figure_dist_nmse_gap.png"),
        ("cos_pred", "cos_outdated", "Cosine gap", "cos_pred - cos_outdated", "figure_dist_cos_gap.png"),
    ]

    for pred_key, out_key, title, xlabel, fname in specs:
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, rows in by_run.items():
            gaps = [r[pred_key] - r[out_key] for r in rows]
            ax.hist(gaps, bins=35, alpha=0.45, label=label)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(out_dir) / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


def parse_run_arg(value):
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    return label.strip(), Path(path.strip())


def parse_value_map(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected label=value, got: {item}")
        label, value = item.split("=", 1)
        out[label.strip()] = float(value)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Parse SoM-MIMO evaluation logs and generate paper-ready CSI/AP summaries."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form label=path/to/log.txt. Can be repeated.",
    )
    parser.add_argument(
        "--ap",
        action="append",
        default=[],
        help="Optional AP override in the form label=27.2213. Useful when AP is not in the parsed file.",
    )
    parser.add_argument(
        "--ap50",
        action="append",
        default=[],
        help="Optional AP50 override in the form label=49.8665.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for CSV, Markdown, and figures.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ap_override = parse_value_map(args.ap)
    ap50_override = parse_value_map(args.ap50)

    all_metric_rows = []
    all_debug_rows = []
    summaries = []

    for run_arg in args.run:
        label, path = parse_run_arg(run_arg)
        metric_rows, debug_rows, ap, ap50 = parse_run(label, path)
        if label in ap_override:
            ap = ap_override[label]
        if label in ap50_override:
            ap50 = ap50_override[label]
        if not metric_rows:
            print(f"[WARN] No [CSI METRIC] lines found in {path}")
        summary = summarize_run(label, metric_rows, debug_rows, ap, ap50)
        summaries.append(summary)
        all_metric_rows.extend(metric_rows)
        all_debug_rows.extend(debug_rows)
        print(
            f"[OK] {label}: csi_metric={len(metric_rows)} csi_debug={len(debug_rows)} "
            f"AP={fmt(ap, 2)} AP50={fmt(ap50, 2)}"
        )

    write_csv(Path(args.out_dir) / "summary.csv", summaries)
    write_csv(Path(args.out_dir) / "csi_metric_rows.csv", all_metric_rows)
    write_csv(Path(args.out_dir) / "csi_debug_rows.csv", all_debug_rows)
    write_markdown(Path(args.out_dir) / "paper_summary.md", summaries)

    if plt is None:
        print("[WARN] matplotlib is not installed; skipped PNG figures.")
    else:
        plot_csi_means(args.out_dir, summaries)
        plot_improvement_rates(args.out_dir, summaries)
        plot_ap_vs_csi(args.out_dir, summaries)
        plot_metric_distributions(args.out_dir, all_metric_rows)

    print(f"[DONE] Wrote analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
