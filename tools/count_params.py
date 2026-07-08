import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from train import setup, Trainer


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_by_keyword(model, keyword):
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        if keyword in name:
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
    return total, trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    class SetupArgs:
        pass

    setup_args = SetupArgs()
    setup_args.config_file = args.config_file
    setup_args.opts = args.opts
    setup_args.resume = False
    setup_args.eval_only = True
    setup_args.num_gpus = 1
    setup_args.num_machines = 1
    setup_args.machine_rank = 0
    setup_args.dist_url = "tcp://127.0.0.1:49152"

    cfg = setup(setup_args)
    model = Trainer.build_model(cfg)

    total, trainable = count_params(model)

    print("=" * 80)
    print(f"Total params:     {total:,}  ({total / 1e6:.3f} M)")
    print(f"Trainable params: {trainable:,}  ({trainable / 1e6:.3f} M)")
    print("=" * 80)

    keywords = [
        "wifo",
        "wifo_adapter",
        "diversity_encoder",
        "diversity_decoder",
        "encoder",
        "decoder",
        "source",
    ]

    for kw in keywords:
        t, tr = count_by_keyword(model, kw)
        if t > 0:
            print(f"{kw:20s}: total={t:,} ({t/1e6:.3f} M), trainable={tr:,} ({tr/1e6:.3f} M)")

    print("=" * 80)
    print("All parameter names containing 'wifo':")
    for name, p in model.named_parameters():
        if "wifo" in name:
            print(f"{name:80s} {p.numel():12,} requires_grad={p.requires_grad}")


if __name__ == "__main__":
    main()