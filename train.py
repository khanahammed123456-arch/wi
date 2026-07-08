import torch
import detectron2.utils.comm as comm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.data import build_detection_train_loader, build_detection_test_loader

from SoM_MIMO import add_mimo_config_city, get_evaluator
from custom_mapper import WiFo_SoM_Mapper


# 可复现设置
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return get_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = WiFo_SoM_Mapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = WiFo_SoM_Mapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_model(cls, cfg):
        model = DefaultTrainer.build_model(cfg)

        freeze_wifo = bool(getattr(cfg.MODEL.MIMO, "FREEZE_WIFO", True))

        # FREEZE_WIFO=True 时只冻结 WiFo 主体，保留 wifo_adapter 可训练。
        if freeze_wifo:
            for name, param in model.named_parameters():
                lower_name = name.lower()
                is_wifo_backbone = (
                    "backbone.wifo." in lower_name
                    and "wifo_adapter" not in lower_name
                )
                if is_wifo_backbone:
                    param.requires_grad = False

            print("\n" + "=" * 60)
            print("Stage 1: REAL WiFo forward + frozen WiFo backbone.")
            print("Train SoM-MIMO + wifo_adapter to adapt predicted CSI.")
            print("=" * 60 + "\n")
        else:
            print("\n" + "=" * 60)
            print("Stage 2 / Joint: REAL WiFo forward + unfrozen WiFo parameters.")
            print("Jointly train WiFo + SoM-MIMO with predicted CSI.")
            print("=" * 60 + "\n")

        # 可选：打印部分关键参数是否可训练，方便核查
        print("[Trainability Check]")
        for name, param in model.named_parameters():
            if (
                "wifo" in name.lower()
                or "wifo_adapter" in name.lower()
                or "backbone.bottom_up" in name.lower()
                or "fpn" in name.lower()
            ):
                print(f"{name}: requires_grad={param.requires_grad}")
        print()

        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        给随机初始化的 WiFo / wifo_adapter 更大学习率。
        目标：
        - backbone / detector 保持较稳
        - WiFo 主体更快学
        - adapter 学得最快
        """
        base_lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        # 如果 yaml 没写，就用默认值
        wifo_lr_mult = float(getattr(cfg.MODEL.MIMO, "WIFO_LR_MULT", 5.0))
        wifo_adapter_lr_mult = float(getattr(cfg.MODEL.MIMO, "WIFO_ADAPTER_LR_MULT", 10.0))
        backbone_lr_mult = float(getattr(cfg.MODEL.MIMO, "BACKBONE_LR_MULT", 1.0))

        params = []
        print("[Optimizer Param Groups]")

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            lr = base_lr * backbone_lr_mult

            # adapter 优先级最高
            if "backbone.wifo_adapter" in name:
                lr = base_lr * wifo_adapter_lr_mult
            elif "backbone.wifo" in name:
                lr = base_lr * wifo_lr_mult

            params.append(
                {
                    "params": [param],
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )

            if (
                "backbone.wifo_adapter" in name
                or "backbone.wifo" in name
                or "backbone.bottom_up" in name
                or "fpn" in name
            ):
                print(f"{name}: lr={lr:.8f}, wd={weight_decay}")

        print()

        optimizer = torch.optim.AdamW(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
        )
        return optimizer


def setup(args):
    """
    创建配置并执行基础初始化
    """
    cfg = get_cfg()

    # 注册 SoM-MIMO / WiFo 扩展配置
    add_mimo_config_city(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=args.resume,
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
