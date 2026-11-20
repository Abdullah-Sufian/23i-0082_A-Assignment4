"""
Baseline pipeline: train the student networks with plain cross-entropy (NO distillation).

Example
  python -m q1_kd.train_baseline --arch all --epochs 200 --amp
  python -m q1_kd.train_baseline --arch mobilenetv2 --epochs 100
"""
from common.engine import make_ce_step
from models.registry import STUDENTS, build_model
from q1_kd.trainer import base_parser, setup, train_and_save


def main():
    p = base_parser("Train student baselines without KD", epochs=200, lr=0.05)
    p.add_argument("--arch", default="all", choices=STUDENTS + ["all"])
    args = p.parse_args()

    device, train_loader, test_loader = setup(args)
    archs = STUDENTS if args.arch == "all" else [args.arch]
    nc = 100 if args.dataset == "cifar100" else 10
    for arch in archs:
        model = build_model(arch, nc).to(device)
        step = make_ce_step(model)
        train_and_save(f"baseline_{arch}", arch, model, step, model, train_loader, test_loader, device, args,
                       extra_meta={"method": "baseline"})


if __name__ == "__main__":
    main()
