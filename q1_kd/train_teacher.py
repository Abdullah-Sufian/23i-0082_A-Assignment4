"""
Train the ResNet-50 teacher on CIFAR-100.

Examples
  # full from-scratch recipe (~77-79% top-1)
  python -m q1_kd.train_teacher --epochs 200 --amp
  # faster: start from ImageNet weights (~80%+ in 30-60 epochs)
  python -m q1_kd.train_teacher --pretrained --epochs 40 --lr 0.05 --amp
"""
from common.engine import evaluate, make_ce_step
from models.registry import build_model
from q1_kd.trainer import base_parser, setup, train_and_save


def main():
    p = base_parser("Train ResNet-50 teacher on CIFAR-100", epochs=200, lr=0.1)
    p.add_argument("--pretrained", action="store_true", help="initialise from ImageNet weights")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--name", default="teacher_resnet50")
    args = p.parse_args()

    device, train_loader, test_loader = setup(args)
    model = build_model("resnet50", 100 if args.dataset == "cifar100" else 10, pretrained=args.pretrained).to(device)
    step = make_ce_step(model, args.label_smoothing)
    train_and_save(args.name, "resnet50", model, step, model, train_loader, test_loader, device, args,
                   extra_meta={"method": "teacher", "pretrained": args.pretrained})


if __name__ == "__main__":
    main()
