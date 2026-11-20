"""
Train student networks with knowledge distillation from the ResNet-50 teacher.

Loss = (1-alpha)*CE + alpha*T^2*KL(soft targets) + beta*feature_loss   (see q1_kd/kd.py)

Examples
  # full KD: soft targets + hard labels + FitNet feature hints
  python -m q1_kd.train_kd --arch all --teacher checkpoints/teacher_resnet50.pt --T 4 --alpha 0.9 --beta 1.0 --amp
  # logit-only KD (ablation)
  python -m q1_kd.train_kd --arch mobilenetv2 --beta 0 --name-suffix _logit
  # attention-transfer feature loss
  python -m q1_kd.train_kd --arch squeezenet --feat-loss attention --beta 1000
"""
from models.registry import STUDENTS, build_model
from q1_kd.kd import Distiller
from q1_kd.trainer import base_parser, load_teacher, setup, train_and_save


def main():
    p = base_parser("Knowledge distillation into lightweight students", epochs=200, lr=0.05)
    p.add_argument("--arch", default="all", choices=STUDENTS + ["all"])
    p.add_argument("--teacher", default="checkpoints/teacher_resnet50.pt")
    p.add_argument("--T", type=float, default=4.0, help="softmax temperature")
    p.add_argument("--alpha", type=float, default=0.9, help="weight of soft-target loss (1-alpha for hard CE)")
    p.add_argument("--beta", type=float, default=1.0, help="weight of feature loss (0 disables it)")
    p.add_argument("--feat-loss", default="fitnet", choices=["fitnet", "attention"])
    p.add_argument("--hints", default="0,1", help="feature hint indices: 0=mid (8x8), 1=final (4x4)")
    p.add_argument("--name-suffix", default="")
    args = p.parse_args()

    device, train_loader, test_loader = setup(args)
    teacher = load_teacher(args.teacher, device)
    archs = STUDENTS if args.arch == "all" else [args.arch]
    nc = 100 if args.dataset == "cifar100" else 10
    hints = [int(h) for h in args.hints.split(",") if h != ""]

    for arch in archs:
        student = build_model(arch, nc).to(device)
        distiller = Distiller(student, teacher, T=args.T, alpha=args.alpha, beta=args.beta,
                              feat_loss=args.feat_loss, hint_layers=hints).to(device)
        # optimise student + feature regressors (teacher is frozen)
        trainable = [p for p in distiller.parameters() if p.requires_grad]
        meta = {"method": "kd", "T": args.T, "alpha": args.alpha, "beta": args.beta,
                "feat_loss": args.feat_loss if args.beta > 0 else "none", "hints": args.hints,
                "teacher": args.teacher}
        train_and_save(f"kd_{arch}{args.name_suffix}", arch, student, distiller, distiller,
                       train_loader, test_loader, device, args, extra_meta=meta, trainable_params=trainable)


if __name__ == "__main__":
    main()
