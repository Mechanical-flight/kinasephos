import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kinasephos")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-assets")
    audit.add_argument("--input", required=True)
    audit.add_argument("--output", required=True)
    for name in ("prepare-data", "cache-teacher", "train", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--checkpoint", default="outputs/smoke/best.pt")
    predict.add_argument("--threshold", type=float, default=0.5)
    export = subparsers.add_parser("export-student")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit-assets":
        from kinasephos.data.inventory import write_inventory_report

        print(json.dumps(write_inventory_report(args.input, args.output), indent=2))
    elif args.command == "train":
        from kinasephos.training.trainer import train_from_config

        print(json.dumps(train_from_config(args.config), indent=2))
    elif args.command == "predict":
        from kinasephos.inference.predictor import predict_csv

        predictions = predict_csv(args.input, args.output, args.checkpoint, args.threshold)
        print(json.dumps({"rows": len(predictions), "output": args.output}, indent=2))
    elif args.command == "export-student":
        from kinasephos.inference.export import export_student

        export_student(args.checkpoint, args.output)
    else:
        raise SystemExit(f"Use the corresponding script for {args.command}; command is scaffolded.")


if __name__ == "__main__":
    main()
