#!/usr/bin/env python
import argparse
import json

from kinasephos.inference.predictor import predict_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="outputs/smoke/best.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    predictions = predict_csv(args.input, args.output, args.checkpoint, args.threshold)
    print(json.dumps({"rows": len(predictions), "output": args.output}, indent=2))
