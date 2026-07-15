#!/usr/bin/env python
import argparse

from kinasephos.inference.export import export_student

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_student(args.checkpoint, args.output)
