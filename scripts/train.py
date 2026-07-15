#!/usr/bin/env python
import argparse
import json

from kinasephos.training.trainer import train_from_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(train_from_config(args.config), indent=2))
