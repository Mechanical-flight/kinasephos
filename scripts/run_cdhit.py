#!/usr/bin/env python
import argparse

from kinasephos.data.clustering import run_cdhit
from kinasephos.utils.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_cdhit(
        config["reference_fasta"],
        f"{config['output_dir']}/reference.cdhit.fasta",
        float(config["cdhit_identity"]),
    )
