#!/usr/bin/env python
import argparse
import json

from kinasephos.data.preprocess import prepare_sites
from kinasephos.utils.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(
        json.dumps(
            prepare_sites(
                config["reference_fasta"],
                config["positive_sites"],
                f"{config['output_dir']}/positive_sites_validated.csv",
            ),
            indent=2,
        )
    )
