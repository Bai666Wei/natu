"""Write out the substitution matrix as align/search/cluster actually see it.

`natu build` writes the UNEXPANDED base matrix: tailoring bonuses (chirality,
N-methylation and, with this branch, reduction state) are applied by
expand_substitution_matrix when the matrix is loaded, in memory, and are never
written back to disk. So a matrix built with a tailoring-aware config looks
byte-identical to one built without it -- the difference only exists at
alignment time. This script applies the expansion and writes the result, so the
bonuses can be inspected before relying on them.

Usage:
  python3 scripts/dump_expanded_matrix.py -m <matrix.tsv> -s <smiles.tsv> \
      -c <config.yaml> -o <expanded.tsv>
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

from natu.cli import load_config
from natu.matrix import expand_substitution_matrix
from natu.variants import get_reduction_state, parse_smiles
from pikachu.general import read_smiles


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--matrix", required=True, type=Path)
    parser.add_argument("-s", "--smiles", required=True, type=Path)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        sys.exit(f"REFUSING to overwrite existing file: {args.output}")

    config = load_config(args.config)
    tailoring = config.get("tailoring", {})
    reduction = tailoring.get("reduction", {})

    base = pd.read_csv(args.matrix, sep="\t", index_col=0)
    expanded = expand_substitution_matrix(base, tailoring, args.smiles)
    expanded.to_csv(args.output, sep="\t")

    print(f"base {base.shape} -> expanded {expanded.shape}")
    print(f"chirality_aware={tailoring.get('chirality', {}).get('chirality_aware_scoring')}  "
          f"n_methylation_aware={tailoring.get('n_methylation', {}).get('n_methylation_aware_scoring')}  "
          f"reduction_aware={reduction.get('reduction_aware_scoring')}")

    if reduction.get("reduction_aware_scoring"):
        print("\nreduction state assigned to each substrate in the SMILES file:")
        for name, smiles in parse_smiles(args.smiles).items():
            print(f"  {name:<36}{get_reduction_state(read_smiles(smiles)).name}")

    changed = 0
    for row in base.index:
        for col in base.columns:
            if abs(expanded.loc[row, col] - base.loc[row, col]) > 1e-9:
                changed += 1
    print(f"\n{changed} of {base.shape[0] * base.shape[1]} original cells changed by expansion")
    if changed == 0:
        print("  (nothing changed: every tailoring axis is either off, or every pair "
              "landed on a state combination scored 0)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
