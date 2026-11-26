"""Print the reduction state natu assigns to every substrate in a SMILES table.

The GroupDefiner patterns behind get_reduction_state are the one part of the
reduction axis that cannot be checked by reading: a pattern that is too
permissive silently collapses two states into one, and a pattern that is too
strict silently drops a monomer to X. Run this against the alphabet you intend
to build a matrix from and read the labels before trusting the matrix.

What to look for:
  * each of the four states appears where you expect it
  * no monomer you meant to label comes back as X
  * amino acids and anything else non-polyketide DO come back as X

Usage: python3 scripts/check_reduction_state.py <smiles.tsv>
"""
import sys
from collections import Counter
from pathlib import Path

from pikachu.general import read_smiles

from natu.variants import get_reduction_state, parse_smiles

path = Path(sys.argv[1] if len(sys.argv) > 1 else "pks_names_smiles.tsv")
name_to_smiles = parse_smiles(path)

counts: Counter = Counter()
unparsed = []
print(f"{'substrate':<36}{'reduction':<10}smiles")
print("-" * 86)
for name, smiles in name_to_smiles.items():
    try:
        state = get_reduction_state(read_smiles(smiles))
    except Exception as exc:                     # a bad SMILES should name itself, not abort the run
        unparsed.append((name, smiles, exc))
        continue
    counts[state.name] += 1
    print(f"{name:<36}{state.name:<10}{smiles}")

print("-" * 86)
print(f"{len(name_to_smiles)} substrates: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
if unparsed:
    print(f"\n{len(unparsed)} could not be parsed:")
    for name, smiles, exc in unparsed:
        print(f"  {name}: {smiles}  ({type(exc).__name__}: {exc})")
    sys.exit(1)
