"""Tests for the polyketide reduction-state tailoring axis.

The axis adds a third tailoring dimension alongside chirality and
N-methylation: how far the beta-carbon of a polyketide extension unit was
reduced (keto -> oh -> ene -> red, the order in which KR, DH and ER act).
It exists because no structural fingerprint tried so far recovers that
order -- ECFP on free acids, ECFP on thioesters and FEATMORGAN all score a
3-step pair HIGHER than a 1-step pair -- while the extender-unit axis is
ranked correctly by FEATMORGAN and is therefore left to the fingerprint.

VERIFICATION STATUS -- please read before trusting a failure here.

The rest of this suite follows the rule that every expected value was
confirmed by running the real function first. That was not possible for
the chemistry half of this file: it was written in an environment without
PIKAChU, so the expected ReductionState values in
TestGetReductionState are REASONED, NOT CONFIRMED. They encode what the
GroupDefiner patterns in natu.variants are intended to match. If one of
them fails, suspect the pattern before suspecting the molecule, and use
scripts/check_reduction_state.py to see what the classifier actually does
across a whole SMILES table.

TestLoadTailoringScores and TestConfig carry no such caveat -- they are
pure dictionary/YAML behaviour with nothing guessed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pikachu.general import read_smiles

from natu.matrix import load_tailoring_scores
from natu.variants import Modification, ReductionState, get_reduction_state

# Taken verbatim from the PKS alphabet built for the NATU x Norine work.
MALONYL_KETO = "CC(=O)CC(=O)O"
MALONYL_OH = "CC(O)CC(=O)O"
MALONYL_ENE = "C/C=C/C(=O)O"
MALONYL_RED = "CCCC(=O)O"
METHYLMALONYL_KETO = "CC(=O)C(C)C(=O)O"
METHYLMALONYL_RED = "CCC(C)C(=O)O"
MALONYL_KETO_THIOESTER = "CC(=O)CC(=O)SC"
MALONYL_RED_THIOESTER = "CCCC(=O)SC"

# From the project's own packaged substrate list.
ALANINE = "C[C@H](N)C(=O)O"
VALINE = "CC(C)[C@H](N)C(=O)O"
NORVALINE = "CCC[C@H](N)C(=O)O"   # same saturated skeleton as a reduced PKS unit


class TestGetReductionState:
    """Classification only: the four states already exist as separate rows in
    the SMILES table, so the function labels them rather than generating them."""

    @pytest.mark.parametrize("smiles,expected", [
        (MALONYL_KETO, ReductionState.KETO),
        (MALONYL_OH, ReductionState.OH),
        (MALONYL_ENE, ReductionState.ENE),
        (MALONYL_RED, ReductionState.RED),
        (METHYLMALONYL_KETO, ReductionState.KETO),
        (METHYLMALONYL_RED, ReductionState.RED),
    ])
    def test_labels_each_reduction_state(self, smiles, expected):
        assert get_reduction_state(read_smiles(smiles)) is expected

    @pytest.mark.parametrize("smiles,expected", [
        (MALONYL_KETO_THIOESTER, ReductionState.KETO),
        (MALONYL_RED_THIOESTER, ReductionState.RED),
    ])
    def test_handles_the_thioester_spelling_too(self, smiles, expected):
        """The alphabet may write C1 as a free acid or as an S-methyl thioester
        standing in for S-ACP, so both forms carry patterns."""
        assert get_reduction_state(read_smiles(smiles)) is expected

    @pytest.mark.parametrize("smiles", [ALANINE, VALINE, NORVALINE])
    def test_amino_acids_are_X(self, smiles):
        """The guarantee this axis rests on: amino acids get X, every X pair
        scores 0, so an NRPS matrix built with this axis enabled is unchanged.

        Norvaline is the case that makes the up-front amino-acid guard
        necessary rather than decorative -- it contains exactly the saturated
        C-C-C-C(=O)O skeleton that the RED pattern matches."""
        assert get_reduction_state(read_smiles(smiles)) is ReductionState.X

    def test_the_four_states_are_distinct(self):
        """A pattern that is too permissive would quietly collapse states into
        one another; this fails loudly if that happens."""
        states = {get_reduction_state(read_smiles(s))
                  for s in (MALONYL_KETO, MALONYL_OH, MALONYL_ENE, MALONYL_RED)}
        assert len(states) == 4


class TestLoadTailoringScores:
    """Pure dictionary behaviour -- nothing guessed below."""

    SCORING = {"keto_keto": 1.5, "keto_oh": 0.0, "keto_red": -3.0, "x_x": 0.0}

    def test_builds_reduction_scores_keyed_by_the_enum(self):
        scores = load_tailoring_scores(self.SCORING, Modification.reduction)
        assert scores[ReductionState.KETO][ReductionState.KETO] == 1.5
        assert scores[ReductionState.KETO][ReductionState.RED] == -3.0
        assert scores[ReductionState.X][ReductionState.X] == 0.0

    def test_is_symmetric(self):
        """Each key is written in both directions, as the other two axes are,
        so the expanded matrix stays symmetric."""
        scores = load_tailoring_scores(self.SCORING, Modification.reduction)
        assert scores[ReductionState.RED][ReductionState.KETO] == -3.0
        assert scores[ReductionState.OH][ReductionState.KETO] == 0.0

    def test_rejects_an_unknown_state_name(self):
        with pytest.raises(KeyError):
            load_tailoring_scores({"keto_diol": 1.0}, Modification.reduction)


class TestConfig:
    """The shipped configs must stay untouched, and the new one must be a
    drop-in -- this is what lets the change be defended as additive."""

    CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "natu" / "data" / "alignment_configurations"

    @pytest.mark.parametrize("name", ["ecfp", "featmorgan", "paras_based", "default", "match_mismatch"])
    def test_existing_configs_do_not_define_the_axis(self, name):
        path = self.CONFIG_DIR / f"{name}.yaml"
        if not path.exists():
            pytest.skip(f"{name}.yaml not present")
        config = yaml.safe_load(path.read_text())
        assert "reduction" not in config.get("tailoring", {})

    def test_pks_config_defines_every_state_pair(self):
        config = yaml.safe_load((self.CONFIG_DIR / "pks_featmorgan.yaml").read_text())
        scoring = config["tailoring"]["reduction"]["scoring"]
        states = ["keto", "oh", "ene", "red", "x"]
        expected = {f"{a}_{b}" for i, a in enumerate(states) for b in states[i:]}
        assert set(scoring) == expected

    def test_pks_config_leaves_non_polyketides_unscored(self):
        """Every pair involving X is 0, which is the whole basis for saying an
        NRPS alphabet is unaffected by enabling this axis."""
        config = yaml.safe_load((self.CONFIG_DIR / "pks_featmorgan.yaml").read_text())
        scoring = config["tailoring"]["reduction"]["scoring"]
        assert all(v == 0.0 for k, v in scoring.items() if "x" in k.split("_"))

    def test_pks_config_penalty_grows_with_step_distance(self):
        config = yaml.safe_load((self.CONFIG_DIR / "pks_featmorgan.yaml").read_text())
        s = config["tailoring"]["reduction"]["scoring"]
        assert s["keto_keto"] > s["keto_oh"] > s["keto_ene"] > s["keto_red"]
        assert s["oh_oh"] > s["oh_ene"] > s["oh_red"]
