"""Tests for WOE binning, information value, and the leakage guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditrisk import config as cfg
from creditrisk.features import WoeEncoder, fit_categorical_scheme, fit_numeric_scheme, interpret_iv


def _separable_frame(n: int = 4000, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    good = rng.normal(0.0, 1.0, n // 2)
    bad = rng.normal(2.0, 1.0, n // 2)
    signal = np.concatenate([good, bad])
    noise = rng.normal(0.0, 1.0, n)
    target = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    return pd.DataFrame({"signal": signal, "noise": noise}), pd.Series(target)


def test_numeric_bins_are_monotonic_in_woe():
    frame, target = _separable_frame()
    scheme = fit_numeric_scheme("signal", frame["signal"], target)
    assert scheme.monotonic is True
    woe = [entry["woe"] for entry in scheme.numeric_bins]
    diffs = np.diff(woe)
    assert (diffs >= -1e-9).all() or (diffs <= 1e-9).all()


def test_informative_feature_has_higher_iv_than_noise():
    frame, target = _separable_frame()
    signal_iv = fit_numeric_scheme("signal", frame["signal"], target).iv
    noise_iv = fit_numeric_scheme("noise", frame["noise"], target).iv
    assert signal_iv > noise_iv
    assert signal_iv > 0.3
    assert noise_iv < 0.05


def test_iv_interpretation_bands():
    assert interpret_iv(0.005) == "unpredictive"
    assert interpret_iv(0.05) == "weak"
    assert interpret_iv(0.25) == "medium"
    assert interpret_iv(0.8) == "suspicious (check for leakage)"


def test_missing_values_get_their_own_evidence_bin():
    frame, target = _separable_frame(n=2000)
    values = frame["signal"].copy()
    values.iloc[:200] = np.nan
    scheme = fit_numeric_scheme("signal", values, target)
    assert scheme.missing_count == 200
    transformed = scheme.transform_series(values)
    assert not transformed.isna().any()
    assert transformed.iloc[:200].nunique() == 1


def test_transform_of_separable_signal_tracks_the_target():
    frame, target = _separable_frame()
    scheme = fit_numeric_scheme("signal", frame["signal"], target)
    transformed = scheme.transform_series(frame["signal"])
    # Defaulters must carry systematically higher WOE than non-defaulters.
    assert transformed[target == 1].mean() > transformed[target == 0].mean()


def test_unseen_category_maps_to_neutral_woe():
    values = pd.Series(["a"] * 500 + ["b"] * 500)
    target = pd.Series([0] * 500 + [1] * 500)
    scheme = fit_categorical_scheme("category", values, target)
    mapped = scheme.transform_series(pd.Series(["never_seen_before"]))
    assert mapped.iloc[0] == pytest.approx(0.0)


def test_rare_categories_are_folded_into_other():
    values = pd.Series(["common"] * 900 + ["rare"] * 20 + ["rarer"] * 20)
    target = pd.Series([0] * 460 + [1] * 480 + [1] * 20 + [0] * 20)
    scheme = fit_categorical_scheme("category", values, target, min_bin_pct=0.05)
    assert "common" in scheme.category_woe
    assert "OTHER" in scheme.category_woe
    assert "rare" not in scheme.category_woe


def test_encoder_never_uses_a_leakage_column():
    """The feature set must be disjoint from the post-origination columns.

    This is the single most important invariant in the project: a PD model that
    can see repayment outcome is worthless in production.
    """
    leakage = set(cfg.LEAKAGE_COLUMNS)
    features = set(cfg.MODEL_FEATURES)
    assert leakage & features == set()
    assert "default_flag" not in features
    assert "out_prncp" not in features
    assert "days_past_due" not in features


def test_selection_drops_weak_features_but_flags_rather_than_drops_strong_ones():
    """A high IV is a review trigger, not an automatic deletion.

    Internal grade and contract rate legitimately carry IV near 0.9 on this book.
    Deleting them on a leakage heuristic would remove the lender's own risk
    assessment from the model, which is worse than the risk the heuristic guards
    against.
    """
    frame, target = _separable_frame()
    encoder = WoeEncoder(numeric_features=["signal", "noise"], categorical_features=[]).fit(frame, target)
    iv_table = encoder.iv_table()
    ivs = dict(zip(iv_table["feature"], iv_table["information_value"], strict=True))

    selected = encoder.selected_features(iv_threshold=0.02)
    assert "signal" in selected  # strong, retained
    assert "noise" not in selected  # weak, dropped on the lower bound

    flagged = encoder.flagged_features(iv_upper=0.5)
    assert ivs["signal"] > 0.5
    assert "signal" in flagged
    assert "noise" not in flagged


def test_encoder_iv_table_is_sorted_descending():
    frame, target = _separable_frame()
    encoder = WoeEncoder(numeric_features=["signal", "noise"], categorical_features=[]).fit(frame, target)
    table = encoder.iv_table()
    values = table["information_value"].tolist()
    assert values == sorted(values, reverse=True)
