"""
precursor_validation.py — machine-checkable guard for "precursor" signals
built on top of `ringdown_resonance()` (ringdown.py).
================================================================================
CONTEXT: `precursor_ringdown_test.py --mode real` already ran the real
test this module reruns: does a feature built from `ringdown_resonance()`
on `fronts()` candidates, computed ONLY from data BEFORE a real large
earthquake, come out higher than the same feature computed in random
background windows (Mann-Whitney U, two-sided)? Real USGS catalog
(M>=6.5, last 5 years), real EarthScope/IRIS waveforms, 8 GSN stations.

**RESULT: REJECTED.** p = 0.9971 (`n_pre_event=40`, `n_background=60`,
mean frac_oscillatory 0.0683 vs 0.0601). See HISTORIA_I_TESTY.md and the
frozen run in `precursor_ringdown_test_output.json`. Same verdict as the
independent `Topology(t)` test in this repo (analyze_topology_resonance_seismic.py).

Until now that negative, real-data result lived ONLY in prose
(HISTORIA_I_TESTY.md) — `ringdown_resonance()` itself had no way of
telling a caller it had been tested as a precursor signal and failed.
This module makes that result a first-class, machine-checkable status
that `ringdown_resonance()` now attaches to every result it returns (see
ringdown.py: `is_validated_precursor`, `precursor_confidence`,
`precursor_validation`) and warns about via `PrecursorValidationWarning`.

This is NOT a new statistical claim and does NOT hardcode the p-value:
`validate_against_catalog()` reruns Mann-Whitney U from scratch on the
real `frac_oscillatory` samples frozen in
`precursor_ringdown_test_output.json` (the same dataset
`precursor_ringdown_test.py --mode real` produced). Regenerate that file
with a fresh real run and this guard picks up the new numbers - it only
hardcodes the dataset location, the statistical test, and the
significance/effect-size decision rule, exactly like `precursor_ringdown_test.py`
already did for the one-off script version of this same comparison.

No hard scipy dependency for the actual test: this module sibling-imports
`mann_whitney_test(..., backend="numpy")` from
`TIMDR-Math-Formalism/timdr_formalism/pipeline.py` instead of keeping its
own copy of the pure numpy/stdlib Mann-Whitney U implementation this
module originally had. UPDATE (2026-09-08): that local implementation
(`_mannwhitney_u_p`) was a THIRD, independent copy of the same math
already living twice in this ecosystem (TIMDR-Math-Formalism's
`pipeline.py` protocol, and this file) - now consolidated into ONE
canonical definition in `pipeline.py` (see that file's header for the
Device Guard story this guard is itself named after), which this module
now calls via the sibling-repo import pattern used throughout this
ecosystem (see `_ensure_timdr_math_formalism_on_path()` below - same
pattern as `meta_adapter.py`'s `_ensure_timdr_meta_dynamics_on_path()`).
This mirrors the repo's existing precaution around module-level scipy
imports (see HISTORIA_I_TESTY.md: `savgol_filter` at module scope
crashed the whole GUI on a machine where Windows Device Guard blocked
scipy's DLLs, even for callers who never used that code path) - this
guard runs on every `ringdown_resonance()` call, so it must not risk the
same failure mode. `pipeline.py` itself now wraps its own `scipy` import
in `try/except` (same fix, applied at the canonical source), so
importing it at all is safe even when scipy is broken; `backend="numpy"`
below additionally guarantees the actual U-test computation never
touches scipy at call time either, matching this module's original
guarantee exactly.

NOTE ON EFFECT SIZE SIGN CONVENTION: `pipeline.py`'s `rank_biserial_effect_size`
uses `r = 2*U/(n_test*n_background) - 1` (r>0 means the FIRST/test group
has higher values) - the opposite sign convention from this module's
former self-computed `effect_r = 1.0 - 2*U/(n1*n2)`. This only ever
flows into `abs(effect_r) >= min_effect_size_r` (magnitude) and a
separately-computed `mean_a > mean_b` (direction) below, never into a
sign-dependent comparison, so the switch is behavior-preserving for
`mannwhitney_validate()`'s pass/fail logic - only the raw number in the
returned dict's `effect_size_r` field flips sign (now matches
`pipeline.py`'s convention, arguably the more intuitive one).
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from functools import lru_cache

import numpy as np

_DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "precursor_ringdown_test_output.json"
)

ALPHA_DEFAULT = 0.05
MIN_EFFECT_SIZE_R_DEFAULT = 0.2  # Cohen's convention: |r| >= 0.2 ~ "small" effect


class PrecursorValidationWarning(UserWarning):
    """A signal built on ringdown_resonance() is being used/returned
    without having passed real-data (USGS catalog + real waveforms)
    validation as an earthquake precursor. See HISTORIA_I_TESTY.md."""


def _ensure_timdr_math_formalism_on_path() -> None:
    """Adds the sibling TIMDR-Math-Formalism folder to sys.path - same
    pattern as `meta_adapter.py`'s `_ensure_timdr_meta_dynamics_on_path()`
    and `bearing_meta_adapter.py`'s `_ensure_siblings_on_path()` in
    TIMDR-Industrial-Predict. TIMDR-Earthquake-Core sits directly in the
    parent directory, so the sibling is exactly one level up."""
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.abspath(os.path.join(here, "..", "TIMDR-Math-Formalism"))
    if not os.path.isdir(sibling):
        raise ImportError(
            "precursor_validation wymaga folderu 'TIMDR-Math-Formalism' jako "
            f"siostry repo TIMDR-Earthquake-Core (szukano w: {sibling})."
        )
    if sibling not in sys.path:
        sys.path.insert(0, sibling)


_ensure_timdr_math_formalism_on_path()

from timdr_formalism.pipeline import mann_whitney_test as _mann_whitney_test  # noqa: E402


def mannwhitney_validate(
    group_a,
    group_b,
    alpha: float = ALPHA_DEFAULT,
    min_effect_size_r: float = MIN_EFFECT_SIZE_R_DEFAULT,
    label_a: str = "pre_event",
    label_b: str = "background",
) -> dict:
    """Reruns the pre-event-vs-background Mann-Whitney U comparison and
    applies a pre-registered decision rule: `validated=True` only if the
    difference is statistically significant (p < alpha), the effect size
    is at least `min_effect_size_r` (rank-biserial correlation, Cohen's
    'small' convention by default), AND the direction matches the
    precursor hypothesis (group_a/pre_event higher than group_b/background).
    Any one of those failing means `validated=False` - fails closed, same
    spirit as `hybrid_trigger()`'s `missing_twist`/`missing_anomaly`
    reasons in timdr_core_earthquake.py: a rejection always carries an
    explicit machine-readable reason, never a silent default."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    if len(a) < 5 or len(b) < 5:
        return {
            "validated": False,
            "p_value": None,
            "effect_size_r": None,
            f"n_{label_a}": int(len(a)),
            f"n_{label_b}": int(len(b)),
            "alpha": alpha,
            "min_effect_size_r": min_effect_size_r,
            "reason": (
                f"za malo probek do wiarygodnego testu ({len(a)} {label_a}, "
                f"{len(b)} {label_b}) - potrzeba >=5 w kazdej grupie"
            ),
        }

    test_result = _mann_whitney_test(a, b, backend="numpy")
    p = test_result.pvalue
    effect_r = test_result.effect_size_r
    n1, n2 = len(a), len(b)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))

    significant = p < alpha
    big_enough_effect = abs(effect_r) >= min_effect_size_r
    direction_ok = mean_a > mean_b
    validated = bool(significant and big_enough_effect and direction_ok)

    if validated:
        reason = (
            f"{label_a} istotnie i namacalnie wyzsze niz {label_b} "
            f"(p={p:.4g}, r={effect_r:.3f})"
        )
    elif not significant:
        reason = (
            f"brak statystycznie istotnej roznicy (p={p:.4g} >= alpha={alpha}) "
            "- WYNIK NEGATYWNY"
        )
    elif not big_enough_effect:
        reason = (
            f"roznica formalnie istotna ale efekt za maly "
            f"(r={effect_r:.3f}, prog={min_effect_size_r})"
        )
    else:
        reason = f"roznica istotna, ale w NIEWLASCIWYM kierunku ({label_b} >= {label_a})"

    return {
        "validated": validated,
        "p_value": float(p),
        "effect_size_r": float(effect_r),
        f"n_{label_a}": n1,
        f"n_{label_b}": n2,
        f"mean_{label_a}": mean_a,
        f"mean_{label_b}": mean_b,
        "alpha": alpha,
        "min_effect_size_r": min_effect_size_r,
        "reason": reason,
    }


def load_real_ringdown_precursor_dataset(path: str | None = None) -> dict:
    """Loads the real `frac_oscillatory` samples from the frozen
    `precursor_ringdown_test.py --mode real` run (real USGS M>=6.5
    catalog, real EarthScope/IRIS waveforms, 8 GSN stations - see
    HISTORIA_I_TESTY.md). This IS the "real catalog data" this repo
    already collected; `validate_against_catalog()` reruns the
    statistical test on it rather than inventing a new dataset."""
    path = path or _DEFAULT_DATASET_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    real = data.get("real_test")
    if not real:
        raise ValueError(
            f"{path} nie zawiera klucza 'real_test' - uruchom najpierw "
            "'python precursor_ringdown_test.py --mode real'"
        )
    pre = [d["frac_oscillatory"] for d in real["pre_event_details"]]
    bg = [d["frac_oscillatory"] for d in real["background_details"]]
    return {"pre_event": pre, "background": bg, "source": path}


def validate_against_catalog(
    real_catalog_data: dict | None = None,
    alpha: float = ALPHA_DEFAULT,
    min_effect_size_r: float = MIN_EFFECT_SIZE_R_DEFAULT,
) -> dict:
    """Reruns the Mann-Whitney U precursor-vs-background comparison
    against real catalog data and returns a validation-status dict.

    `real_catalog_data`: dict with 'pre_event' and 'background' lists of
    `frac_oscillatory` values (same shape as
    `load_real_ringdown_precursor_dataset()`'s return). If omitted,
    loads the frozen real USGS+EarthScope run already in this repo
    (`precursor_ringdown_test_output.json`) - the real data this repo's
    own real-data test already collected, not synthetic data.
    """
    if real_catalog_data is None:
        real_catalog_data = load_real_ringdown_precursor_dataset()
    result = mannwhitney_validate(
        real_catalog_data["pre_event"], real_catalog_data["background"],
        alpha=alpha, min_effect_size_r=min_effect_size_r,
    )
    result["source"] = real_catalog_data.get("source", "provided")
    return result


@lru_cache(maxsize=1)
def get_ringdown_precursor_validation_status() -> dict:
    """Cached (computed once per process, lazily - no import-time cost)
    validation status of `ringdown_resonance()` as an earthquake
    PRECURSOR signal. A missing/corrupt validation dataset is never
    treated as "validated" (fail-closed)."""
    try:
        return validate_against_catalog()
    except Exception as e:
        return {
            "validated": False,
            "p_value": None,
            "effect_size_r": None,
            "reason": (
                f"nie udalo sie zaladowac/przeliczyc walidacji na realnych danych "
                f"({type(e).__name__}: {e}) - domyslnie NIEZWALIDOWANE (fail-closed)"
            ),
        }


def reset_validation_cache() -> None:
    """Test/debug helper: clears the cached status so the next call to
    get_ringdown_precursor_validation_status() recomputes it (e.g. after
    regenerating precursor_ringdown_test_output.json, or in tests that
    inject a different dataset)."""
    get_ringdown_precursor_validation_status.cache_clear()


def warn_if_unvalidated(context: str = "ringdown_resonance() jako sygnal precursor") -> dict:
    """Emits a PrecursorValidationWarning (once per process per call
    site, standard python warnings behaviour) if the cached validation
    status says this signal is not validated as a real-world precursor.
    Returns the status dict either way, so callers can also inspect it
    programmatically instead of relying on the warning."""
    status = get_ringdown_precursor_validation_status()
    if not status.get("validated", False):
        warnings.warn(
            f"{context}: NIEZWALIDOWANE wzgledem realnych danych (USGS "
            f"M>=6.5 + EarthScope/IRIS, 8 stacji GSN - patrz HISTORIA_I_TESTY.md "
            f"i precursor_ringdown_test_output.json). {status.get('reason', 'brak powodu')}. "
            "Traktuj jako confidence=0 do celow PREDYKCYJNYCH - to narzedzie jest "
            "OPISOWE (post-event), nie predyktorem trzesien ziemi.",
            PrecursorValidationWarning,
            stacklevel=3,
        )
    return status
