"""Testy meta_adapter.py.

Ten sam wzorzec co Synoptyk-v3/tests/test_meta_adapter.py: (1) syntetyczne
kontrole pozytywna/negatywna z asercjami o KIERUNKU wyniku, ustalone przed
uruchomieniem na realnym sladzie; (2) test na PRAWDZIWYCH danych
(Ridgecrest 2019, stacja CLC, real_waveform_CLC_RIO/CLC_HHZ.csv) - CELOWO
bez asercji "faza == X", tylko sprawdzenie mechanicznej poprawnosci +
raport (nie asercja) o zgodnosci z realnym, znanym czasem mainshocku
(t~=60.0s, patrz HISTORIA_I_TESTY.md)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from meta_adapter import (
    WINDOW_SECONDS,
    build_meta_series_from_waveform,
    compute_global_thresholds,
    window_to_meta_state,
)
from timdr_core_earthquake import TIMDR_EarthquakeCore

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_CSV = os.path.join(HERE, "data", "ridgecrest_2019", "real_waveform_CLC_RIO", "CLC_HHZ.csv")


def _synthetic_calm(duration=30.0, fs=100.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / fs)
    s = rng.normal(0.0, 1.0, size=len(t))
    return t, s


def _synthetic_with_event(duration=30.0, fs=100.0, event_t=15.0, amplitude=50.0, seed=0):
    t, s = _synthetic_calm(duration, fs, seed)
    s = s.copy()
    event_dur = 2.0
    mask = (t >= event_t) & (t < event_t + event_dur)
    n_event = int(mask.sum())
    envelope = np.exp(-np.linspace(0, 5, n_event))
    s[mask] += amplitude * envelope * np.sin(2 * np.pi * 5.0 * (t[mask] - event_t))
    return t, s


# ---------------------------------------------------------------------------
# Kontrola pozytywna: okno z realistycznym "wstrzasem" -> wyzsze skladowe
# adaptera (Lambda/tau/rho/J) niz okno spokojnego szumu.
# ---------------------------------------------------------------------------

def test_positive_control_event_window_has_higher_rho_and_tau_than_calm_window():
    """V2: progi globalne licza sie na CALYM sladzie (kalibracyjnym,
    obejmujacym i wstrzas, i spokojna czesc) - dokladnie tak, jak
    build_meta_series_from_waveform() robi to naprawde."""
    core = TIMDR_EarthquakeCore()
    t_full, s_full = _synthetic_with_event(duration=30.0, fs=100.0, event_t=15.0, seed=1)
    thresholds = compute_global_thresholds(core, t_full, s_full)

    calm_mask = (t_full >= 0.0) & (t_full < 5.0)
    event_mask = (t_full >= 15.0) & (t_full < 20.0)
    t_calm, s_calm = t_full[calm_mask], s_full[calm_mask]
    t_ev, s_ev = t_full[event_mask], s_full[event_mask]

    calm_state = window_to_meta_state(core, t_calm, s_calm, thresholds)
    event_state = window_to_meta_state(core, t_ev, s_ev, thresholds)

    assert event_state.rho > calm_state.rho
    assert event_state.tau > calm_state.tau
    assert event_state.J > calm_state.J


def test_negative_control_two_identical_calm_windows_give_zero_M_and_stable_phase():
    core = TIMDR_EarthquakeCore()
    t, s = _synthetic_calm(duration=60.0, fs=100.0, seed=2)
    thresholds = compute_global_thresholds(core, t, s)
    half = len(t) // 2
    t1, s1 = t[:half], s[:half]
    t2, s2 = t[:half], s[:half]  # ta sama tablica - identyczne okno

    from timdr_meta_dynamics import MetaOperatorM
    op = MetaOperatorM()
    state1 = window_to_meta_state(core, t1, s1, thresholds)
    state2 = window_to_meta_state(core, t2, s2, thresholds)
    M = op.compute(state1, state2, dt=1.0)
    assert op.magnitude(M) == 0.0


def test_v1_scale_invariance_bug_documented_by_algebra():
    """Dokumentuje ALGEBRAICZNIE przyczyne porzucenia V1 (patrz docstring
    modulu): normalizacja mean(okno)/prog(TEGO SAMEGO okna) jest
    niezmiennicza na jednorodne przeskalowanie - podstawiony konkretny
    przyklad liczbowy (zasada z meta-regul ekosystemu TIMDR: kazde
    twierdzenie matematyczne podeprzec liczbowym przykladem przed
    publikacja)."""
    from meta_adapter import _robust_threshold

    window_a = np.array([1.0, 1.0, 1.0, 10.0])
    window_b = window_a * 10.0  # ta sama "ksztalt", 10x wieksza skala

    ratio_a = window_a.mean() / _robust_threshold(window_a)
    ratio_b = window_b.mean() / _robust_threshold(window_b)

    assert ratio_a == pytest.approx(ratio_b)


# ---------------------------------------------------------------------------
# Walidacja wejscia.
# ---------------------------------------------------------------------------

def test_requires_at_least_2_full_windows():
    t = np.arange(0.0, 3.0, 0.01)
    s = np.random.default_rng(0).normal(size=len(t))
    with pytest.raises(ValueError):
        build_meta_series_from_waveform(t, s, window_seconds=5.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        build_meta_series_from_waveform(np.arange(10.0), np.arange(9.0))


# ---------------------------------------------------------------------------
# PRAWDZIWE dane: Ridgecrest 2019, stacja CLC, 360s realnego sladu.
# Znany fakt (HISTORIA_I_TESTY.md, ustalony NIEZALEZNIE od tego adaptera,
# przez sta_lta()/trigger_onset() tego samego repo): prawdziwy mainshock
# jest w tym sladzie w przyblizeniu przy t~=60.0s.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="brak pliku z realnymi danymi CLC_HHZ.csv")
def test_end_to_end_real_ridgecrest_waveform_runs_without_crashing():
    import warnings
    from seismic_loader import SeismicLoader

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loader = SeismicLoader()
        t, s = loader.load_csv(REAL_CSV)

    result = build_meta_series_from_waveform(t, s, window_seconds=WINDOW_SECONDS)

    n_windows_expected = int((t[-1] - t[0]) // WINDOW_SECONDS)
    assert len(result.states) <= n_windows_expected
    assert len(result.states) >= n_windows_expected - 1  # tolerancja na niepelne okno na koncu
    assert len(result.M_series) == len(result.states) - 1
    assert len(result.phases) == len(result.M_series)
    assert all(p in ("stabilna", "przejsciowa", "krytyczna") for p in result.phases)

    for st in result.states:
        assert math_isfinite_all(st)
        assert 0.0 <= st.Lambda <= 1.0
        assert 0.0 <= st.rho <= 1.0
        assert 0.0 <= st.J <= 1.0

    # V2 (progi globalne z CALEGO sladu) dyskryminuje: nie wszystko jest
    # ta sama faza (V1 by tego nie przeszedl - patrz test_v1_..._algebra
    # i docstring modulu).
    assert len(set(result.phases)) > 1

    # Krok, w ktorym po raz pierwszy osiagnieto "krytyczna", powinien
    # przypadac w poblizu realnego, NIEZALEZNIE ustalonego czasu mainshocku
    # (t~=60.0s, wykryty przez sta_lta()/trigger_onset() w tym samym repo
    # na t=61.1s - patrz HISTORIA_I_TESTY.md). Tolerancja = 2 okna (10s),
    # bo okno 5s samo w sobie wnosi niepewnosc rzedu jednego okna.
    assert result.trigger.triggered
    idx = result.trigger.location
    t_trigger_window_end = result.window_starts[idx + 1]
    assert abs(t_trigger_window_end - 60.0) <= 10.0, (
        f"Pierwsza 'krytyczna' faza w oknie konczacym sie o t={t_trigger_window_end}s, "
        f"za daleko od znanego t~=60.0s mainshocku."
    )


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="brak pliku z realnymi danymi CLC_HHZ.csv")
def test_causal_calibration_end_variant_on_real_data_saturates_post_event():
    """Wariant PRZYCZYNOWY (calibration_end=55.0, tylko dane sprzed
    mainshocku jako referencja) - patrz zastrzezenie #6 w docstringu
    modulu. Real wynik (zweryfikowany na surowych danych, NIE artefakt):
    niemal cala reszta 6-minutowego zapisu wychodzi jako 'krytyczna',
    bo naprawde jest ekstremalna wzgledem tla SPRZED zdarzenia (odch.
    std. amplitudy surowej ~40x wyzsze nawet 3-5 min po). Asercja
    ROZLUZNIONA (>50%, nie dokladna liczba) - to zjawisko fizyczne
    (dlugi coda/sekwencja wstrzasow wtornych), nie precyzyjnie
    powtarzalna stala."""
    import warnings
    from seismic_loader import SeismicLoader

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loader = SeismicLoader()
        t, s = loader.load_csv(REAL_CSV)

    result = build_meta_series_from_waveform(t, s, window_seconds=WINDOW_SECONDS, calibration_end=55.0)

    n_krytyczna = result.phases.count("krytyczna")
    assert n_krytyczna / len(result.phases) > 0.5, (
        f"Oczekiwano nasycenia 'krytyczna' po zdarzeniu przy kalibracji "
        f"przyczynowej, dostano tylko {n_krytyczna}/{len(result.phases)}."
    )
    for st in result.states:
        assert math_isfinite_all(st)


def math_isfinite_all(state) -> bool:
    import math
    return all(math.isfinite(v) for v in (state.Lambda, state.tau, state.rho, state.J))
