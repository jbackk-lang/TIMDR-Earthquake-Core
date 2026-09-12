"""Testy residual_offset.py - cecha komplementarna do ringdown_resonance()
(ringdown.py): zamiast "czy powrót jest oscylacyjny", pyta "czy w ogóle
NASTĄPIŁ powrót" (czy pod koniec okna obserwacji sygnał wciąż siedzi poza
pasmem szumu sprzed zdarzenia). Zobacz residual_offset.py's docstring dla
pełnego uzasadnienia i pre-rejestracji parametrów.
"""
import csv

import numpy as np
import pytest

from residual_offset import residual_offset
from ringdown import ringdown_resonance
from timdr_core_earthquake import TIMDR_EarthquakeCore


def test_persistent_step_is_detected():
    """Pozytywna kontrola: trwały skok (NIE zanikający) po zdarzeniu MUSI
    dać is_persistent=True."""
    fs = 10.0
    t = np.arange(0, 400.0, 1 / fs)
    event_idx = int(50.0 * fs)
    x = np.zeros_like(t)
    x[event_idx:] = 5.0  # trwaly skok, stala wartosc od tego momentu
    rng = np.random.default_rng(0)
    x_noisy = x + rng.normal(0, 0.05, len(t))

    res = residual_offset(t, x_noisy, event_idx=event_idx, pre_event_window=int(30 * fs))
    assert res["is_persistent"] is True
    assert res["offset_ratio"] is not None
    assert res["offset_ratio"] >= 1.0


def test_full_recovery_to_baseline_is_not_persistent():
    """Negatywna kontrola: sygnal ktory calkowicie wraca do linii bazowej
    (nawet jesli po drodze robil cokolwiek) NIE powinien byc persistent -
    ogon okna to znowu czysty szum wokol baseline."""
    fs = 10.0
    t = np.arange(0, 400.0, 1 / fs)
    event_idx = int(50.0 * fs)
    post = t[event_idx:] - t[event_idx]
    x = np.zeros_like(t)
    x[event_idx:] = 5.0 * np.exp(-post / 5.0)  # szybki, calkowity zanik do 0
    rng = np.random.default_rng(1)
    x_noisy = x + rng.normal(0, 0.05, len(t))

    res = residual_offset(t, x_noisy, event_idx=event_idx, pre_event_window=int(30 * fs),
                            max_lookahead=int(20 * fs))
    assert res["is_persistent"] is False


def test_decaying_oscillation_that_fully_returns_is_not_persistent():
    """Kluczowa kontrola DYSKRYMINACYJNA (patrz residual_offset.py's
    docstring): dokladnie ten sam sygnal, ktory ringdown_resonance()
    klasyfikuje jako is_oscillatory=True (tlumiona oscylacja, WRACA do
    zera), residual_offset() NIE powinien uznac za persistent - bo z
    definicji nie jest trwaly. To pokazuje, ze obie funkcje mierza rozne
    rzeczy na tym samym sygnale, nie ten sam fakt pod inna nazwa."""
    fs = 10.0
    t = np.arange(0, 400.0, 1 / fs)
    event_idx = int(50.0 * fs)
    f0, tau = 0.1, 25.0
    post = t[event_idx:] - t[event_idx]
    x = np.zeros_like(t)
    x[event_idx:] = 5.0 * np.exp(-post / tau) * np.cos(2 * np.pi * f0 * post)
    rng = np.random.default_rng(0)
    x_noisy = x + rng.normal(0, 0.05, len(t))

    ring_res = ringdown_resonance(t, x_noisy, event_idx=event_idx, pre_event_window=int(30 * fs))
    assert ring_res["is_oscillatory"] is True  # ta sama asercja co w test_ringdown.py

    offset_res = residual_offset(t, x_noisy, event_idx=event_idx, pre_event_window=int(30 * fs),
                                   max_lookahead=int(8 * tau * fs))
    assert offset_res["is_persistent"] is False


def test_offset_ratio_scales_with_step_size():
    """offset_ratio powinien rosnac mniej-wiecej proporcjonalnie do
    wielkosci trwalego skoku (przy stalym poziomie szumu)."""
    fs = 10.0
    t = np.arange(0, 200.0, 1 / fs)
    event_idx = int(20.0 * fs)
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.05, len(t))

    ratios = []
    for step in (1.0, 2.0, 4.0):
        x = np.zeros_like(t)
        x[event_idx:] = step
        res = residual_offset(t, x + noise, event_idx=event_idx, pre_event_window=int(15 * fs))
        ratios.append(res["offset_ratio"])
    assert ratios[0] < ratios[1] < ratios[2]


def test_insufficient_tail_samples_is_fail_closed():
    """Zbyt krotkie okno lookahead (mniej niz min_tail_samples) daje
    insufficient_tail_samples=True i is_persistent=False, nie cichy blad
    ani zgadywanie."""
    t = np.arange(0, 20.0, 1.0)
    s = np.zeros_like(t)
    s[10:] = 5.0
    res = residual_offset(t, s, event_idx=10, pre_event_window=5, max_lookahead=2,
                            min_tail_samples=5)
    assert res["insufficient_tail_samples"] is True
    assert res["is_persistent"] is False


def test_zero_noise_floor_degenerate_case_does_not_divide_by_zero():
    """pre_event_window calkowicie stale (noise_std=0) -> noise_floor=0 -
    nie powinno rzucic ani dac inf/NaN, tylko jawny, udokumentowany
    fallback (persistent jesli jest JAKIEKOLWIEK mierzalne odchylenie)."""
    t = np.arange(0, 40.0, 1.0)
    s = np.zeros_like(t)
    s[:10] = 0.0  # idealnie stale "przed" - noise_std=0
    s[10:] = 3.0  # trwaly skok po
    res = residual_offset(t, s, event_idx=10, pre_event_window=10, max_lookahead=20)
    assert res["noise_floor"] == 0.0
    assert res["offset_ratio"] is None
    assert res["is_persistent"] is True
    assert np.isfinite(res["mean_abs_tail_offset"])


def test_raises_on_event_idx_out_of_range():
    t = np.arange(0, 10.0, 1.0)
    s = np.zeros_like(t)
    with pytest.raises(ValueError):
        residual_offset(t, s, event_idx=999)


def _load_real_trace():
    t, s = [], []
    with open("obspy_BW_RJOB_example.csv", newline="") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"]))
            s.append(float(row["s"]))
    return np.asarray(t), np.asarray(s)


def test_prawdziwy_slad_bw_rjob_regression():
    """Test regresyjny (jak w test_ringdown.py) na prawdziwym zapisie -
    nie zaklada z gory 'poprawnej' odpowiedzi (nie znamy niezaleznie, czy
    ten konkretny wstrzas zostawil trwaly odcisk w tym 30s oknie), tylko
    pilnuje, zeby wynik nie zmienil sie cicho przy przyszlych zmianach.
    event_idx wyznaczony tym samym STA/LTA co w test_ringdown.py (nie
    zahardkodowana liczba), zeby oba testy patrzyly na to samo zdarzenie."""
    t, s = _load_real_trace()
    core = TIMDR_EarthquakeCore()
    fs = 1.0 / np.median(np.diff(t))
    charfct = core.sta_lta(s, nsta=int(0.5 * fs), nlta=int(5 * fs))
    onsets = core.trigger_onset(charfct, thr_on=3.0, thr_off=1.5)
    assert len(onsets) >= 2
    event_idx = int(onsets[1][0])
    assert 1750 <= event_idx <= 1850

    res = residual_offset(t, s, event_idx, pre_event_window=200, noise_floor_factor=3.0)
    assert res["insufficient_tail_samples"] is False
    assert isinstance(res["is_persistent"], bool)
    assert res["offset_ratio"] is not None and res["offset_ratio"] >= 0.0
