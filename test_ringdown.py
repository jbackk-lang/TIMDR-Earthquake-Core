"""Testy ringdown.py - port 1:1 z universal-state-analyzer/timdr_core/ringdown.py
(patrz tam po pełną walidację numeryczną: odzyskana częstotliwość/tłumienie
zgodne z teorią na tłumionym oscylatorze o znanym f0/tau, plus historia
znalezionych/naprawionych błędów). Tu: powtórzona kluczowa walidacja w
skali sejsmicznej (0.01-1 Hz, niższe fs) + test na PRAWDZIWYM zapisie
(obspy_BW_RJOB_example.csv), dokumentujący wprost próg-czułość opisaną
w docstringu modułu.
"""
import csv

import numpy as np
import pytest

from ringdown import ringdown_resonance
from timdr_core_earthquake import TIMDR_EarthquakeCore


def test_underdamped_recovers_known_frequency_and_damping_skala_sejsmiczna():
    fs = 10.0  # 10 probek/s - typowe dla LHZ/BHZ pasma dlugookresowego
    t = np.arange(0, 400.0, 1 / fs)
    event_idx = int(50.0 * fs)
    f0, tau = 0.1, 25.0  # rzad wielkosci: fala powierzchniowa/rezonans lokalny 0.01-1 Hz
    post = t[event_idx:] - t[event_idx]
    x = np.zeros_like(t)
    x[event_idx:] = 5.0 * np.exp(-post / tau) * np.cos(2 * np.pi * f0 * post)
    rng = np.random.default_rng(0)
    x_noisy = x + rng.normal(0, 0.05, len(t))

    res = ringdown_resonance(t, x_noisy, event_idx=event_idx, pre_event_window=int(30 * fs))
    assert res["is_oscillatory"] is True
    assert res["frequency_hz"] == pytest.approx(f0, rel=0.05)
    zeta_theory = 1.0 / np.sqrt((2 * np.pi * f0 * tau) ** 2 + 1)
    assert res["damping_ratio"] == pytest.approx(zeta_theory, rel=0.3)


def test_monotonic_recovery_is_not_oscillatory():
    fs = 10.0
    t = np.arange(0, 400.0, 1 / fs)
    event_idx = int(50.0 * fs)
    post = t[event_idx:] - t[event_idx]
    x = np.zeros_like(t)
    x[event_idx:] = 5.0 * np.exp(-post / 20.0)
    rng = np.random.default_rng(1)
    x_noisy = x + rng.normal(0, 0.05, len(t))

    # UWAGA: max_lookahead ograniczony do ~8*tau po zdarzeniu. Bez tego
    # ograniczenia dlugi "ogon" czystego szumu PO calkowitym zaniku
    # sygnalu (tutaj: ~250s przy tau=20s) losowo przecina pasmo szumu
    # wielokrotnie z samej definicji (spacer losowy wokol 0) i falszywie
    # zglasza is_oscillatory=True - to NIE jest blad funkcji, to
    # artefakt zbyt dlugiego okna analizy wzgledem faktycznego czasu
    # zaniku (zweryfikowano empirycznie przy pisaniu tego testu).
    # `max_lookahead` istnieje dokladnie po to, zeby ograniczyc analize
    # do sensownego okna po zdarzeniu.
    res = ringdown_resonance(
        t, x_noisy, event_idx=event_idx,
        pre_event_window=int(30 * fs),
        max_lookahead=int(8 * 20.0 * fs),
    )
    assert res["is_oscillatory"] is False


def _load_real_trace():
    t, s = [], []
    with open("obspy_BW_RJOB_example.csv", newline="") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"]))
            s.append(float(row["s"]))
    return np.asarray(t), np.asarray(s)


def test_prawdziwy_slad_bw_rjob_progi_zgodne_z_udokumentowana_czuloscia():
    """Powtarza test wykonany na prawdziwym zapisie (BW.RJOB, standardowy
    przyklad tutorialowy ObsPy - autentyczny lokalny wstrzas) opisany w
    README/docstringu modulu: STA/LTA (nsta=0.5s, nlta=5s, thr_on=3.0,
    thr_off=1.5 - zweryfikowany wobec ObsPy w test_timdr_core_earthquake.py)
    znajduje DWA wyzwolenia w tym 30-sekundowym sladzie; drugie
    (glowny wstrzas, ~t=17.9s) jest tym analizowanym dalej. is_oscillatory
    zalezy od progu szumu - False przy domyslnym noise_floor_factor>=2.0,
    True przy poluzowaniu do 1.5/1.0. To NIE jest test "poprawnosci" w
    sensie znanej z gory odpowiedzi (nie znamy niezaleznie prawdziwej
    czestotliwosci tego wstrzasu) - to jest test REGRESJI, ktory ma
    pilnowac, zeby ta udokumentowana, uczciwie zaraportowana wrazliwosc
    na prog nie zmienila sie cicho przy przyszlych zmianach kodu."""
    t, s = _load_real_trace()
    core = TIMDR_EarthquakeCore()
    fs = 1.0 / np.median(np.diff(t))
    charfct = core.sta_lta(s, nsta=int(0.5 * fs), nlta=int(5 * fs))
    onsets = core.trigger_onset(charfct, thr_on=3.0, thr_off=1.5)
    assert len(onsets) >= 2
    event_idx = int(onsets[1][0])
    assert 1750 <= event_idx <= 1850  # t ~ 17.5-18.5s, glowny wstrzas w tym sladzie

    res_default = ringdown_resonance(t, s, event_idx, pre_event_window=200, noise_floor_factor=3.0)
    assert res_default["is_oscillatory"] is False

    res_loose = ringdown_resonance(t, s, event_idx, pre_event_window=200, noise_floor_factor=1.5)
    assert res_loose["is_oscillatory"] is True
    assert res_loose["period_s"] == pytest.approx(4.0, abs=2.0)
