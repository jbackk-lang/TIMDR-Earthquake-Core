"""
ringdown.py — czy powrót do poziomu odniesienia PO zdarzeniu jest
REZONANSOWY (oscylacyjny)
================================================================================
Port 1:1 (bez zmian w matematyce) z jbackk-lang/universal-state-analyzer
(`timdr_core/ringdown.py`), TIMDR-Grid-Monitor, analizator-gieldowy-v3 i
deliverable_timdr_finanse - tam metoda jest zweryfikowana numerycznie na
tłumionym oscylatorze o znanej częstotliwości/stałej czasowej (odzyskana
częstotliwość i tłumienie zgodne z teorią, patrz tamte README/historia
commitów za pełną listę znalezionych i naprawionych błędów: chatter przy
wysokim fs, zdegenerowane granice segmentów, mediana zamiast średniej).

KONTEKST SEJSMICZNY (dlaczego tu w ogóle jest): po impulsie (wstrząsie)
skorupa Ziemi odchyla się od stanu równowagi i wraca do niego - pytanie
brzmi, czy ten powrót jest oscylacyjny (swobodne oscylacje/fale
powierzchniowe/rezonans lokalny - realne, dobrze udokumentowane zjawisko
w sejsmologii, np. "dzwonienie" całej Ziemi po bardzo dużych wstrząsach)
czy monotoniczny.

**UWAGA - TO JEST NARZĘDZIE OPISOWE (POST-EVENT), NIE PREDYKCYJNE.**
Sama ta funkcja analizuje powrót do równowagi PO zdarzeniu (potrzebuje
`event_idx` - momentu perturbacji) - nie mówi nic o tym, czy coś podobnego
da się wykryć PRZED przyszłym wstrząsem. Test predykcyjności (czy
cokolwiek zbudowanego na tej funkcji wykazuje sygnał w oknie PRZED
prawdziwym wstrząsem, silniejszy niż w losowym tle) jest w
`precursor_ringdown_test.py` w tym samym repo - i jest ZUPEŁNIE OSOBNYM
pytaniem od tego, co robi `ringdown_resonance()` samo w sobie. Patrz
README.md, sekcja o teście predykcyjności `Topology(t)`, dla precedensu
tego samego rodzaju testu na innej matematyce w tym repo - wynik tamtego
testu był NEGATYWNY.

NIEZWALIDOWANE NA DANYCH SEJSMICZNYCH w chwili portowania tej funkcji -
zweryfikowane wyłącznie na syntetycznym, czystym tłumionym oscylatorze
(jak we wszystkich innych portach). Realne sejsmogramy po wstrząsie to
superpozycja wielu nakładających się trybów (fala P, S, powierzchniowa,
koda) - metoda zero-crossing zakłada z grubsza JEDEN dominujący tryb
oscylacji; test na prawdziwym, choć krótkim (30s) zapisie lokalnego
wstrząsu (`obspy_BW_RJOB_example.csv`, ten sam ślad co w
`test_sta_lta_i_trigger_onset_zgodne_z_obspy`) dał wynik WRAŻLIWY na próg
szumu (`noise_floor_factor`): dla głównego wstrząsu w tym śladzie (drugie
wyzwolenie STA/LTA, t≈17.9s - patrz `test_ringdown.py`) przy
noise_floor_factor>=2.0 wychodzi `is_oscillatory=False`, przy poluzowaniu
do 1.0-1.5 wychodzi `True` z okresem ~4-6s - bez niezależnego pomiaru
"prawdziwej" częstotliwości tego konkretnego wstrząsu nie da się
stwierdzić, która odpowiedź jest poprawna. To jest udokumentowane
ograniczenie, nie ukryta wada.
"""
from __future__ import annotations

import numpy as np


def ringdown_resonance(
    t,
    s,
    event_idx: int,
    baseline: float | None = None,
    pre_event_window: int = 10,
    max_lookahead: int | None = None,
    noise_floor_factor: float = 3.0,
) -> dict:
    """Analizuje powrót `s` do poziomu odniesienia PO indeksie `event_idx`.
    Patrz docstring modułu i universal-state-analyzer/timdr_core/ringdown.py
    po pełne uzasadnienie parametrów i metody.

    Zwraca dict: baseline, noise_floor, is_oscillatory, n_crossings,
    n_peaks_used, period_s, frequency_hz, log_decrement, damping_ratio,
    peak_times, peak_amplitudes.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(s)
    if n == 0 or not (0 <= event_idx < n):
        raise ValueError(f"event_idx={event_idx} poza zakresem serii o długości {n}")

    pre_start = max(0, event_idx - pre_event_window)
    pre_samples = s[pre_start:event_idx]

    if baseline is None:
        baseline = float(np.mean(pre_samples)) if len(pre_samples) else float(s[event_idx])

    noise_std = float(np.std(pre_samples)) if len(pre_samples) >= 2 else 0.0
    noise_floor = noise_floor_factor * noise_std

    end = n if max_lookahead is None else min(n, event_idx + max_lookahead)
    t_post = t[event_idx:end]
    d = s[event_idx:end] - baseline

    result: dict = {
        "baseline": float(baseline),
        "noise_floor": float(noise_floor),
        "is_oscillatory": False,
        "n_crossings": 0,
        "n_peaks_used": 0,
        "period_s": None,
        "frequency_hz": None,
        "log_decrement": None,
        "damping_ratio": None,
        "peak_times": [],
        "peak_amplitudes": [],
    }

    if len(d) < 3:
        return result

    band = noise_floor
    confirmed_idx: list[int] = []
    state = 0
    for i in range(len(d)):
        if d[i] > band:
            new_state = 1
        elif d[i] < -band:
            new_state = -1
        else:
            continue
        if new_state != state:
            confirmed_idx.append(i)
            state = new_state

    crossing_times: list[float] = []
    for prev_i, cur_i in zip(confirmed_idx[:-1], confirmed_idx[1:]):
        found = None
        for k in range(prev_i, cur_i):
            if d[k] == 0 or (d[k] > 0) != (d[k + 1] > 0):
                frac = 0.0 if d[k] == 0 else -d[k] / (d[k + 1] - d[k])
                found = float(t_post[k] + frac * (t_post[k + 1] - t_post[k]))
                break
        if found is None:
            found = float((t_post[prev_i] + t_post[cur_i]) / 2.0)
        crossing_times.append(found)

    bounds_idx = sorted(set([0] + confirmed_idx + [len(d) - 1]))
    peak_times: list[float] = []
    peak_amps: list[float] = []
    for a, b in zip(bounds_idx[:-1], bounds_idx[1:]):
        if b < a:
            continue
        seg = d[a:b + 1]
        local_idx = int(np.argmax(np.abs(seg)))
        peak_times.append(float(t_post[a + local_idx]))
        peak_amps.append(float(seg[local_idx]))

    used_crossings = crossing_times

    result["n_crossings"] = len(used_crossings)
    result["n_peaks_used"] = len(peak_amps)
    result["peak_times"] = peak_times
    result["peak_amplitudes"] = peak_amps

    if len(used_crossings) >= 2 and len(peak_amps) >= 2:
        result["is_oscillatory"] = True

        crossing_diffs = np.diff(used_crossings)
        if len(crossing_diffs) and np.median(crossing_diffs) > 0:
            period = 2.0 * float(np.median(crossing_diffs))
            result["period_s"] = period
            result["frequency_hz"] = 1.0 / period

        log_ratios = []
        for i in range(len(peak_amps) - 2):
            a, b = peak_amps[i], peak_amps[i + 2]
            if np.sign(a) == np.sign(b) and a != 0 and b != 0:
                ratio = abs(a) / abs(b)
                if ratio > 0:
                    log_ratios.append(np.log(ratio))
        if log_ratios:
            delta = float(np.mean(log_ratios))
            result["log_decrement"] = delta
            result["damping_ratio"] = float(delta / np.sqrt(4 * np.pi ** 2 + delta ** 2))

    return result
