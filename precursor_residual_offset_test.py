"""
precursor_residual_offset_test.py — czy residual_offset() (residual_offset.py)
ma jakąkolwiek moc PREDYKCYJNĄ dla trzęsień ziemi?
================================================================================
KONTEKST: to jest DRUGI test tego samego rodzaju co
`precursor_ringdown_test.py`, na tym samym realnym katalogu/stacjach, ale z
inną cechą. `precursor_ringdown_test.py` odrzucił `frac_oscillatory`
(ułamek mikro-zdarzeń, których powrót do bazy jest OSCYLACYJNY) jako
prekursor (p=0.997). Ta sesja wskazała, że `frac_oscillatory` mierzy
przeciwieństwo tego, co konceptualnie miałoby być "informacją" w tym
ekosystemie (ta część delty, która NIE wraca do zera - patrz
residual_offset.py's docstring) - więc odrzucenie oscylacyjnego ringdown
NIE JEST automatycznie odrzuceniem hipotezy "mikro-wstrząsy przed dużym
wstrząsem zostawiają trwalszy ślad niż w tle". To osobna hipoteza,
wymagająca OSOBNEGO testu na tych samych realnych danych - to jest ten
test.

Operacjonalizacja (analogiczna do precursor_ringdown_test.py, patrz tamten
plik dla uzasadnienia współdzielonych parametrów - WINDOW_HOURS,
RELIABLE_STATIONS, fronts() progi, itd. są tu CELOWO IDENTYCZNE, importowane
wprost stamtąd, żeby porównanie było uczciwe: różni się TYLKO cecha liczona
na tych samych kandydatach z fronts(), nie zestaw danych ani sposób ich
zbierania):

  1-2. Identycznie jak w precursor_ringdown_test.py: okno WINDOW_HOURS
     kończące się w momencie wstrząsu, kandydaci z fronts().
  3. Dla każdego kandydata policz `residual_offset()` (TA sama
     pre_event_window/max_lookahead co ringdown - patrz
     RINGDOWN_PRE_EVENT_WINDOW/RINGDOWN_MAX_LOOKAHEAD, nazwy zaimportowane
     wprost, nie skopiowane, żeby nie rozjechały się przypadkiem).
  4. Cecha okna: `frac_persistent` = ułamek kandydatów z
     `is_persistent=True` (0.0 jeśli brak kandydatów).
  5-6. Jak w precursor_ringdown_test.py: N_BACKGROUND okien tła, Mann-Whitney U.

DODATKOWA KONTROLA DYSKRYMINACYJNA (tego nie ma w precursor_ringdown_test.py,
bo tam nie było potrzeby - jest tu, żeby UCZCIWIE sprawdzić, że ta cecha
naprawdę mierzy coś INNEGO niż frac_oscillatory, a nie jest tym samym
klasyfikatorem pod inną nazwą): wstrzyknięcie DOKŁADNIE TEJ SAMEJ tłumionej
oscylacji, której używa precursor_ringdown_test.py jako swojej kontroli
pozytywnej (oscylacja WRACA do zera), NIE POWINNO podnosić frac_persistent -
bo z definicji ten sygnał w końcu wraca w okolice bazy. Jeśli ta kontrola
nie przejdzie, residual_offset() nie jest konceptualnie odrębny od
ringdown_resonance() i cały ten test byłby bez sensu.

UCZCIWOŚĆ: dokładnie ta sama dyscyplina co precursor_ringdown_test.py - jeśli
wynik na realnych danych wyjdzie negatywny, to jest PRAWIDŁOWY wynik,
raportowany wprost, nie ukrywany ani nie retuszowany.

WYMAGANIA: identyczne jak precursor_ringdown_test.py (obspy, requests, scipy,
prawdziwy dostęp do earthquake.usgs.gov i EARTHSCOPE - NIEOSIĄGALNE z tego
sandboxa, sprawdzone bezpośrednio, patrz tamten plik). `--mode synthetic`
(domyślny) nie wymaga sieci i BYŁ uruchomiony w tej sesji.

URUCHOMIENIE:
    python precursor_residual_offset_test.py                    # sanity-check syntetyczny
    python precursor_residual_offset_test.py --mode real         # pelny test na realnych danych

Wynik: precursor_residual_offset_test_output.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from residual_offset import residual_offset
from timdr_core_earthquake import TIMDR_EarthquakeCore

# Reużyte WPROST (nie skopiowane) z precursor_ringdown_test.py - ten sam
# katalog/stacje/progi fronts()/okna czasowe, żeby porównanie dwóch cech
# było uczciwe (różni się TYLKO cecha, nie dane ani ich zbieranie).
from precursor_ringdown_test import (
    WINDOW_HOURS,
    LEAD_HOURS,
    EXCLUSION_DAYS,
    FRONTS_TWIST_THRESHOLD,
    FRONTS_ANOMALY_FACTOR,
    RINGDOWN_PRE_EVENT_WINDOW,
    RINGDOWN_MAX_LOOKAHEAD,
    RELIABLE_STATIONS,
    MIN_MAGNITUDE_DEFAULT,
    nearest_station,
    fetch_usgs_catalog,
    fetch_window,
    has_nearby_significant_event,
    _synthetic_window,
)


def _synthetic_window_with_persistent_injection(n=600, fs=1.0, seed=0):
    """Jak _synthetic_window(colored=True) z precursor_ringdown_test.py
    (tlo AR(1) + kilka niet lumionych mikro-skokow, zeby fronts() mial co
    znalezc), ale z DODATKOWYM trwalym przesunieciem od pewnego momentu w
    oknie - dla end-to-end diagnostyki residual_offset() przez PELNY
    pipeline fronts()+residual_offset(), analogicznie do
    end_to_end_fronts_pipeline_diagnostic w precursor_ringdown_test.py,
    ktore tam uzywalo inject_oscillation=True. _synthetic_window() sam nie
    ma opcji wstrzykniecia trwalego skoku (tylko oscylacyjny), wiec ta
    funkcja dodaje go osobno, PO wygenerowaniu identycznego tla."""
    t, s = _synthetic_window(n=n, fs=fs, seed=seed, inject_oscillation=False, colored=True)
    rng = np.random.default_rng(seed + 999_999)
    jump_at = int(rng.integers(RINGDOWN_PRE_EVENT_WINDOW + 5, n - 150))
    s = s.copy()
    s[jump_at:] += rng.choice([-1, 1]) * rng.uniform(5.0, 9.0)
    return t, s

RESIDUAL_TAIL_FRACTION = 0.2
RESIDUAL_PERSISTENCE_THRESHOLD = 1.0


# ---------------------------------------------------------------------
# Cecha okna: frac_persistent z residual_offset() na kandydatach z fronts()
# ---------------------------------------------------------------------

def residual_window_feature(t: np.ndarray, s: np.ndarray, core: TIMDR_EarthquakeCore) -> dict:
    """Lustrzane odbicie ringdown_window_feature() z precursor_ringdown_test.py,
    ta sama procedura znajdowania kandydatów (fronts()), inna cecha na
    każdym z nich (residual_offset zamiast ringdown_resonance)."""
    if len(t) < RINGDOWN_PRE_EVENT_WINDOW + 10:
        return {"frac_persistent": 0.0, "n_candidates": 0}

    fronts_idx, _, _ = core.fronts(
        t, s, twist_threshold=FRONTS_TWIST_THRESHOLD, anomaly_factor=FRONTS_ANOMALY_FACTOR,
    )
    usable = [int(i) for i in fronts_idx if i >= RINGDOWN_PRE_EVENT_WINDOW]
    if not usable:
        return {"frac_persistent": 0.0, "n_candidates": 0}

    n_persist = 0
    for idx in usable:
        res = residual_offset(
            t, s, idx,
            pre_event_window=RINGDOWN_PRE_EVENT_WINDOW,
            max_lookahead=RINGDOWN_MAX_LOOKAHEAD,
            tail_fraction=RESIDUAL_TAIL_FRACTION,
            persistence_threshold=RESIDUAL_PERSISTENCE_THRESHOLD,
        )
        if res["is_persistent"]:
            n_persist += 1
    return {"frac_persistent": n_persist / len(usable), "n_candidates": len(usable)}


# ---------------------------------------------------------------------
# TRYB SYNTETYCZNY
# ---------------------------------------------------------------------

def _direct_persistence_probe(seed: int, alpha: float = 0.9, n_probe: int = 15,
                                inject_step: bool = False) -> float:
    """Analogiczne do precursor_ringdown_test.py's _direct_classifier_probe,
    ale wstrzykuje TRWAŁY (nie-zanikający) skok zamiast tłumionej oscylacji -
    dokladnie to, co residual_offset() powinien wykrywać."""
    rng = np.random.default_rng(seed)
    n = 600
    t = np.arange(n, dtype=float)
    white = rng.normal(0, 1.0, n)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = alpha * s[i - 1] + white[i]

    probe_rng = np.random.default_rng(seed + 500_000)
    idxs = probe_rng.integers(RINGDOWN_PRE_EVENT_WINDOW + 10, n - 130, size=n_probe)
    n_persist = 0
    for idx in idxs:
        idx = int(idx)
        if inject_step:
            # Trwały skok: NIE zanika (w przeciwienstwie do tlumionej
            # oscylacji ringdown-testu) - to jest dokladnie "nieodwracalny
            # odcisk", ktory residual_offset() ma wykryc.
            s = s.copy()
            s[idx:] += rng.choice([-1, 1]) * rng.uniform(5.0, 9.0)
        res = residual_offset(t, s, idx, pre_event_window=RINGDOWN_PRE_EVENT_WINDOW,
                               max_lookahead=RINGDOWN_MAX_LOOKAHEAD,
                               tail_fraction=RESIDUAL_TAIL_FRACTION,
                               persistence_threshold=RESIDUAL_PERSISTENCE_THRESHOLD)
        if res["is_persistent"]:
            n_persist += 1
    return n_persist / n_probe


def _discrimination_probe_decaying_oscillation(seed: int, alpha: float = 0.9, n_probe: int = 15) -> float:
    """KONTROLA DYSKRYMINACYJNA (patrz docstring modulu): wstrzykuje
    DOKLADNIE ta sama tlumiona oscylacje co
    precursor_ringdown_test.py::_direct_classifier_probe(inject_at_probe=True)
    - sygnal ktory WRACA do zera - i sprawdza, ze residual_offset() TEGO
    NIE lapie jako "persistent" (bo z definicji nie jest trwaly). Jesli ta
    probka wyjdzie wysoka, residual_offset() nie mierzy niczego nowego."""
    rng = np.random.default_rng(seed)
    n = 600
    t = np.arange(n, dtype=float)
    white = rng.normal(0, 1.0, n)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = alpha * s[i - 1] + white[i]

    probe_rng = np.random.default_rng(seed + 500_000)
    idxs = probe_rng.integers(RINGDOWN_PRE_EVENT_WINDOW + 10, n - 130, size=n_probe)
    n_persist = 0
    for idx in idxs:
        idx = int(idx)
        post = np.arange(min(150, n - idx), dtype=float)
        f0 = rng.uniform(0.05, 0.2)
        tau = rng.uniform(15, 40)
        s = s.copy()
        s[idx:idx + len(post)] += 6.0 * np.exp(-post / tau) * np.cos(2 * np.pi * f0 * post)
        res = residual_offset(t, s, idx, pre_event_window=RINGDOWN_PRE_EVENT_WINDOW,
                               max_lookahead=RINGDOWN_MAX_LOOKAHEAD,
                               tail_fraction=RESIDUAL_TAIL_FRACTION,
                               persistence_threshold=RESIDUAL_PERSISTENCE_THRESHOLD)
        if res["is_persistent"]:
            n_persist += 1
    return n_persist / n_probe


def run_synthetic_selftest(n_windows=30) -> dict:
    from scipy.stats import mannwhitneyu

    core = TIMDR_EarthquakeCore()

    # --- TEST GLOWNY: wstrzykniety TRWALY skok vs czyste tlo AR(1) - MUSI
    # dac istotna roznice, inaczej klasyfikator is_persistent nie dziala.
    direct_with_signal = [_direct_persistence_probe(i, inject_step=True) for i in range(n_windows)]
    direct_background_a = [_direct_persistence_probe(1000 + i, inject_step=False) for i in range(n_windows)]
    stat_pos, p_pos = mannwhitneyu(direct_with_signal, direct_background_a, alternative="two-sided")

    # --- KONTROLA NEGATYWNA GLOWNA: tlo-vs-tlo, NIE MOZE dac falszywie
    # istotnego wyniku.
    direct_background_b = [_direct_persistence_probe(2000 + i, inject_step=False) for i in range(n_windows)]
    stat_neg, p_neg = mannwhitneyu(direct_background_a, direct_background_b, alternative="two-sided")

    ok_positive_control = p_pos < 0.05
    ok_negative_control = p_neg >= 0.05

    # --- KONTROLA DYSKRYMINACYJNA (patrz docstring modulu i funkcji powyzej):
    # tlumiona oscylacja (ktora WRACA do zera) NIE powinna byc czesto
    # klasyfikowana jako "persistent" - to udowadnia, ze ta cecha mierzy
    # cos INNEGO niz frac_oscillatory z precursor_ringdown_test.py.
    decaying_oscillation_probe = [_discrimination_probe_decaying_oscillation(3000 + i) for i in range(n_windows)]
    mean_decaying_oscillation_persistence = float(np.mean(decaying_oscillation_probe))
    # Prog a priori: srednio mniej niz 30% falszywych "persistent" na
    # sygnale ktory z definicji NIE jest trwaly - hojny margines (nie
    # oczekujemy dokladnie 0%, bo ogon tlumionej oscylacji tuz po skoku
    # moze chwilowo przekroczyc noise_floor zanim zdazy zanikn ac, ale
    # WIEKSZOSC probek powinna wrocic do bazy w ustalonym oknie lookahead).
    discrimination_ok = mean_decaying_oscillation_persistence < 0.3

    # --- DIAGNOSTYKA DODATKOWA (informacyjna, NIE bramkuje pipeline_verified):
    # PELNY pipeline fronts()+residual_offset() na tle AR(1) Z i BEZ
    # wstrzykniecia trwalego skoku - odpowiednik
    # end_to_end_fronts_pipeline_diagnostic w precursor_ringdown_test.py.
    with_signal_pipeline = [residual_window_feature(*_synthetic_window_with_persistent_injection(seed=i), core)["frac_persistent"]
                            for i in range(n_windows)]
    background_pipeline = [residual_window_feature(*_synthetic_window(seed=1000 + i, inject_oscillation=False, colored=True), core)["frac_persistent"]
                           for i in range(n_windows)]
    n_nonzero_pipeline = int(np.sum(np.array(with_signal_pipeline + background_pipeline) > 0))

    return {
        "positive_control_p_value": float(p_pos),
        "positive_control_detected_injected_signal": bool(ok_positive_control),
        "negative_control_p_value": float(p_neg),
        "negative_control_no_false_alarm": bool(ok_negative_control),
        "mean_frac_persistent_direct_with_signal": float(np.mean(direct_with_signal)),
        "mean_frac_persistent_direct_background": float(np.mean(direct_background_a)),
        "discrimination_vs_decaying_oscillation_mean_persistence": mean_decaying_oscillation_persistence,
        "discrimination_ok": bool(discrimination_ok),
        "pipeline_verified": bool(ok_positive_control and ok_negative_control and discrimination_ok),
        "end_to_end_fronts_pipeline_diagnostic": {
            "note": "Diagnostyka pomocnicza (nie bramkuje pipeline_verified): pelny fronts()+residual_offset na tle AR(1) z i bez wstrzknietego trwalego skoku. fronts() jest konserwatywny na tym tle (mala liczba niezerowych probek w obu grupach, patrz precursor_ringdown_test.py), wiec ten test ma mniejsza moc niz bezposrednia kalibracja klasyfikatora powyzej.",
            "n_nonzero_samples_of_60": n_nonzero_pipeline,
            "mean_frac_persistent_with_signal": float(np.mean(with_signal_pipeline)),
            "mean_frac_persistent_background": float(np.mean(background_pipeline)),
        },
    }


# ---------------------------------------------------------------------
# TRYB REALNY - identyczny szkielet co precursor_ringdown_test.py, inna cecha
# ---------------------------------------------------------------------

def run_real_test(min_magnitude: float, years: int, n_background: int, max_pre_events: int,
                   waveform_timeout: float = 20.0, bg_time_budget_s: float = 240.0) -> dict:
    import random
    import time
    from scipy.stats import mannwhitneyu
    from obspy.clients.fdsn import Client

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years)

    print(f"[1/4] Pobieram realny katalog USGS (M>={min_magnitude}, {years} lat)...")
    events = fetch_usgs_catalog(min_magnitude, start, end)
    print(f"      -> {len(events)} zdarzen")
    if not events:
        raise RuntimeError("Katalog USGS pusty dla podanych parametrow - nie ma czego testowac.")

    if max_pre_events and len(events) > max_pre_events:
        sample_rng = random.Random(7)
        events = sample_rng.sample(events, max_pre_events)
        print(f"      (probka {max_pre_events} z {len(events)}+ zdarzen, seed=7)")

    client = Client("EARTHSCOPE", timeout=waveform_timeout)
    core = TIMDR_EarthquakeCore()

    print(f"[2/4] Licze cechy PRE-EVENT dla {len(events)} realnych wstrzasow...")
    pre_features, pre_meta = [], []
    t_start = time.time()
    for i, ev in enumerate(events):
        station = nearest_station(ev["lat"], ev["lon"])
        window_end = ev["time"] - timedelta(hours=LEAD_HOURS)
        elapsed = time.time() - t_start
        print(f"      [{i+1}/{len(events)}] {ev['id']} ({station['sta']}, M{ev['mag']}) - {elapsed:.0f}s uplynelo...", end=" ", flush=True)
        try:
            t, s = fetch_window(client, station, window_end, WINDOW_HOURS, 0.0)
        except Exception as e:
            print(f"pominieto ({type(e).__name__}: {e})")
            continue
        feat = residual_window_feature(t, s, core)
        pre_features.append(feat["frac_persistent"])
        pre_meta.append({"event_id": ev["id"], "station": station["sta"], **feat})
        print(f"OK (frac_persistent={feat['frac_persistent']:.3f}, n_candidates={feat['n_candidates']})")

    print(f"[3/4] Licze cechy TLA dla max {n_background} losowych okien...")
    rng = random.Random(42)
    bg_features, bg_meta, attempts = [], [], 0
    t_start = time.time()
    while (len(bg_features) < n_background
           and attempts < n_background * 20
           and (time.time() - t_start) < bg_time_budget_s):
        attempts += 1
        station = rng.choice(RELIABLE_STATIONS)
        candidate = start + timedelta(seconds=rng.uniform(0, (end - start).total_seconds()))
        elapsed = time.time() - t_start
        print(f"      [{len(bg_features)+1}/{n_background}, proba {attempts}, {elapsed:.0f}/{bg_time_budget_s:.0f}s] "
              f"{candidate.date()} ({station['sta']})...", end=" ", flush=True)
        try:
            if has_nearby_significant_event(candidate, EXCLUSION_DAYS, station):
                print("pominieto (blisko M>=4.5 w promieniu stacji)")
                continue
        except Exception as e:
            print(f"pominieto sprawdzenie wykluczenia ({type(e).__name__}: {e})")
            continue
        try:
            t, s = fetch_window(client, station, candidate, WINDOW_HOURS, 0.0)
        except Exception as e:
            print(f"pominieto ({type(e).__name__}: {e})")
            continue
        feat = residual_window_feature(t, s, core)
        bg_features.append(feat["frac_persistent"])
        bg_meta.append({"station": station["sta"], "time": candidate.isoformat(), **feat})
        print(f"OK (frac_persistent={feat['frac_persistent']:.3f}, n_candidates={feat['n_candidates']})")

    time_budget_hit = (time.time() - t_start) >= bg_time_budget_s and len(bg_features) < n_background

    print(f"      -> {len(pre_features)} okien pre-event, {len(bg_features)} okien tla")
    if len(pre_features) < 5 or len(bg_features) < 5:
        raise RuntimeError("Za malo udanych okien do sensownego testu statystycznego (potrzeba >=5 w kazdej grupie).")

    print("[4/4] Test Manna-Whitneya U (pre-event vs tlo)...")
    stat, p_value = mannwhitneyu(pre_features, bg_features, alternative="two-sided")

    return {
        "n_pre_event_windows": len(pre_features),
        "n_background_windows": len(bg_features),
        "mean_frac_persistent_pre_event": float(np.mean(pre_features)),
        "mean_frac_persistent_background": float(np.mean(bg_features)),
        "mannwhitney_p_value": float(p_value),
        "significant_at_0_05": bool(p_value < 0.05),
        "pre_event_higher_than_background": bool(np.mean(pre_features) > np.mean(bg_features)),
        "background_time_budget_hit": bool(time_budget_hit),
        "pre_event_details": pre_meta,
        "background_details": bg_meta,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--min-magnitude", type=float, default=MIN_MAGNITUDE_DEFAULT)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--n-background", type=int, default=60)
    ap.add_argument("--max-pre-events", type=int, default=40)
    ap.add_argument("--waveform-timeout", type=float, default=20.0)
    ap.add_argument("--background-time-budget-s", type=float, default=240.0)
    ap.add_argument("--out", default="precursor_residual_offset_test_output.json")
    args = ap.parse_args()

    print("=" * 70)
    print("KROK 1: sanity-check na danych SYNTETYCZNYCH (zawsze, bez sieci)")
    print("=" * 70)
    synth = run_synthetic_selftest()
    for k, v in synth.items():
        print(f"  {k}: {v}")

    result = {"synthetic_selftest": synth}

    if not synth["pipeline_verified"]:
        print("\n!!! Sanity-check NIE PRZESZEDL.")
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        sys.exit(1)

    print("\nSanity-check PRZESZEDL: klasyfikator wykrywa trwaly skok, nie daje")
    print("falszywych alarmow tlo-vs-tlo, i (kluczowe) NIE reaguje na tlumiona")
    print("oscylacje, ktora WRACA do zera - a wiec mierzy cos innego niz")
    print("frac_oscillatory z precursor_ringdown_test.py.")

    if args.mode == "real":
        print("\n" + "=" * 70)
        print("KROK 2: test na REALNYCH danych (USGS + EarthScope/IRIS)")
        print("=" * 70)
        try:
            real = run_real_test(args.min_magnitude, args.years, args.n_background,
                                  args.max_pre_events, args.waveform_timeout,
                                  args.background_time_budget_s)
        except Exception as e:
            print(f"\nBLAD podczas testu na realnych danych: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2, default=str)
            sys.exit(2)

        result["real_test"] = real
        print()
        print("-" * 70)
        print("WYNIK (realne dane):")
        print(f"  pre-event frac_persistent (srednia): {real['mean_frac_persistent_pre_event']:.4f}")
        print(f"  tlo frac_persistent (srednia):        {real['mean_frac_persistent_background']:.4f}")
        print(f"  p-value (Mann-Whitney U):              {real['mannwhitney_p_value']:.4f}")
        if real["significant_at_0_05"] and real["pre_event_higher_than_background"]:
            print("  => Statystycznie istotna ROZNICA, pre-event WYZSZE niz tlo.")
            print("     To jest WSTEPNA przeslanka, NIE dowod predykcyjnosci - wymaga")
            print("     replikacji na niezaleznym zbiorze zdarzen.")
        elif real["significant_at_0_05"]:
            print("  => Statystycznie istotna roznica, ale TLO wyzsze niz pre-event -")
            print("     to NIE wspiera hipotezy predykcyjnej (kierunek odwrotny).")
        else:
            print("  => WYNIK NEGATYWNY: brak statystycznie istotnej roznicy.")
        print("-" * 70)
    else:
        print("\n(Tryb 'synthetic' - zeby uruchomic pelny test na realnych danych:")
        print(" python precursor_residual_offset_test.py --mode real)")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nPelny wynik zapisany do: {args.out}")


if __name__ == "__main__":
    main()
