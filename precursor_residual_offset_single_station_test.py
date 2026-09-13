"""
precursor_residual_offset_single_station_test.py — czy frac_persistent
(residual_offset.py) replikuje się, gdy zawęzić analizę do JEDNEJ stacji
(CTAO), zamiast puli 8 stacji GSN z precursor_residual_offset_test.py?
================================================================================
DLACZEGO TEN PLIK ISTNIEJE I DLACZEGO STACJA=CTAO NIE JEST WYBOREM "NA ŚLEPO"
(przeczytaj to PRZED uruchomieniem --mode real - to jest kluczowe dla
uczciwej interpretacji wyniku):

Poprzedni test (`precursor_residual_offset_test.py --mode real`, wynik w
`precursor_residual_offset_test_output.json`, opisany w
`HISTORIA_I_TESTY.md`) dał p=0.0251, r≈0.11 ("mały") na PULI 8 stacji GSN.
Rozbicie tego wyniku PO STACJACH (policzone z już zapisanych danych, bez
nowego pobierania - patrz git log tej sesji) pokazało:

  - CTAO niesie POŁOWĘ wszystkich okien pre-event (20 z 40) - bo to
    najbliższa stacja referencyjna dla ogromnej liczby wstrząsów M>=6.5 w
    zachodnim Pacyfiku/Indonezji (region wyjątkowo aktywny sejsmicznie),
    NIE dlatego, że ktoś ją wybrał.
  - CTAO niesie 4 z 5 WSZYSTKICH niezerowych okien `frac_persistent` w
    całej próbie pre-event.
  - Reanaliza WYŁĄCZNIE na CTAO (n=20 pre-event vs n=10 tło, dwustronny
    Mann-Whitney, ta sama metodologia): p=0.1466, r=0.20 ("mały") -
    NIEISTOTNE przy progu 0.05, ale w TYM SAMYM kierunku (średnia
    pre-event 0.0162 vs tło 0.0000 - tło na samym CTAO było idealnie
    płaskie). Prawdopodobny powód nieistotności: mała próba tła (tylko
    10 okien CTAO w oryginalnym teście, bo tło było losowane równo po
    8 stacjach), nie sprzeczny kierunek efektu.

WNIOSEK UCZCIWOŚCI: to NIE jest "wybierzmy stację, która zadziałała" w
sensie data-snoopingu na NOWYCH danych - CTAO jest testowana PONOWNIE tu,
tymi samymi już-pobranymi liczbami powyżej jako uzasadnieniem, a właściwy
test w tym pliku pobiera NOWE, w większości NIEZALEŻNE okna (data inne niż
poprzedni seed=7 przebieg, patrz niżej). Ale trzeba to nazwać wprost: wybór
stacji BYŁ poinformowany wynikiem poprzedniego przebiegu, więc "pozytywny"
wynik tutaj NIE jest niezależnym, ślepym potwierdzeniem - to ukierunkowana
replikacja/pogłębienie NAJSILNIEJSZEGO tropu, nie nowe, niezależne odkrycie.
Prawdziwe ślepe potwierdzenie wymagałoby powtórzenia tego protokołu na
INNEJ, z góry wybranej (nie post-hoc) pojedynczej stacji.

CO JEST TU FAKTYCZNIE INNE NIŻ ZWYKŁE PRZEFILTROWANIE STAREGO WYNIKU:
  1. `nearest_station()` (identyczna, geograficzna logika z
     precursor_ringdown_test.py, NIEZMIENIONA) nadal decyduje, które
     realne wstrząsy "należą" do CTAO - zdarzenia, dla których najbliższa
     z 8 stacji referencyjnych NIE jest CTAO, są pomijane. To zachowuje
     fizyczny sens (analizujemy mikrosejsmiczność W POBLIŻU danej stacji,
     nie przypadkowy szum stacji odległej od wstrząsu).
  2. `max_pre_events` podniesione z 40 do 150 (patrz DEFAULT_MAX_PRE_EVENTS
     niżej) - skoro tylko ~50% globalnej próby przechodzi filtr CTAO
     (obserwacja z poprzedniego przebiegu, NIE gwarancja), podniesienie
     puli wejściowej ma dać WIĘCEJ niż poprzednie 20 okien pre-event, nie
     tyle samo przefiltrowane.
  3. Tło NIE jest już losowane równo po 8 stacjach (`rng.choice(RELIABLE_STATIONS)`)
     - tu ZAWSZE jest to CTAO, więc przy tym samym budżecie czasu powinno
       dać WIĘCEJ niż poprzednie 10 okien tła (główne źródło niskiej mocy
       testu w reanalizie powyżej).
  4. Inny plik wyjściowy (`precursor_residual_offset_single_station_test_output.json`)
     - NIE nadpisuje oryginalnego wyniku puli 8 stacji.

WSZYSTKO INNE (WINDOW_HOURS, LEAD_HOURS, EXCLUSION_DAYS, progi fronts(),
definicja residual_offset(), test Manna-Whitneya dwustronny + rozmiar
efektu rank-biserial) jest IMPORTOWANE WPROST z
precursor_residual_offset_test.py / precursor_ringdown_test.py, celowo
NIEZMIENIONE - różni się TYLKO zakres stacji, żeby porównanie było uczciwe.

URUCHOMIENIE:
    python precursor_residual_offset_single_station_test.py                  # sanity-check syntetyczny (bez sieci)
    python precursor_residual_offset_single_station_test.py --mode real       # pelny test, tylko CTAO

Wymaga: obspy, requests, scipy, prawdziwy dostep do earthquake.usgs.gov i
EarthScope - NIEOSIAGALNE z tego sandboxa (sprawdzone bezposrednio w
poprzednich przebiegach tej sesji).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from timdr_core_earthquake import TIMDR_EarthquakeCore
from precursor_residual_offset_test import residual_window_feature, run_synthetic_selftest
from precursor_ringdown_test import (
    WINDOW_HOURS,
    LEAD_HOURS,
    EXCLUSION_DAYS,
    RELIABLE_STATIONS,
    MIN_MAGNITUDE_DEFAULT,
    nearest_station,
    fetch_usgs_catalog,
    fetch_window,
    has_nearby_significant_event,
)

# ---------------------------------------------------------------------
# PARAMETRY ZAMROŻONE PRZED URUCHOMIENIEM NA REALNYCH DANYCH
# ---------------------------------------------------------------------
TARGET_STATION_ID = "CTAO"
TARGET_STATION = next(st for st in RELIABLE_STATIONS if st["sta"] == TARGET_STATION_ID)
DEFAULT_MAX_PRE_EVENTS = 150   # patrz punkt 2 w docstringu modulu
DEFAULT_N_BACKGROUND = 40      # patrz punkt 3 w docstringu modulu
DEFAULT_BG_TIME_BUDGET_S = 400.0  # wyzsze niz oryginalne 240s - jedna stacja
                                    # zamiast 8 moze miec inny profil sukcesu/failure


def run_real_test_single_station(
    min_magnitude: float,
    years: int,
    n_background: int,
    max_pre_events: int,
    waveform_timeout: float = 20.0,
    bg_time_budget_s: float = DEFAULT_BG_TIME_BUDGET_S,
) -> dict:
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

    print(f"[2/4] Filtruje do zdarzen, dla ktorych najblizsza stacja = {TARGET_STATION_ID}, "
          f"licze cechy PRE-EVENT...")
    pre_features, pre_meta = [], []
    n_skipped_other_station = 0
    t_start = time.time()
    for i, ev in enumerate(events):
        station = nearest_station(ev["lat"], ev["lon"])
        if station["sta"] != TARGET_STATION_ID:
            n_skipped_other_station += 1
            continue
        window_end = ev["time"] - timedelta(hours=LEAD_HOURS)
        elapsed = time.time() - t_start
        print(f"      [{i+1}/{len(events)}] {ev['id']} (M{ev['mag']}) - {elapsed:.0f}s uplynelo...", end=" ", flush=True)
        try:
            t, s = fetch_window(client, station, window_end, WINDOW_HOURS, 0.0)
        except Exception as e:
            print(f"pominieto ({type(e).__name__}: {e})")
            continue
        feat = residual_window_feature(t, s, core)
        pre_features.append(feat["frac_persistent"])
        pre_meta.append({"event_id": ev["id"], "station": station["sta"], **feat})
        print(f"OK (frac_persistent={feat['frac_persistent']:.3f}, n_candidates={feat['n_candidates']})")

    print(f"      -> {len(pre_features)} okien pre-event na {TARGET_STATION_ID} "
          f"({n_skipped_other_station} zdarzen pominietych - najblizsza stacja byla inna)")

    print(f"[3/4] Licze cechy TLA dla max {n_background} losowych okien na {TARGET_STATION_ID} "
          f"(budzet czasu: {bg_time_budget_s:.0f}s)...")
    rng = random.Random(42)
    bg_features, bg_meta, attempts = [], [], 0
    t_start = time.time()
    while (len(bg_features) < n_background
           and attempts < n_background * 20
           and (time.time() - t_start) < bg_time_budget_s):
        attempts += 1
        station = TARGET_STATION  # ZAWSZE ta sama stacja - to jest sedno tego skryptu
        candidate = start + timedelta(seconds=rng.uniform(0, (end - start).total_seconds()))
        elapsed = time.time() - t_start
        print(f"      [{len(bg_features)+1}/{n_background}, proba {attempts}, {elapsed:.0f}/{bg_time_budget_s:.0f}s] "
              f"{candidate.date()}...", end=" ", flush=True)
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

    print(f"      -> {len(pre_features)} okien pre-event, {len(bg_features)} okien tla (oba na {TARGET_STATION_ID})")
    if len(pre_features) < 5 or len(bg_features) < 5:
        raise RuntimeError("Za malo udanych okien do sensownego testu statystycznego (potrzeba >=5 w kazdej grupie).")

    print("[4/4] Test Manna-Whitneya U (pre-event vs tlo, tylko CTAO)...")
    stat, p_value = mannwhitneyu(pre_features, bg_features, alternative="two-sided")

    n1, n2 = len(pre_features), len(bg_features)
    effect_size = 2.0 * float(stat) / (n1 * n2) - 1.0
    abs_r = abs(effect_size)
    if abs_r < 0.1:
        effect_size_label = "pomijalny"
    elif abs_r < 0.3:
        effect_size_label = "maly"
    elif abs_r < 0.5:
        effect_size_label = "sredni"
    else:
        effect_size_label = "duzy"

    return {
        "target_station": TARGET_STATION_ID,
        "n_events_skipped_other_nearest_station": n_skipped_other_station,
        "n_pre_event_windows": len(pre_features),
        "n_background_windows": len(bg_features),
        "mean_frac_persistent_pre_event": float(np.mean(pre_features)),
        "mean_frac_persistent_background": float(np.mean(bg_features)),
        "mannwhitney_p_value": float(p_value),
        "significant_at_0_05": bool(p_value < 0.05),
        "pre_event_higher_than_background": bool(np.mean(pre_features) > np.mean(bg_features)),
        "rank_biserial_effect_size": effect_size,
        "effect_size_label": effect_size_label,
        "background_time_budget_hit": bool(time_budget_hit),
        "pre_event_details": pre_meta,
        "background_details": bg_meta,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--min-magnitude", type=float, default=MIN_MAGNITUDE_DEFAULT)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--n-background", type=int, default=DEFAULT_N_BACKGROUND)
    ap.add_argument("--max-pre-events", type=int, default=DEFAULT_MAX_PRE_EVENTS)
    ap.add_argument("--waveform-timeout", type=float, default=20.0)
    ap.add_argument("--background-time-budget-s", type=float, default=DEFAULT_BG_TIME_BUDGET_S)
    ap.add_argument("--out", default="precursor_residual_offset_single_station_test_output.json")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Stacja docelowa: {TARGET_STATION_ID} - patrz docstring modulu DLACZEGO ta")
    print("stacja i dlaczego to NIE jest slepy, niezalezny test.")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("KROK 1: sanity-check na danych SYNTETYCZNYCH (identyczny co w tescie")
    print("puli 8 stacji - ta sama cecha/klasyfikator, wiec ten sam test)")
    print("=" * 70)
    synth = run_synthetic_selftest()
    for k, v in synth.items():
        print(f"  {k}: {v}")

    result = {"target_station": TARGET_STATION_ID, "synthetic_selftest": synth}

    if not synth["pipeline_verified"]:
        print("\n!!! Sanity-check NIE PRZESZEDL.")
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        sys.exit(1)

    print("\nSanity-check PRZESZEDL.")

    if args.mode == "real":
        print("\n" + "=" * 70)
        print(f"KROK 2: test na REALNYCH danych, WYLACZNIE stacja {TARGET_STATION_ID}")
        print("=" * 70)
        try:
            real = run_real_test_single_station(
                args.min_magnitude, args.years, args.n_background,
                args.max_pre_events, args.waveform_timeout,
                args.background_time_budget_s,
            )
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
        print(f"WYNIK (realne dane, tylko {TARGET_STATION_ID}):")
        print(f"  n pre-event: {real['n_pre_event_windows']}, n tlo: {real['n_background_windows']}")
        print(f"  pre-event frac_persistent (srednia): {real['mean_frac_persistent_pre_event']:.4f}")
        print(f"  tlo frac_persistent (srednia):        {real['mean_frac_persistent_background']:.4f}")
        print(f"  p-value (Mann-Whitney U, dwustronny):  {real['mannwhitney_p_value']:.4f}")
        print(f"  rozmiar efektu (rank-biserial r):       {real['rank_biserial_effect_size']:.3f} ({real['effect_size_label']})")
        print()
        print("  PAMIETAJ (patrz docstring modulu): wybor stacji CTAO byl poinformowany")
        print("  wynikiem poprzedniego testu na puli 8 stacji - to NIE jest niezalezne,")
        print("  slepe potwierdzenie, tylko ukierunkowana replikacja najsilniejszego tropu.")
        if real["significant_at_0_05"] and real["pre_event_higher_than_background"]:
            print("  => Statystycznie istotna ROZNICA, pre-event WYZSZE niz tlo - wzmacnia")
            print("     wczesniejsza (niezistotna) obserwacje na tej samej stacji, ale wciaz")
            print("     wymaga replikacji na INNEJ, z gory wybranej stacji, zanim uznamy to")
            print("     za cokolwiek wiecej niz trop.")
        elif real["significant_at_0_05"]:
            print("  => Statystycznie istotna roznica, ale TLO wyzsze niz pre-event -")
            print("     kierunek odwrotny, NIE wspiera hipotezy.")
        else:
            print("  => Nadal NIEISTOTNE. Przy tak rzadkim sygnale (patrz oryginalny test:")
            print("     5 z 40 okien mialo frac_persistent>0) nawet wieksza probka na jednej")
            print("     stacji moze nie wystarczyc - to uczciwy wynik negatywny, nie porazka")
            print("     skryptu.")
        print("-" * 70)
    else:
        print("\n(Tryb 'synthetic' - zeby uruchomic pelny test na realnych danych:")
        print(" python precursor_residual_offset_single_station_test.py --mode real)")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nPelny wynik zapisany do: {args.out}")


if __name__ == "__main__":
    main()
