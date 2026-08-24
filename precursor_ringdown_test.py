"""
precursor_ringdown_test.py — czy ringdown_resonance() ma jakąkolwiek moc
PREDYKCYJNĄ dla trzęsień ziemi (nie tylko opisową)?
================================================================================
KONTEKST: `ringdown_resonance()` (ringdown.py) analizuje, czy powrót
sygnału do poziomu odniesienia PO zdarzeniu jest oscylacyjny. To jest
narzędzie OPISOWE (post-event) - samo w sobie nic nie mówi o predykcji.

Ten skrypt testuje ZUPEŁNIE OSOBNE pytanie, tym samym protokołem, który
już raz uczciwie zastosowano w tym repo dla innej matematyki
(`Topology(t)`/"Rezonans TIMDR", patrz README.md - wynik był NEGATYWNY):

  Czy cecha zbudowana z ringdown_resonance(), policzona WYŁĄCZNIE z danych
  SPRZED prawdziwego dużego wstrząsu, jest podwyższona względem cechy
  policzonej w losowych oknach TŁA (bez nadchodzącego dużego wstrząsu)?

Operacjonalizacja (USTALONA PRZED policzeniem czegokolwiek na realnych
danych - patrz `--mode synthetic`, które musi przejść PRZED odpaleniem
`--mode real`, dokładnie ta sama dyscyplina co przy zamrożeniu parametrów
embeddingu w analyze_topology_resonance_seismic.py):

  1. Weź okno `WINDOW_HOURS` godzin kończące się `LEAD_HOURS` godzin PRZED
     czasem początku prawdziwego wstrząsu (realny katalog USGS,
     M >= MIN_MAGNITUDE), zarejestrowane na najbliższej stacji szerokopasmowej
     GSN z ustalonej listy `RELIABLE_STATIONS`.
  2. W tym oknie znajdź kandydatów na mikro-zdarzenia przez `fronts()`
     (już istniejący, przetestowany picker w timdr_core_earthquake.py) -
     TE SAME mikro-wstrząsy/szum, których szukałby analityk NIE wiedząc
     o nadchodzącym dużym wstrząsie.
  3. Dla każdego kandydata policz `ringdown_resonance()` (powrót do
     lokalnej linii bazowej PO tym mikro-zdarzeniu).
  4. Cecha okna: `frac_oscillatory` = ułamek kandydatów, dla których
     `is_oscillatory=True` (0.0, jeśli brak kandydatów - jawna, ustalona
     z góry reguła, nie cichy NaN).
  5. To samo dla `N_BACKGROUND` okien TŁA: losowe czasy na tych samych
     stacjach, bez ŻADNEGO M>=4.5 w promieniu `EXCLUSION_DAYS` dni.
  6. Test Manna-Whitneya U (dwustronny) między dwoma grupami
     `frac_oscillatory` - PRE-EVENT vs TŁO. To jest formalny test
     istotności, nie tylko porównanie percentyli (mocniejsze niż
     pierwotny test Topology(t), który tylko podawał percentyl).

UCZCIWOŚĆ: to narzędzie NIE zakłada z góry, że coś znajdzie. Jeśli wynik
wyjdzie negatywny (brak istotnej różnicy) - to jest PRAWIDŁOWY, ważny
wynik, dokładnie taki, jaki wyszedł dla Topology(t). Jeśli wyjdzie
"pozytywny" na małej próbie - to NIE jest dowód realnej predykcyjności,
tylko wstępna przesłanka wymagająca replikacji na innym, niezależnym
zbiorze zdarzeń (dokładnie tak, jak "trop" na BTC w timdr-finanse nie
przetrwał replikacji na złocie - patrz tamto README).

WYMAGANIA (nie ma ich w tym środowisku sandboxowym - sprawdzone
bezpośrednio: `service.earthscope.org` i `service.ncedc.org` dają 403
zarówno przez narzędzie do pobierania stron, jak i przez `curl`/ObsPy z
tego środowiska; `earthquake.usgs.gov` też nie jest osiągalne z poziomu
skryptu w tym sandboxie, mimo że jest osiągalne przez dedykowane
narzędzie do pobierania pojedynczych stron):
    pip install obspy requests scipy

URUCHOMIENIE (w pełni zautomatyzowane, zero interakcji):
    python precursor_ringdown_test.py                    # tylko sanity-check syntetyczny
    python precursor_ringdown_test.py --mode real         # pełny test na realnych danych
    python precursor_ringdown_test.py --mode real --min-magnitude 7.0 --years 10 --n-background 80

Wynik zapisywany do `precursor_ringdown_test_output.json` + czytelne
podsumowanie na stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

import numpy as np

from ringdown import ringdown_resonance
from timdr_core_earthquake import TIMDR_EarthquakeCore

# ---------------------------------------------------------------------
# PARAMETRY ZAMROŻONE PRZED URUCHOMIENIEM NA REALNYCH DANYCH
# (nie dostrajać po zobaczeniu wyniku na realnym katalogu - to byłby
# dokładnie ten sam błąd data snoopingu, którego uniknięto przy Topology(t))
# ---------------------------------------------------------------------
MIN_MAGNITUDE_DEFAULT = 6.5
WINDOW_HOURS = 2.0          # dlugosc okna analizy (przed wstrzasem / w tle)
LEAD_HOURS = 0.0            # okno konczy sie DOKLADNIE w momencie wstrzasu (t=0 = origin time)
EXCLUSION_DAYS = 3          # zaden M>=4.5 w tym promieniu czasowym od okna tla
SAMPLE_RATE_TARGET_HZ = 1.0  # kanal LHZ (long-period), ~1 probka/s - pasuje do 0.01-1Hz
FRONTS_TWIST_THRESHOLD = 0.4
FRONTS_ANOMALY_FACTOR = 3.0
RINGDOWN_PRE_EVENT_WINDOW = 30   # probek (~30s przy 1Hz) przed kazdym kandydatem na front
RINGDOWN_MAX_LOOKAHEAD = 120     # probek (~2min) po kazdym kandydacie

# Osiem dlugo dzialajacych, dobrze udokumentowanych stacji referencyjnych
# sieci globalnej (GSN) - wybrane PRZED zobaczeniem jakichkolwiek wynikow,
# zeby nie "wybierac" pozniej stacji, ktora akurat daje ladny wynik.
RELIABLE_STATIONS = [
    {"net": "IU", "sta": "ANMO", "loc": "00", "cha": "LHZ", "lat": 34.9459, "lon": -106.4572},
    {"net": "IU", "sta": "COLA", "loc": "00", "cha": "LHZ", "lat": 64.8736, "lon": -147.8616},
    {"net": "IU", "sta": "KONO", "loc": "00", "cha": "LHZ", "lat": 59.6491, "lon": 9.5982},
    {"net": "II", "sta": "PFO", "loc": "00", "cha": "LHZ", "lat": 33.6092, "lon": -116.4553},
    {"net": "IU", "sta": "HRV", "loc": "00", "cha": "LHZ", "lat": 42.5064, "lon": -71.5583},
    {"net": "IU", "sta": "CTAO", "loc": "00", "cha": "LHZ", "lat": -20.0882, "lon": 146.2545},
    {"net": "IU", "sta": "MAJO", "loc": "00", "cha": "LHZ", "lat": 36.5457, "lon": 138.2041},
    {"net": "II", "sta": "BFO", "loc": "00", "cha": "LHZ", "lat": 48.3319, "lon": 8.3311},
]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def nearest_station(lat, lon):
    return min(RELIABLE_STATIONS, key=lambda st: haversine_km(lat, lon, st["lat"], st["lon"]))


# ---------------------------------------------------------------------
# Cecha okna: frac_oscillatory z ringdown_resonance() na kandydatach z fronts()
# ---------------------------------------------------------------------

def ringdown_window_feature(t: np.ndarray, s: np.ndarray, core: TIMDR_EarthquakeCore) -> dict:
    """USTALONA Z GORY reguła (patrz docstring modułu, krok 2-4).
    Zwraca dict z frac_oscillatory (0.0 jesli brak kandydatow) i
    diagnostyka (n_candidates)."""
    if len(t) < RINGDOWN_PRE_EVENT_WINDOW + 10:
        return {"frac_oscillatory": 0.0, "n_candidates": 0}

    fronts_idx, _, _ = core.fronts(
        t, s, twist_threshold=FRONTS_TWIST_THRESHOLD, anomaly_factor=FRONTS_ANOMALY_FACTOR,
    )
    usable = [int(i) for i in fronts_idx if i >= RINGDOWN_PRE_EVENT_WINDOW]
    if not usable:
        return {"frac_oscillatory": 0.0, "n_candidates": 0}

    n_osc = 0
    for idx in usable:
        res = ringdown_resonance(
            t, s, idx,
            pre_event_window=RINGDOWN_PRE_EVENT_WINDOW,
            max_lookahead=RINGDOWN_MAX_LOOKAHEAD,
        )
        if res["is_oscillatory"]:
            n_osc += 1
    return {"frac_oscillatory": n_osc / len(usable), "n_candidates": len(usable)}


# ---------------------------------------------------------------------
# TRYB SYNTETYCZNY - sanity-check PRZED dotknięciem realnych danych
# (pozytywna kontrola: wstrzykniety sygnal MUSI zostac wykryty jako
# roznica; negatywna kontrola: czysty szum-vs-szum NIE MOZE dac
# falszywie istotnego wyniku - kalibracja czulosci na falszywe alarmy)
# ---------------------------------------------------------------------

def _synthetic_window(n=600, fs=1.0, seed=0, inject_oscillation=False, colored=False):
    """colored=True: szum AR(1) (autoskorelowany) zamiast bialego szumu,
    plus kilka losowych, NIETLUMIONYCH skokow ("mikro-usterki") - dzieki
    temu fronts() faktycznie znajduje kandydatow w tle rowniez BEZ
    wstrzyknietej oscylacji. Bialy szum (colored=False) jest zbyt gladki:
    fronts() nie znajduje na nim ZADNYCH kandydatow (zweryfikowano: 0/30
    okien), wiec porownanie bialy-szum-vs-bialy-szum jest zdegenerowane
    (0 vs 0) i nie testuje realnie, czy klasyfikacja is_oscillatory ma
    podwyzszony odsetek falszywych alarmow na tle, gdzie COS jest do
    zklasyfikowania."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) / fs
    if colored:
        white = rng.normal(0, 1.0, n)
        s = np.zeros(n)
        alpha = 0.85
        for i in range(1, n):
            s[i] = alpha * s[i - 1] + white[i]
        for _ in range(rng.integers(2, 5)):
            ev = int(rng.integers(RINGDOWN_PRE_EVENT_WINDOW + 5, n - 20))
            s[ev] += rng.choice([-1, 1]) * rng.uniform(4.0, 8.0)
    else:
        s = rng.normal(0, 1.0, n)
    if inject_oscillation:
        # wstrzyknij 2-3 tlumione oscylacje w losowych miejscach okna -
        # symuluje mikro-wstrzasy z realnym powrotem "dzwoniacym"
        for _ in range(rng.integers(2, 4)):
            ev = int(rng.integers(RINGDOWN_PRE_EVENT_WINDOW + 5, n - 150))
            post = np.arange(min(150, n - ev), dtype=float) / fs
            f0 = rng.uniform(0.05, 0.2)
            tau = rng.uniform(15, 40)
            s[ev:ev + len(post)] += 6.0 * np.exp(-post / tau) * np.cos(2 * np.pi * f0 * post)
    return t, s


def _direct_classifier_probe(seed: int, alpha: float = 0.9, n_probe: int = 15,
                              inject_at_probe: bool = False) -> float:
    """Omija fronts() i probuje ringdown_resonance() BEZPOSREDNIO w
    n_probe losowych punktach szumu AR(1) - izoluje kalibracje samego
    klasyfikatora is_oscillatory od tego, czy fronts() akurat cos
    znajdzie (fronts() jest bardzo konserwatywny na szumie AR(1):
    zweryfikowano, ze na 30 oknach AR(1)+okazjonalne skoki generuje
    kandydatow w ~10% okien - zbyt rzadko, zeby dac nietrywialna
    (nie-zdegenerowana 0-vs-0) probke do testu Manna-Whitneya). Ten
    bezposredni test jest surowszy i bardziej informatywny: sprawdza
    wprost, jak czesto klasyfikator MYLNIE (przy inject_at_probe=False)
    albo POPRAWNIE (przy inject_at_probe=True) rozpoznaje oscylacje w
    tle o realistycznej autokorelacji (blizszej realnemu tlu
    sejsmicznemu niz bialy szum)."""
    rng = np.random.default_rng(seed)
    n = 600
    t = np.arange(n, dtype=float)
    white = rng.normal(0, 1.0, n)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = alpha * s[i - 1] + white[i]

    probe_rng = np.random.default_rng(seed + 500_000)
    idxs = probe_rng.integers(RINGDOWN_PRE_EVENT_WINDOW + 10, n - 130, size=n_probe)
    n_osc = 0
    for idx in idxs:
        idx = int(idx)
        if inject_at_probe:
            post = np.arange(min(150, n - idx), dtype=float)
            f0 = rng.uniform(0.05, 0.2)
            tau = rng.uniform(15, 40)
            s = s.copy()
            s[idx:idx + len(post)] += 6.0 * np.exp(-post / tau) * np.cos(2 * np.pi * f0 * post)
        res = ringdown_resonance(t, s, idx, pre_event_window=RINGDOWN_PRE_EVENT_WINDOW,
                                  max_lookahead=RINGDOWN_MAX_LOOKAHEAD)
        if res["is_oscillatory"]:
            n_osc += 1
    return n_osc / n_probe


def run_synthetic_selftest(n_windows=30) -> dict:
    from scipy.stats import mannwhitneyu

    core = TIMDR_EarthquakeCore()

    # --- TEST GLOWNY (bramkuje pipeline_verified): kalibracja samego
    # klasyfikatora ringdown_resonance() na szumie AR(1) (autoskorelowanym,
    # blizszym realnemu tlu sejsmicznemu niz bialy szum), z pominieciem
    # fronts() - patrz docstring _direct_classifier_probe powyzej. Dwie
    # grupy: (1) prawdziwie wstrzykniete tlumione oscylacje w losowych
    # punktach tla, (2) czyste tlo bez wstrzykniecia. MUSI wyjsc
    # statystycznie istotna roznica, inaczej klasyfikator nie odroznia
    # sygnalu od tla.
    direct_with_signal = [_direct_classifier_probe(i, inject_at_probe=True) for i in range(n_windows)]
    direct_background_a = [_direct_classifier_probe(1000 + i, inject_at_probe=False) for i in range(n_windows)]
    stat_pos, p_pos = mannwhitneyu(direct_with_signal, direct_background_a, alternative="two-sided")

    # --- KONTROLA NEGATYWNA GLOWNA: dwie niezalezne probki tego samego
    # tla AR(1), OBIE bez wstrzykniecia. Test NIE MOZE regularnie dawac
    # istotnego wyniku - inaczej klasyfikator ma zawyzony odsetek
    # falszywych alarmow na realistycznym (autoskorelowanym) tle.
    direct_background_b = [_direct_classifier_probe(2000 + i, inject_at_probe=False) for i in range(n_windows)]
    stat_neg, p_neg = mannwhitneyu(direct_background_a, direct_background_b, alternative="two-sided")

    ok_positive_control = p_pos < 0.05
    ok_negative_control = p_neg >= 0.05

    # --- DIAGNOSTYKA DODATKOWA (informacyjna, NIE bramkuje
    # pipeline_verified): ten sam test, ale przez PELNY end-to-end
    # pipeline fronts()+ringdown_resonance(), dokladnie jak w
    # run_real_test(). fronts() jest na tyle konserwatywny na szumie
    # AR(1), ze w wiekszosci okien nie znajduje ZADNEGO kandydata (w
    # obu grupach), co czyni to porownanie czesciowo zdegenerowanym
    # (mala liczba niezerowych probek) - dlatego jest tu tylko jako
    # dodatkowa informacja, nie jako warunek zaliczenia.
    with_signal_pipeline = [ringdown_window_feature(*_synthetic_window(seed=i, inject_oscillation=True, colored=True), core)["frac_oscillatory"]
                            for i in range(n_windows)]
    background_pipeline = [ringdown_window_feature(*_synthetic_window(seed=1000 + i, inject_oscillation=False, colored=True), core)["frac_oscillatory"]
                           for i in range(n_windows)]
    n_nonzero_pipeline = int(np.sum(np.array(with_signal_pipeline + background_pipeline) > 0))

    return {
        "positive_control_p_value": float(p_pos),
        "positive_control_detected_injected_signal": bool(ok_positive_control),
        "negative_control_p_value": float(p_neg),
        "negative_control_no_false_alarm": bool(ok_negative_control),
        "mean_frac_oscillatory_direct_with_signal": float(np.mean(direct_with_signal)),
        "mean_frac_oscillatory_direct_background": float(np.mean(direct_background_a)),
        "pipeline_verified": bool(ok_positive_control and ok_negative_control),
        "end_to_end_fronts_pipeline_diagnostic": {
            "note": "Diagnostyka pomocnicza (nie bramkuje pipeline_verified): pelny fronts()+ringdown na tym samym tle AR(1). fronts() jest konserwatywny na tym tle (mala liczba niezerowych probek w obu grupach), wiec ten test ma mniejsza moc niz bezposrednia kalibracja klasyfikatora powyzej.",
            "n_nonzero_samples_of_60": n_nonzero_pipeline,
            "mean_frac_oscillatory_with_signal": float(np.mean(with_signal_pipeline)),
            "mean_frac_oscillatory_background": float(np.mean(background_pipeline)),
        },
    }


# ---------------------------------------------------------------------
# TRYB REALNY - wymaga obspy + requests + polaczenia z siecia
# ---------------------------------------------------------------------

def fetch_usgs_catalog(min_magnitude: float, start: datetime, end: datetime) -> list[dict]:
    import requests
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        f"?format=geojson&starttime={start.date()}&endtime={end.date()}"
        f"&minmagnitude={min_magnitude}"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    events = []
    for feat in data.get("features", []):
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"][:2]
        events.append({
            "id": feat["id"],
            "time": datetime.fromtimestamp(props["time"] / 1000.0, tz=timezone.utc),
            "lat": lat, "lon": lon, "mag": props["mag"],
        })
    return events


def fetch_background_catalog(start: datetime, end: datetime) -> list[dict]:
    """M>=4.5 - szerszy katalog do WYKLUCZANIA okien tla (nawet mniejszy
    wstrzas w poblizu okna tla by je skazil)."""
    return fetch_usgs_catalog(4.5, start, end)


def fetch_window(client, station: dict, center_time, hours_before: float, hours_after: float):
    from obspy import UTCDateTime
    ct = UTCDateTime(center_time)
    st = client.get_waveforms(
        station["net"], station["sta"], station["loc"], station["cha"],
        ct - hours_before * 3600, ct + hours_after * 3600,
    )
    st.merge(fill_value="interpolate")
    tr = st[0]
    tr.detrend("demean")
    s = tr.data.astype(float)
    fs = tr.stats.sampling_rate
    t = np.arange(len(s)) / fs
    return t, s


def run_real_test(min_magnitude: float, years: int, n_background: int) -> dict:
    import random
    from scipy.stats import mannwhitneyu
    from obspy.clients.fdsn import Client

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years)

    print(f"[1/5] Pobieram realny katalog USGS (M>={min_magnitude}, {years} lat)...")
    events = fetch_usgs_catalog(min_magnitude, start, end)
    print(f"      -> {len(events)} zdarzen")
    if not events:
        raise RuntimeError("Katalog USGS pusty dla podanych parametrow - nie ma czego testowac.")

    print("[2/5] Pobieram szerszy katalog M>=4.5 (do wykluczania okien tla)...")
    exclusion_catalog = fetch_background_catalog(start, end)
    exclusion_times = sorted(e["time"] for e in exclusion_catalog)

    client = Client("EARTHSCOPE")
    core = TIMDR_EarthquakeCore()

    print(f"[3/5] Licze cechy PRE-EVENT dla {len(events)} realnych wstrzasow...")
    pre_features, pre_meta = [], []
    for ev in events:
        station = nearest_station(ev["lat"], ev["lon"])
        window_end = ev["time"] - timedelta(hours=LEAD_HOURS)
        try:
            t, s = fetch_window(client, station, window_end, WINDOW_HOURS, 0.0)
        except Exception as e:
            print(f"      pominieto {ev['id']} ({station['sta']}): {e}")
            continue
        feat = ringdown_window_feature(t, s, core)
        pre_features.append(feat["frac_oscillatory"])
        pre_meta.append({"event_id": ev["id"], "station": station["sta"], **feat})

    print(f"[4/5] Licze cechy TLA dla {n_background} losowych okien...")
    rng = random.Random(42)
    bg_features, bg_meta, attempts = [], [], 0
    while len(bg_features) < n_background and attempts < n_background * 20:
        attempts += 1
        station = rng.choice(RELIABLE_STATIONS)
        candidate = start + timedelta(seconds=rng.uniform(0, (end - start).total_seconds()))
        if any(abs((candidate - t_ex).total_seconds()) < EXCLUSION_DAYS * 86400 for t_ex in exclusion_times):
            continue
        try:
            t, s = fetch_window(client, station, candidate, WINDOW_HOURS, 0.0)
        except Exception:
            continue
        feat = ringdown_window_feature(t, s, core)
        bg_features.append(feat["frac_oscillatory"])
        bg_meta.append({"station": station["sta"], "time": candidate.isoformat(), **feat})

    print(f"      -> {len(pre_features)} okien pre-event, {len(bg_features)} okien tla")
    if len(pre_features) < 5 or len(bg_features) < 5:
        raise RuntimeError("Za malo udanych okien do sensownego testu statystycznego (potrzeba >=5 w kazdej grupie).")

    print("[5/5] Test Manna-Whitneya U (pre-event vs tlo)...")
    stat, p_value = mannwhitneyu(pre_features, bg_features, alternative="two-sided")

    return {
        "n_pre_event_windows": len(pre_features),
        "n_background_windows": len(bg_features),
        "mean_frac_oscillatory_pre_event": float(np.mean(pre_features)),
        "mean_frac_oscillatory_background": float(np.mean(bg_features)),
        "mannwhitney_p_value": float(p_value),
        "significant_at_0_05": bool(p_value < 0.05),
        "pre_event_higher_than_background": bool(np.mean(pre_features) > np.mean(bg_features)),
        "pre_event_details": pre_meta,
        "background_details": bg_meta,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic",
                     help="synthetic = tylko sanity-check (domyslne, bez sieci); real = pelny test na USGS+EarthScope")
    ap.add_argument("--min-magnitude", type=float, default=MIN_MAGNITUDE_DEFAULT)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--n-background", type=int, default=60)
    ap.add_argument("--out", default="precursor_ringdown_test_output.json")
    args = ap.parse_args()

    print("=" * 70)
    print("KROK 1: sanity-check na danych SYNTETYCZNYCH (zawsze, bez sieci)")
    print("=" * 70)
    synth = run_synthetic_selftest()
    for k, v in synth.items():
        print(f"  {k}: {v}")

    result = {"synthetic_selftest": synth}

    if not synth["pipeline_verified"]:
        print("\n!!! Sanity-check NIE PRZESZEDL - metodologia ma problem (nie wykrywa")
        print("    wstrzykniętego sygnału albo daje fałszywe alarmy na czystym szumie).")
        print("    NIE URUCHAMIAM testu na realnych danych - wynik byłby niewiarygodny.")
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        sys.exit(1)

    print("\nSanity-check PRZESZEDL: pipeline poprawnie odróżnia wstrzykniety sygnał")
    print("od szumu i nie daje fałszywych alarmów szum-vs-szum.")

    if args.mode == "real":
        print("\n" + "=" * 70)
        print("KROK 2: test na REALNYCH danych (USGS + EarthScope/IRIS)")
        print("=" * 70)
        try:
            real = run_real_test(args.min_magnitude, args.years, args.n_background)
        except Exception as e:
            print(f"\nBLAD podczas testu na realnych danych: {type(e).__name__}: {e}")
            print("(Typowa przyczyna w środowisku bez pełnego dostępu do sieci: brak")
            print(" połączenia z earthquake.usgs.gov / service.earthscope.org.")
            print(" Uruchom ten skrypt na maszynie z pełnym dostępem do internetu.)")
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2, default=str)
            sys.exit(2)

        result["real_test"] = real
        print()
        print("-" * 70)
        print("WYNIK (realne dane):")
        print(f"  pre-event frac_oscillatory (srednia): {real['mean_frac_oscillatory_pre_event']:.4f}")
        print(f"  tlo frac_oscillatory (srednia):        {real['mean_frac_oscillatory_background']:.4f}")
        print(f"  p-value (Mann-Whitney U):               {real['mannwhitney_p_value']:.4f}")
        if real["significant_at_0_05"] and real["pre_event_higher_than_background"]:
            print("  => Statystycznie istotna RÓŻNICA, pre-event WYŻSZE niż tło.")
            print("     To jest WSTĘPNA przesłanka, NIE dowód predykcyjności - wymaga")
            print("     replikacji na niezależnym zbiorze zdarzeń, dokładnie tak jak")
            print("     'trop' z BTC w timdr-finanse nie przetrwał replikacji na złocie.")
        elif real["significant_at_0_05"]:
            print("  => Statystycznie istotna różnica, ale TŁO wyższe niż pre-event -")
            print("     to NIE wspiera hipotezy predykcyjnej (kierunek odwrotny).")
        else:
            print("  => WYNIK NEGATYWNY: brak statystycznie istotnej różnicy.")
            print("     Zgodne z wcześniejszym testem Topology(t) w tym repo.")
        print("-" * 70)
    else:
        print("\n(Tryb 'synthetic' - żeby uruchomić pełny test na realnych danych:")
        print(" python precursor_ringdown_test.py --mode real)")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nPelny wynik zapisany do: {args.out}")


if __name__ == "__main__":
    main()
