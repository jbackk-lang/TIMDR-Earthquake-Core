"""
analyze_topology_resonance_seismic.py — czy R(t)=|Flow|*|Twist|*|Defect|*
Topology podnosi sie PRZED poczatkiem wstrzasu (a nie dopiero w trakcie)?
==============================================================================
Dla sejsmiki pytanie "czy to predyktuje" ma inny ksztalt niz dla finansow:
`fronts()`/`hybrid_trigger()` juz DETEKTUJA wstrzas, ktory sie zaczyna -
to nie jest predykcja, to potwierdzenie. Jedyny uczciwy sposob zapytania
"czy R(t) predyktuje" w tym kontekscie: czy R(t), liczone WYLACZNIE z
danych sprzed etykietowanego poczatku wstrzasu (fronts()/rampy w demo),
jest SYSTEMATYCZNIE podwyzszone w oknie TUZ PRZED startem, w porownaniu
do losowych okien z tla (bez zadnego zdarzenia)? Jesli tak - to byloby
realne, uzyteczne wczesne ostrzeganie. Jesli nie (R(t) rosnie dopiero
PO starcie, razem z detekcja) - to R(t) jest kolejnym detektorem, nie
predyktorem.

Dane: 2 scenariusze demo z gui_app.py z prawdziwym, ZNANYM poczatkiem
zdarzenia (nie trzeba etykiety zgadywac - sami skonstruowalismy sygnal):
- demo_earthquake: rampa zaczyna sie w probce `start` (nagly poczatek)
- demo_drift: rampa zaczyna sie w probce `ramp_start` (stopniowy poczatek)
Plus demo_noise jako kontrola negatywna (brak zdarzenia w ogole - test
czy metoda nie halucynuje "przedwstrzasowego" sygnalu tam, gdzie nic sie
nie dzieje).

Test: PERMUTACYJNY, nie tylko srednia. Porownujemy R(t) (i osobno samo
Topology(t)) w oknie PRZED-startem vs w wielu losowych oknach z tla,
zeby ocenic, czy roznica jest wieksza niz przypadkowa.
"""
import numpy as np
from timdr_core_earthquake import TIMDR_EarthquakeCore
from topology_features import topology_series, zscore_causal

TOPOLOGY_WINDOW, TOPOLOGY_EMBED_DIM, TOPOLOGY_DELAY = 60, 3, 3
PRE_WINDOW = 30  # ile probek "przed startem" liczymy jako okno przedwstrzasowe


# Skopiowane 1:1 z gui_app.py (bez importu, zeby nie ciagnac zaleznosci od
# tkinter do czysto analitycznego skryptu bez GUI).
def demo_earthquake(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * 0.01
    s = rng.normal(0, 0.05, n) + 0.002 * t
    start = n // 2 - 20
    ramp = np.concatenate([np.linspace(0, 3.0, 20), np.linspace(3.0, 0, 20)])
    s[start:start + 40] += ramp
    glitch_idx = max(10, start - 60)
    s[glitch_idx] = 8.0
    return t, s


def demo_drift(n=500, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * 0.01
    s = rng.normal(0, 0.03, n)
    ramp_start, ramp_len = 200, 120
    s[ramp_start:ramp_start + ramp_len] += np.linspace(0, 4.0, ramp_len)
    s[ramp_start + ramp_len:] += 4.0
    return t, s


def demo_noise(n=400, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * 0.01
    s = rng.normal(0, 0.05, n)
    return t, s


def causal_R_series(t, s):
    """Liczy Flow/Twist/Defect/Topology KAUZALNIE w kazdym punkcie i
    (uzywajac tylko s[:i+1]/t[:i+1]) i sklada w R(t). Kosztowne (O(n^2)
    bo Flow/Twist/anomalies sa przeliczane od zera w kazdym punkcie -
    ale to jedyny sposob, zeby bylo naprawde kauzalne, bez lookaheadu z
    wygladzania/normalizacji na calym sygnale)."""
    core = TIMDR_EarthquakeCore()
    n = len(s)
    flow_raw = np.zeros(n)
    twist_raw = np.zeros(n)
    defect_raw = np.zeros(n)
    topo_raw = np.zeros(n)

    for i in range(5, n):  # 5 minimalna historia dla flow/twist
        past_t, past_s = t[:i + 1], s[:i + 1]
        fg = core.flow(past_t, past_s)
        flow_raw[i] = fg[-1] if len(fg) else 0.0

        _, tw_strength = core.twist(fg, past_t, threshold=0.0)
        twist_raw[i] = tw_strength[-1] if len(tw_strength) else 0.0

        _, resid, _ = core.anomalies(past_t, past_s, factor=3.0)
        defect_raw[i] = resid[-1] if len(resid) else 0.0

        tail = past_s[-TOPOLOGY_WINDOW:]
        topo = topology_series(tail, window=TOPOLOGY_WINDOW,
                                embed_dim=TOPOLOGY_EMBED_DIM, delay=TOPOLOGY_DELAY)
        topo_raw[i] = topo['topology'][-1] if len(topo['topology']) else 0.0

    flow_n = np.clip(zscore_causal(np.abs(flow_raw)), 0, None)
    twist_n = np.clip(zscore_causal(np.abs(twist_raw)), 0, None)
    defect_n = np.clip(zscore_causal(np.abs(defect_raw)), 0, None)
    topo_n = np.clip(zscore_causal(topo_raw), 0, None)
    R = flow_n * twist_n * defect_n * topo_n
    return dict(flow=flow_raw, twist=twist_raw, defect=defect_raw,
                topology=topo_raw, R=R)


def pre_event_test(signal_name, values, event_start, pre_window=PRE_WINDOW,
                    n_perm=5000, seed=0, exclude_around_event=40):
    n = len(values)
    pre_lo = max(0, event_start - pre_window)
    pre_hi = event_start
    if pre_hi - pre_lo < 5:
        print(f"  [{signal_name}] za malo historii przed zdarzeniem, pomijam")
        return
    pre_vals = values[pre_lo:pre_hi]
    pre_mean = np.mean(pre_vals)

    # tlo: wszystkie mozliwe okna tej samej dlugosci, z wylaczeniem stref
    # wokol zdarzenia (zeby nie "podpowiadac" tlu wartosci z samego
    # narastania/szczytu wstrzasu)
    excl_lo = max(0, event_start - exclude_around_event)
    excl_hi = min(n, event_start + exclude_around_event)
    candidates = [i for i in range(pre_window, n)
                  if not (excl_lo - pre_window <= i <= excl_hi)]
    if len(candidates) < 20:
        print(f"  [{signal_name}] za malo okien tla, pomijam")
        return

    rng = np.random.default_rng(seed)
    bg_means = np.empty(n_perm)
    for p in range(n_perm):
        end = candidates[rng.integers(0, len(candidates))]
        bg_means[p] = np.mean(values[end - pre_window:end])

    pctl = float(np.mean(bg_means < pre_mean) * 100)
    print(f"  [{signal_name}] srednia w oknie przed-startem ({pre_window} probek) = {pre_mean:.4f}  "
          f"| tlo (n={n_perm} losowych okien): srednia={bg_means.mean():.4f}, std={bg_means.std():.4f} "
          f"| percentyl okna przed-startem wzgledem tla = {pctl:.1f} "
          f"({'PODWYZSZONE' if pctl > 95 else 'NIE odstaje istotnie' if 5 <= pctl <= 95 else 'OBNIZONE'})")
    return pctl


def main():
    scenarios = [
        ("Earthquake (nagly poczatek, front=start)", *demo_earthquake(), None),
        ("Gradual drift (stopniowy poczatek)", *demo_drift(), None),
        ("Background noise (brak zdarzenia - kontrola)", *demo_noise(), None),
    ]
    # znane (skonstruowane przez nas) indeksy startu zdarzenia:
    event_starts = {
        "Earthquake (nagly poczatek, front=start)": 400 // 2 - 20,
        "Gradual drift (stopniowy poczatek)": 200,
        "Background noise (brak zdarzenia - kontrola)": None,
    }

    for label, t, s, _ in scenarios:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        res = causal_R_series(t, s)
        event_start = event_starts[label]

        if event_start is None:
            # kontrola negatywna: porownaj losowe okno z "poczatku" (np. w
            # polowie sygnalu) z reszta tla - nie powinno byc podwyzszone
            event_start = len(s) // 2
            print("  (brak realnego zdarzenia - test formalny na srodku sygnalu, "
                  "oczekiwany wynik: brak istotnego podwyzszenia)")

        print(" -- R(t) (pelny iloczyn) --")
        pre_event_test(label, res['R'], event_start)
        print(" -- Topology(t) sama --")
        pre_event_test(label, res['topology'], event_start)


if __name__ == "__main__":
    main()
