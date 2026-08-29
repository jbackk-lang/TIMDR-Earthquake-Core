"""
CLI — uruchom pełną analizę TIMDR-Earthquake-Core na pliku CSV bez
pisania własnego kodu.

Użycie:
    python cli.py dane.csv
    python cli.py dane.csv --t-col time --s-col amplitude
    python cli.py dane.csv --nsta 25 --nlta 100 --out raport.json
    python cli.py dane.csv --no-clean   # pomiń SeismicLoader (dane już czyste)

Plik wejściowy: CSV z dwiema kolumnami (domyślnie "t","s") — znaczniki
czasu w sekundach i amplituda. Przykład: obspy_BW_RJOB_example.csv w
tym repo (prawdziwy ślad sejsmiczny, tutorial ObsPy).
"""
import argparse
import json
import sys

import numpy as np

from seismic_loader import SeismicLoader
from timdr_core_earthquake import TIMDR_EarthquakeCore


def analyze(t, s, nsta=25, nlta=100, sta_lta_thr_on=3.0, sta_lta_thr_off=1.0,
            twist_threshold=0.4, anomaly_factor=3.0):
    """Uruchamia pełny pipeline i zwraca wynik jako zwykły dict (JSON-friendly)."""
    core = TIMDR_EarthquakeCore()

    flow_grad = core.flow(t, s)
    twist_pts, twist_strength = core.twist(flow_grad, t, threshold=twist_threshold)
    fronts, _, _ = core.fronts(t, s, twist_threshold=twist_threshold, anomaly_factor=anomaly_factor)
    events_raw = core.classify_anomalies(t, s, factor=anomaly_factor)
    events = [
        {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in e.items()}
        for e in events_raw
    ]

    ratio = core.sta_lta(s, nsta, nlta)
    onsets = core.trigger_onset(ratio, thr_on=sta_lta_thr_on, thr_off=sta_lta_thr_off)
    confirmed, rejected = core.hybrid_trigger(
        t, s, nsta=nsta, nlta=nlta,
        twist_threshold=twist_threshold, anomaly_factor=anomaly_factor,
        sta_lta_thr_on=sta_lta_thr_on, sta_lta_thr_off=sta_lta_thr_off,
    )

    def onset_to_time(idx_pair):
        a, b = idx_pair
        return {"start_t": float(t[a]), "end_t": float(t[b]), "start_idx": int(a), "end_idx": int(b)}

    return {
        "n_samples": len(t),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "twist_points": {"n": len(twist_pts), "times": [float(t[i]) for i in twist_pts[:50]]},
        "fronts": {"n": len(fronts), "times": [float(t[i]) for i in fronts[:50]]},
        "classified_events": events[:50],
        "sta_lta": {
            "params": {"nsta": nsta, "nlta": nlta, "thr_on": sta_lta_thr_on, "thr_off": sta_lta_thr_off},
            "n_onsets": len(onsets),
            "onsets": [onset_to_time(o) for o in onsets[:50]],
        },
        "hybrid_trigger": {
            "n_confirmed": len(confirmed),
            "n_rejected": len(rejected),
            "confirmed": [onset_to_time(o) for o in confirmed[:50]],
            "rejected_reasons": [
                ("missing_twist" if r["missing_twist"] else "") +
                ("+" if r["missing_twist"] and r["missing_anomaly"] else "") +
                ("missing_anomaly" if r["missing_anomaly"] else "")
                for r in rejected[:50]
            ],
        },
        "note": (
            "hybrid_trigger redukuje false-positive przez potwierdzenie twist+anomaly, "
            "ale NIE jest zwalidowany wzgledem katalogu prawdziwych zdarzen (patrz README). "
            "Podczas gestych sekwencji (roj wstrzasow wtornych) rozdzielczosc detekcji "
            "pojedynczych zdarzen spada - patrz README, sekcja 'Roj wstrzasow wtornych'."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analiza sejsmiczna TIMDR-Earthquake-Core z linii poleceń.")
    parser.add_argument("csv_path", help="Ścieżka do pliku CSV (kolumny czas/amplituda)")
    parser.add_argument("--t-col", default="t", help="Nazwa kolumny czasu (domyślnie: t)")
    parser.add_argument("--s-col", default="s", help="Nazwa kolumny amplitudy (domyślnie: s)")
    parser.add_argument("--nsta", type=int, default=25, help="Okno STA w próbkach (domyślnie: 25)")
    parser.add_argument("--nlta", type=int, default=100, help="Okno LTA w próbkach (domyślnie: 100)")
    parser.add_argument("--thr-on", type=float, default=3.0, help="Próg włączenia STA/LTA (domyślnie: 3.0)")
    parser.add_argument("--thr-off", type=float, default=1.0, help="Próg wyłączenia STA/LTA (domyślnie: 1.0)")
    parser.add_argument("--no-clean", action="store_true", help="Pomiń SeismicLoader (dane już czyste)")
    parser.add_argument("--out", default=None, help="Zapisz pełny wynik JSON do pliku")
    args = parser.parse_args(argv)

    if args.no_clean:
        import csv as csv_mod
        t, s = [], []
        with open(args.csv_path, newline="") as f:
            for row in csv_mod.DictReader(f):
                t.append(float(row[args.t_col]))
                s.append(float(row[args.s_col]))
        t, s = np.asarray(t), np.asarray(s)
    else:
        loader = SeismicLoader()
        t, s = loader.load_csv(args.csv_path, t_col=args.t_col, s_col=args.s_col)

    if len(t) < args.nlta:
        print(f"[UWAGA] Sygnał ma {len(t)} próbek, mniej niż okno LTA ({args.nlta}) — "
              f"STA/LTA zwróci same zera. Zmniejsz --nlta.", file=sys.stderr)

    result = analyze(t, s, nsta=args.nsta, nlta=args.nlta,
                      sta_lta_thr_on=args.thr_on, sta_lta_thr_off=args.thr_off)

    print(f"Plik: {args.csv_path}  |  {result['n_samples']} próbek, {result['duration_s']:.2f}s")
    print(f"Fronty wstrząsów (fronts): {result['fronts']['n']}")
    print(f"Zdarzenia sklasyfikowane (classify_anomalies): {len(result['classified_events'])}")
    print(f"STA/LTA onsets: {result['sta_lta']['n_onsets']}")
    print(f"hybrid_trigger: {result['hybrid_trigger']['n_confirmed']} potwierdzonych, "
          f"{result['hybrid_trigger']['n_rejected']} odrzuconych")
    if result["sta_lta"]["onsets"]:
        print("\nPierwsze wykryte zdarzenia (STA/LTA):")
        for o in result["sta_lta"]["onsets"][:10]:
            print(f"  t={o['start_t']:.2f}s -> {o['end_t']:.2f}s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nPełny wynik zapisany do {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
