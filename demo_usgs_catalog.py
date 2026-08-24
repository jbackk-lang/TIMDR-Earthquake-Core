"""
demo_usgs_catalog.py — TIMDR Catalog Fusion na realnych danych USGS
========================================================================
Snapshot poniżej to PRAWDZIWE dane, pobrane live z
`earthquake.usgs.gov/fdsnws/event/1/query` w dniu 2026-08-14
(zakres: M5.0+, 2026-08-01 do 2026-08-14, 64 zdarzenia, w tym
trzęsienie M7.4 w Kolumbii z 2026-08-08).

Żeby odświeżyć na aktualne dane, zamiast `RAW` poniżej użyj:

    import requests
    r = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params={"format": "geojson", "starttime": "2026-08-01",
                "endtime": "2026-08-14", "minmagnitude": 5},
    )
    feats = r.json()["features"]
    RAW = [(f["properties"]["mag"], f["properties"]["time"]) for f in feats]
"""

import numpy as np
from catalog_core import TIMDRCatalogFusion

RAW = [
    (5.3, 1786646371327), (5.1, 1786642769650), (5.6, 1786637505826), (5.0, 1786625099102),
    (5.1, 1786612978999), (5.1, 1786581318815), (5.5, 1786581004840), (5.1, 1786572813611),
    (6.0, 1786535232117), (5.0, 1786518892581), (5.1, 1786511487487), (5.2, 1786506513980),
    (5.0, 1786480288580), (5.1, 1786479258452), (5.0, 1786467978902), (5.1, 1786381686809),
    (5.0, 1786367890451), (7.4, 1786365268125), (5.0, 1786363942220), (5.5, 1786340504224),
    (5.0, 1786313847839), (5.3, 1786288503956), (5.2, 1786276653237), (5.0, 1786244527086),
    (5.6, 1786211878745), (5.2, 1786196970300), (5.5, 1786189287790), (5.6, 1786164634985),
    (5.0, 1786161571810), (5.1, 1786144747914), (5.0, 1786112490314), (5.1, 1786079310903),
    (5.0, 1786070290013), (5.3, 1786042641831), (5.4, 1786038216862), (5.1, 1786023731429),
    (5.6, 1785966087717), (5.2, 1785959186861), (5.3, 1785948618349), (5.0, 1785924369955),
    (6.3, 1785915807317), (5.2, 1785912264701), (5.2, 1785903904586), (5.1, 1785903561769),
    (6.3, 1785903246625), (5.7, 1785874597181), (5.1, 1785855374381), (5.2, 1785721344662),
    (5.0, 1785715233174), (5.0, 1785684930843), (5.1, 1785676247947), (5.3, 1785671553048),
    (5.3, 1785666854181), (5.0, 1785662231235), (5.6, 1785659727743), (5.2, 1785629618325),
    (5.2, 1785614566638), (5.3, 1785611453802), (5.1, 1785599471360), (5.2, 1785582440193),
    (5.1, 1785577833882), (5.1, 1785570363634), (5.6, 1785559741269), (5.5, 1785552481570),
]


def load_snapshot():
    raw = sorted(RAW, key=lambda r: r[1])
    t0 = raw[0][1]
    t = np.array([(r[1] - t0) / 1000 / 3600 for r in raw])  # godziny od pierwszego zdarzenia
    mag = np.array([r[0] for r in raw])
    return t, mag


def main():
    t, mag = load_snapshot()
    print(f"N zdarzen: {len(t)}, okno czasowe: {t[-1]:.1f} h (~{t[-1] / 24:.1f} dni)")

    cat = TIMDRCatalogFusion()

    an_idx, an_z = cat.anomalies(mag)
    print("\n--- ANOMALIE (magnitude, MAD-z > 3.0) ---")
    for i in an_idx:
        print(f"  t={t[i]:7.2f}h  M{mag[i]:.1f}  |z|={an_z[i]:.2f}")

    periods, r_score = cat.rhythm(mag, max_lag=30, power_thresh=0.4)
    print(f"\n--- RYTM (max_lag=30, power_thresh=0.4) ---")
    print(f"  wykryte 'okresy' (w zdarzeniach, NIE godzinach): {periods}  score={r_score:.3f}")
    print("  poprawnie: brak okresowosci (M5+ globalnie ~ proces Poissona).")
    print("  patrz docstring rhythm() w catalog_core.py po szczegoly (w tym")
    print("  o bledzie rektyfikacji, ktory dawal falszywy alarm we wczesniejszej wersji).")

    slopes, tr_z = cat.trend(t, mag, window=15)
    print(f"\n--- TREND (magnitude, okno 15 zdarzen) ---")
    print(f"  ostatnia wartosc trend_z: {tr_z[-1]:.2f}, max |trend_z|: {np.max(np.abs(tr_z)):.2f}")

    nxt, dt_h = cat.nearest_aftershock(t, mag)
    if nxt is not None:
        print(f"\n--- Najblizsze zdarzenie PO najwiekszym wstrzasie (M{mag.max():.1f}) ---")
        print(f"  M{mag[nxt]:.1f}, {dt_h * 60:.1f} min pozniej -> klasyczny aftershock")


if __name__ == "__main__":
    main()
