"""
Pierwszy realny test ringdown_resonance() i sta_lta()/trigger_onset() na
PRAWDZIWYM, ciaglym sejsmogramie (nie syntetycznej fali, nie katalogu
magnitud) - dane od uzytkownika, pobrane samodzielnie przez ObsPy z
EarthScope/IRIS na jego wlasnym komputerze (sandbox blokowal ten serwer),
stacje CI.CLC i CI.RIO, kanal HHZ, 2019-07-06 03:18:52.998 - 03:24:52.998
(6 minut, 100Hz, 36001 probek), obejmujace prawdziwy mainshock M7.1
Ridgecrest (2019-07-06T03:19:53.040Z).

PRE-REJESTRACJA (przed uruchomieniem):
- event_idx: obliczony z realnego czasu mainshocku wzgledem realnego
  czasu startu sladu (nie zgadywany, nie dopasowywany po zobaczeniu danych).
- ringdown_resonance(): domyslne noise_floor_factor=3.0 jako WYNIK GLOWNY;
  dodatkowo sweep {1.0, 1.5, 2.0, 3.0} zgodnie z udokumentowana w module
  wrazliwoscia na ten parametr (nie chowamy tej wrazliwosci).
- sta_lta()/trigger_onset(): nsta=100 (1s), nlta=1000 (10s), thr_on=3.5,
  thr_off=1.0 - te same wartosci co w kazdym innym tescie w tej sesji,
  bez tuningu pod te konkretne dane.
- Uruchomienie: RAZ, obie stacje, bez zmiany parametrow po zobaczeniu wyniku.
"""
import sys
import os; sys.path.insert(0, os.path.dirname(__file__))
from obspy import read, UTCDateTime
import numpy as np
from timdr_core_earthquake import TIMDR_EarthquakeCore
from ringdown import ringdown_resonance

MAINSHOCK = UTCDateTime("2019-07-06T03:19:53.040")

FILES = {
    "CLC": "data/ridgecrest_2019/real_waveform_CLC_RIO/CLC_HHZ.mseed",
    "RIO": "data/ridgecrest_2019/real_waveform_CLC_RIO/RIO_HHZ.mseed",
}

core = TIMDR_EarthquakeCore()

for name, path in FILES.items():
    st = read(path)
    tr = st[0]
    fs = tr.stats.sampling_rate
    s = tr.data.astype(float)
    n = len(s)
    t = np.arange(n) / fs

    event_idx = int(round((MAINSHOCK - tr.stats.starttime) * fs))
    print(f"=========== STACJA {name} ({tr.id}) ===========")
    print(f"n={n} fs={fs}Hz start={tr.stats.starttime} event_idx={event_idx} "
          f"(t_event={event_idx/fs:.2f}s)")

    # --- ringdown_resonance ---
    print("\n-- ringdown_resonance() --")
    for nff in (1.0, 1.5, 2.0, 3.0):
        r = ringdown_resonance(t, s, event_idx, pre_event_window=500,
                                noise_floor_factor=nff)
        print(f"  noise_floor_factor={nff}: is_oscillatory={r['is_oscillatory']} "
              f"n_crossings={r['n_crossings']} n_peaks={r['n_peaks_used']} "
              f"period_s={r['period_s']} freq_hz={r['frequency_hz']} "
              f"damping_ratio={r['damping_ratio']}")

    # --- sta_lta / trigger_onset ---
    print("\n-- sta_lta() / trigger_onset() --")
    nsta, nlta = int(1.0 * fs), int(10.0 * fs)
    ratio = core.sta_lta(s, nsta, nlta)
    onsets = core.trigger_onset(ratio, thr_on=3.5, thr_off=1.0)
    print(f"  nsta={nsta} nlta={nlta} thr_on=3.5 thr_off=1.0")
    print(f"  n_onsets={len(onsets)}")
    for a, b in onsets:
        print(f"    onset: {a/fs:.2f}s - {b/fs:.2f}s  (peak ratio in window: {ratio[a:b+1].max():.2f})")
    print()
