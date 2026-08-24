import numpy as np
from seismic_loader import SeismicLoader
from timdr_core_earthquake import TIMDR_EarthquakeCore

rng = np.random.default_rng(0)
n = 300
t = np.arange(n, dtype=float) * 0.01
s = rng.normal(0, 0.05, n) + 0.002 * t          # tlo + drobny dryf
s[140:180] += np.concatenate([np.linspace(0, 3.0, 20), np.linspace(3.0, 0, 20)])  # prawdziwy wstrzas
s[80] = 8.0                                      # izolowany glitch czujnika

loader = SeismicLoader()  # domyslnie: sortowanie, detrend, despike, normalizacja
t_clean, s_clean = loader.load_waveform(s, t)

print("Przed czyszczeniem: glitch(t=0.8s)=", round(s[80], 2), " wstrzas_peak=", round(s[159], 2))
print("Po czyszczeniu:      glitch(t=0.8s)=", round(s_clean[80], 2), " wstrzas_peak=", round(s_clean[159], 2))

core = TIMDR_EarthquakeCore()
flow_grad = core.flow(t_clean, s_clean)
twist_pts, _ = core.twist(flow_grad, t_clean)
fronts, _, _ = core.fronts(t_clean, s_clean)
print("\nWykryte fronty (poczatek wstrzasu):", fronts[:5], "...")

# STA/LTA - klasyczny picker (wlasna implementacja, zweryfikowana z ObsPy - patrz README)
nsta, nlta = 25, 100  # w probkach (0.25s / 1.0s przy 100Hz)
ratio = core.sta_lta(s_clean, nsta, nlta)
onsets = core.trigger_onset(ratio, thr_on=3.0, thr_off=1.0)
print("STA/LTA trigger on/off (indeksy probek):", onsets.tolist())
