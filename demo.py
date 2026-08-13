import numpy as np
from timdr_core_earthquake import TIMDR_EarthquakeCore

core = TIMDR_EarthquakeCore()

rng = np.random.default_rng(2)
n = 300
t = np.arange(n) * 0.01
s = rng.normal(0, 0.02, n)
s[150:200] += np.linspace(0, 8.0, 50)  # narastajacy wstrzas

flow_grad = core.flow(t, s)
twist_pts, twist_strength = core.twist(flow_grad, t)
smooth = core.trm(t, s)
anomaly_pts, residuals, th = core.anomalies(t, s)
fronts, _, _ = core.fronts(t, s)

print("Liczba probek:", n)
print("Twist points (pierwsze 10):", twist_pts[:10])
print("Anomaly points (pierwsze 10):", anomaly_pts[:10])
print("Fronts (poczatek wstrzasu):", fronts[:5], "..." if len(fronts) > 5 else "")
print("Prog anomalii (MAD):", round(th, 4))
