"""
topology_features.py — Topology(t): kauzalna cecha topologiczna sygnalu
=========================================================================
Operacyjna (POLICZALNA) wersja pojecia "Topology(t)" z dyskusji TIMDR
(segmenty / granice / petle / wezly / przerwy).

METODA: sliding-window (delay/Takens) embedding + Vietoris-Rips
persistent homology, tzw. SW1PerS (Sliding Windows 1-Persistence
Scoring; Perea & Harer, "Sliding Windows and Persistence", Found.
Comput. Math. 2015) — publikowana, ustalona metoda do wykrywania
struktury cyklicznej (okresowosci) w pojedynczym szeregu czasowym.
NIE jest to metoda wymyslona pod ten test.

PIERWSZA PROBA (udokumentowana, zeby bylo widac, ze nie przeszukiwalem
parametrow az "zadzialalo" na WLASCIWYCH danych): naiwny graf k-NN +
circuit rank (E-V+C) na tym samym embeddingu NIE odrozniał sygnalu
okresowego od czystego szumu (oba dawaly podobne beta1, zdominowane
przez sama liczbe k, nie przez geometrie). Test sanity-check (sinus vs
szum vs trend liniowy, WYLACZNIE syntetyczne dane, nie te uzywane w
analizie predykcyjnej) na prawdziwej homologii uporczywej (gudhi,
Vietoris-Rips) dal poprawna, oczekiwana jakosciowo separacje: sinus ->
1 trwaly cykl H1 (wysoka "persistence"), szum -> kilkanascie
krotkotrwalych, szum-podobnych cykli, trend liniowy -> 0 cykli. Na tej
podstawie (embed_dim=3, delay=3, window=60) zbudowany jest ponizszy
kod - parametry ZAMROZONE PRZED policzeniem czegokolwiek na realnych
danych BTC/zlota/sejsmiki, zeby uniknac data snoopingu.

Dla kazdego t, uzywajac WYLACZNIE danych z przeszlosci [t-window+1, t]
(kauzalnie):
1. Delay embedding: punkt i = [s[i], s[i-delay], s[i-2*delay]]
2. Vietoris-Rips + homologia uporczywa (wymiar 1 = petle)
3. Topology(t) = max_persistence najbardziej trwalego cyklu H1
   (0, jesli brak zadnego cyklu) - to jest SW1PerS score.
   Dodatkowo n_bars (liczba cykli w ogole) jako cecha pomocnicza.
"""

import numpy as np
import gudhi


def _delay_embed(window_vals, embed_dim, delay):
    n = len(window_vals)
    m = n - (embed_dim - 1) * delay
    if m <= embed_dim:  # za malo punktow na sensowny kompleks Ripsa
        return None
    pts = np.zeros((m, embed_dim))
    for d in range(embed_dim):
        pts[:, d] = window_vals[d * delay: d * delay + m]
    return pts


def _h1_features(pts):
    rc = gudhi.RipsComplex(points=pts)
    st = rc.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    h1 = np.array(st.persistence_intervals_in_dimension(1))
    if len(h1) == 0:
        return 0.0, 0
    finite = h1[np.isfinite(h1[:, 1])]
    if len(finite) == 0:
        return 0.0, 0
    pers = finite[:, 1] - finite[:, 0]
    return float(pers.max()), int(len(pers))


def topology_series(s, window=60, embed_dim=3, delay=3):
    """
    Kauzalny szereg "Topology(t)" (SW1PerS: max persistence H1) i
    n_bars (liczba cykli) dla sygnalu s. Dla i bez wystarczajacej
    historii zwraca 0.0 (brak wykrywalnej struktury cyklicznej - nie
    NaN, zeby dalo sie od razu uzywac w dalszych obliczeniach).
    """
    s = np.asarray(s, dtype=float)
    n = len(s)
    topology = np.zeros(n)
    n_bars = np.zeros(n)
    min_window = (embed_dim - 1) * delay + embed_dim + 1

    for i in range(n):
        lo = max(0, i - window + 1)
        seg = s[lo:i + 1]
        if len(seg) < min_window:
            continue
        pts = _delay_embed(seg, embed_dim, delay)
        if pts is None:
            continue
        mp, nb = _h1_features(pts)
        topology[i] = mp
        n_bars[i] = nb

    return dict(topology=topology, n_bars=n_bars)


def zscore_causal(x, min_hist=20):
    """Z-score liczony KAUZALNIE - w kazdym punkcie i uzywa wylacznie
    x[:i+1] (rozszerzajace sie okno historii), nie calego szeregu naraz
    (to drugie bylby lookaheadem: normalizacja "z przyszlosci")."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        if i < min_hist:
            continue
        hist = x[:i + 1]
        mu, sd = np.mean(hist), np.std(hist)
        out[i] = (x[i] - mu) / sd if sd > 1e-12 else 0.0
    return out
