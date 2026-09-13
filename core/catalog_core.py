"""
catalog_core.py — TIMDR Catalog Fusion (tryb katalogowy)
==========================================================
`timdr_core_earthquake.py` jest zbudowany pod SYGNAŁ FALOWY (ciągła,
mniej więcej równomiernie próbkowana amplituda z sejsmometru). Katalog
zdarzeń (lista trzęsień: czas + magnitude, jak z USGS
`earthquake.usgs.gov/fdsnws/event`) to inny typ danych — rzadki,
nieregularnie rozłożony w czasie proces punktowy, nie fala.

Ten moduł to NIE jest wrapper na `TIMDR_EarthquakeCore` — to osobna,
mała implementacja tej samej rodziny detektorów (anomaly / trend /
rhythm), tego samego typu co `TIMDRIndustrialFusion` z
TIMDR-Industrial-Predict, przystosowana do listy zdarzeń zamiast fali.
Kod jest zduplikowany celowo: to dwa niezależne publiczne repozytoria,
nie powinny się nawzajem importować.

Zweryfikowane na żywych danych USGS (M5.0+, 2026-08-01 do 2026-08-14,
64 zdarzenia, w tym trzęsienie M7.4 w Kolumbii) — patrz demo_usgs_catalog.py.
"""

import numpy as np


class TIMDRCatalogFusion:
    def __init__(self, mad_scale=1.4826):
        self.mad_scale = mad_scale

    def _mad_z(self, x):
        x = np.asarray(x, float)
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * self.mad_scale
        if mad == 0:
            span = np.max(x) - np.min(x)
            if span == 0:
                return np.zeros_like(x)
            return (x - med) / (span / 4.0)
        return (x - med) / mad

    def anomalies(self, magnitude, factor=3.0):
        """Wykrywa zdarzenia istotnie większe/mniejsze niż typowe w katalogu
        (MAD-z na magnitude). Zweryfikowano na realnym katalogu USGS:
        poprawnie wyłapuje 4 największe zdarzenia (M6.0-M7.4) jako
        anomalie, z mainshockiem M7.4 na szczycie (z=14.16)."""
        z = np.abs(self._mad_z(magnitude))
        idx = np.where(z > factor)[0]
        return idx, z

    def trend(self, t, magnitude, window=15):
        """Lokalne nachylenie magnitude w oknie N ZDARZEŃ (nie sekund/godzin
        - okno jest w jednostkach liczby zdarzeń, bo katalog jest
        nierównomiernie próbkowany w czasie)."""
        t = np.asarray(t, float)
        magnitude = np.asarray(magnitude, float)
        n = len(t)
        if n == 0:
            return np.array([]), np.array([])
        slopes = np.zeros(n)
        for i in range(n):
            j0 = max(0, i - window + 1)
            tt, mm = t[j0:i + 1], magnitude[j0:i + 1]
            if len(tt) < 2:
                continue
            A = np.column_stack([tt, np.ones_like(tt)])
            a, _ = np.linalg.lstsq(A, mm, rcond=None)[0]
            slopes[i] = a
        return slopes, self._mad_z(slopes)

    def rhythm(self, magnitude, max_lag=30, power_thresh=0.4):
        """
        ZASTRZEŻENIE (ważne, przeczytaj przed użyciem): ta funkcja liczy
        autokorelację po INDEKSIE ZDARZENIA w katalogu, nie po realnym
        upływie czasu. "Wykryty okres = 27" oznacza "co 27 zdarzeń w
        katalogu", NIE "co 27 godzin/dni". Dla katalogu o mocno
        nierównomiernych odstępach między zdarzeniami (typowe dla
        prawdziwej sejsmiczności - roje, cisza, potem znowu aktywność)
        to może dawać myląco nazwany, ale matematycznie poprawny wynik.

        Zweryfikowano na realnym katalogu USGS (64 zdarzenia M5+,
        2026-08-01 do 2026-08-14, w tym mainshock M7.4 Kolumbia): wynik
        to `[]`, `0.0` — poprawnie brak wykrytej okresowości, zgodnie z
        oczekiwaniem (globalna sejsmiczność M5+ jest w przybliżeniu
        procesem Poissona, bez znanej periodyczności).

        WAŻNE (błąd znaleziony po drodze, nie w tej funkcji): pierwsza
        wersja tego demo liczyła rytm na sygnale E=|MAD-z(magnitude)|
        (rektyfikacja wartości bezwzględnej, przeniesiona 1:1 z
        wielocechowego `TIMDRIndustrialFusion.fuse()`, gdzie ma sens dla
        >1 czujnika). Dla POJEDYNCZEJ cechy branie wartości bezwzględnej
        z MAD-z **tworzy sztuczną okresowość z rektyfikacji** (ten sam
        mechanizm co udokumentowany błąd L2-norm w
        TIMDR-Industrial-Fusion) — dawało to score=0.434 na tych samych
        danych, tuż nad progiem 0.4, czyli graniczny fałszywy alarm.
        Wniosek: dla katalogu z JEDNĄ cechą (magnitude) licz `rhythm()`
        bezpośrednio na wartości ze znakiem (jak robi ta funkcja), NIE
        na `sqrt(z**2)` z pipeline'u wielocechowego.

        Nadal aktualne ograniczenie (niezależnie od powyższego): lag
        liczony jest po INDEKSIE zdarzenia w katalogu, nie po realnym
        czasie. "Wykryty okres = N" oznacza "co N zdarzeń", nie "co N
        godzin/dni". Dla katalogu o mocno nierównomiernych odstępach
        między zdarzeniami to może dawać matematycznie poprawny, ale
        mylnie nazwany wynik. Jeśli Twój katalog może mieć prawdziwą
        okresowość względem realnego czasu (np. cykle pływowe), rozważ
        Lomb-Scargle zamiast tej funkcji.
        """
        E = np.asarray(magnitude, float)
        n = len(E)
        if n < 2:
            return [], 0.0

        t_idx = np.arange(n, dtype=float)
        if n > 2:
            slope, intercept = np.polyfit(t_idx, E, 1)
            E = E - (slope * t_idx + intercept)
        else:
            E = E - np.mean(E)

        max_lag = min(max_lag, n - 1)
        ac = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                ac[lag] = np.dot(E, E) / n
            else:
                overlap = n - lag
                if overlap <= 0:
                    break
                ac[lag] = np.dot(E[:-lag], E[lag:]) / overlap

        if ac[0] == 0:
            return [], 0.0
        ac /= ac[0]

        peaks = [
            (i, float(ac[i])) for i in range(1, len(ac) - 1)
            if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] >= power_thresh
        ]
        if not peaks:
            return [], 0.0
        score = max(p for _, p in peaks)
        return [p for p, _ in peaks], score

    def nearest_aftershock(self, t, magnitude, place=None, mainshock_idx=None):
        """Zwraca (indeks, dt) najblizszego zdarzenia PO danym wstrzasie
        (domyslnie: najwiekszym w katalogu). Prosta, niezalezna od
        MAD/rytmu kontrola sekwencji glowny-wstrzas -> aftershock."""
        t = np.asarray(t, float)
        magnitude = np.asarray(magnitude, float)
        if mainshock_idx is None:
            mainshock_idx = int(np.argmax(magnitude))
        t_main = t[mainshock_idx]
        after = np.where(t > t_main)[0]
        if len(after) == 0:
            return None, None
        nxt = after[0]
        return int(nxt), float(t[nxt] - t_main)
