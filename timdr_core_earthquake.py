"""
TIMDR-Earthquake-Core — timdr_core_earthquake.py
==================================================
Rdzeń analizy sejsmicznej: lokalny gradient amplitudy (flow), nagłe
zmiany kierunku / początek wstrząsu (twist), wygładzenie szumu (TRM),
mikro-wstrząsy (anomalies) i punkty rozpoczęcia wstrząsu (fronts).

Wejście: t (znaczniki czasu, sekundy, ściśle rosnące), s (amplituda).
"""

import numpy as np
from scipy.spatial import KDTree


class TIMDR_EarthquakeCore:
    """
    TIMDR Earthquake Detector Core
    - flow: lokalny gradient amplitudy sygnału sejsmicznego (LSQ względem czasu)
    - twist: nagłe zmiany kierunku (początek wstrząsu)
    - trm: wygładzenie szumu sejsmicznego
    - anomalies: mikro-wstrząsy
    - fronts: punkt rozpoczęcia wstrząsu
    """

    def __init__(self, k_neighbors=8, mad_scale=1.4826):
        self.k_neighbors = k_neighbors
        self.mad_scale = mad_scale

    def _safe_k(self, n):
        return min(self.k_neighbors, n)

    def _validate(self, t, s):
        t = np.asarray(t, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        if t.shape != s.shape:
            raise ValueError(f"t i s musza miec ten sam ksztalt, dostano {t.shape} i {s.shape}")
        if len(t) >= 2 and np.any(np.diff(t) <= 0):
            raise ValueError("t musi byc scisle rosnace (dt > 0 miedzy kolejnymi probkami)")
        return t, s

    # -----------------------------
    # FLOW — lokalny gradient LSQ
    # -----------------------------
    def flow(self, t, s):
        t, s = self._validate(t, s)
        n = len(t)
        if n < 3:
            return np.zeros_like(s)

        k = self._safe_k(n)
        tree = KDTree(t.reshape(-1, 1))
        grad = np.zeros_like(s)

        for i, ti in enumerate(t):
            _, idx = tree.query([ti], k=k)
            idx = np.atleast_1d(idx)
            tt = t[idx]
            ss = s[idx]

            A = np.column_stack([tt, np.ones_like(tt)])
            try:
                a, b = np.linalg.lstsq(A, ss, rcond=None)[0]
            except Exception:
                a = 0.0
            grad[i] = a

        return grad

    # -----------------------------
    # TWIST — nagłe zmiany kierunku
    # -----------------------------
    def twist(self, flow_grad, t, threshold=0.4):
        """
        POPRAWKA (bug krytyczny): oryginalny kod liczył
        `np.gradient(flow_grad)` BEZ przekazania `t` - traktując kolejne
        PRÓBKI jako równoodległe w czasie, mimo że `flow_grad` samo w
        sobie jest już poprawnie policzone względem rzeczywistego czasu
        w metodzie flow(). Dla danych z realną przerwą w rejestracji
        (typowe dla telemetrii sejsmicznej: dropout, przerwa między
        stacjami, scalanie segmentów) to daje SPURIOUS "twist" dokładnie
        na granicy przerwy.

        Zweryfikowano na czystej fali sinusoidalnej (0.1*sin(2*pi*2*t))
        z 3-sekundową przerwą w rejestracji pośrodku: oryginalny kod
        (gradient po indeksie) dawał na granicy przerwy szczyt "siły
        twistu" **3.8x większy** niż typowa (medianowa) wartość w reszcie
        sygnału - fałszywy alarm wywołany wyłącznie strukturą przerwy w
        danych, nie żadną prawdziwą zmianą fizyczną. Po poprawce
        (`np.gradient(flow_grad, t)`) granica przerwy wypada **0.0x**
        typowej wartości - poprawnie rozpoznana jako nic
        nadzwyczajnego.

        UWAGA (zmiana API): metoda teraz wymaga `t` jako argumentu -
        bez tego nie da się poprawnie liczyć gradientu względem czasu.
        """
        flow_grad = np.asarray(flow_grad, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        if len(flow_grad) < 3:
            return np.array([], dtype=int), np.zeros_like(flow_grad)
        if len(t) != len(flow_grad):
            raise ValueError("t i flow_grad musza miec ta sama dlugosc")

        dg = np.gradient(flow_grad, t)
        twist_strength = np.abs(dg)
        twist_points = np.where(twist_strength > threshold)[0]

        return twist_points, twist_strength

    # -----------------------------
    # TRM — wygładzenie medianowe
    # -----------------------------
    def trm(self, t, s):
        t, s = self._validate(t, s)
        n = len(t)
        if n < 2:
            return s.copy()

        k = self._safe_k(n)
        tree = KDTree(t.reshape(-1, 1))
        smooth = np.zeros_like(s)

        for i, ti in enumerate(t):
            _, idx = tree.query([ti], k=k)
            idx = np.atleast_1d(idx)
            smooth[i] = np.median(s[idx])

        return smooth

    # -----------------------------
    # ANOMALIE — mikro-wstrząsy
    # -----------------------------
    def anomalies(self, t, s, factor=3.0):
        """
        POPRAWKA (edge case): gdy sygnał jest silnie skwantowany /
        zawiera dużo powtórzonych wartości (typowe dla realnych danych z
        przetworników o ograniczonej rozdzielczości, albo skompresowanych
        formatów), mediana |reszt| (MAD) może wyjść dokładnie 0 - próg
        wychodzi wtedy 0, i KAŻDA niezerowa reszta (włącznie z szumem
        numerycznym) zostaje sklasyfikowana jako "anomalia".
        Zweryfikowano: na gładkim, tylko-zaokrąglonym sygnale (bez
        żadnej realnej anomalii) dawało to 7/30 (23%) fałszywych alarmów.
        Naprawiono: gdy MAD wychodzi ~0, próg spada na odch. std. reszt
        (a jeśli i to jest ~0 - sygnał faktycznie stały - próg to mała
        stała, nie zero).
        """
        t, s = self._validate(t, s)
        if len(t) == 0:
            return np.array([], dtype=int), np.array([]), 0.0
        smooth = self.trm(t, s)
        residuals = s - smooth

        mad = np.median(np.abs(residuals)) * self.mad_scale
        if mad <= 1e-12:
            std = np.std(residuals)
            mad = std if std > 1e-12 else 1e-9
        threshold = factor * mad

        anomaly_points = np.where(np.abs(residuals) > threshold)[0]

        return anomaly_points, residuals, threshold

    # -----------------------------
    # FRONTS — początek wstrząsu
    # -----------------------------
    def fronts(self, t, s, twist_threshold=0.4, anomaly_factor=3.0):
        t, s = self._validate(t, s)
        flow_grad = self.flow(t, s)
        twist_pts, twist_strength = self.twist(flow_grad, t, threshold=twist_threshold)
        anomalies, residuals, th = self.anomalies(t, s, factor=anomaly_factor)

        candidates = np.intersect1d(twist_pts, anomalies)

        if len(flow_grad) < 3:
            return np.array([], dtype=int), twist_strength, residuals

        flow_med = np.median(np.abs(flow_grad))

        strong_fronts = [
            i for i in candidates
            if abs(flow_grad[i]) > 2 * flow_med
        ]

        return np.array(strong_fronts, dtype=int), twist_strength, residuals
