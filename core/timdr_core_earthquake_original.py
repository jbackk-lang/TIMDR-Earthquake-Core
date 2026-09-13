import numpy as np
from scipy.spatial import KDTree

class TIMDR_EarthquakeCore:
    def __init__(self, k_neighbors=8, mad_scale=1.4826):
        self.k_neighbors = k_neighbors
        self.mad_scale = mad_scale

    def _safe_k(self, n):
        return min(self.k_neighbors, n)

    def flow(self, t, s):
        n = len(t)
        if n < 3:
            return np.zeros_like(s)
        k = self._safe_k(n)
        tree = KDTree(t.reshape(-1, 1))
        grad = np.zeros_like(s)
        for i, ti in enumerate(t):
            _, idx = tree.query([ti], k=k)
            tt = t[idx]
            ss = s[idx]
            A = np.column_stack([tt, np.ones_like(tt)])
            try:
                a, b = np.linalg.lstsq(A, ss, rcond=None)[0]
            except Exception:
                a = 0.0
            grad[i] = a
        return grad

    def twist(self, flow_grad, threshold=0.4):
        if len(flow_grad) < 3:
            return np.array([], dtype=int), np.zeros_like(flow_grad)
        dg = np.gradient(flow_grad)
        twist_strength = np.abs(dg)
        twist_points = np.where(twist_strength > threshold)[0]
        return twist_points, twist_strength

    def trm(self, t, s):
        n = len(t)
        if n < 2:
            return s.copy()
        k = self._safe_k(n)
        tree = KDTree(t.reshape(-1, 1))
        smooth = np.zeros_like(s)
        for i, ti in enumerate(t):
            _, idx = tree.query([ti], k=k)
            smooth[i] = np.median(s[idx])
        return smooth

    def anomalies(self, t, s, factor=3.0):
        smooth = self.trm(t, s)
        residuals = s - smooth
        mad = np.median(np.abs(residuals)) * self.mad_scale
        threshold = factor * mad
        anomaly_points = np.where(np.abs(residuals) > threshold)[0]
        return anomaly_points, residuals, threshold

    def fronts(self, t, s):
        flow_grad = self.flow(t, s)
        twist_pts, twist_strength = self.twist(flow_grad)
        anomalies, residuals, th = self.anomalies(t, s)
        candidates = np.intersect1d(twist_pts, anomalies)
        if len(flow_grad) < 3:
            return np.array([], dtype=int), twist_strength, residuals
        flow_med = np.median(np.abs(flow_grad))
        strong_fronts = [i for i in candidates if abs(flow_grad[i]) > 2 * flow_med]
        return np.array(strong_fronts, dtype=int), twist_strength, residuals
