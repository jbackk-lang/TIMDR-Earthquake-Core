import numpy as np
import json
import csv

class SeismicLoader:
    def __init__(self, normalize=True, detrend=True, clip_outliers=True):
        self.normalize = normalize
        self.detrend = detrend
        self.clip_outliers = clip_outliers

    def load_csv(self, path, t_col="t", s_col="s"):
        t = []
        s = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t.append(float(row[t_col]))
                    s.append(float(row[s_col]))
                except Exception:
                    continue
        t = np.asarray(t)
        s = np.asarray(s)
        return self._postprocess(t, s)

    def load_json_api(self, json_string):
        obj = json.loads(json_string)
        t = []
        s = []
        for item in obj.get("data", []):
            try:
                t.append(float(item["t"]))
                s.append(float(item["s"]))
            except Exception:
                continue
        return self._postprocess(np.asarray(t), np.asarray(s))

    def load_waveform(self, s, t=None):
        s = np.asarray(s, dtype=float)
        if t is None:
            t = np.arange(len(s), dtype=float)
        else:
            t = np.asarray(t, dtype=float)
        return self._postprocess(t, s)

    def _postprocess(self, t, s):
        if len(s) == 0:
            return t, s
        if self.detrend and len(s) > 2:
            slope = np.polyfit(t, s, 1)[0]
            s = s - slope * t
        if self.clip_outliers:
            mean = np.mean(s)
            std = np.std(s)
            s = np.clip(s, mean - 5 * std, mean + 5 * std)
        if self.normalize:
            maxv = np.max(np.abs(s))
            if maxv > 0:
                s = s / maxv
        return t, s
