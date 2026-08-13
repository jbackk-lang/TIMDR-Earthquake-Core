"""
TIMDR-Earthquake-Core — GUI (Tkinter)
========================================
Prosty, samodzielny interfejs graficzny do TIMDR_EarthquakeCore +
SeismicLoader. Uruchom przez `run.bat` (Windows) albo bezpośrednio:
`python gui_app.py`.

Wymaga tylko standardowej biblioteki Pythona (tkinter) + numpy/scipy/
matplotlib (instalowane automatycznie przez run.bat).
"""

import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from timdr_core_earthquake import TIMDR_EarthquakeCore
from seismic_loader import SeismicLoader


def make_demo_signal(n=400, seed=0):
    """Syntetyczny sygnal demo: szum tla + narastajacy wstrzas + pojedynczy glitch."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * 0.01
    s = rng.normal(0, 0.05, n) + 0.002 * t
    start = n // 2 - 20
    ramp = np.concatenate([np.linspace(0, 3.0, 20), np.linspace(3.0, 0, 20)])
    s[start:start + 40] += ramp
    glitch_idx = max(10, start - 60)
    s[glitch_idx] = 8.0
    return t, s


class TimdrEarthquakeGUI(tk.Tk):
    COLORS = {
        "bg": "#f4f6f8",
        "accent": "#1e88e5",
        "danger": "#e53935",
        "ok": "#43a047",
        "warn": "#fb8c00",
    }

    def __init__(self):
        super().__init__()
        self.title("TIMDR-Earthquake-Core")
        self.geometry("1180x760")
        self.configure(bg=self.COLORS["bg"])
        self.minsize(900, 600)

        self.t_raw = None
        self.s_raw = None
        self.t = None
        self.s = None

        self._build_style()
        self._build_layout()

    # ------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=self.COLORS["bg"])
        style.configure("TLabel", background=self.COLORS["bg"], font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=self.COLORS["bg"], font=("Segoe UI", 15, "bold"))
        style.configure("Section.TLabel", background=self.COLORS["bg"], font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("TLabelframe", background=self.COLORS["bg"], font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background=self.COLORS["bg"], font=("Segoe UI", 10, "bold"))
        style.configure("TEntry", padding=4)

    # ------------------------------------------------------------
    def _build_layout(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=16, pady=(14, 6))
        ttk.Label(header, text="🌍 TIMDR-Earthquake-Core", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="  flow · twist · TRM · anomalie · fronty",
                  foreground="#607d8b").pack(side="left", padx=(4, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=16, pady=8)

        left = ttk.Frame(body, width=300)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        self._build_controls(left)
        self._build_plot(right)

        self.status_var = tk.StringVar(value="Gotowy. Wczytaj CSV albo wygeneruj sygnał demo.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken",
                                anchor="w", padding=(8, 4))
        status_bar.pack(fill="x", side="bottom")

    # ------------------------------------------------------------
    def _build_controls(self, parent):
        data_frame = ttk.Labelframe(parent, text="1. Dane wejściowe", padding=10)
        data_frame.pack(fill="x", pady=(0, 10))

        ttk.Button(data_frame, text="📂 Wczytaj CSV...", command=self.on_load_csv).pack(fill="x", pady=3)
        ttk.Button(data_frame, text="🎲 Wygeneruj sygnał demo", command=self.on_load_demo).pack(fill="x", pady=3)

        self.t_col_var = tk.StringVar(value="t")
        self.s_col_var = tk.StringVar(value="s")
        col_row = ttk.Frame(data_frame)
        col_row.pack(fill="x", pady=(6, 0))
        ttk.Label(col_row, text="kolumna t:").grid(row=0, column=0, sticky="w")
        ttk.Entry(col_row, textvariable=self.t_col_var, width=8).grid(row=0, column=1, padx=4)
        ttk.Label(col_row, text="kolumna s:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Entry(col_row, textvariable=self.s_col_var, width=8).grid(row=0, column=3, padx=4)

        self.data_info_var = tk.StringVar(value="Brak wczytanych danych.")
        ttk.Label(data_frame, textvariable=self.data_info_var, wraplength=250,
                  foreground="#607d8b").pack(fill="x", pady=(6, 0))

        prep_frame = ttk.Labelframe(parent, text="2. Preprocessing (SeismicLoader)", padding=10)
        prep_frame.pack(fill="x", pady=(0, 10))

        self.detrend_var = tk.BooleanVar(value=True)
        self.despike_var = tk.BooleanVar(value=True)
        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prep_frame, text="Detrend (usuń dryf)", variable=self.detrend_var).pack(anchor="w")
        ttk.Checkbutton(prep_frame, text="Despike (usuń odosobnione skoki)", variable=self.despike_var).pack(anchor="w")
        ttk.Checkbutton(prep_frame, text="Normalizacja amplitudy", variable=self.normalize_var).pack(anchor="w")

        param_frame = ttk.Labelframe(parent, text="3. Parametry detekcji", padding=10)
        param_frame.pack(fill="x", pady=(0, 10))

        self.k_var = tk.StringVar(value="8")
        # UWAGA: domyslny prog "0.4" z timdr_core_earthquake.py byl
        # skalibrowany na innej skali danych - dla znormalizowanego
        # sygnalu demo (amplituda w [-1,1], 100Hz) dawal >90% probek
        # jako "twist" (median |twist| na tym sygnale ~10, nie ~0.4).
        # Wartosc "20" ponizej dobrana tak, by demo dawalo czytelny,
        # ilustracyjny wynik - i tak wymaga przestrojenia pod realne dane.
        self.twist_thr_var = tk.StringVar(value="20")
        self.anomaly_factor_var = tk.StringVar(value="3.0")

        self._param_row(param_frame, "k_neighbors:", self.k_var)
        self._param_row(param_frame, "próg twist:", self.twist_thr_var)
        self._param_row(param_frame, "factor anomalii:", self.anomaly_factor_var)

        ttk.Button(parent, text="▶  Uruchom analizę", style="Accent.TButton",
                   command=self.on_analyze).pack(fill="x", pady=(4, 10))

        results_frame = ttk.Labelframe(parent, text="Wyniki", padding=10)
        results_frame.pack(fill="both", expand=True)
        self.results_text = tk.Text(results_frame, height=12, width=32, font=("Consolas", 9),
                                     bg="white", relief="flat", wrap="word")
        self.results_text.pack(fill="both", expand=True)
        self.results_text.insert("1.0", "Wyniki analizy pojawią się tutaj.")
        self.results_text.configure(state="disabled")

    def _param_row(self, parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=14).pack(side="left")
        ttk.Entry(row, textvariable=var, width=10).pack(side="left")

    # ------------------------------------------------------------
    def _build_plot(self, parent):
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(7.5, 7), dpi=100)
        self.axes = self.fig.subplots(4, 1, sharex=True)
        self.fig.subplots_adjust(hspace=0.35, left=0.09, right=0.98, top=0.96, bottom=0.07)
        self._draw_placeholder()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    def _draw_placeholder(self):
        titles = ["Sygnał", "flow (lokalny gradient)", "|twist|", "residuum + anomalie/fronty"]
        for ax, title in zip(self.axes, titles):
            ax.clear()
            ax.set_ylabel(title, fontsize=9)
            ax.grid(alpha=0.2)
        self.axes[-1].set_xlabel("czas (s)")
        self.axes[0].set_title("Wczytaj dane i uruchom analizę", fontsize=10, color="#90a4ae")

    # ------------------------------------------------------------
    def on_load_csv(self):
        path = filedialog.askopenfilename(
            title="Wybierz plik CSV",
            filetypes=[("CSV", "*.csv"), ("Wszystkie pliki", "*.*")],
        )
        if not path:
            return
        try:
            loader = SeismicLoader(normalize=False, detrend=False, clip_outliers=False)
            t, s = loader.load_csv(path, t_col=self.t_col_var.get(), s_col=self.s_col_var.get())
            if len(t) == 0:
                raise ValueError("Plik nie zawiera zadnych poprawnych wierszy.")
            self.t_raw, self.s_raw = t, s
            self.data_info_var.set(f"Wczytano {len(t)} próbek z {path.split('/')[-1].split(chr(92))[-1]}")
            self.status_var.set(f"Wczytano {len(t)} próbek. Kliknij 'Uruchom analizę'.")
        except Exception as exc:
            messagebox.showerror("Błąd wczytywania CSV", str(exc))

    def on_load_demo(self):
        self.t_raw, self.s_raw = make_demo_signal()
        self.data_info_var.set(f"Wygenerowano sygnał demo: {len(self.t_raw)} próbek "
                                f"(wstrząs + izolowany glitch czujnika)")
        self.status_var.set("Wygenerowano dane demo. Kliknij 'Uruchom analizę'.")

    # ------------------------------------------------------------
    def on_analyze(self):
        if self.t_raw is None or self.s_raw is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj CSV albo wygeneruj sygnał demo.")
            return
        try:
            k = int(self.k_var.get())
            twist_thr = float(self.twist_thr_var.get())
            anomaly_factor = float(self.anomaly_factor_var.get())
        except ValueError:
            messagebox.showerror("Błędne parametry", "k_neighbors, próg twist i factor anomalii musza byc liczbami.")
            return

        try:
            loader = SeismicLoader(
                normalize=self.normalize_var.get(),
                detrend=self.detrend_var.get(),
                clip_outliers=self.despike_var.get(),
            )
            t, s = loader.load_waveform(self.s_raw.copy(), self.t_raw.copy())
            self.t, self.s = t, s

            core = TIMDR_EarthquakeCore(k_neighbors=k)
            flow_grad = core.flow(t, s)
            twist_pts, twist_strength = core.twist(flow_grad, t, threshold=twist_thr)
            anomaly_pts, residuals, th = core.anomalies(t, s, factor=anomaly_factor)
            fronts, _, _ = core.fronts(t, s, twist_threshold=twist_thr, anomaly_factor=anomaly_factor)

            self._plot_results(t, s, flow_grad, twist_strength, twist_pts, residuals, anomaly_pts, fronts, twist_thr)
            self._show_results(t, twist_pts, anomaly_pts, fronts, th)
            self.status_var.set(
                f"Analiza zakończona: {len(twist_pts)} pkt. twist, {len(anomaly_pts)} anomalii, {len(fronts)} frontów."
            )
        except Exception:
            messagebox.showerror("Błąd analizy", traceback.format_exc(limit=3))

    # ------------------------------------------------------------
    def _plot_results(self, t, s, flow_grad, twist_strength, twist_pts, residuals, anomaly_pts, fronts, twist_thr):
        for ax in self.axes:
            ax.clear()
            ax.grid(alpha=0.2)

        self.axes[0].plot(t, s, color=self.COLORS["accent"], lw=0.9)
        self.axes[0].set_ylabel("sygnał", fontsize=9)

        self.axes[1].plot(t, flow_grad, color="#8e24aa", lw=0.9)
        self.axes[1].set_ylabel("flow", fontsize=9)

        self.axes[2].plot(t, twist_strength, color=self.COLORS["warn"], lw=0.9)
        self.axes[2].axhline(twist_thr, color="gray", ls="--", lw=1)
        if len(twist_pts):
            self.axes[2].scatter(t[twist_pts], twist_strength[twist_pts],
                                  color=self.COLORS["danger"], s=14, zorder=5, label="twist")
            self.axes[2].legend(fontsize=7, loc="upper right")
        self.axes[2].set_ylabel("|twist|", fontsize=9)

        self.axes[3].plot(t, residuals, color=self.COLORS["ok"], lw=0.9)
        if len(anomaly_pts):
            self.axes[3].scatter(t[anomaly_pts], residuals[anomaly_pts],
                                  color=self.COLORS["danger"], s=14, zorder=5, label="anomalia")
        if len(fronts):
            self.axes[3].scatter(t[fronts], residuals[fronts],
                                  color="black", s=70, marker="*", zorder=6, label="front")
        if len(anomaly_pts) or len(fronts):
            self.axes[3].legend(fontsize=7, loc="upper right")
        self.axes[3].set_ylabel("residuum", fontsize=9)
        self.axes[3].set_xlabel("czas (s)")

        self.fig.suptitle("")
        self.canvas.draw()

    def _show_results(self, t, twist_pts, anomaly_pts, fronts, threshold):
        lines = []
        lines.append(f"Liczba próbek: {len(t)}")
        lines.append(f"Zakres czasu: {t[0]:.3f} - {t[-1]:.3f} s")
        lines.append("")
        lines.append(f"Punkty twist: {len(twist_pts)}")
        lines.append(f"Anomalie (mikro-wstrząsy): {len(anomaly_pts)}")
        lines.append(f"Próg anomalii (MAD): {threshold:.4f}")
        lines.append("")
        lines.append(f"Fronty (start wstrząsu): {len(fronts)}")
        if len(fronts):
            first = fronts[0]
            lines.append(f"  pierwszy front: t={t[first]:.3f}s (idx={first})")
            if len(fronts) > 1:
                lines.append(f"  ostatni front:  t={t[fronts[-1]]:.3f}s (idx={fronts[-1]})")
        else:
            lines.append("  (brak wykrytych frontów)")

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "\n".join(lines))
        self.results_text.configure(state="disabled")


def main():
    app = TimdrEarthquakeGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
