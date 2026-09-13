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
        ttk.Label(header, text="  flow · twist · TRM · anomalie · fronty · STA/LTA",
                  foreground="#607d8b").pack(side="left", padx=(4, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=16, pady=8)

        left_container = ttk.Frame(body, width=300)
        left_container.pack(side="left", fill="y", padx=(0, 12))
        left_container.pack_propagate(False)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        left = self._build_scrollable_left(left_container)
        self._build_controls(left)
        self._build_plot(right)

        self.status_var = tk.StringVar(value="Gotowy. Wczytaj CSV albo wygeneruj sygnał demo.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken",
                                anchor="w", padding=(8, 4))
        status_bar.pack(fill="x", side="bottom")

    def _build_scrollable_left(self, container):
        """Lewa kolumna (dane/preprocessing/parametry/wyniki) w wielu
        wypadkach nie mieści się w wysokości okna (np. mniejszy monitor,
        więcej pól po dodaniu STA/LTA) - bez przewijania przycisk
        'Uruchom analizę' i panel wyników potrafiły wypaść poza widoczny
        obszar. Owijamy zawartość w Canvas + Scrollbar, żeby zawsze
        dało się do nich przewinąć niezależnie od wysokości okna."""
        canvas = tk.Canvas(container, bg=self.COLORS["bg"], highlightthickness=0,
                            width=284)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw", width=284)

        def _on_inner_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(inner_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            delta = -1 * (event.delta // 120) if event.delta else (1 if event.num == 5 else -1)
            canvas.yview_scroll(int(delta), "units")

        canvas.bind("<Enter>", lambda _e: (canvas.bind_all("<MouseWheel>", _on_mousewheel),
                                            canvas.bind_all("<Button-4>", _on_mousewheel),
                                            canvas.bind_all("<Button-5>", _on_mousewheel)))
        canvas.bind("<Leave>", lambda _e: (canvas.unbind_all("<MouseWheel>"),
                                            canvas.unbind_all("<Button-4>"),
                                            canvas.unbind_all("<Button-5>")))

        return inner

    # ------------------------------------------------------------
    def _build_controls(self, parent):
        # Przycisk analizy i pasek statusu na samej górze, żeby zawsze
        # były widoczne bez przewijania - reszta (dane/preprocessing/
        # parametry/wyniki) jest niżej, w przewijalnej kolumnie.
        ttk.Button(parent, text="▶  Uruchom analizę", style="Accent.TButton",
                   command=self.on_analyze).pack(fill="x", pady=(0, 10))

        data_frame = ttk.Labelframe(parent, text="1. Dane wejściowe", padding=8)
        data_frame.pack(fill="x", pady=(0, 8))

        btn_row = ttk.Frame(data_frame)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="📂 Wczytaj CSV...", command=self.on_load_csv).pack(
            side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(btn_row, text="🎲 Demo", command=self.on_load_demo).pack(
            side="left", expand=True, fill="x", padx=(3, 0))

        self.t_col_var = tk.StringVar(value="t")
        self.s_col_var = tk.StringVar(value="s")
        col_row = ttk.Frame(data_frame)
        col_row.pack(fill="x", pady=(6, 0))
        ttk.Label(col_row, text="kol. t:").grid(row=0, column=0, sticky="w")
        ttk.Entry(col_row, textvariable=self.t_col_var, width=6).grid(row=0, column=1, padx=4)
        ttk.Label(col_row, text="kol. s:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Entry(col_row, textvariable=self.s_col_var, width=6).grid(row=0, column=3, padx=4)

        self.data_info_var = tk.StringVar(value="Brak wczytanych danych.")
        ttk.Label(data_frame, textvariable=self.data_info_var, wraplength=250,
                  foreground="#607d8b").pack(fill="x", pady=(4, 0))

        prep_frame = ttk.Labelframe(parent, text="2. Preprocessing", padding=8)
        prep_frame.pack(fill="x", pady=(0, 8))

        self.detrend_var = tk.BooleanVar(value=True)
        self.despike_var = tk.BooleanVar(value=True)
        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prep_frame, text="Detrend", variable=self.detrend_var).pack(anchor="w")
        ttk.Checkbutton(prep_frame, text="Despike", variable=self.despike_var).pack(anchor="w")
        ttk.Checkbutton(prep_frame, text="Normalizacja amplitudy", variable=self.normalize_var).pack(anchor="w")

        param_frame = ttk.Labelframe(parent, text="3. Parametry detekcji", padding=8)
        param_frame.pack(fill="x", pady=(0, 8))

        self.k_var = tk.StringVar(value="8")
        # UWAGA: domyslny prog "0.4" z timdr_core_earthquake.py byl
        # skalibrowany na innej skali danych - dla znormalizowanego
        # sygnalu demo (amplituda w [-1,1], 100Hz) dawal >90% probek
        # jako "twist" (median |twist| na tym sygnale ~10, nie ~0.4).
        # Wartosc "20" ponizej dobrana tak, by demo dawalo czytelny,
        # ilustracyjny wynik - i tak wymaga przestrojenia pod realne dane.
        self.twist_thr_var = tk.StringVar(value="20")
        self.anomaly_factor_var = tk.StringVar(value="3.0")

        self._param_grid(param_frame, [
            ("k_neighbors:", self.k_var), ("próg twist:", self.twist_thr_var),
            ("factor anom.:", self.anomaly_factor_var),
        ])

        stalta_frame = ttk.Labelframe(parent, text="4. STA/LTA (picker)", padding=8)
        stalta_frame.pack(fill="x", pady=(0, 8))

        # Domyslne 25/100 probek = 0.25s/1.0s przy 100Hz (jak w demo.py,
        # zweryfikowane wobec ObsPy - patrz README). Progi 3.0/1.0 to
        # standardowe wartosci startowe dla classic STA/LTA - wymagaja
        # przestrojenia pod realny szum tla i czulosc sensora.
        self.nsta_var = tk.StringVar(value="25")
        self.nlta_var = tk.StringVar(value="100")
        self.thr_on_var = tk.StringVar(value="3.0")
        self.thr_off_var = tk.StringVar(value="1.0")

        self._param_grid(stalta_frame, [
            ("nsta:", self.nsta_var), ("nlta:", self.nlta_var),
            ("próg wł.:", self.thr_on_var), ("próg wył.:", self.thr_off_var),
        ])

        results_frame = ttk.Labelframe(parent, text="Wyniki", padding=8)
        results_frame.pack(fill="both", expand=True, pady=(0, 4))
        self.results_text = tk.Text(results_frame, height=12, width=32, font=("Consolas", 9),
                                     bg="white", relief="flat", wrap="word")
        self.results_text.pack(fill="both", expand=True)
        self.results_text.insert("1.0", "Wyniki analizy pojawią się tutaj.")
        self.results_text.configure(state="disabled")

    def _param_grid(self, parent, label_var_pairs, per_row=2):
        """Kompaktowa siatka pól parametrów, `per_row` na wiersz -
        zastępuje wcześniejsze układanie jednego pola pod drugim, które
        przy większej liczbie parametrów (STA/LTA dodało 4 kolejne pola)
        zbytnio wydłużało lewą kolumnę."""
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        for i, (label, var) in enumerate(label_var_pairs):
            r, c = divmod(i, per_row)
            cell = ttk.Frame(grid)
            cell.grid(row=r, column=c, sticky="w", padx=(0, 10), pady=2)
            ttk.Label(cell, text=label, width=10).pack(side="left")
            ttk.Entry(cell, textvariable=var, width=7).pack(side="left")

    # ------------------------------------------------------------
    def _build_plot(self, parent):
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(7.5, 8.5), dpi=100)
        self.axes = self.fig.subplots(5, 1, sharex=True)
        self.fig.subplots_adjust(hspace=0.4, left=0.09, right=0.98, top=0.96, bottom=0.06)
        self._draw_placeholder()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    def _draw_placeholder(self):
        titles = ["Sygnał", "flow (lokalny gradient)", "|twist|", "residuum + anomalie/fronty",
                  "STA/LTA ratio"]
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
            nsta = int(self.nsta_var.get())
            nlta = int(self.nlta_var.get())
            thr_on = float(self.thr_on_var.get())
            thr_off = float(self.thr_off_var.get())
        except ValueError:
            messagebox.showerror(
                "Błędne parametry",
                "k_neighbors, próg twist, factor anomalii, nsta, nlta, próg włącz i próg wyłącz musza byc liczbami."
            )
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
            ratio = core.sta_lta(s, nsta, nlta)
            onsets = core.trigger_onset(ratio, thr_on=thr_on, thr_off=thr_off)

            self._plot_results(t, s, flow_grad, twist_strength, twist_pts, residuals, anomaly_pts, fronts,
                                twist_thr, ratio, onsets, thr_on, thr_off)
            self._show_results(t, twist_pts, anomaly_pts, fronts, th, ratio, onsets)
            self.status_var.set(
                f"Analiza zakończona: {len(twist_pts)} pkt. twist, {len(anomaly_pts)} anomalii, "
                f"{len(fronts)} frontów, {len(onsets)} wyzwoleń STA/LTA."
            )
        except Exception:
            messagebox.showerror("Błąd analizy", traceback.format_exc(limit=3))

    # ------------------------------------------------------------
    def _plot_results(self, t, s, flow_grad, twist_strength, twist_pts, residuals, anomaly_pts, fronts,
                       twist_thr, ratio, onsets, thr_on, thr_off):
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

        self.axes[4].plot(t, ratio, color="#00838f", lw=0.9)
        self.axes[4].axhline(thr_on, color=self.COLORS["danger"], ls="--", lw=1, label="próg włącz")
        self.axes[4].axhline(thr_off, color=self.COLORS["warn"], ls="--", lw=1, label="próg wyłącz")
        for k_evt, (i_start, i_end) in enumerate(onsets):
            self.axes[4].axvspan(t[i_start], t[i_end], color=self.COLORS["danger"], alpha=0.15,
                                  label="wyzwolenie" if k_evt == 0 else None)
        self.axes[4].legend(fontsize=7, loc="upper right")
        self.axes[4].set_ylabel("STA/LTA", fontsize=9)
        self.axes[4].set_xlabel("czas (s)")

        self.fig.suptitle("")
        self.canvas.draw()

    def _show_results(self, t, twist_pts, anomaly_pts, fronts, threshold, ratio, onsets):
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

        lines.append("")
        lines.append(f"Wyzwolenia STA/LTA: {len(onsets)}")
        if len(onsets):
            i_start, i_end = onsets[0]
            lines.append(f"  pierwsze: t={t[i_start]:.3f}s -> {t[i_end]:.3f}s (idx={i_start}-{i_end})")
            if len(onsets) > 1:
                i_start, i_end = onsets[-1]
                lines.append(f"  ostatnie: t={t[i_start]:.3f}s -> {t[i_end]:.3f}s (idx={i_start}-{i_end})")
        else:
            lines.append("  (brak wyzwoleń - sprawdź progi włącz/wyłącz albo nsta/nlta)")

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "\n".join(lines))
        self.results_text.configure(state="disabled")


def main():
    app = TimdrEarthquakeGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
