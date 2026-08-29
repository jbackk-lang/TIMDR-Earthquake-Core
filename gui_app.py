"""
TIMDR-Earthquake-Core — GUI (Tkinter)
========================================
Simple, self-contained graphical interface for TIMDR_EarthquakeCore +
SeismicLoader. Run via `run.bat` (Windows) or directly:
`python gui_app.py`.

Requires only the Python standard library (tkinter) plus numpy/scipy/
matplotlib (installed automatically by run.bat).

IMPORTANT — scope of this tool: everything here DETECTS and CLASSIFIES
features already present in a signal you feed it (an onset that already
started, an anomaly that already happened, a picker trigger on energy
that's already there). None of it FORECASTS an earthquake before it
occurs. Short-term earthquake prediction (a reliable precursor signal
that fires meaningfully before rupture) is an open, unsolved problem in
seismology — this tool does not attempt it and nothing below should be
read as doing so.
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


# ============================================================
# Demo scenarios — several synthetic signals illustrating
# different situations the core is meant to tell apart.
# ============================================================

def demo_earthquake(n=400, seed=0):
    """Background noise + a real ramping-up-then-down shock + one
    isolated sensor glitch (single-sample outlier)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * 0.01
    s = rng.normal(0, 0.05, n) + 0.002 * t
    start = n // 2 - 20
    ramp = np.concatenate([np.linspace(0, 3.0, 20), np.linspace(3.0, 0, 20)])
    s[start:start + 40] += ramp
    glitch_idx = max(10, start - 60)
    s[glitch_idx] = 8.0
    return t, s


def demo_dropout(n=400, seed=1):
    """A sensor gets stuck at a constant value for a while (telemetry /
    hardware dropout), then recovers - not a real ground motion."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * 0.01
    s = rng.normal(0, 0.05, n)
    stuck_start = n // 2 - 10
    s[stuck_start:stuck_start + 40] = 1.2
    return t, s


def demo_drift(n=500, seed=2):
    """Level rises gradually over ~1.2s (not a sudden jump) and stays at
    the new level - e.g. a slow-building swarm rather than a single
    sharp onset."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * 0.01
    s = rng.normal(0, 0.03, n)
    ramp_start, ramp_len = 200, 120
    s[ramp_start:ramp_start + ramp_len] += np.linspace(0, 4.0, ramp_len)
    s[ramp_start + ramp_len:] += 4.0
    return t, s


def demo_noise(n=400, seed=3):
    """Pure background noise, no event at all - useful to check the
    detector stays quiet (no false fronts/triggers) on a boring signal."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * 0.01
    s = rng.normal(0, 0.05, n)
    return t, s


DEMO_SCENARIOS = {
    "Earthquake + sensor glitch": demo_earthquake,
    "Stuck sensor (dropout)": demo_dropout,
    "Gradual drift (no sudden onset)": demo_drift,
    "Background noise only (no event)": demo_noise,
}

ANOMALY_TYPE_COLORS = {
    "impuls": "#e53935",
    "spike": "#fb8c00",
    "step": "#8e24aa",
    "drift": "#6d4c41",
    "dropout": "#455a64",
}


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
        self.geometry("1220x780")
        self.configure(bg=self.COLORS["bg"])
        self.minsize(920, 620)

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
        header.pack(fill="x", padx=16, pady=(14, 2))
        ttk.Label(header, text="🌍 TIMDR-Earthquake-Core", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="  flow · twist · TRM (median/adaptive/savgol) · anomalies · "
                                "classify · fronts · STA/LTA · hybrid trigger",
                  foreground="#607d8b").pack(side="left", padx=(4, 0))

        note = ttk.Frame(self)
        note.pack(fill="x", padx=16, pady=(0, 6))
        ttk.Label(note, text="Detection & classification of features already in the signal — "
                              "NOT earthquake prediction/forecasting.",
                  foreground="#b71c1c", font=("Segoe UI", 8, "italic")).pack(side="left")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=16, pady=8)

        left_container = ttk.Frame(body, width=345)
        left_container.pack(side="left", fill="y", padx=(0, 12))
        left_container.pack_propagate(False)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        left = self._build_scrollable_left(left_container)
        self._build_controls(left)
        self._build_plot(right)

        self.status_var = tk.StringVar(value="Ready. Load a CSV file or generate a demo signal.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken",
                                anchor="w", padding=(8, 4))
        status_bar.pack(fill="x", side="bottom")

    def _build_scrollable_left(self, container):
        """The left column (data/preprocessing/parameters/results) often
        doesn't fit in the window height (e.g. smaller monitor, more
        fields after adding STA/LTA + hybrid trigger) - without
        scrolling, the 'Run analysis' button and results panel could end
        up outside the visible area. We wrap the content in a
        Canvas + Scrollbar so it's always reachable regardless of window
        height."""
        # POPRAWKA (panel parametrow chowal sie czesciowo za paskiem
        # przewijania): canvas i inner-frame mialy NIEZALEZNIE zahardkodowana
        # szerokosc 294px, dobrana recznie pod zalozenie 96 DPI / brak
        # skalowania Windows. Przy skalowaniu DPI >100% (bardzo typowe na
        # laptopach) rzeczywista szerokosc kontenera i paska przewijania w
        # fizycznych pikselach jest inna niz przyjeta w tej stalej - canvas
        # bywal SZERSZY niz realnie dostepna przestrzen obok paska, wiec
        # jego prawa krawedz (a z nia czesc etykiet/pol) renderowala sie
        # POD paskiem przewijania. Naprawiono podwojnie:
        #  1. Jawne wlaczenie DPI-awareness dla calego procesu (patrz
        #     main() nizej) - Tkinter bez tego jest DPI-nieswiadomy na
        #     Windows i caly interfejs jest wtedy skalowany "na sile" przez
        #     system operacyjny, co psuje dokladnie tego typu wyliczenia.
        #  2. Usuniecie zahardkodowanej szerokosci 294 w dwoch miejscach -
        #     canvas dostaje szerokosc z pack(fill="both", expand=True)
        #     kontenera (ktory sam ma stala szerokosc, patrz
        #     left_container w _build_layout), a inner-frame i tak jest
        #     juz dynamicznie dopasowywany do biezacej szerokosci canvasa
        #     w _on_canvas_configure ponizej - te dwie stale liczby nigdy
        #     nie musialy byc zahardkodowane osobno.
        # POPRAWKA 2 (te same objawy - urwane pierwsze znaki etykiet -
        # utrzymywaly sie mimo poprawki #1 powyzej): canvas jest tylko
        # PIONOWO przewijalny (jeden Scrollbar, orient="vertical"), ale
        # widget tk.Canvas ma WBUDOWANE domyslne bindingi klawiatury/
        # gestow (strzalki, Shift+kolko myszy/gest poziomy na trackpadzie),
        # ktore przewijaja go takze W POZIOMIE (xview) - mimo ze nie ma
        # widocznego poziomego paska, ktorym dalby sie to cofnac. Jesli
        # canvas kiedykolwiek dostanie fokus klawiatury (np. Tab) albo
        # trackpad wysle gest poziomy, xview przesuwa sie o kilka pikseli
        # w prawo i ZOSTAJE tak juz na stale - obcinajac lewa krawedz
        # WSZYSTKICH etykiet jednakowo (dokladnie to bylo widac na
        # zrzucie ekranu). Naprawiono: canvas nie przyjmuje fokusu
        # klawiatury (takefocus=0) i jego xview jest wymuszane na 0 przy
        # kazdym przeliczeniu layoutu.
        canvas = tk.Canvas(container, bg=self.COLORS["bg"], highlightthickness=0, takefocus=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.xview_moveto(0)

        def _on_canvas_configure(event):
            canvas.itemconfigure(inner_id, width=event.width)
            canvas.xview_moveto(0)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        # Blokada gestow/klawiszy przewijajacych poziomo ten konkretnie
        # canvas (celowo tylko pionowy scroll jest tu wspierany).
        canvas.bind("<Left>", lambda e: "break")
        canvas.bind("<Right>", lambda e: "break")
        canvas.bind("<Shift-MouseWheel>", lambda e: "break")

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
        # Analyze button and status stay at the very top so they're
        # always visible without scrolling - everything else
        # (data/preprocessing/parameters/results) is below, in a
        # scrollable column.
        ttk.Button(parent, text="▶  Run analysis", style="Accent.TButton",
                   command=self.on_analyze).pack(fill="x", pady=(0, 10))

        data_frame = ttk.Labelframe(parent, text="1. Input data", padding=8)
        data_frame.pack(fill="x", pady=(0, 8))

        ttk.Button(data_frame, text="📂 Load CSV...", command=self.on_load_csv).pack(fill="x")

        self.demo_scenario_var = tk.StringVar(value=next(iter(DEMO_SCENARIOS)))
        ttk.Combobox(data_frame, textvariable=self.demo_scenario_var,
                     values=list(DEMO_SCENARIOS.keys()), state="readonly").pack(
            fill="x", pady=(6, 0))
        ttk.Button(data_frame, text="🎲 Generate demo", command=self.on_load_demo).pack(
            fill="x", pady=(4, 0))

        self.t_col_var = tk.StringVar(value="t")
        self.s_col_var = tk.StringVar(value="s")
        col_row = ttk.Frame(data_frame)
        col_row.pack(fill="x", pady=(6, 0))
        ttk.Label(col_row, text="t column:").grid(row=0, column=0, sticky="w")
        ttk.Entry(col_row, textvariable=self.t_col_var, width=6).grid(row=0, column=1, padx=4)
        ttk.Label(col_row, text="s column:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Entry(col_row, textvariable=self.s_col_var, width=6).grid(row=0, column=3, padx=4)

        self.data_info_var = tk.StringVar(value="No data loaded.")
        ttk.Label(data_frame, textvariable=self.data_info_var, wraplength=260,
                  foreground="#607d8b").pack(fill="x", pady=(4, 0))

        prep_frame = ttk.Labelframe(parent, text="2. Preprocessing", padding=8)
        prep_frame.pack(fill="x", pady=(0, 8))

        self.detrend_var = tk.BooleanVar(value=True)
        self.despike_var = tk.BooleanVar(value=True)
        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prep_frame, text="Detrend", variable=self.detrend_var).pack(anchor="w")
        ttk.Checkbutton(prep_frame, text="Despike", variable=self.despike_var).pack(anchor="w")
        ttk.Checkbutton(prep_frame, text="Normalize amplitude", variable=self.normalize_var).pack(anchor="w")

        param_frame = ttk.Labelframe(parent, text="3. Detection parameters", padding=8)
        param_frame.pack(fill="x", pady=(0, 8))

        self.k_var = tk.StringVar(value="8")
        # NOTE: the "0.4" default threshold in timdr_core_earthquake.py
        # was calibrated for a different data scale - for the
        # normalized demo signal (amplitude in [-1,1], 100Hz) it flagged
        # >90% of samples as "twist" (median |twist| on this signal
        # ~10, not ~0.4). "20" below was picked so the demo gives a
        # readable, illustrative result - it still needs retuning for
        # real data.
        self.twist_thr_var = tk.StringVar(value="20")
        self.anomaly_factor_var = tk.StringVar(value="3.0")

        # POPRAWKA (nazwy pol byly wewnetrznym zargonem TIMDR, niezrozumialym
        # dla sejsmologa bez czytania kodu): "k_neighbors" -> "Smoothing k"
        # (rozmiar okna mediany uzywanej jako linia bazowa/TRM), "anom.
        # factor" -> "MAD factor" (to dokladnie mnoznik Median Absolute
        # Deviation z klasycznej detekcji odstajacych probek - standardowy
        # termin w QC danych sejsmicznych, patrz kod: threshold = factor*mad).
        self._param_grid(param_frame, [
            ("Smoothing k:", self.k_var), ("Twist thr.:", self.twist_thr_var),
            ("MAD factor:", self.anomaly_factor_var),
        ])

        trm_row = ttk.Frame(param_frame)
        trm_row.pack(fill="x", pady=(4, 0))
        ttk.Label(trm_row, text="Baseline method:", width=15).pack(side="left")
        self.trm_method_var = tk.StringVar(value="median")
        ttk.Combobox(trm_row, textvariable=self.trm_method_var,
                     values=["median", "adaptive", "savgol"], state="readonly",
                     width=10).pack(side="left")
        ttk.Label(param_frame,
                  text="(overlay only, for comparison - anomalies()/fronts() always "
                       "use the standard median smoothing internally)",
                  wraplength=260, foreground="#90a4ae", font=("Segoe UI", 8)).pack(
            fill="x", pady=(2, 0))

        stalta_frame = ttk.Labelframe(parent, text="4. STA/LTA (picker)", padding=8)
        stalta_frame.pack(fill="x", pady=(0, 8))

        # Default 25/100 samples = 0.25s/1.0s at 100Hz (as in demo.py,
        # verified against ObsPy - see README). Thresholds 3.0/1.0 are
        # standard starting values for classic STA/LTA - they need
        # retuning for real sensor noise/sensitivity.
        self.nsta_var = tk.StringVar(value="25")
        self.nlta_var = tk.StringVar(value="100")
        self.thr_on_var = tk.StringVar(value="3.0")
        self.thr_off_var = tk.StringVar(value="1.0")

        self._param_grid(stalta_frame, [
            ("STA window:", self.nsta_var), ("LTA window:", self.nlta_var),
            ("Trigger ON:", self.thr_on_var), ("Trigger OFF:", self.thr_off_var),
        ])

        self.hybrid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(stalta_frame, text="Hybrid trigger (require twist + anomaly confirmation)",
                         variable=self.hybrid_var).pack(anchor="w", pady=(6, 0))
        tol_row = ttk.Frame(stalta_frame)
        tol_row.pack(fill="x", pady=(2, 0))
        ttk.Label(tol_row, text="tolerance (samples):", width=17).pack(side="left")
        self.tolerance_var = tk.StringVar(value="5")
        ttk.Entry(tol_row, textvariable=self.tolerance_var, width=6).pack(side="left")

        results_frame = ttk.Labelframe(parent, text="Results", padding=8)
        results_frame.pack(fill="both", expand=True, pady=(0, 4))
        self.results_text = tk.Text(results_frame, height=14, width=34, font=("Consolas", 9),
                                     bg="white", relief="flat", wrap="word")
        self.results_text.pack(fill="both", expand=True)
        self.results_text.insert("1.0", "Analysis results will appear here.")
        self.results_text.configure(state="disabled")

    def _param_grid(self, parent, label_var_pairs, per_row=2):
        """Compact grid of parameter fields, `per_row` per row - replaces
        stacking one field under another, which with more parameters
        (STA/LTA added 4 more fields) made the left column too tall.

        POPRAWKA 3 (cyfry w polach 2. kolumny chowaly sie POD paskiem
        przewijania - "twist thr.", "nlta", "off thr." na zrzucie
        ekranu uzytkownika): etykieta miala ZAHARDKODOWANA stala
        szerokosc 13 znakow nawet tam, gdzie najdluzsza etykieta w danej
        grupie byla krotsza (np. "nsta:"/"nlta:"/"off thr.:" w grupie
        STA/LTA - max 9 znakow, nie 13). Marnowana przestrzen w kolumnie
        1 wypychala kolumne 2 poza prawa krawedz dostepnej szerokosci
        panelu (~290-300px), czyli dokladnie pod pasek przewijania.
        Naprawiono: szerokosc etykiety liczona dynamicznie z najdluzszej
        etykiety W TEJ KONKRETNEJ grupie, plus lekko zmniejszone pole
        wpisywania i odstep miedzy kolumnami - razem daje margines,
        ktory wczesniej byl zerowy albo ujemny.
        """
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        label_w = max(len(label) for label, _ in label_var_pairs)
        for i, (label, var) in enumerate(label_var_pairs):
            r, c = divmod(i, per_row)
            cell = ttk.Frame(grid)
            cell.grid(row=r, column=c, sticky="w", padx=(0, 6), pady=2)
            ttk.Label(cell, text=label, width=label_w).pack(side="left")
            ttk.Entry(cell, textvariable=var, width=6).pack(side="left")

    # ------------------------------------------------------------
    def _build_plot(self, parent):
        # POPRAWKA (uzytkownik poprosil o wieksze wykresy + pasek przewijania
        # po prawej): wczesniej figura byla SCISKANA do wysokosci okna (5
        # wykresow w stosie robilo sie coraz mniejsze na mniejszych
        # ekranach). Teraz pasek narzedzi (zoom/pan/save) zostaje NA STALE
        # widoczny u gory, a sama figura ma wiekszy, STALY rozmiar bazowy
        # (13in wysokosci zamiast 8.5in) i jest opakowana w pionowy
        # Canvas+Scrollbar (dokladnie ten sam wzorzec co panel parametrow po
        # lewej) - gdy wykresy sa wyzsze niz okno, przewija sie je zamiast
        # pomniejszac. Szerokosc nadal sledzi dostepna przestrzen dynamicznie.
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill="x", side="top")

        scroll_area = ttk.Frame(parent)
        scroll_area.pack(fill="both", expand=True)

        plot_canvas = tk.Canvas(scroll_area, bg=self.COLORS["bg"], highlightthickness=0, takefocus=0)
        plot_scrollbar = ttk.Scrollbar(scroll_area, orient="vertical", command=plot_canvas.yview)
        plot_canvas.configure(yscrollcommand=plot_scrollbar.set)
        plot_scrollbar.pack(side="right", fill="y")
        plot_canvas.pack(side="left", fill="both", expand=True)

        plot_inner = ttk.Frame(plot_canvas)
        plot_inner_id = plot_canvas.create_window((0, 0), window=plot_inner, anchor="nw")

        self.fig = Figure(figsize=(9.0, 13.0), dpi=100)
        self.axes = self.fig.subplots(5, 1, sharex=True)
        self.fig.subplots_adjust(hspace=0.4, left=0.08, right=0.98, top=0.97, bottom=0.045)
        self._draw_placeholder()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_inner)
        canvas_widget = self.canvas.get_tk_widget()
        # fill="x" (bez expand/vertical fill) celowo: szerokosc ma sledzic
        # kontener, wysokosc ma zostac przy swoim naturalnym (duzym) rozmiarze
        # zamiast byc scisniana do widocznego obszaru.
        canvas_widget.pack(fill="x")
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        def _on_plot_inner_configure(_event):
            plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))
            plot_canvas.xview_moveto(0)

        def _on_plot_canvas_configure(event):
            plot_canvas.itemconfigure(plot_inner_id, width=event.width)

        plot_inner.bind("<Configure>", _on_plot_inner_configure)
        plot_canvas.bind("<Configure>", _on_plot_canvas_configure)
        # Ta sama blokada przypadkowego poziomego przewiniecia co w panelu
        # po lewej (patrz _build_scrollable_left) - tu tylko pionowo.
        plot_canvas.bind("<Left>", lambda e: "break")
        plot_canvas.bind("<Right>", lambda e: "break")
        plot_canvas.bind("<Shift-MouseWheel>", lambda e: "break")

        def _on_plot_mousewheel(event):
            delta = -1 * (event.delta // 120) if event.delta else (1 if event.num == 5 else -1)
            plot_canvas.yview_scroll(int(delta), "units")

        plot_canvas.bind("<Enter>", lambda _e: (plot_canvas.bind_all("<MouseWheel>", _on_plot_mousewheel),
                                                 plot_canvas.bind_all("<Button-4>", _on_plot_mousewheel),
                                                 plot_canvas.bind_all("<Button-5>", _on_plot_mousewheel)))
        plot_canvas.bind("<Leave>", lambda _e: (plot_canvas.unbind_all("<MouseWheel>"),
                                                 plot_canvas.unbind_all("<Button-4>"),
                                                 plot_canvas.unbind_all("<Button-5>")))

        # Debounce (after) bo <Configure> odpala sie wielokrotnie w trakcie
        # przeciagania krawedzi okna.
        self._resize_job = None
        canvas_widget.bind("<Configure>", self._on_plot_resize)

    def _on_plot_resize(self, event):
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(120, lambda w=event.width: self._apply_plot_resize(w))

    def _apply_plot_resize(self, width):
        self._resize_job = None
        if width < 100:
            return
        dpi = self.fig.get_dpi()
        new_w = width / dpi
        if abs(new_w - self.fig.get_figwidth()) < 0.05:
            return
        # Tylko szerokosc sledzi dostepna przestrzen - wysokosc jest celowo
        # stala (patrz _build_plot), zeby wykresy nie kurczyly sie na
        # mniejszych oknach; nadmiar wysokosci jest przewijany.
        self.fig.set_size_inches(new_w, self.fig.get_figheight())
        self.canvas.draw_idle()

    def _draw_placeholder(self):
        titles = ["signal", "flow (local gradient)", "|twist|", "residual + anomalies/fronts",
                  "STA/LTA ratio"]
        for ax, title in zip(self.axes, titles):
            ax.clear()
            ax.set_ylabel(title, fontsize=9)
            ax.grid(alpha=0.2)
        self.axes[-1].set_xlabel("time (s)")
        self.axes[0].set_title("Load data and run analysis", fontsize=10, color="#90a4ae")

    # ------------------------------------------------------------
    def on_load_csv(self):
        path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            loader = SeismicLoader(normalize=False, detrend=False, clip_outliers=False)
            t, s = loader.load_csv(path, t_col=self.t_col_var.get(), s_col=self.s_col_var.get())
            if len(t) == 0:
                raise ValueError("The file contains no valid rows.")
            self.t_raw, self.s_raw = t, s
            fname = path.replace("\\", "/").split("/")[-1]
            self.data_info_var.set(f"Loaded {len(t)} samples from {fname}")
            self.status_var.set(f"Loaded {len(t)} samples. Click 'Run analysis'.")
        except Exception as exc:
            messagebox.showerror("CSV loading error", str(exc))

    def on_load_demo(self):
        scenario = self.demo_scenario_var.get()
        gen = DEMO_SCENARIOS.get(scenario, demo_earthquake)
        self.t_raw, self.s_raw = gen()
        self.data_info_var.set(f"Generated demo signal: {len(self.t_raw)} samples "
                                f"({scenario})")
        self.status_var.set(f"Generated demo data ({scenario}). Click 'Run analysis'.")

    # ------------------------------------------------------------
    def on_analyze(self):
        if self.t_raw is None or self.s_raw is None:
            messagebox.showwarning("No data", "Load a CSV file or generate a demo signal first.")
            return
        try:
            k = int(self.k_var.get())
            twist_thr = float(self.twist_thr_var.get())
            anomaly_factor = float(self.anomaly_factor_var.get())
            nsta = int(self.nsta_var.get())
            nlta = int(self.nlta_var.get())
            thr_on = float(self.thr_on_var.get())
            thr_off = float(self.thr_off_var.get())
            tolerance = int(self.tolerance_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid parameters",
                "k_neighbors, twist threshold, anomaly factor, nsta, nlta, "
                "on/off threshold and tolerance must all be numbers."
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
            events = core.classify_anomalies(t, s, factor=anomaly_factor)

            try:
                smooth_preview = core.trm(t, s, method=self.trm_method_var.get())
            except Exception:
                smooth_preview = None

            ratio = core.sta_lta(s, nsta, nlta)
            hybrid_enabled = self.hybrid_var.get()
            if hybrid_enabled:
                confirmed, rejected = core.hybrid_trigger(
                    t, s, nsta, nlta, twist_threshold=twist_thr,
                    anomaly_factor=anomaly_factor, sta_lta_thr_on=thr_on,
                    sta_lta_thr_off=thr_off, tolerance=tolerance,
                )
            else:
                confirmed = core.trigger_onset(ratio, thr_on=thr_on, thr_off=thr_off)
                rejected = []

            self._plot_results(t, s, flow_grad, twist_strength, twist_pts, residuals,
                                anomaly_pts, fronts, twist_thr, ratio, confirmed, rejected,
                                thr_on, thr_off, events, smooth_preview, hybrid_enabled)
            self._show_results(t, twist_pts, anomaly_pts, fronts, th, confirmed, rejected,
                                events, hybrid_enabled)
            self.status_var.set(
                f"Analysis done: {len(twist_pts)} twist pts, {len(anomaly_pts)} anomalies, "
                f"{len(fronts)} fronts, {len(confirmed)} STA/LTA "
                f"{'confirmed' if hybrid_enabled else 'triggers'}."
            )
        except Exception:
            messagebox.showerror("Analysis error", traceback.format_exc(limit=3))

    # ------------------------------------------------------------
    def _plot_results(self, t, s, flow_grad, twist_strength, twist_pts, residuals, anomaly_pts,
                       fronts, twist_thr, ratio, confirmed, rejected, thr_on, thr_off, events,
                       smooth_preview, hybrid_enabled):
        for ax in self.axes:
            ax.clear()
            ax.grid(alpha=0.2)

        self.axes[0].plot(t, s, color=self.COLORS["accent"], lw=0.9, label="signal")
        if smooth_preview is not None:
            self.axes[0].plot(t, smooth_preview, color="#455a64", lw=1.1, ls="--",
                               alpha=0.8, label=f"TRM preview ({self.trm_method_var.get()})")
            self.axes[0].legend(fontsize=7, loc="upper right")
        self.axes[0].set_ylabel("signal", fontsize=9)

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
        seen_types = set()
        for ev in events:
            a, b = ev["start"], ev["end"]
            local = np.argmax(np.abs(residuals[a:b + 1])) + a
            etype = ev["type"]
            color = ANOMALY_TYPE_COLORS.get(etype, self.COLORS["danger"])
            label = etype if etype not in seen_types else None
            seen_types.add(etype)
            self.axes[3].scatter(t[local], residuals[local], color=color, s=22,
                                  zorder=5, label=label)
        if len(fronts):
            self.axes[3].scatter(t[fronts], residuals[fronts],
                                  color="black", s=70, marker="*", zorder=6, label="front")
        if seen_types or len(fronts):
            self.axes[3].legend(fontsize=7, loc="upper right", ncol=2)
        self.axes[3].set_ylabel("residual", fontsize=9)

        self.axes[4].plot(t, ratio, color="#00838f", lw=0.9)
        self.axes[4].axhline(thr_on, color=self.COLORS["danger"], ls="--", lw=1, label="on threshold")
        self.axes[4].axhline(thr_off, color=self.COLORS["warn"], ls="--", lw=1, label="off threshold")
        conf_label = "confirmed" if hybrid_enabled else "trigger"
        for k_evt, (i_start, i_end) in enumerate(confirmed):
            self.axes[4].axvspan(t[i_start], t[i_end], color=self.COLORS["ok"] if hybrid_enabled
                                  else self.COLORS["danger"], alpha=0.18,
                                  label=conf_label if k_evt == 0 else None)
        if hybrid_enabled:
            for k_evt, r in enumerate(rejected):
                self.axes[4].axvspan(t[r["start"]], t[r["end"]], color="#9e9e9e", alpha=0.18,
                                      label="rejected" if k_evt == 0 else None)
        self.axes[4].legend(fontsize=7, loc="upper right")
        self.axes[4].set_ylabel("STA/LTA", fontsize=9)
        self.axes[4].set_xlabel("time (s)")

        self.fig.suptitle("")
        self.canvas.draw()

    def _show_results(self, t, twist_pts, anomaly_pts, fronts, threshold, confirmed, rejected,
                       events, hybrid_enabled):
        lines = []
        lines.append(f"Samples: {len(t)}")
        lines.append(f"Time range: {t[0]:.3f} - {t[-1]:.3f} s")
        lines.append("")
        lines.append(f"Twist points: {len(twist_pts)}")
        lines.append(f"Anomalies (micro-events): {len(anomaly_pts)}")
        lines.append(f"Anomaly threshold (MAD): {threshold:.4f}")

        if events:
            counts = {}
            for ev in events:
                counts[ev["type"]] = counts.get(ev["type"], 0) + 1
            breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            lines.append(f"Anomaly shape breakdown: {breakdown}")

        lines.append("")
        lines.append(f"Fronts (shock onset): {len(fronts)}")
        if len(fronts):
            first = fronts[0]
            lines.append(f"  first front: t={t[first]:.3f}s (idx={first})")
            if len(fronts) > 1:
                lines.append(f"  last front:  t={t[fronts[-1]]:.3f}s (idx={fronts[-1]})")
        else:
            lines.append("  (no fronts detected)")

        lines.append("")
        if hybrid_enabled:
            lines.append(f"STA/LTA confirmed (hybrid): {len(confirmed)}")
            lines.append(f"STA/LTA rejected (missing twist/anomaly): {len(rejected)}")
            if rejected:
                miss_twist = sum(1 for r in rejected if r["missing_twist"])
                miss_anom = sum(1 for r in rejected if r["missing_anomaly"])
                lines.append(f"  missing twist: {miss_twist}, missing anomaly: {miss_anom}")
        else:
            lines.append(f"STA/LTA triggers: {len(confirmed)}")
        if len(confirmed):
            i_start, i_end = confirmed[0]
            lines.append(f"  first: t={t[i_start]:.3f}s -> {t[i_end]:.3f}s (idx={i_start}-{i_end})")
            if len(confirmed) > 1:
                i_start, i_end = confirmed[-1]
                lines.append(f"  last:  t={t[i_start]:.3f}s -> {t[i_end]:.3f}s (idx={i_start}-{i_end})")
        else:
            lines.append("  (no triggers - check on/off thresholds or nsta/nlta)")

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "\n".join(lines))
        self.results_text.configure(state="disabled")


def _enable_windows_dpi_awareness():
    """POPRAWKA: Tkinter na Windows jest domyslnie DPI-nieswiadomy - przy
    skalowaniu ekranu >100% (bardzo czeste na laptopach, np. 125%/150%)
    caly interfejs jest wtedy rozciagany "na sile" przez system operacyjny
    zamiast renderowany natywnie w realnej rozdzielczosci. To byla
    prawdopodobna przyczyna zgloszonego bledu "panel parametrow chowa sie
    za paskiem przewijania" - stale liczone w px (patrz
    _build_scrollable_left) nie odpowiadaly fizycznym pikselom na ekranie.
    Bezpieczny no-op na innych systemach / starszych Windows (opakowane w
    try/except, zaden blad nie przerywa startu aplikacji)."""
    import sys
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()  # starsze Windows (Vista-8)
        except Exception:
            pass  # np. brak shcore.dll - nie blokujemy startu aplikacji


def main():
    _enable_windows_dpi_awareness()
    app = TimdrEarthquakeGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
