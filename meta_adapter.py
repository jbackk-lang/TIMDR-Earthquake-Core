"""meta_adapter.py -- adapter seismic_state -> MetaState (integracja z
TIMDR-META-DYNAMICS), trzecia realna integracja formalizmu Lambda-tau-rho-J
(pierwsza - finansowa w analizator-gieldowy-v3, druga - pogodowa w
Synoptyk-v3/membrane/meta_adapter.py). Ten sam wzorzec sys.path
(folder-siostra), inny sygnal wejsciowy: realny, ciagly slad sejsmiczny
zamiast siatki przestrzennej.

KONTEKST: TIMDR_EarthquakeCore (timdr_core_earthquake.py) juz liczy
dokladnie te sygnaly, ktore GIA-TIMDR nazywa gałęzią M/S (flow=tempo/
gradient, twist=skręt sygnałowy, anomalies=anomalia, trm=struktura/
baseline) - NIE nowe znaczenie, tylko ponowne uzycie juz istniejacych,
przetestowanych funkcji tego repo, okienkowanych w czasie zeby dac SERIE
MetaState (jeden na okno), zamiast pojedynczego stanu na cala 6-minutowa
probke.

LEKCJA Z Synoptyk-v3/membrane/meta_adapter.py ZASTOSOWANA OD RAZU (nie
odkrywana ponownie od zera): V1 tamtego adaptera uzyl SUROWYCH liczb
komorek dla rho/J, co zdominowalo sume |Lambda|+|tau|+|rho|+|J| i dalo
zero mocy dyskryminujacej classify_phase() (wszystko wyszlo "krytyczna").
Tu WSZYSTKIE cztery skladowe byly zaprojektowane jako bezwymiarowe [0,1]
OD RAZU (nie jako pozniejsza poprawka).

===========================================================================
WERSJA 1 (2026-09-08, PORZUCONA - zachowana dla historii): kazde okno
normalizowane WZGLEDEM SAMEGO SIEBIE - tau = sredni |flow_grad| w oknie /
(mediana+k*MAD TEGO SAMEGO |flow_grad| w TYM SAMYM oknie), J = fraction
probek gdzie |d(flow_grad)/dt| > mediana+k*MAD TEGO SAMEGO okna. Uruchomione
end-to-end na realnym sladzie (Ridgecrest 2019, stacja CLC, 360s, 72 okna
po 5s) - REALNY WYNIK: WSZYSTKIE 71 krokow, bez wyjatku (wliczajac okno
obejmujace prawdziwy mainshock t~=60s), wyszly jako "stabilna" (|M| w
waskim pasmie ~0.005-0.07, prog stabilna<0.1) - dokladnie ODWROTNY
problem niz w Synoptyk-v3 V1 (tam wszystko "krytyczna", tu wszystko
"stabilna") - zero mocy dyskryminujacej w OBU kierunkach.

PRZYCZYNA (znaleziona algebraicznie, nie zgadywaniem): normalizacja
srednia(okno)/prog(TEGO SAMEGO okna) jest NIEZMIENNICZA na jednorodne
przeskalowanie calego okna. Jesli kazda wartosc w oknie pomnozyc przez
stala c (okno c-krotnie bardziej "aktywne" wszedzie rownomiernie), to
mediana i MAD tego okna TEZ rosna c-krotnie, wiec prog=mediana+k*MAD
rosnie c-krotnie, a stosunek srednia/prog zostaje DOKLADNIE bez zmian -
tau w V1 mierzyl KSZTALT rozkladu wewnatrz okna (cwiczenie: podstaw
konkretny przyklad liczbowy - okno [1,1,1,10] i okno [10,10,10,100],
oba maja identyczny stosunek mean/(median+k*mad) mimo 10x rozniej w
absolutnej skali), NIE poziom aktywnosci wzgledem reszty sladu, ktory
mial reprezentowac. Realny slad ma wystarczajaco jednorodny "ksztalt"
szumu tla we WSZYSTKICH oknach (w tym oknie z mainshockiem, ktory trwa
kilka sekund w oknie 5s, wiec nie dominuje calego okna), wiec V1 dawal
podobny tau wszedzie (~0.25-0.38) niezaleznie od realnej aktywnosci.

WERSJA 2 (2026-09-08, AKTUALNA) - PROGI GLOBALNE zamiast per-okno:
`compute_global_thresholds()` liczy `flow_threshold`/`twist_threshold`/
`anomaly_threshold` RAZ na CALYM 360s sladzie (przed podzieleniem na
okna, przed policzeniem jakiegokolwiek MetaState/M-serii - nadal
pre-rejestracja w doslownym sensie, tylko przesunieta o jeden poziom:
teraz to PROGI sa ustalone raz dla calego przebiegu, a nie ponownie
liczone per-okno). Kazde okno jest teraz PORONYWANE DO WSPOLNEGO punktu
odniesienia zamiast do samego siebie - okno o realnie podwyzszonej
aktywnosci wzgledem CALEGO sladu wychyli sie ponad wspolny prog, zamiast
byc automatycznie znormalizowane do wlasnego poziomu.

===========================================================================
PRE-REJESTRACJA MAPOWANIA V2 (ustalone PRZED ponownym uruchomieniem
M-serii/classify_phase() na tym samym realnym sladzie - protokol
numerologii/formalizmu, skill timdr-signal-framework):

Okno: `WINDOW_SECONDS` (domyslnie 5.0s) sasiadujacych, NIENAKLADAJACYCH
SIE probek sladu (partycjonowanie, nie przesuwne okno - kolejne stany
S_meta(t) musza byc od siebie niezalezne probkowo, zeby M=d/dt(S) mial
sens jako "zmiana miedzy kolejnymi, osobnymi obserwacjami").

    Lambda (struktura)      = high-frequency fraction widma amplitudy W
                              TYM OKNIE (FFT, |widmo|^2, gorna polowa
                              pasm/(gorna+dolna)) - DOSLOWNIE ten sam
                              wzor co w Synoptyk-v3 (radial_power_spectrum),
                              1D zamiast 2D. Bez zmian wzgledem V1 - to
                              jest wewnetrznie ulamek [0,1] z definicji,
                              wiec NIE cierpi na problem niezmienniczosci
                              opisany wyzej (dzieli energie PRZEZ SIEBIE
                              SAMA, nie przez zewnetrzny prog).

    tau (transformacja)     = sredni |flow_grad| W TYM OKNIE / GLOBALNY
                              flow_threshold (mediana+k*MAD z |flow_grad|
                              policzonego na CALYM sladzie, k=3.5 - ta
                              sama stala co DEFECT_K w Synoptyk-v3, dla
                              spojnosci ekosystemu).

    rho (anomalia)          = fraction probek W TYM OKNIE, gdzie
                              |s - core.trm(s)| > GLOBALNY
                              anomaly_threshold (mediana+k*MAD residuow
                              z CALEGO sladu, k=3.5 - ten sam wzor co
                              wewnatrz core.anomalies(), ale prog
                              policzony raz globalnie zamiast ponownie
                              per-okno).

    J (operator punktowy)   = fraction probek W TYM OKNIE, gdzie
                              |d(flow_grad)/dt| > GLOBALNY twist_threshold
                              (mediana+k*MAD z |d(flow_grad)/dt| na CALYM
                              sladzie).

UCZCIWE ZASTRZEZENIA:
  1. To jest JEDNO z mozliwych mapowan - inny wybor (np. dluzsze/krotsze
     okno, inne k) dalby inny liczbowo wynik.
  2. Progi classify_phase() (0.1/1.0) sa PRZENIESIONE bez zmian z
     TIMDR-META-DYNAMICS, NIE skalibrowane na tym sygnale.
  3. k=3.5 (prog robust dla tau/J) i factor=3.5 (prog anomalii dla rho) sa
     wyborami konserwatywnymi dla spojnosci z reszta ekosystemu TIMDR
     (Synoptyk-v3 DEFECT_K), NIE niezalezna kalibracja na sejsmice.
  4. Jedna realna 6-minutowa probka (Ridgecrest 2019, stacja CLC) z
     JEDNYM znanym zdarzeniem (mainshock, t~=60.0s wg USGS, wykryty przez
     sta_lta()/trigger_onset() tego repo na t=61.1s - patrz
     HISTORIA_I_TESTY.md) to JEDEN przyklad, nie kalibracja. Ewentualna
     zgodnosc miedzy szczytem |M|/faza a t~=60s jest obserwacja, NIE
     dowodem predykcyjnosci ani wykrywalnosci w ogolnosci - dokladnie
     tak jak ringdown.py w tym samym repo juz uczciwie odroznia
     "opisowe post-event" od "predykcyjne" (patrz jego docstring).
  5. Progi globalne domyslnie uzywaja CALEGO sladu (wliczajac sam
     mainshock) do wyznaczenia "typowego" poziomu - w realnym zastosowaniu
     online (wykrywanie NA BIEZACO) progi musialyby byc liczone tylko z
     danych SPRZED momentu decyzji, inaczej to forma wyciekania informacji
     z przyszlosci do przeszlosci. Domyslny wariant nadaje sie do analizy
     POST-HOC (offline) calego zapisu, NIE do zywej detekcji w czasie
     rzeczywistym - to samo rozroznienie, ktore ringdown.py juz robi miedzy
     "opisowe" i "predykcyjne". Parametr `calibration_end` w
     `build_meta_series_from_waveform()` daje przyczynowy wariant
     (kalibracja WYLACZNIE z danych sprzed `calibration_end`) - patrz #6.
  6. ZBADANE (na wyrazna prosbe, nie domyslnie porzucone): czy uzycie
     CALEGO sladu do kalibracji (#5) nie jest w praktyce "przejsciem
     miedzy dwoma rezimami" ukrytym w jednej statystyce. Odpowiedz:
     TAK, dosl. w tym sensie, ze pre-event (t<55s) i post-event sa
     statystycznie DRASTYCZNIE rozne rezimy - zweryfikowane na SUROWYCH
     (nieprzetworzonych) danych: odch. std. amplitudy = ~9 400 przed
     zdarzeniem, ~2.3-5.5 MILIONA w pierwszych ~90s po, wciaz ~390 000
     (40x wiecej niz tlo) w ostatnich 100s 6-minutowego zapisu - to
     PRAWDZIWA fizyka (silny ruch gruntu + gesta sekwencja wstrzasow
     wtornych M7.1, mozliwe nasycenie czujnika blisko epicentrum), NIE
     artefakt przetwarzania (detrend/despike/normalize). Konsekwencja:
     kalibracja (calibration_end=None, cale 360s) daje krzywa
     narastanie-i-ZANIK (faza "krytyczna" dokladnie w kroku z mainshockiem,
     potem stopniowy powrot do "stabilna" w ciagu ~90s - patrz test
     end-to-end), a kalibracja WYLACZNIE przyczynowa (calibration_end=55.0,
     tylko dane sprzed zdarzenia) daje wynik NIEMAL BINARNY: 56/67 krokow
     "krytyczna", bo doslownie CALA reszta zapisu (300+ z 360s) jest
     ekstremalna wzgledem tla sprzed zdarzenia - obie odpowiedzi sa
     POPRAWNE, tylko odpowiadaja na inne pytanie ("jak ewoluuje aktywnosc
     w obrebie zdarzenia" vs "czy jestem juz w rezimie po-katastroficznym").
     Zaden wariant nie zostal porzucony na rzecz drugiego - oba zostaja w
     kodzie (`calibration_end` parametr), z tym zastrzezeniem widocznym
     w obu miejscach.
===========================================================================
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from timdr_core_earthquake import TIMDR_EarthquakeCore

MAD_TO_STD = 1.4826  # ta sama stala co Synoptyk-v3/membrane/defects.py
ROBUST_K = 3.5        # ta sama wartosc co Synoptyk-v3 DEFECT_K, patrz zastrzezenie #3
ANOMALY_FACTOR = 3.5  # jw., przekazywane do core.anomalies()


def _ensure_timdr_meta_dynamics_on_path() -> None:
    """Dodaje folder-siostre TIMDR-META-DYNAMICS do sys.path - ten sam
    wzorzec co Synoptyk-v3/membrane/meta_adapter.py i
    analizator-gieldowy-v3/meta_dynamics_module.py. TIMDR-Earthquake-Core
    lezy BEZPOSREDNIO w katalogu nadrzednym (nie w podfolderze jak
    membrane/ w Synoptyk-v3), wiec siostra jest o JEDEN poziom wyzej."""
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.join(here, "..", "TIMDR-META-DYNAMICS")
    sibling = os.path.abspath(sibling)

    if not os.path.isdir(sibling):
        raise ImportError(
            "meta_adapter wymaga folderu 'TIMDR-META-DYNAMICS' jako "
            f"siostry repo TIMDR-Earthquake-Core (szukano w: {sibling})."
        )
    if sibling not in sys.path:
        sys.path.insert(0, sibling)


_ensure_timdr_meta_dynamics_on_path()

from timdr_meta_dynamics import MetaState, MetaOperatorM  # noqa: E402
from analysis.meta_map import MetaMap  # noqa: E402
from analysis.meta_trigger import MetaTrigger, MetaTriggerResult  # noqa: E402


WINDOW_SECONDS = 5.0


@dataclass
class SeismicMetaResult:
    window_starts: List[float]     # czas poczatku kazdego okna [s], dlugosc n
    states: List[MetaState]        # S_meta(t) per okno, dlugosc n
    M_series: List[MetaState]      # M(t) = dS/dt, dlugosc n-1
    phases: List[str]              # faza per krok M, dlugosc n-1
    trigger: MetaTriggerResult


def _robust_threshold(values: np.ndarray, k: float = ROBUST_K) -> float:
    """Mediana + k*MAD(przeskalowany) - identyczna logika co
    Synoptyk-v3/membrane/defects.py::robust_threshold(), reimplementowana
    tu lokalnie (repo-do-repo, nie import) zeby meta_adapter.py zostal
    samowystarczalny w TYM repo."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) * MAD_TO_STD
    return med + k * mad


def _high_freq_fraction(s_window: np.ndarray) -> float:
    """Ulamek energii widmowej w gornej polowie pasm czestotliwosci -
    1D odpowiednik Synoptyk-v3/membrane/spectrum.py::radial_power_spectrum
    (tam 2D/promieniowe, tu 1D wzdluz czasu)."""
    n = len(s_window)
    if n < 4:
        return 0.0
    centered = s_window - np.mean(s_window)
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2
    half = len(power) // 2
    low = float(power[:half].sum())
    high = float(power[half:].sum())
    total = low + high
    return high / total if total > 0 else 0.0


@dataclass
class GlobalThresholds:
    """Progi policzone RAZ na CALYM sladzie (nie per-okno) - patrz
    NAPRAWA V2 w docstringu modulu: normalizacja mean(okno)/prog(TEGO
    SAMEGO okna) jest algebraicznie niezmiennicza na jednorodne
    przeskalowanie calego okna (mean i prog rosna razem), wiec NIE
    wykrywa wcale wzrostu aktywnosci, tylko zmiane KSZTALTU rozkladu.
    Prog wspolny dla wszystkich okien naprawia to - okno o realnie
    podwyzszonej aktywnosci wzgledem CALEGO sladu wychyli sie ponad
    prog, zamiast byc znormalizowane do samego siebie."""
    flow_threshold: float
    twist_threshold: float
    anomaly_threshold: float


def compute_global_thresholds(core: TIMDR_EarthquakeCore, t: np.ndarray, s: np.ndarray) -> GlobalThresholds:
    """Liczy progi robust na PODANYM (t,s) - PRZED podzieleniem na okna i
    PRZED policzeniem jakiegokolwiek MetaState/M-serii - pre-rejestracja
    w dosl. sensie: te liczby sa ustalone zanim jakikolwiek wynik koncowy
    (faza/|M|) w ogole istnieje.

    UWAGA (dwa rozne, oba uzasadnione wybory `(t,s)` przekazywanego tutaj -
    patrz `build_meta_series_from_waveform(calibration_end=...)` i
    zastrzezenie #5/#6 w docstringu modulu):

    (a) CALY slad (dlugosc trwania calego zapisu, `calibration_end=None`) -
        daje wglad w STRUKTURE zdarzenia (narastanie i ZANIK aktywnosci w
        czasie), ale prog jest wtedy MIESZANKA statystyk sprzed i po
        zdarzenia - "lookahead": w zywym systemie nie mialbys jeszcze
        dostepu do danych z przyszlosci wzgledem chwili decyzji.

    (b) TYLKO okres SPRZED zdarzenia (`calibration_end=<czas>`) - przyczynowe
        (dokladnie to, co realny system online widzialby w danym momencie),
        ale na PRAWDZIWYM sladzie Ridgecrest 2019 (stacja CLC) daje niemal
        BINARNY wynik: cala reszta 6-minutowego zapisu (300+ z 360s) wychodzi
        jako "anomalna" wzgledem spokojnego tla - NIE dlatego, ze to artefakt
        (zweryfikowano na SUROWYCH, nieprzetworzonych danych: odch. std.
        amplitudy skacze z ~9 400 przed zdarzeniem do ~2-5 mln tuz po i
        POZOSTAJE ~40x wyzsze nawet 3-5 minut pozniej - to prawdziwa fizyka
        silnego ruchu gruntu + sekwencji wstrzasow wtornych M7.1, nie blad
        przetwarzania), tylko dlatego, ze prog (a) i prog (b) odpowiadaja na
        RÓŻNE pytania: (a) "jak ewoluuje aktywnosc w obrebie calego zapisu"
        vs (b) "czy jestem juz w rezimie po-katastroficznym, czy jeszcze w
        rezimie tla" - oba sa poprawne, zadne nie jest bledne, ale dają
        jakosciowo inna odpowiedz z tych samych danych."""
    flow_grad_all = core.flow(t, s)
    flow_threshold = _robust_threshold(np.abs(flow_grad_all))

    twist_strength_all = np.abs(np.gradient(flow_grad_all, t))
    twist_threshold = _robust_threshold(twist_strength_all)

    smooth_all = core.trm(t, s)
    residuals_all = s - smooth_all
    mad = float(np.median(np.abs(residuals_all))) * MAD_TO_STD
    if mad <= 1e-12:
        std = float(np.std(residuals_all))
        mad = std if std > 1e-12 else 1e-9
    anomaly_threshold = ANOMALY_FACTOR * mad

    return GlobalThresholds(
        flow_threshold=flow_threshold,
        twist_threshold=twist_threshold,
        anomaly_threshold=anomaly_threshold,
    )


def window_to_meta_state(
    core: TIMDR_EarthquakeCore,
    t_window: np.ndarray,
    s_window: np.ndarray,
    thresholds: GlobalThresholds,
) -> MetaState:
    """Mapowanie jednego okna (t,s) -> jeden MetaState, wzgledem progow
    GLOBALNYCH (`thresholds`, policzonych raz na calym sladzie - patrz
    compute_global_thresholds). Wzory zamrozone w PRE-REJESTRACJI/NAPRAWA
    V2 na gorze pliku."""
    n = len(s_window)

    Lambda = _high_freq_fraction(s_window)

    flow_grad = core.flow(t_window, s_window)
    abs_flow = np.abs(flow_grad)
    tau = float(abs_flow.mean()) / thresholds.flow_threshold if thresholds.flow_threshold > 0 else 0.0

    if n >= 3:
        twist_strength = np.abs(np.gradient(flow_grad, t_window))
        n_twist = int(np.sum(twist_strength > thresholds.twist_threshold))
    else:
        n_twist = 0
    J = n_twist / n if n > 0 else 0.0

    smooth = core.trm(t_window, s_window)
    residuals = s_window - smooth
    n_anomaly = int(np.sum(np.abs(residuals) > thresholds.anomaly_threshold))
    rho = n_anomaly / n if n > 0 else 0.0

    return MetaState(Lambda=Lambda, tau=tau, rho=rho, J=J)


def build_meta_series_from_waveform(
    t: np.ndarray,
    s: np.ndarray,
    window_seconds: float = WINDOW_SECONDS,
    core: Optional[TIMDR_EarthquakeCore] = None,
    dt: Optional[float] = None,
    calibration_end: Optional[float] = None,
) -> SeismicMetaResult:
    """Dzieli (t,s) na kolejne, NIENAKLADAJACE SIE okna dlugosci
    `window_seconds`, liczy MetaState per okno, potem M-serie/fazy/trigger.

    `dt` (czas miedzy kolejnymi OKNAMI, nie probkami) domyslnie =
    window_seconds - okna sa wprost sasiadujace, wiec to najbardziej
    dosłowna interpretacja "jeden krok czasu miedzy stanami".

    `calibration_end` (domyslnie None = kalibruj na CALYM sladzie) -
    jesli podane, progi globalne sa liczone WYLACZNIE z probek
    `t < calibration_end`, a stosowane do WSZYSTKICH okien (rowniez
    tych po `calibration_end`). To jest przyczynowy/"online-bezpieczny"
    wariant kalibracji (patrz uzasadnienie i realny wynik na Ridgecrest
    2019 w docstringu `compute_global_thresholds`) - NIE domyslny, bo
    wymaga znajomosci, KIEDY konczy sie "spokojny" okres referencyjny,
    czego adapter sam z siebie nie zgaduje. Oba warianty sa
    rownoprawne/oba zachowane (patrz zastrzezenie #5/#6) - zaden nie
    zastepuje drugiego.

    Ostatnie, niepelne okno (gdy dlugosc sladu nie jest calkowita
    wielokrotnoscia window_seconds) jest ODRZUCANE, nie dopelniane -
    niepelne okno mialoby inna liczbe probek niz reszta, co
    zaburzyloby porownywalnosc statystyk robust (mediana/MAD) miedzy
    oknami."""
    t = np.asarray(t, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if len(t) != len(s):
        raise ValueError(f"t i s musza miec ta sama dlugosc, dostano {len(t)} i {len(s)}")
    if len(t) < 2:
        raise ValueError("Potrzeba >= 2 probek")

    if core is None:
        core = TIMDR_EarthquakeCore()
    if dt is None:
        dt = window_seconds

    # Progi GLOBALNE - domyslnie z calego sladu, albo (calibration_end
    # podane) wylacznie z okresu referencyjnego SPRZED tego czasu -
    # zawsze PRZED podzieleniem na okna, przed policzeniem
    # jakiegokolwiek MetaState/M-serii. Patrz GlobalThresholds/NAPRAWA V2
    # i uwaga (a)/(b) w docstringu compute_global_thresholds.
    if calibration_end is None:
        thresholds = compute_global_thresholds(core, t, s)
    else:
        calib_mask = t < calibration_end
        if calib_mask.sum() < 4:
            raise ValueError(
                f"calibration_end={calibration_end} zostawia tylko "
                f"{int(calib_mask.sum())} probek referencyjnych (< 4)."
            )
        thresholds = compute_global_thresholds(core, t[calib_mask], s[calib_mask])

    t0 = t[0]
    duration = t[-1] - t0
    n_windows = int(duration // window_seconds)
    if n_windows < 2:
        raise ValueError(
            f"Za krotki slad ({duration:.1f}s) na >= 2 pelne okna po "
            f"{window_seconds}s - dostano {n_windows}."
        )

    window_starts: List[float] = []
    states: List[MetaState] = []
    for i in range(n_windows):
        w_start = t0 + i * window_seconds
        w_end = w_start + window_seconds
        mask = (t >= w_start) & (t < w_end)
        t_win, s_win = t[mask], s[mask]
        if len(t_win) < 4:
            continue
        window_starts.append(w_start)
        states.append(window_to_meta_state(core, t_win, s_win, thresholds))

    if len(states) < 2:
        raise ValueError(f"Za malo pelnych okien z wystarczajaca liczba probek ({len(states)} < 2)")

    meta_operator = MetaOperatorM()
    M_series: List[MetaState] = []
    for i in range(len(states) - 1):
        M_series.append(meta_operator.compute(states[i], states[i + 1], dt))

    meta_map = MetaMap(meta_operator)
    phases = meta_map.detect_transitions(M_series)
    trigger = MetaTrigger().analyze(phases)

    return SeismicMetaResult(
        window_starts=window_starts, states=states,
        M_series=M_series, phases=phases, trigger=trigger,
    )
