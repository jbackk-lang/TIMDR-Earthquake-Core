"""
omori_forecast.py — probabilistyczna prognoza wstrząsów wtórnych (rozkład
czasowy Omori-Utsu + rozkład magnitud Gutenberga-Richtera), dopasowana do
REALNEGO katalogu (czas, magnituda) zdarzeń.
================================================================================
CO TO JEST I CZYM NIE JEST (przeczytaj PRZED użyciem/pokazaniem wyniku
użytkownikowi - patrz też rozmowa, w której to powstało: pytanie wprost o
czerwone/zielone ostrzeżenie "będą kolejne/koniec" w gui_app.py):

To jest USTALONA, OPERACYJNA metoda sejsmologiczna, nie nowa matematyka
TIMDR:
  - zmodyfikowane prawo Omoriego (Omori 1894 / Utsu 1961): tempo wstrząsów
    wtórnych maleje jak K/(t+c)^p po głównym wstrząsie.
  - MLE Ogaty (Ogata 1983, "Estimation of the parameters in the modified
    Omori formula...") do dopasowania K, c, p do realnych czasów zdarzeń.
  - rozkład Gutenberga-Richtera (1944): log10 N(M>=m) = a - b*m.
  - MLE Akiego (Aki 1965) dla b-value.
  - połączenie obu w prognozę prawdopodobieństwa (Reasenberg-Jones 1989) -
    dokładnie ta metoda stoi za operacyjnymi prognozami wstrząsów wtórnych
    USGS ("Aftershock Forecast").

To NIE jest prognoza GŁÓWNEGO wstrząsu - ten problem (wiarygodny prekursor,
który odpala się PRZED zerwaniem uskoku) pozostaje otwarty w sejsmologii i
ten moduł go nie rusza (patrz baner w gui_app.py: "NOT earthquake
prediction/forecasting" - ten baner mówi o TYM problemie). To, co ten
moduł robi, to prognoza TEMPA już-rozpoczętej sekwencji wstrząsów
WTÓRNYCH - i to WYŁĄCZNIE probabilistycznie (np. "23% szans na >=1
zdarzenie M>=4.0 w najbliższej godzinie"), NIGDY jako binarne "tak/nie
będzie kolejny" lub "sekwencja się skończyła" - żaden model statystyczny
nie daje takiej pewności, więc ten moduł jej nie udaje.

WEJŚCIE: realny katalog (czas, magnituda) zdarzeń - NIE surowy przebieg
falowy i NIE triggery wykryte przez STA/LTA tego repo. Powód:
test_aftershock_swarm_detection.py pokazał, że hybrid_trigger/STA-LTA
gubi 9-60% zdarzeń podczas gęstego roju (blisko skupione wstrząsy wtórne
zlewają się w jedno okno) - dopasowywanie Omoriego do NIEPEŁNEGO zbioru
systematycznie zaniżyłoby tempo. Dlatego ten moduł celowo NIE korzysta z
detekcji tego repo, tylko z niezależnego, kompletniejszego katalogu.

METODA (trzy kroki, każdy z osobną funkcją niżej):
  1. `fit_omori_utsu()` - profile MLE: K wyliczane analitycznie z (c,p)
     (bo maksimum względem K ma zamkniętą postać), (c,p) maksymalizowane
     numerycznie (L-BFGS-B, kilka punktów startowych dla odporności).
  2. `fit_gutenberg_richter_b()` - MLE Akiego: b = log10(e)/(mean(M)-Mc).
  3. `forecast_probability()` - oczekiwana liczba zdarzeń M>=Mc w oknie
     [t1,t2] z całki tempa Omoriego, przeskalowana rozkładem GR do
     M>=m (m>=Mc), P(>=1) = 1-exp(-oczekiwana_liczba) (założenie procesu
     Poissona o zmiennym w czasie tempie - to samo założenie co w
     operacyjnym modelu Reasenberg-Jones/USGS).

ZNALEZIONA WŁASNOŚĆ STATYSTYCZNA (odkryta WŁASNYM sanity-checkiem na
symulowanym procesie o znanych K,c,p, PRZED dotknięciem realnych danych -
patrz test_omori_forecast.py): pojedyncze parametry (K,c,p) z dopasowania
MLE mogą być SILNIE niepewne nawet przy setkach zdarzeń i poprawnej
zbieżności optymalizatora (dobrze udokumentowany w literaturze
kompromis c-p, np. Wang 2010) - dla części symulowanych prób konkretne
(K,c,p) różniło się 2-7x od prawdziwych wartości, MIMO osiągnięcia
LEPSZEGO (nie gorszego) log-likelihood niż prawdziwe parametry - to nie
błąd optymalizacji, to prawdziwa niepewność estymacji przy skończonej
próbie. NATOMIAST scałkowana prognoza (`expected_count_omori()` na oknie
podobnej długości do zaobserwowanego) była SYSTEMATYCZNIE stabilniejsza
(zwykle w granicach 0.5-1.1x prawdziwej wartości) niż surowe parametry -
kompromis c-p częściowo się znosi przy całkowaniu. WNIOSEK: ten moduł (i
panel GUI zbudowany na nim) NIE powinien prezentować pojedynczych
K/c/p jako precyzyjnych "faktów" - powinien pokazywać PROGNOZĘ
(oczekiwaną liczbę / prawdopodobieństwo), i to z zastrzeżeniem, że jest
to szacunek, nie pewnik.

BRAMKA MIN-N I MC (fail-closed, USTALONE PRZED policzeniem czegokolwiek na
realnym katalogu Ridgecrest - patrz test_omori_ridgecrest_real.py dla
uczciwego uruchomienia i raportu, zamrożone w tym samym commicie co ten
plik, PRZED odpaleniem tamtego skryptu):
  - MIN_EVENTS_FOR_FIT = 20: poniżej tego liczba stopni swobody na
    3-parametrowe dopasowanie MLE jest zbyt mała - zwraca
    insufficient_data=True zamiast niepewnego wyniku.
  - MC_DEFAULT = 3.5: rozsądny, zaokrąglony wybór typowej krótkoterminowej
    magnitudy zupełności katalogu tuż po dużym wstrząsie w dobrze
    zinstrumentowanym regionie (SCSN/Kalifornia - kontekst danych
    Ridgecrest) - małe wstrząsy giną w kodzie fal większych zdarzeń w
    pierwszych minutach/godzinach. Nie dostrojone do konkretnego wyniku
    na tym katalogu - wybrane PRZED policzeniem, ile zdarzeń go
    przekracza.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MIN_EVENTS_FOR_FIT = 20
MC_DEFAULT = 3.5
P_BOUNDS = (0.3, 3.0)


# ---------------------------------------------------------------------
# 1. Omori-Utsu (czasowy rozkład tempa wstrzasow wtornych)
# ---------------------------------------------------------------------

def _omori_integral(T: float, c: float, p: float) -> float:
    """I(c,p) = integral_0^T (t+c)^-p dt - czesc calki niezalezna od K
    (K profilowane analitycznie, patrz docstring modulu)."""
    if abs(p - 1.0) < 1e-8:
        return float(np.log(T + c) - np.log(c))
    return float(((T + c) ** (1.0 - p) - c ** (1.0 - p)) / (1.0 - p))


def _omori_neg_profile_loglik(params, t_rel: np.ndarray, T: float) -> float:
    """params = (log_c, p) - optymalizacja w log-przestrzeni dla c, bo c
    rozpina rzedy wielkosci i optymalizacja w liniowej skali czesto grzeznie
    w lokalnym minimum (zweryfikowano empirycznie: patrz test_omori_forecast.py
    ::test_mle_fit_recovers_known_parameters_multi_seed, ktory bez tego
    padal na kilku z 8 ziaren)."""
    log_c, p = params
    c = float(np.exp(log_c))
    if c <= 0 or not np.isfinite(c):
        return 1e12
    I = _omori_integral(T, c, p)
    if I <= 0 or not np.isfinite(I):
        return 1e12
    n = len(t_rel)
    K_hat = n / I
    if K_hat <= 0 or not np.isfinite(K_hat):
        return 1e12
    # profilowany LL = n*log(K_hat) - p*sum(log(t_i+c)) - K_hat*I
    #              = n*log(n) - n*log(I) - p*sum(log(t_i+c)) - n
    ll = n * np.log(n) - n * np.log(I) - p * float(np.sum(np.log(t_rel + c))) - n
    if not np.isfinite(ll):
        return 1e12
    return -ll


@dataclass
class OmoriFit:
    n_events: int
    T_obs_s: float
    K: float
    c_s: float
    p: float
    converged: bool
    neg_loglik: float
    insufficient_data: bool = False
    fit_at_boundary: bool = False
    # ZNALEZIONE PRZED uzyciem na realnych danych (patrz backtest w
    # commicie tego pliku, test_omori_ridgecrest_real.py): dopasowania na
    # KROTKICH oknach obserwacji (np. pierwsze 10-30 min po glownym
    # wstrzasie) czesto ladowaly p DOKLADNIE na granicy P_BOUNDS (0.3 lub
    # 3.0) - to sygnal, ze model jest zle zidentyfikowany dla tej proby
    # (za malo krzywizny zaobserwowanej), NIE realna wlasnosc fizyczna.
    # fit_at_boundary=True oznacza "nie ufaj temu dopasowaniu do
    # ekstrapolacji" - flagowane automatycznie w fit_omori_utsu().


def fit_omori_utsu(t_rel_s: np.ndarray, T_obs_s: float,
                    min_events: int = MIN_EVENTS_FOR_FIT) -> OmoriFit:
    """t_rel_s: czasy zdarzen (sekundy) wzgledem czasu odniesienia t=0
    (zwykle glownego wstrzasu), WSZYSTKIE > 0, w oknie obserwacji
    [0, T_obs_s]. Zwraca dopasowane K (na sekunde), c (sekundy), p
    (bezwymiarowe) metoda MLE Ogaty (profile likelihood dla K, numeryczna
    optymalizacja dla c,p z kilku punktow startowych)."""
    t_rel_s = np.asarray(t_rel_s, dtype=float)
    n = len(t_rel_s)
    if n < min_events:
        return OmoriFit(n_events=n, T_obs_s=float(T_obs_s), K=float("nan"),
                         c_s=float("nan"), p=float("nan"), converged=False,
                         neg_loglik=float("nan"), insufficient_data=True)
    if np.any(t_rel_s <= 0) or np.any(t_rel_s > T_obs_s):
        raise ValueError("t_rel_s musi byc w (0, T_obs_s] - zdarzenia PRZED "
                          "lub PO oknie obserwacji nie naleza do tej calki.")

    from scipy.optimize import minimize

    # ZNALEZIONY PROBLEM (przed uzyciem na jakichkolwiek prawdziwych danych,
    # wykryte wlasnym sanity-checkiem na symulowanym procesie o znanych
    # parametrach): optymalizacja w LINIOWEJ przestrzeni dla c, z tylko 3x3
    # siatka startowa, potrafila zbiegac do zlego lokalnego minimum (K_hat
    # przeszacowane o rzad wielkosci na ok. 2 z 8 ziaren testowych, mimo
    # scipy zglaszajacego success=True). Naprawione dwoma rzeczami:
    # (1) optymalizacja w PRZESTRZENI log(c) (c rozpina rzedy wielkosci,
    #     patrz median_dt_candidates ponizej - liniowa siatka nie pokrywala
    #     jej dobrze), (2) szersza siatka startowa (5 log_c x 4 p zamiast
    #     3x3 = 20 startow zamiast 9).
    median_dt = float(np.median(np.diff(np.sort(t_rel_s)))) if n > 1 else T_obs_s / n
    c0_candidates = [max(median_dt * f, 1e-4) for f in (0.01, 0.1, 1.0, 5.0, 20.0)]
    p0_candidates = [0.6, 0.9, 1.1, 1.5]

    log_c_bounds = (np.log(1e-4), np.log(T_obs_s * 20))
    best = None
    for c0 in c0_candidates:
        for p0 in p0_candidates:
            res = minimize(
                _omori_neg_profile_loglik, x0=[np.log(c0), p0], args=(t_rel_s, T_obs_s),
                method="L-BFGS-B",
                bounds=[log_c_bounds, P_BOUNDS],
            )
            if best is None or res.fun < best.fun:
                best = res

    c_hat, p_hat = float(np.exp(best.x[0])), float(best.x[1])
    I = _omori_integral(T_obs_s, c_hat, p_hat)
    K_hat = n / I
    p_eps = (P_BOUNDS[1] - P_BOUNDS[0]) * 1e-3
    at_boundary = (p_hat <= P_BOUNDS[0] + p_eps) or (p_hat >= P_BOUNDS[1] - p_eps)

    return OmoriFit(n_events=n, T_obs_s=float(T_obs_s), K=float(K_hat),
                     c_s=c_hat, p=p_hat, converged=bool(best.success),
                     neg_loglik=float(best.fun), insufficient_data=False,
                     fit_at_boundary=bool(at_boundary))


def expected_count_omori(fit: OmoriFit, t_from_s: float, t_to_s: float) -> float:
    """Oczekiwana liczba zdarzen M>=Mc (ta magnituda, na ktorej fit byl
    policzony) w oknie [t_from_s, t_to_s] (obie wzgledem tego samego t=0
    co przy fitowaniu, t_to_s > t_from_s >= 0). To calka tempa Omoriego,
    NIE wymaga, zeby okno bylo w przyszlosci - dziala tez retrospektywnie
    (uzyteczne do testow/kalibracji)."""
    if fit.insufficient_data:
        raise ValueError("fit.insufficient_data=True - nie ma parametrow do calkowania.")
    I_to = _omori_integral(t_to_s, fit.c_s, fit.p)
    I_from = _omori_integral(t_from_s, fit.c_s, fit.p)
    return float(fit.K * (I_to - I_from))


# ---------------------------------------------------------------------
# 2. Gutenberg-Richter (rozklad magnitud)
# ---------------------------------------------------------------------

@dataclass
class GRFit:
    n_events: int
    mc: float
    b: float
    b_stderr: float
    insufficient_data: bool = False


def fit_gutenberg_richter_b(magnitudes: np.ndarray, mc: float,
                             min_events: int = MIN_EVENTS_FOR_FIT) -> GRFit:
    """MLE Akiego (1965): b = log10(e) / (mean(M) - Mc), dla M>=Mc.
    SE(b) w przyblizeniu b/sqrt(n) (Aki 1965) - podane obok, bo sam
    b-value bez niepewnosci jest niekompletna informacja (ten sam powod,
    dla ktorego kazdy test w tym repo podaje rozmiar efektu obok p-value)."""
    m = np.asarray(magnitudes, dtype=float)
    m = m[m >= mc]
    n = len(m)
    if n < min_events:
        return GRFit(n_events=n, mc=float(mc), b=float("nan"),
                     b_stderr=float("nan"), insufficient_data=True)
    mean_excess = float(np.mean(m) - mc)
    if mean_excess <= 0:
        return GRFit(n_events=n, mc=float(mc), b=float("nan"),
                     b_stderr=float("nan"), insufficient_data=True)
    b = float(np.log10(np.e) / mean_excess)
    b_stderr = float(b / np.sqrt(n))
    return GRFit(n_events=n, mc=float(mc), b=b, b_stderr=b_stderr, insufficient_data=False)


def gr_fraction_at_least(b: float, mc: float, m_target: float) -> float:
    """P(M>=m_target | M>=mc) = 10^(-b*(m_target-mc)) dla m_target>=mc
    (ogon wykladniczy rozkladu Gutenberga-Richtera)."""
    if m_target < mc:
        raise ValueError(f"m_target={m_target} < mc={mc} - poza zakresem, w ktorym GR byl dopasowany.")
    return float(10.0 ** (-b * (m_target - mc)))


# ---------------------------------------------------------------------
# 3. Polaczona prognoza probabilistyczna (Reasenberg-Jones)
# ---------------------------------------------------------------------

@dataclass
class ForecastResult:
    t_from_s: float
    t_to_s: float
    m_target: float
    expected_count: float
    probability_at_least_one: float


def forecast_probability(omori_fit: OmoriFit, gr_fit: GRFit,
                          t_from_s: float, t_to_s: float, m_target: float) -> ForecastResult:
    """Oczekiwana liczba i P(>=1) zdarzenia M>=m_target w [t_from_s, t_to_s],
    zakladajac proces Poissona o tempie Omoriego przeskalowanym rozkladem
    GR (Reasenberg-Jones 1989) - DOKLADNIE ta kombinacja stoi za
    operacyjnymi prognozami wstrzasow wtornych USGS."""
    if omori_fit.insufficient_data or gr_fit.insufficient_data:
        raise ValueError("Nie mozna prognozowac - jeden z dwoch fitow ma insufficient_data=True.")
    expected_mc = expected_count_omori(omori_fit, t_from_s, t_to_s)
    frac = gr_fraction_at_least(gr_fit.b, gr_fit.mc, m_target)
    expected_target = expected_mc * frac
    p_at_least_one = 1.0 - float(np.exp(-expected_target))
    return ForecastResult(t_from_s=t_from_s, t_to_s=t_to_s, m_target=m_target,
                           expected_count=expected_target,
                           probability_at_least_one=p_at_least_one)


# ---------------------------------------------------------------------
# 4. Tempo chwilowe + klasyfikacja poziomu ("czerwony/zolty/zielony")
# ---------------------------------------------------------------------
# DLACZEGO NIE P(>=1 zdarzenie) JAKO WSKAZNIK POZIOMU: P(>=1) w oknie o
# ROSNACEJ dlugosci dazy do 1 dla kazdego dodatniego tempa, wiec dla
# wystarczajaco dlugiego okna prognozy ZAWSZE wyjdzie "czerwony" - to
# odkryte WPROST na realnym katalogu Ridgecrest (patrz
# test_omori_ridgecrest_real.py): P(>=1 M>=4 w nastepnych 73 min) = 100%,
# CIAGLE, nawet gdy tempo juz wyraznie opada. Dlatego poziom liczony jest
# z TEMPA CHWILOWEGO (zdarzenia/godzine w danym momencie t) - malejacego
# monotonicznie w czasie, porownywalnego pomiedzy roznymi momentami
# sekwencji, bez tego artefaktu nasycenia.
#
# PROGI (USTALONE PRZED zobaczeniem, jak klasyfikuja realna sekwencje
# Ridgecrest - patrz ten sam plik testu backtestowego): przyblizone,
# zaokraglone tempo "ile razy dziennie/godzinowo zdarzenie M>=RISK_MAGNITUDE
# jest prawdopodobne":
#   CZERWONY: >=0.5/h (~1 co <2h)
#   ZOLTY:    0.05-0.5/h (~1 co 2-20h)
#   ZIELONY:  <0.05/h (~rzadziej niz raz na ~20h)
# Nie jest to "sekwencja definitywnie SIE SKONCZYLA" (Omori nie modeluje
# powrotu do tla regionalnego - patrz docstring modulu) - ZIELONY znaczy
# tylko "tempo Omoriego spadlo ponizej tego progu", nie "zero ryzyka".
RISK_MAGNITUDE_DEFAULT = 4.0
RATE_THRESHOLD_RED_PER_HOUR = 0.5
RATE_THRESHOLD_YELLOW_PER_HOUR = 0.05

TIER_RED = "czerwony"
TIER_YELLOW = "zolty"
TIER_GREEN = "zielony"


def instantaneous_rate_per_hour(omori_fit: OmoriFit, gr_fit: GRFit,
                                 t_s: float, m_target: float = RISK_MAGNITUDE_DEFAULT) -> float:
    """Tempo CHWILOWE (nie scalkowane w oknie) zdarzen M>=m_target w
    momencie t_s (sekundy od t=0), w zdarzeniach/godzine: lambda(t) =
    K/(t+c)^p * GR_frac(m_target), przeliczone z /s na /h. To pochodna
    Lambda(t), nie jej calka - stad brak artefaktu nasycenia P(>=1)->1."""
    if omori_fit.insufficient_data or gr_fit.insufficient_data:
        raise ValueError("Nie mozna liczyc tempa - jeden z dwoch fitow ma insufficient_data=True.")
    rate_per_s_mc = omori_fit.K / (t_s + omori_fit.c_s) ** omori_fit.p
    frac = gr_fraction_at_least(gr_fit.b, gr_fit.mc, m_target)
    return float(rate_per_s_mc * frac * 3600.0)


def classify_risk_tier(rate_per_hour: float) -> str:
    """Progi zamrozone PRZED zobaczeniem, jak klasyfikuja realny katalog
    Ridgecrest - patrz komentarz nad RATE_THRESHOLD_* powyzej."""
    if rate_per_hour >= RATE_THRESHOLD_RED_PER_HOUR:
        return TIER_RED
    if rate_per_hour >= RATE_THRESHOLD_YELLOW_PER_HOUR:
        return TIER_YELLOW
    return TIER_GREEN


@dataclass
class RiskStatus:
    t_s: float
    m_target: float
    rate_per_hour: float
    tier: str
    is_extrapolated_beyond_observation: bool
    fit_unreliable: bool  # True jesli omori_fit.fit_at_boundary


def current_risk_status(omori_fit: OmoriFit, gr_fit: GRFit, t_s: float,
                         m_target: float = RISK_MAGNITUDE_DEFAULT) -> RiskStatus:
    """Wygodne opakowanie dla panelu GUI: tempo + poziom + dwie flagi
    uczciwosci (czy t_s wykracza poza zaobserwowane okno = ekstrapolacja
    mniej pewna; czy sam fit ladowal na granicy parametrow = fit
    niewiarygodny niezaleznie od t_s) - patrz docstring modulu i
    OmoriFit.fit_at_boundary."""
    rate = instantaneous_rate_per_hour(omori_fit, gr_fit, t_s, m_target)
    tier = classify_risk_tier(rate)
    return RiskStatus(
        t_s=t_s, m_target=m_target, rate_per_hour=rate, tier=tier,
        is_extrapolated_beyond_observation=bool(t_s > omori_fit.T_obs_s),
        fit_unreliable=bool(omori_fit.fit_at_boundary),
    )
