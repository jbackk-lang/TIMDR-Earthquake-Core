"""
test_omori_ridgecrest_real.py — omori_forecast.py na REALNYM katalogu
Ridgecrest 2019 (data/ridgecrest_2019/*.txt, lokalne pliki, BEZ internetu).
================================================================================
KONTEKST: pytanie użytkownika wprost - "na ile wcześniej sprawdź na danych
ostrzegałoby przed wstrząsem, trzymajmy się tego czasu jeśli po wstrząsie
będzie dłuższy żółte ostrzeżenie, praktycznie zielony nigdy" - ten plik to
sprawdza EMPIRYCZNIE na jedynym realnym katalogu dostępnym w tym repo
(283 realne zdarzenia, prawdziwe czasy/magnitudy z USGS FDSN, patrz
gui_app.py::demo_ridgecrest_real() dla providencji), nie tylko na symulacji.

TRZY OSOBNE PYTANIA, TRZY OSOBNE TESTY:

1. `test_early_window_fits_are_unreliable_and_overconfident` - UCZCIWY
   WYNIK NEGATYWNY, odkryty PRZED napisaniem panelu GUI: dopasowanie
   Omoriego do TYLKO pierwszych 10-30 minut po wstrząsie ląduje z p na
   granicy P_BOUNDS (fit_at_boundary=True) i MASYWNIE przeszacowuje
   przyszłe tempo (np. z 10 minut danych: oczekiwane 110 zdarzeń M>=4 w
   kolejnej godzinie, rzeczywiście wystąpiło 17 - przeszacowanie ~6x).
   To bezpośrednia odpowiedź na "na ile wcześniej ostrzegałoby": TA
   METODA, na TYM katalogu, NIE daje wiarygodnego ostrzeżenia z samych
   pierwszych ~30 minut - potrzeba więcej czasu obserwacji, zanim
   dopasowanie się ustabilizuje (patrz punkt 2).

2. `test_full_window_fit_is_well_behaved` - dopasowanie do PEŁNYCH 73.5
   minuty (146 zdarzeń M>=3.5) NIE ląduje na granicy, i daje b-value=1.08
   - bardzo blisko uniwersalnej wartości b≈1.0 znanej z literatury
   sejsmologicznej (Gutenberg-Richter dla większości regionów) - dobry
   znak, że przy wystarczającej próbie dopasowanie jest sensowne.

3. `test_tier_stays_red_throughout_observed_window_and_far_beyond` -
   potwierdza wprost intuicję użytkownika: w CAŁYM zaobserwowanym oknie
   (73.5 min) tempo chwilowe M>=4.0 pozostaje w strefie CZERWONEJ (nigdy
   nie spada nawet do żółtej), i pozostaje czerwone jeszcze wiele godzin
   dalej w ekstrapolacji - "praktycznie zielony nigdy" w krótkiej skali
   czasu jest empirycznie poprawne dla wstrząsu tej wielkości (M7.1).

4. `test_extrapolation_to_independent_later_real_data_is_roughly_consistent`
   - NIEZALEŻNA kontrola: drugi, osobny realny plik
   (ridgecrest_raw_isolated.txt, 46 zdarzeń, ~26 dni po wstrząsie,
   sekwencja "background-ish") - ekstrapolacja dopasowania z pierwszych
   73.5 minut na +26 dni przewiduje oczekiwane 0.011 zdarzenia M>=4 w
   24h - a w tym oknie REALNIE zaobserwowano 0 takich zdarzeń (max
   magnituda 3.17 z 46 zdarzeń). To JEDEN punkt walidacji poza próbą
   (n=1 sekwencja) - NIE systematyczny backtest, ale uczciwy, pozytywny
   sygnał, że mimo niepewności pojedynczych parametrów (patrz
   omori_forecast.py docstring), ekstrapolowana prognoza trafia w
   realny rząd wielkości.

MC=3.5 i MIN_EVENTS_FOR_FIT=20 sa importowane wprost z omori_forecast.py -
NIE redefiniowane tutaj, zeby nie bylo pokusy dostrojenia ich do tego
konkretnego katalogu.
"""
import os
from datetime import datetime

import numpy as np
import pytest

from omori_forecast import (
    fit_omori_utsu,
    fit_gutenberg_richter_b,
    forecast_probability,
    instantaneous_rate_per_hour,
    classify_risk_tier,
    current_risk_status,
    MC_DEFAULT,
    MIN_EVENTS_FOR_FIT,
    RISK_MAGNITUDE_DEFAULT,
    TIER_RED,
    TIER_YELLOW,
    TIER_GREEN,
)

MAINSHOCK_T0 = datetime.fromisoformat("2019-07-06T03:19:53.040000+00:00")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ridgecrest_2019")


def _load_catalog(filename):
    path = os.path.join(DATA_DIR, filename)
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts, mag = line.split(",")
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            events.append((t, float(mag)))
    events.sort()
    return events


@pytest.fixture(scope="module")
def dense_catalog():
    events = _load_catalog("ridgecrest_raw_dense.txt")
    after = [(t, m) for t, m in events if t > MAINSHOCK_T0]
    rel_s = np.array([(t - MAINSHOCK_T0).total_seconds() for t, m in after])
    mags = np.array([m for t, m in after])
    return rel_s, mags


def test_dense_catalog_loads_as_expected(dense_catalog):
    """Sanity check danych wejsciowych samych w sobie, nie modelu."""
    rel_s, mags = dense_catalog
    assert len(rel_s) > 200  # 280 zdarzen po glownym wstrzasie, patrz docstring
    assert np.all(rel_s > 0)
    assert mags.max() < 7.1  # M7.1 to sam glowny wstrzas, nie jest w tej liscie


def test_early_window_fits_are_unreliable_and_overconfident(dense_catalog):
    """UCZCIWY WYNIK NEGATYWNY (patrz docstring modulu, punkt 1): fit z
    tylko pierwszych 10 minut przeszacowuje tempo o rzad wielkosci i
    laduje na granicy P_BOUNDS."""
    rel_s, mags = dense_catalog
    Tc = 10 * 60.0
    mask = rel_s <= Tc
    sel = mags[mask] >= MC_DEFAULT
    t_fit = rel_s[mask][sel]
    assert len(t_fit) >= MIN_EVENTS_FOR_FIT  # ma wystarczajaco do proby (nie insufficient_data)

    fit = fit_omori_utsu(t_fit, Tc)
    gr = fit_gutenberg_richter_b(mags[mask], MC_DEFAULT)
    assert fit.fit_at_boundary is True  # UDOKUMENTOWANY problem, nie ukryty

    horizon_s = 3600.0
    res = forecast_probability(fit, gr, Tc, Tc + horizon_s, m_target=4.0)
    actual = int(np.sum((rel_s > Tc) & (rel_s <= Tc + horizon_s) & (mags >= 4.0)))
    # NIE twierdzimy dokladnej liczby (zmieni sie gdyby ktos przetunowal
    # optymalizator) - twierdzimy KIERUNEK i SKALE przeszacowania, ktore
    # jest uczciwym, odtwarzalnym wnioskiem.
    assert res.expected_count > actual * 3, (
        f"oczekiwano wyraznego przeszacowania (>3x) na tak krotkim oknie, "
        f"dostano oczek.={res.expected_count:.2f} vs rzeczywiste={actual}"
    )


def test_full_window_fit_is_well_behaved(dense_catalog):
    """Kontrast z powyzszym: pelne okno (146 zdarzen, 73.5 min) NIE laduje
    na granicy, i daje b-value bliski uniwersalnemu b=1.0."""
    rel_s, mags = dense_catalog
    T_full = float(rel_s.max())
    sel = mags >= MC_DEFAULT
    fit = fit_omori_utsu(rel_s[sel], T_full)
    gr = fit_gutenberg_richter_b(mags, MC_DEFAULT)

    assert fit.insufficient_data is False
    assert fit.fit_at_boundary is False
    assert gr.insufficient_data is False
    assert 0.7 < gr.b < 1.4  # szeroki, ale sensowny przedzial wokol b~1.0


def test_tier_stays_red_throughout_observed_window_and_far_beyond(dense_catalog):
    """Bezposrednia odpowiedz na pytanie uzytkownika: w calym
    zaobserwowanym oknie tempo M>=4.0 jest CZERWONE (nigdy nie spada do
    zoltego), i pozostaje czerwone znacznie dalej w ekstrapolacji."""
    rel_s, mags = dense_catalog
    T_full = float(rel_s.max())
    sel = mags >= MC_DEFAULT
    fit = fit_omori_utsu(rel_s[sel], T_full)
    gr = fit_gutenberg_richter_b(mags, MC_DEFAULT)

    # w calym zaobserwowanym oknie: sprobkuj gesto, wszedzie CZERWONY
    for t_s in np.linspace(60.0, T_full, 20):
        status = current_risk_status(fit, gr, t_s, m_target=RISK_MAGNITUDE_DEFAULT)
        assert status.tier == TIER_RED, f"t={t_s/60:.1f}min: {status.tier}, oczekiwano CZERWONY"
        assert status.is_extrapolated_beyond_observation is False

    # 6h po wstrzasie (ekstrapolacja, oznaczona jako taka) - dalej czerwony
    status_6h = current_risk_status(fit, gr, 6 * 3600.0, m_target=RISK_MAGNITUDE_DEFAULT)
    assert status_6h.tier == TIER_RED
    assert status_6h.is_extrapolated_beyond_observation is True

    # ~1 dzien: powinno juz zejsc z czerwonego (patrz recznie policzone
    # tempo w commicie tego pliku: ~0.16/h w okolicy 24h -> ZOLTY)
    status_1d = current_risk_status(fit, gr, 24 * 3600.0, m_target=RISK_MAGNITUDE_DEFAULT)
    assert status_1d.tier in (TIER_YELLOW, TIER_RED)  # nie twierdzimy dokladnej minuty przejscia


def test_extrapolation_to_independent_later_real_data_is_roughly_consistent(dense_catalog):
    """NIEZALEZNA kontrola na DRUGIM realnym pliku (~26 dni pozniej,
    sekwencja rozproszona do tla) - patrz docstring modulu, punkt 4."""
    rel_s, mags = dense_catalog
    T_full = float(rel_s.max())
    sel = mags >= MC_DEFAULT
    fit = fit_omori_utsu(rel_s[sel], T_full)
    gr = fit_gutenberg_richter_b(mags, MC_DEFAULT)

    isolated = _load_catalog("ridgecrest_raw_isolated.txt")
    assert len(isolated) >= 20  # niezalezna proba nie jest trywialnie mala
    iso_mags = np.array([m for _, m in isolated])
    n_m4_isolated = int(np.sum(iso_mags >= 4.0))
    assert n_m4_isolated == 0  # RZECZYWISTY fakt z realnych danych (max M=3.17)

    t_from = (isolated[0][0] - MAINSHOCK_T0).total_seconds()
    t_to = (isolated[-1][0] - MAINSHOCK_T0).total_seconds()
    assert t_from > 25 * 86400  # ~26 dni po wstrzasie, spoza zaobserwowanego okna

    res = forecast_probability(fit, gr, t_from, t_to, m_target=4.0)
    # Ekstrapolacja z 73.5 minuty danych o ~26 dni do przodu jest z
    # zalozenia niepewna (patrz omori_forecast.py docstring) - nie
    # twierdzimy precyzji, tylko ze RZAD WIELKOSCI zgadza sie z
    # "prawie na pewno zero" (rzeczywisty wynik).
    assert res.expected_count < 1.0
    assert res.probability_at_least_one < 0.5
