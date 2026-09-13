"""
test_omori_forecast.py — testy syntetyczne omori_forecast.py, PRZED
dotknięciem realnego katalogu Ridgecrest (patrz test_omori_ridgecrest_real.py).

Zawiera kontrolę pozytywną (symulowany proces Omori-Utsu/Gutenberg-Richter
o ZNANYCH parametrach - odzyskanie musi być w rozsądnej tolerancji),
bramki min-N (fail-closed), i test WŁASNOŚCI odkrytej podczas budowy tego
modułu (patrz jego docstring): surowe (K,c,p) mogą być niestabilne, ale
scałkowana prognoza jest stabilniejsza - to udokumentowane, nie ukryte.
"""
import numpy as np
import pytest

from core.omori_forecast import (
    fit_omori_utsu,
    expected_count_omori,
    fit_gutenberg_richter_b,
    gr_fraction_at_least,
    forecast_probability,
    instantaneous_rate_per_hour,
    classify_risk_tier,
    current_risk_status,
    _omori_integral,
    OmoriFit,
    GRFit,
    MIN_EVENTS_FOR_FIT,
    RATE_THRESHOLD_RED_PER_HOUR,
    RATE_THRESHOLD_YELLOW_PER_HOUR,
    TIER_RED,
    TIER_YELLOW,
    TIER_GREEN,
)


def _simulate_omori(K, c, p, T, seed):
    """Symulacja niejednorodnego procesu Poissona o tempie K/(t+c)^p na
    [0,T] metoda odwrocenia skumulowanej intensywnosci (time-change
    theorem): punkty jednostkowego procesu Poissona na osi Lambda(t) sa
    odwracane z powrotem na os czasu przez bisekcje (Lambda jest scisle
    rosnaca, wiec bisekcja jest bezpieczna i dokladna)."""
    rng = np.random.default_rng(seed)

    def Lam(t):
        return K * _omori_integral(t, c, p)

    Lam_T = Lam(T)
    times = []
    S = 0.0
    while True:
        S += rng.exponential(1.0)
        if S > Lam_T:
            break
        lo, hi = 0.0, T
        for _ in range(60):
            mid = (lo + hi) / 2
            if Lam(mid) < S:
                lo = mid
            else:
                hi = mid
        times.append((lo + hi) / 2)
    return np.array(times)


def _simulate_gr_magnitudes(b, mc, n, seed):
    """M = Mc + Exp(rate=b*ln(10)) - dokladnie rozklad GR (gestosc
    f(m) = b*ln(10)*10^{-b*(m-Mc)})."""
    rng = np.random.default_rng(seed)
    return mc + rng.exponential(1.0 / (b * np.log(10)), size=n)


# ---------------------------------------------------------------------
# Omori-Utsu: calka, bramka min-N
# ---------------------------------------------------------------------

def test_omori_integral_matches_numeric_integration_p_not_1():
    from scipy.integrate import quad
    c, p, T = 15.0, 1.3, 500.0
    analytic = _omori_integral(T, c, p)
    numeric, _ = quad(lambda t: (t + c) ** -p, 0, T)
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_omori_integral_matches_numeric_integration_p_equals_1():
    from scipy.integrate import quad
    c, p, T = 15.0, 1.0, 500.0
    analytic = _omori_integral(T, c, p)
    numeric, _ = quad(lambda t: (t + c) ** -p, 0, T)
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_fit_omori_returns_insufficient_data_below_min_events():
    t = np.array([10.0, 20.0, 30.0])  # far below MIN_EVENTS_FOR_FIT
    fit = fit_omori_utsu(t, T_obs_s=1000.0)
    assert fit.insufficient_data is True
    assert fit.n_events == 3


def test_fit_omori_rejects_events_outside_observation_window():
    t = np.concatenate([np.full(MIN_EVENTS_FOR_FIT, 5.0), [2000.0]])  # 2000 > T
    with pytest.raises(ValueError):
        fit_omori_utsu(t, T_obs_s=1000.0)


# ---------------------------------------------------------------------
# KONTROLA POZYTYWNA: proces o znanych parametrach musi zostac wykryty
# (nie dokladnie odzyskany co do parametru - patrz docstring modulu o
# niepewnosci K/c/p - ale w rozsadnej tolerancji, i ZAWSZE ze zbieznoscia)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_mle_fit_converges_on_simulated_omori_process(seed):
    true_K, true_c, true_p, T = 40.0, 20.0, 1.1, 5000.0
    t = _simulate_omori(true_K, true_c, true_p, T, seed)
    fit = fit_omori_utsu(t, T)
    assert fit.n_events >= MIN_EVENTS_FOR_FIT
    assert fit.converged is True
    assert fit.p > 0  # w granicach P_BOUNDS, nie zdegenerowane
    assert fit.c_s > 0
    assert fit.K > 0


@pytest.mark.parametrize("seed", range(6))
def test_forecast_expected_count_within_reasonable_range_of_truth(seed):
    """Kluczowa wlasnosc udokumentowana w module: scalkowana prognoza
    (nie surowe K/c/p) powinna byc w rozsadnej odleglosci od prawdy, nawet
    gdy pojedyncze parametry sie roznia. Tolerancja 0.3x-3x (szeroka,
    ustalona PRZED zobaczeniem tych konkretnych liczb - patrz sanity-check
    w commicie tego pliku, gdzie najgorszy z 12 ziaren dal ok. 0.5x)."""
    true_K, true_c, true_p, T = 40.0, 20.0, 1.1, 5000.0
    t = _simulate_omori(true_K, true_c, true_p, T, seed)
    fit = fit_omori_utsu(t, T)
    true_fit = OmoriFit(n_events=0, T_obs_s=T, K=true_K, c_s=true_c, p=true_p,
                         converged=True, neg_loglik=0.0)
    exp_fit = expected_count_omori(fit, T, T + 2000.0)
    exp_true = expected_count_omori(true_fit, T, T + 2000.0)
    ratio = exp_fit / exp_true
    assert 0.3 <= ratio <= 3.0, f"seed={seed}: ratio={ratio} poza rozsadnym zakresem"


def test_expected_count_omori_raises_on_insufficient_data():
    fit = fit_omori_utsu(np.array([1.0, 2.0]), T_obs_s=10.0)
    assert fit.insufficient_data
    with pytest.raises(ValueError):
        expected_count_omori(fit, 0.0, 10.0)


def test_expected_count_monotonic_in_window_length():
    true_K, true_c, true_p, T = 40.0, 20.0, 1.1, 5000.0
    t = _simulate_omori(true_K, true_c, true_p, T, seed=1)
    fit = fit_omori_utsu(t, T)
    e1 = expected_count_omori(fit, T, T + 500.0)
    e2 = expected_count_omori(fit, T, T + 2000.0)
    e3 = expected_count_omori(fit, T, T + 10000.0)
    assert e1 < e2 < e3  # dluzsze okno = wiecej oczekiwanych zdarzen (tempo>0 wszedzie)


# ---------------------------------------------------------------------
# Gutenberg-Richter b-value
# ---------------------------------------------------------------------

@pytest.mark.parametrize("true_b,seed", [(0.9, 0), (1.0, 1), (1.1, 2)])
def test_gr_b_value_recovery(true_b, seed):
    mc = 3.5
    mags = _simulate_gr_magnitudes(true_b, mc, n=2000, seed=seed)
    fit = fit_gutenberg_richter_b(mags, mc)
    assert not fit.insufficient_data
    assert fit.b == pytest.approx(true_b, rel=0.15)  # duza probka, ciasna tolerancja


def test_gr_min_events_gate():
    mags = np.array([3.6, 3.7, 3.8])
    fit = fit_gutenberg_richter_b(mags, mc=3.5)
    assert fit.insufficient_data is True


def test_gr_fraction_at_least_decreasing_in_magnitude():
    b, mc = 1.0, 3.5
    f1 = gr_fraction_at_least(b, mc, 4.0)
    f2 = gr_fraction_at_least(b, mc, 5.0)
    f3 = gr_fraction_at_least(b, mc, 6.0)
    assert 1.0 > f1 > f2 > f3 > 0.0


def test_gr_fraction_at_least_rejects_below_mc():
    with pytest.raises(ValueError):
        gr_fraction_at_least(1.0, mc=3.5, m_target=3.0)


def test_gr_fraction_at_least_mc_itself_is_one():
    assert gr_fraction_at_least(1.0, mc=3.5, m_target=3.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------
# Prognoza polaczona (forecast_probability)
# ---------------------------------------------------------------------

def _make_calibrated_fits(seed=0):
    true_K, true_c, true_p, T = 40.0, 20.0, 1.1, 5000.0
    t = _simulate_omori(true_K, true_c, true_p, T, seed)
    omori_fit = fit_omori_utsu(t, T)
    mags = _simulate_gr_magnitudes(1.0, mc=3.5, n=2000, seed=seed + 100)
    gr_fit = fit_gutenberg_richter_b(mags, mc=3.5)
    return omori_fit, gr_fit, T


def test_forecast_probability_bounded_in_0_1():
    omori_fit, gr_fit, T = _make_calibrated_fits()
    result = forecast_probability(omori_fit, gr_fit, T, T + 5000.0, m_target=4.0)
    assert 0.0 <= result.probability_at_least_one <= 1.0
    assert result.expected_count >= 0.0


def test_forecast_probability_higher_for_lower_magnitude_threshold():
    omori_fit, gr_fit, T = _make_calibrated_fits()
    r_low = forecast_probability(omori_fit, gr_fit, T, T + 3000.0, m_target=4.0)
    r_high = forecast_probability(omori_fit, gr_fit, T, T + 3000.0, m_target=5.5)
    assert r_low.probability_at_least_one > r_high.probability_at_least_one


def test_forecast_probability_higher_for_longer_window():
    omori_fit, gr_fit, T = _make_calibrated_fits()
    r_short = forecast_probability(omori_fit, gr_fit, T, T + 500.0, m_target=4.0)
    r_long = forecast_probability(omori_fit, gr_fit, T, T + 10000.0, m_target=4.0)
    assert r_long.probability_at_least_one > r_short.probability_at_least_one


def test_forecast_probability_raises_if_either_fit_insufficient():
    omori_fit, gr_fit, T = _make_calibrated_fits()
    bad_gr = GRFit(n_events=1, mc=3.5, b=float("nan"), b_stderr=float("nan"), insufficient_data=True)
    with pytest.raises(ValueError):
        forecast_probability(omori_fit, bad_gr, T, T + 1000.0, m_target=4.0)


# ---------------------------------------------------------------------
# Tempo chwilowe + klasyfikacja poziomu (czerwony/zolty/zielony) - patrz
# omori_forecast.py's docstring dla uzasadnienia, dlaczego to tempo
# chwilowe, nie P(>=1) w rosnacym oknie (ktore nasyca sie do 1).
# ---------------------------------------------------------------------

def test_instantaneous_rate_decreasing_in_time():
    omori_fit, gr_fit, T = _make_calibrated_fits()
    r1 = instantaneous_rate_per_hour(omori_fit, gr_fit, T)
    r2 = instantaneous_rate_per_hour(omori_fit, gr_fit, T * 3)
    r3 = instantaneous_rate_per_hour(omori_fit, gr_fit, T * 10)
    assert r1 > r2 > r3 >= 0.0


def test_instantaneous_rate_raises_on_insufficient_fit():
    bad_omori = OmoriFit(n_events=1, T_obs_s=10.0, K=float("nan"), c_s=float("nan"),
                          p=float("nan"), converged=False, neg_loglik=float("nan"),
                          insufficient_data=True)
    _, gr_fit, T = _make_calibrated_fits()
    with pytest.raises(ValueError):
        instantaneous_rate_per_hour(bad_omori, gr_fit, T)


@pytest.mark.parametrize("rate,expected_tier", [
    (10.0, TIER_RED),
    (RATE_THRESHOLD_RED_PER_HOUR, TIER_RED),
    (0.2, TIER_YELLOW),
    (RATE_THRESHOLD_YELLOW_PER_HOUR, TIER_YELLOW),
    (0.001, TIER_GREEN),
    (0.0, TIER_GREEN),
])
def test_classify_risk_tier_thresholds(rate, expected_tier):
    assert classify_risk_tier(rate) == expected_tier


def test_current_risk_status_flags_extrapolation_beyond_observed_window():
    omori_fit, gr_fit, T = _make_calibrated_fits()
    status_within = current_risk_status(omori_fit, gr_fit, T * 0.5)
    status_beyond = current_risk_status(omori_fit, gr_fit, T * 5.0)
    assert status_within.is_extrapolated_beyond_observation is False
    assert status_beyond.is_extrapolated_beyond_observation is True


def test_current_risk_status_carries_fit_reliability_flag():
    omori_fit, gr_fit, T = _make_calibrated_fits()
    status = current_risk_status(omori_fit, gr_fit, T)
    assert status.fit_unreliable == omori_fit.fit_at_boundary
