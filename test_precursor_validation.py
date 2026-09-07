"""Testy precursor_validation.py - mechanizmu, ktory sprawia, ze
`ringdown_resonance()` SAMO wie, ze zostalo przetestowane jako sygnal
precursor na realnych danych USGS+EarthScope i ODRZUCONE (p=0.997,
patrz HISTORIA_I_TESTY.md i precursor_ringdown_test_output.json).

Sprawdzane tu:
1. Ponowne przeliczenie testu Manna-Whitneya na zamrozonych, REALNYCH
   danych (nie syntetycznych) daje ten sam werdykt co oryginalny bieg
   `precursor_ringdown_test.py --mode real`: NIEZWALIDOWANE.
2. Wlasna implementacja testu Manna-Whitneya (bez scipy) daje wyniki
   zgodne ze scipy.stats.mannwhitneyu (jesli scipy jest dostepne).
3. Pozytywna/negatywna kontrola mechanizmu walidacji na danych
   syntetycznych z jawna separacja/brakiem separacji.
4. `ringdown_resonance()` faktycznie surfaceuje ten status (nie tylko
   modul precursor_validation w izolacji) i emituje ostrzezenie.
5. Nic z powyzszego nie psuje istniejacych testow ringdown.py
   (test_ringdown.py) - te testy nadal przechodza bez modyfikacji.
"""
import warnings

import numpy as np
import pytest

import precursor_validation as pv
from precursor_validation import (
    PrecursorValidationWarning,
    get_ringdown_precursor_validation_status,
    load_real_ringdown_precursor_dataset,
    mannwhitney_validate,
    reset_validation_cache,
    validate_against_catalog,
    warn_if_unvalidated,
)
from ringdown import ringdown_resonance


@pytest.fixture(autouse=True)
def _clear_cache():
    """Kazdy test dostaje swiezo przeliczony status - inaczej pierwszy
    test, ktory go obliczy, zamrozi wynik dla calej reszty sesji pytest
    (lru_cache jest per-proces, nie per-test)."""
    reset_validation_cache()
    yield
    reset_validation_cache()


# ---------------------------------------------------------------------
# 1. Realne dane (juz zebrane w tym repo) - musi odtworzyc znany,
#    odrzucony wynik.
# ---------------------------------------------------------------------

def test_realny_zbior_danych_ma_znane_rozmiary():
    data = load_real_ringdown_precursor_dataset()
    assert len(data["pre_event"]) == 40
    assert len(data["background"]) == 60


def test_walidacja_na_realnych_danych_potwierdza_znany_odrzucony_wynik():
    """To jest sedno mechanizmu: uruchomiony na TYCH SAMYCH realnych
    danych USGS+EarthScope co oryginalny test, `validate_against_catalog()`
    musi dojsc do tego samego wniosku - brak istotnej roznicy, p bliskie
    0.997 (oryginal: 0.9970551895863518), sygnal NIEZWALIDOWANY."""
    status = validate_against_catalog()
    assert status["validated"] is False
    # Wlasna (bez-scipy) implementacja Manna-Whitneya moze dac lekko inna
    # wartosc p niz oryginalny bieg scipy (asymptotyczna aproksymacja bez
    # poprawki na ciaglosc) - liczy sie, ze WNIOSEK jest ten sam co w
    # oryginalnym biegu (p=0.9970551895863518): mocno, wyraznie
    # nieistotne, nie 4 miejsce po przecinku.
    assert status["p_value"] > 0.7
    assert status["n_pre_event"] == 40
    assert status["n_background"] == 60
    assert "NEGATYWNY" in status["reason"] or "negatyw" in status["reason"].lower() or status["p_value"] > 0.5


def test_cached_status_jest_niezwalidowany():
    status = get_ringdown_precursor_validation_status()
    assert status["validated"] is False
    assert status["p_value"] is not None
    assert status["p_value"] > 0.7  # znany wynik: p~0.997, mocno niepredykcyjny


# ---------------------------------------------------------------------
# 2. Wlasna implementacja Manna-Whitneya (bez scipy) - zgodnosc ze scipy,
#    jesli jest dostepne w srodowisku testowym.
# ---------------------------------------------------------------------

def test_mannwhitney_u_p_zgodny_ze_scipy_gdy_dostepne():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 40)
    b = rng.normal(0.3, 1, 60)  # umiarkowana roznica, z lekkimi remisami po zaokragleniu
    a = np.round(a, 2)
    b = np.round(b, 2)

    u_mine, p_mine = pv._mannwhitney_u_p(a, b)
    u_scipy, p_scipy = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")

    assert u_mine == pytest.approx(u_scipy, abs=1e-6)
    assert p_mine == pytest.approx(p_scipy, abs=0.01)


# ---------------------------------------------------------------------
# 3. Kontrola pozytywna/negatywna samego mechanizmu decyzyjnego.
# ---------------------------------------------------------------------

def test_mannwhitney_validate_pozytywna_kontrola_wyrazna_separacja():
    rng = np.random.default_rng(1)
    pre_event = rng.normal(5.0, 0.5, 30)   # wyraznie WYZSZE
    background = rng.normal(0.0, 0.5, 30)
    result = mannwhitney_validate(pre_event, background)
    assert result["validated"] is True
    assert result["p_value"] < 0.05
    assert abs(result["effect_size_r"]) >= 0.2


def test_mannwhitney_validate_negatywna_kontrola_brak_roznicy():
    rng = np.random.default_rng(2)
    pre_event = rng.normal(0.0, 1.0, 40)
    background = rng.normal(0.0, 1.0, 60)
    result = mannwhitney_validate(pre_event, background)
    assert result["validated"] is False


def test_mannwhitney_validate_za_malo_probek_jest_niezwalidowane():
    result = mannwhitney_validate([0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4])
    assert result["validated"] is False
    assert result["p_value"] is None
    assert "za malo" in result["reason"]


def test_validate_against_catalog_z_wlasnymi_danymi_moze_wyjsc_zwalidowane():
    """Mechanizm nie jest na sztywno "zawsze False" - jesli PODSTAWIONE
    dane maja realna, silna separacje, walidacja przechodzi. To pokazuje,
    ze `validated=False` na realnym zbiorze USGS+EarthScope to wynik
    TESTU, nie sztywna stala."""
    rng = np.random.default_rng(3)
    fake_real_data = {
        "pre_event": list(rng.normal(0.9, 0.05, 25)),
        "background": list(rng.normal(0.1, 0.05, 25)),
        "source": "synthetic-positive-control-for-test",
    }
    status = validate_against_catalog(fake_real_data)
    assert status["validated"] is True


def test_brakujacy_plik_danych_jest_fail_closed(tmp_path):
    missing = tmp_path / "nie_istnieje.json"
    with pytest.raises(FileNotFoundError):
        load_real_ringdown_precursor_dataset(str(missing))

    # get_ringdown_precursor_validation_status() lapie DOWOLNY wyjatek z
    # walidacji (brakujacy plik, uszkodzony JSON, ...) i zwraca status
    # fail-closed (`validated=False`) zamiast wywalac cala aplikacje -
    # symulujemy to bezposrednio podmieniajac sciezke w cache'u.
    reset_validation_cache()
    orig_path = pv._DEFAULT_DATASET_PATH
    try:
        pv._DEFAULT_DATASET_PATH = str(missing)
        status = get_ringdown_precursor_validation_status()
        assert status["validated"] is False
        assert "nie udalo" in status["reason"] or "FileNotFoundError" in status["reason"]
    finally:
        pv._DEFAULT_DATASET_PATH = orig_path
        reset_validation_cache()


# ---------------------------------------------------------------------
# 4. ringdown_resonance() faktycznie surfaceuje ten status.
# ---------------------------------------------------------------------

def _damped_oscillation(seed=0):
    fs = 10.0
    t = np.arange(0, 400.0, 1 / fs)
    event_idx = int(50.0 * fs)
    f0, tau = 0.1, 25.0
    post = t[event_idx:] - t[event_idx]
    x = np.zeros_like(t)
    x[event_idx:] = 5.0 * np.exp(-post / tau) * np.cos(2 * np.pi * f0 * post)
    rng = np.random.default_rng(seed)
    x_noisy = x + rng.normal(0, 0.05, len(t))
    return t, x_noisy, event_idx


def test_ringdown_resonance_surfaceuje_status_walidacji_precursor():
    t, s, event_idx = _damped_oscillation()
    res = ringdown_resonance(t, s, event_idx=event_idx, pre_event_window=300)

    # Matematyka ringdown dziala jak dotychczas (regresja):
    assert res["is_oscillatory"] is True

    # NOWE: status walidacji jako sygnalu precursor jest zawsze obecny
    # i uczciwie NIEZWALIDOWANY (znany wynik na realnych danych).
    assert res["is_validated_precursor"] is False
    assert res["precursor_confidence"] == 0.0
    assert isinstance(res["precursor_validation"], dict)
    assert res["precursor_validation"]["validated"] is False
    assert res["precursor_validation"]["p_value"] > 0.7


def test_ringdown_resonance_surfaceuje_status_takze_na_krotkiej_serii():
    """Sciezka wczesnego return (len(d) < 3) tez musi niesc status - to
    byla najbardziej prawdopodobna luka do przeoczenia przy dodawaniu
    pol tylko na koncu funkcji."""
    t = np.array([0.0, 1.0])
    s = np.array([0.0, 0.0])
    res = ringdown_resonance(t, s, event_idx=1, pre_event_window=1)
    assert res["is_validated_precursor"] is False
    assert res["precursor_confidence"] == 0.0
    assert "precursor_validation" in res


def test_ringdown_resonance_emituje_ostrzezenie():
    t, s, event_idx = _damped_oscillation(seed=7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ringdown_resonance(t, s, event_idx=event_idx, pre_event_window=300)
    assert any(issubclass(w.category, PrecursorValidationWarning) for w in caught)


def test_warn_if_unvalidated_zwraca_status_bez_wywalania_wyjatku():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        status = warn_if_unvalidated()
    assert status["validated"] is False
