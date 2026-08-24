import numpy as np
import pytest
from catalog_core import TIMDRCatalogFusion
from demo_usgs_catalog import load_snapshot


@pytest.fixture
def cat():
    return TIMDRCatalogFusion()


def test_anomalies_pusty_katalog(cat):
    idx, z = cat.anomalies(np.array([]))
    assert len(idx) == 0
    assert len(z) == 0


def test_anomalies_wykrywa_wyrazny_wiekszy_wstrzas(cat):
    rng = np.random.default_rng(0)
    mag = rng.normal(5.0, 0.15, 50)
    mag[30] = 7.5  # wyrazny mainshock
    idx, z = cat.anomalies(mag, factor=3.0)
    assert 30 in idx
    assert z[30] == np.max(z)


def test_trend_rosnacy_ciag_magnitude(cat):
    t = np.arange(30, dtype=float)
    mag = 5.0 + 0.05 * t  # narastajaca aktywnosc
    slopes, z = cat.trend(t, mag, window=10)
    assert slopes[-1] > 0


def test_rhythm_brak_okresowosci_na_losowym_procesie(cat):
    """Kontrola: na czysto losowych (nieskorelowanych) magnitude nie
    powinno byc silnej, powtarzalnej okresowosci - sprawdzamy to na
    duzej probie (200 zdarzen), zeby odroznic prawdziwy false-positive
    rate od przypadkowego trafienia na malej probie."""
    rng = np.random.default_rng(42)
    mag = rng.normal(5.2, 0.3, 200)
    periods, score = cat.rhythm(mag, max_lag=30, power_thresh=0.4)
    assert score < 0.6, f"nieoczekiwanie silny 'rytm' na losowych danych: score={score}"


def test_rhythm_wykrywa_prawdziwa_okresowosc_po_indeksie(cat):
    n = 120
    idx = np.arange(n, dtype=float)
    mag = 5.0 + 0.3 * np.sin(2 * np.pi * idx / 12)
    periods, score = cat.rhythm(mag, max_lag=30, power_thresh=0.4)
    assert 12 in periods
    assert score > 0.4


def test_nearest_aftershock_znajduje_najblizsze_zdarzenie(cat):
    t = np.array([0.0, 1.0, 5.0, 5.7, 20.0])
    mag = np.array([5.0, 5.1, 7.4, 5.0, 5.2])  # mainshock na indeksie 2
    idx, dt = cat.nearest_aftershock(t, mag)
    assert idx == 3
    assert dt == pytest.approx(0.7)


def test_nearest_aftershock_brak_zdarzen_po_mainshocku(cat):
    t = np.array([0.0, 1.0, 2.0])
    mag = np.array([5.0, 5.1, 7.4])  # mainshock ostatni
    idx, dt = cat.nearest_aftershock(t, mag)
    assert idx is None
    assert dt is None


# ============================================================
# Regresja na realnym snapshocie USGS (64 zdarzenia M5+,
# 2026-08-01 do 2026-08-14, w tym mainshock M7.4 Kolumbia)
# ============================================================

def test_snapshot_usgs_wczytuje_sie_poprawnie():
    t, mag = load_snapshot()
    assert len(t) == 64
    assert np.all(np.diff(t) >= 0)
    assert mag.max() == pytest.approx(7.4)


def test_snapshot_usgs_anomalie_wylapuja_najwieksze_wstrzasy(cat):
    """Zweryfikowano na zywych danych: detektor poprawnie wskazuje 5
    najwiekszych zdarzen w katalogu (M5.7, dwa M6.3, M6.0, M7.4) jako
    anomalie, z mainshockiem M7.4 na szczycie."""
    t, mag = load_snapshot()
    idx, z = cat.anomalies(mag, factor=3.0)
    assert len(idx) == 5
    mainshock_idx = int(np.argmax(mag))
    assert mainshock_idx in idx
    assert z[mainshock_idx] == np.max(z)


def test_snapshot_usgs_rytm_brak_falszywego_alarmu(cat):
    """Regresja: na realnym katalogu M5+ (proces zblizony do Poissona,
    brak znanej okresowosci) rhythm() liczony BEZPOSREDNIO na magnitude
    (ze znakiem) poprawnie nie znajduje zadnej okresowosci. Wczesniejsza
    wersja tego demo liczyla rytm na E=|MAD-z(magnitude)| (rektyfikacja
    przeniesiona z wielocechowego fuse()) i dawala falszywy, graniczny
    alarm score=0.434 - patrz docstring rhythm() w catalog_core.py."""
    t, mag = load_snapshot()
    periods, score = cat.rhythm(mag, max_lag=30, power_thresh=0.4)
    assert periods == []
    assert score == 0.0


def test_snapshot_usgs_aftershock_po_mainshocku_m74(cat):
    """Zweryfikowano na zywych danych: pierwsze zdarzenie po mainshocku
    M7.4 (Kolumbia) to M5.0 ok. 44 minuty pozniej - typowy aftershock."""
    t, mag = load_snapshot()
    idx, dt_h = cat.nearest_aftershock(t, mag)
    assert idx is not None
    assert mag[idx] == pytest.approx(5.0)
    assert dt_h * 60 == pytest.approx(43.7, abs=0.2)
