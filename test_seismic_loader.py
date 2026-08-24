import numpy as np
import pytest
import csv as csvmod
import json
import warnings
from seismic_loader import SeismicLoader


@pytest.fixture
def loader():
    return SeismicLoader(normalize=False, detrend=False, clip_outliers=False)


def test_load_waveform_bez_t_generuje_indeks(loader):
    t, s = loader.load_waveform([1.0, 2.0, 3.0])
    assert np.allclose(t, [0, 1, 2])


def test_detrend_usuwa_pelny_trend_liniowy():
    """Regresja: oryginalny detrend zostawial przesuniecie DC."""
    loader = SeismicLoader(normalize=False, detrend=True, clip_outliers=False)
    t = np.arange(50, dtype=float)
    s = 5.0 + 2.0 * t
    _, s2 = loader.load_waveform(s, t)
    assert np.allclose(s2, 0.0, atol=1e-8), f"detrend zostawil przesuniecie: {s2[0]}"


def test_despike_nie_obcina_prawdziwego_impulsu():
    """
    Regresja: globalny 5-sigma clip obcinal pojedynczy, wyrazny impuls
    wstrzasu (5.0 przy tle std=0.05) do ~1.48 (70% redukcja). Lokalny
    despike powinien zachowac znacznie wiecej sygnalu niz stara metoda,
    a przede wszystkim - jego skuteczność nie powinna zależeć od
    proporcji pliku, ktora zajmuje zdarzenie.
    """
    rng = np.random.default_rng(0)
    n = 300
    t = np.arange(n, dtype=float) * 0.01
    s = rng.normal(0, 0.05, n)
    ramp = np.concatenate([np.linspace(0, 3.0, 20), np.linspace(3.0, 0, 20)])
    s[140:180] += ramp

    loader = SeismicLoader(normalize=False, detrend=False, clip_outliers=True)
    _, s_out = loader.load_waveform(s, t)
    peak_idx = 140 + int(np.argmax(ramp))
    assert s_out[peak_idx] > s[peak_idx] * 0.9, (
        f"prawdziwy, trwajacy wstrzas zostal obciety: {s_out[peak_idx]:.3f} vs {s[peak_idx]:.3f}"
    )


def test_despike_obcina_izolowany_glitch():
    rng = np.random.default_rng(0)
    n = 300
    t = np.arange(n, dtype=float) * 0.01
    s = rng.normal(0, 0.05, n)
    s[80] = 8.0  # pojedyncza probka - typowa usterka czujnika

    loader = SeismicLoader(normalize=False, detrend=False, clip_outliers=True)
    _, s_out = loader.load_waveform(s, t)
    assert s_out[80] < 4.0


def test_load_csv_zla_nazwa_kolumny_rzuca_blad(tmp_path):
    path = tmp_path / "data.csv"
    with open(path, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(["time", "amplitude"])
        w.writerow([0.0, 1.0])
        w.writerow([0.1, 2.0])

    loader = SeismicLoader()
    with pytest.raises(ValueError):
        loader.load_csv(str(path))  # domyslne t_col="t"/s_col="s" nie istnieja


def test_load_csv_poprawne_kolumny(tmp_path):
    path = tmp_path / "data.csv"
    with open(path, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(["t", "s"])
        for i in range(10):
            w.writerow([i * 0.1, float(i)])

    loader = SeismicLoader(normalize=False, detrend=False, clip_outliers=False)
    t, s = loader.load_csv(str(path))
    assert len(t) == 10
    assert np.allclose(s, np.arange(10, dtype=float))


def test_load_json_api_zly_format_rzuca_blad():
    loader = SeismicLoader()
    bad_json = json.dumps({"data": [{"time": 0, "value": 1}, {"time": 1, "value": 2}]})
    with pytest.raises(ValueError):
        loader.load_json_api(bad_json)


def test_load_json_api_poprawny_format():
    loader = SeismicLoader(normalize=False, detrend=False, clip_outliers=False)
    good_json = json.dumps({"data": [{"t": 0, "s": 1.0}, {"t": 1, "s": 2.0}, {"t": 2, "s": 3.0}]})
    t, s = loader.load_json_api(good_json)
    assert np.allclose(t, [0, 1, 2])
    assert np.allclose(s, [1.0, 2.0, 3.0])


def test_postprocess_sortuje_po_czasie(loader):
    t, s = loader.load_waveform(s=[3.0, 1.0, 2.0], t=[2.0, 0.0, 1.0])
    assert np.allclose(t, [0.0, 1.0, 2.0])
    assert np.allclose(s, [1.0, 2.0, 3.0])


def test_postprocess_usuwa_duplikaty_czasu():
    loader = SeismicLoader(normalize=False, detrend=False, clip_outliers=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t, s = loader.load_waveform(s=[1.0, 2.0, 3.0], t=[0.0, 0.0, 1.0])
        assert len(t) == 2
        assert any("duplikat" in str(x.message) for x in w)


def test_normalize_dziala(loader2=None):
    loader = SeismicLoader(normalize=True, detrend=False, clip_outliers=False)
    t, s = loader.load_waveform([1.0, -4.0, 2.0])
    assert np.isclose(np.max(np.abs(s)), 1.0)


def test_pusty_sygnal_nie_crashuje():
    loader = SeismicLoader()
    t, s = loader.load_waveform([])
    assert len(t) == 0 and len(s) == 0


def test_integracja_z_earthquake_core(tmp_path):
    """Loader -> TIMDR_EarthquakeCore powinno dzialac end-to-end bez bledow."""
    from timdr_core_earthquake import TIMDR_EarthquakeCore

    rng = np.random.default_rng(1)
    n = 200
    t_raw = np.arange(n, dtype=float) * 0.01
    s_raw = rng.normal(0, 0.05, n) + 0.001 * t_raw  # z drobnym dryfem
    s_raw[100:130] += np.linspace(0, 3.0, 30)

    loader = SeismicLoader()
    t, s = loader.load_waveform(s_raw, t_raw)

    core = TIMDR_EarthquakeCore()
    flow_grad = core.flow(t, s)
    twist_pts, _ = core.twist(flow_grad, t)
    fronts, _, _ = core.fronts(t, s)
    assert not np.any(np.isnan(flow_grad))
