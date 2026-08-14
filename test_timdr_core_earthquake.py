import numpy as np
import pytest
from timdr_core_earthquake import TIMDR_EarthquakeCore


@pytest.fixture
def core():
    return TIMDR_EarthquakeCore()


def test_walidacja_ksztaltu(core):
    with pytest.raises(ValueError):
        core.flow(np.array([0, 1, 2]), np.array([0, 1]))


def test_walidacja_nierosnacy_czas(core):
    with pytest.raises(ValueError):
        core.flow(np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0]))


def test_nie_crashuje_na_n1_n2(core):
    for n in [0, 1, 2]:
        t = np.arange(n) * 0.01
        s = np.ones(n)
        flow_grad = core.flow(t, s)
        assert len(flow_grad) == n
        tw, ts = core.twist(flow_grad, t)
        assert len(tw) == 0
        anomalies, resid, th = core.anomalies(t, s)
        assert not np.any(np.isnan(resid))
        fronts, _, _ = core.fronts(t, s)
        assert len(fronts) == 0


def test_twist_wymaga_teraz_t(core):
    """Regresja API: twist() musi teraz przyjac t (bug: liczyl po indeksie)."""
    import inspect
    sig = inspect.signature(core.twist)
    assert "t" in sig.parameters


def test_twist_brak_falszywego_alarmu_na_przerwie_rejestracji(core):
    """
    Kluczowa regresja: fala sinusoidalna z 3-sekundowa przerwa w
    rejestracji (typowy dropout telemetrii). Prawdziwa 'sila twistu'
    na granicy przerwy powinna byc NIEWYROZNIAJACA SIE (blisko lub
    ponizej mediany), nie sztucznym szczytem.
    """
    freq = 2.0
    t = np.concatenate([np.arange(0, 30) * 0.01, np.arange(30, 60) * 0.01 + 3.0])
    s = np.sin(2 * np.pi * freq * t)

    flow_grad = core.flow(t, s)
    _, twist_strength = core.twist(flow_grad, t, threshold=0.4)

    gap_region = twist_strength[28:31]
    rest = np.concatenate([twist_strength[2:27], twist_strength[32:57]])
    median_rest = np.median(rest)

    assert np.max(gap_region) <= 2 * median_rest, (
        f"granica przerwy ({np.max(gap_region):.4f}) nadal wyraznie odstaje "
        f"od typowej wartosci reszty sygnalu ({median_rest:.4f})"
    )


def test_fronts_brak_falszywego_frontu_na_gladkiej_fali_z_przerwa(core):
    freq = 2.0
    t = np.concatenate([np.arange(0, 30) * 0.01, np.arange(30, 60) * 0.01 + 3.0])
    rng = np.random.default_rng(0)
    s = np.sin(2 * np.pi * freq * t) * 0.1 + rng.normal(0, 0.01, 60)
    fronts, _, _ = core.fronts(t, s)
    assert len(fronts) == 0


def test_anomalies_brak_falszywych_alarmow_na_skwantowanym_sygnale(core):
    """
    Regresja edge case'u MAD==0: silnie zaokraglony (skwantowany)
    sygnal bez zadnej realnej anomalii dawal w oryginale prog=0 i
    falszywe alarmy na kazdej niezerowej reszcie.
    """
    t = np.arange(30) * 0.01
    s = np.round(np.sin(2 * np.pi * 1.0 * t), 1)
    anomalies, resid, th = core.anomalies(t, s, factor=3.0)
    assert th > 0.0, "prog wyszedl 0 - MAD==0 edge case nie naprawiony"
    assert len(anomalies) <= 2, f"zbyt duzo falszywych anomalii na gladkim skwantowanym sygnale: {len(anomalies)}"


def test_anomalies_wykrywa_prawdziwy_mikro_wstrzas(core):
    rng = np.random.default_rng(1)
    n = 200
    t = np.arange(n) * 0.01
    s = rng.normal(0, 0.05, n)
    s[100] += 1.0  # wyrazny mikro-wstrzas
    anomalies, resid, th = core.anomalies(t, s, factor=3.0)
    assert 100 in anomalies


def test_fronts_wykrywa_prawdziwy_poczatek_wstrzasu(core):
    rng = np.random.default_rng(2)
    n = 300
    t = np.arange(n) * 0.01
    s = rng.normal(0, 0.02, n)
    s[150:200] += np.linspace(0, 8.0, 50)  # narastajacy wstrzas
    fronts, twist_strength, resid = core.fronts(t, s)
    assert len(fronts) > 0
    assert any(145 <= f <= 165 for f in fronts)


def test_trm_wygladza_szum(core):
    rng = np.random.default_rng(3)
    n = 100
    t = np.arange(n) * 0.01
    s = np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.3, n)
    smooth = core.trm(t, s)
    clean = np.sin(2 * np.pi * 1.0 * t)
    assert np.std(smooth - clean) < np.std(s - clean)


# ============================================================
# STA/LTA (wlasna implementacja, zweryfikowana wzgledem ObsPy)
# ============================================================

def test_sta_lta_ksztalt_i_zakres_startowy(core):
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, 200)
    ratio = core.sta_lta(s, nsta=5, nlta=20)
    assert len(ratio) == len(s)
    assert np.allclose(ratio[:19], 0.0)


def test_sta_lta_wykrywa_wzrost_energii(core):
    n = 500
    s = np.ones(n) * 0.01
    s[300:320] = 5.0  # nagly wzrost energii
    ratio = core.sta_lta(s, nsta=5, nlta=50)
    assert ratio[305] > ratio[100]
    assert ratio[305] > 5.0


def test_sta_lta_nlta_wieksze_niz_sygnal(core):
    s = np.ones(10)
    ratio = core.sta_lta(s, nsta=2, nlta=50)
    assert np.allclose(ratio, 0.0)


def test_trigger_onset_wykrywa_pojedyncze_zdarzenie(core):
    charfct = np.array([0.1, 0.2, 2.0, 3.0, 2.5, 0.3, 0.1, 0.1])
    onsets = core.trigger_onset(charfct, thr_on=1.5, thr_off=0.5)
    assert len(onsets) == 1
    assert onsets[0][0] == 2  # pierwszy indeks >= thr_on


def test_trigger_onset_pusty_gdy_brak_przekroczenia(core):
    charfct = np.array([0.1, 0.2, 0.3, 0.2])
    onsets = core.trigger_onset(charfct, thr_on=1.5, thr_off=0.5)
    assert len(onsets) == 0


def test_trigger_onset_dwa_oddzielne_zdarzenia(core):
    charfct = np.array([0.1, 2.0, 0.1, 0.1, 0.1, 2.0, 0.1])
    onsets = core.trigger_onset(charfct, thr_on=1.5, thr_off=0.5)
    assert len(onsets) == 2


def test_sta_lta_i_trigger_onset_zgodne_z_obspy(core):
    """
    Kluczowy test wiarygodności: wlasna implementacja STA/LTA i
    trigger_onset porownana bit-po-bicie z referencyjna implementacja
    ObsPy na prawdziwych danych sejsmicznych (przyklad dolaczony do
    ObsPy). Pomijany automatycznie, jesli ObsPy nie jest zainstalowane -
    nie jest to twarda zaleznosc tego repo.
    """
    obspy = pytest.importorskip("obspy")
    from obspy.signal.trigger import classic_sta_lta as obspy_sta_lta
    from obspy.signal.trigger import trigger_onset as obspy_trigger_onset

    st = obspy.read()
    for tr in st:
        data = tr.data.astype(np.float64)
        df = tr.stats.sampling_rate
        for sta_s, lta_s in [(2.5, 10.0), (1.0, 20.0)]:
            nsta, nlta = int(sta_s * df), int(lta_s * df)
            ref_ratio = obspy_sta_lta(data, nsta, nlta)
            mine_ratio = core.sta_lta(data, nsta, nlta)
            assert np.allclose(ref_ratio, mine_ratio, atol=1e-6), (
                f"sta_lta niezgodne z ObsPy dla nsta={nsta}, nlta={nlta}"
            )

            for thr_on, thr_off in [(1.5, 0.5), (1.2, 0.8)]:
                ref_onsets = obspy_trigger_onset(ref_ratio, thr_on, thr_off)
                mine_onsets = core.trigger_onset(ref_ratio, thr_on, thr_off)
                assert np.array_equal(np.asarray(ref_onsets), mine_onsets), (
                    f"trigger_onset niezgodne z ObsPy dla thr=({thr_on},{thr_off})"
                )
