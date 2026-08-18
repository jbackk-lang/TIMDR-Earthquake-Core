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


# ============================================================
# Helper _nearest_k_bounds (zamiennik KDTree) — punkt 2
# ============================================================

def test_nearest_k_bounds_zgodny_z_kdtree_przy_przerwie(core):
    """Regresja: nowy helper musi dawac dokladnie to samo sasiedztwo co
    stary KDTree, TAKZE przy nierownomiernym probkowaniu z przerwa - to
    najbardziej krytyczny przypadek, bo tu sztywne okno po indeksie by sie
    wysypalo."""
    from scipy.spatial import KDTree
    rng = np.random.default_rng(42)
    t = np.sort(rng.uniform(0, 1, 40))
    t = np.concatenate([t, t[-1] + 3.0 + np.sort(rng.uniform(0, 1, 40))])
    k = core._safe_k(len(t))
    tree = KDTree(t.reshape(-1, 1))
    for i, ti in enumerate(t):
        _, idx = tree.query([ti], k=k)
        idx_old = set(np.atleast_1d(idx).tolist())
        lo, hi = core._nearest_k_bounds(t, i, k)
        assert set(range(lo, hi + 1)) == idx_old, f"niezgodnosc przy probce {i}"


def test_flow_i_trm_bez_regresji_po_usunieciu_kdtree(core):
    """flow()/trm() musza dalej dzialac identycznie (do precyzji lstsq) po
    zamianie KDTree na helper - w tym na danych z przerwa."""
    rng = np.random.default_rng(7)
    n = 100
    t = np.sort(rng.uniform(0, 10, n))
    s = rng.normal(size=n)
    flow_grad = core.flow(t, s)
    smooth = core.trm(t, s)
    assert len(flow_grad) == n
    assert len(smooth) == n
    assert not np.any(np.isnan(flow_grad))
    assert not np.any(np.isnan(smooth))


# ============================================================
# TRM: method="adaptive" / method="savgol" — punkt 4
# ============================================================

def test_trm_nieznana_metoda_rzuca_wyjatek(core):
    with pytest.raises(ValueError):
        core.trm(np.arange(10) * 0.01, np.ones(10), method="cos_tam")


def test_trm_adaptive_wyglasza_szum(core):
    rng = np.random.default_rng(3)
    n = 150
    t = np.arange(n) * 0.01
    clean = np.sin(2 * np.pi * 1.0 * t)
    s = clean + rng.normal(0, 0.3, n)
    smooth = core.trm(t, s, method="adaptive")
    assert np.std(smooth - clean) < np.std(s - clean)


def test_trm_adaptive_mniej_wygladza_wokol_prawdziwego_skoku(core):
    """Idea adaptacji: w miejscu prawdziwej duzej zmiany okno ma byc
    mniejsze (mniej 'usztywnia' skok) niz w spokojnym tle."""
    rng = np.random.default_rng(4)
    n = 200
    t = np.arange(n) * 0.01
    s = rng.normal(0, 0.02, n)
    s[100:] += 5.0  # trwaly skok (step)
    fixed = core.trm(t, s, method="median")
    adaptive = core.trm(t, s, method="adaptive")
    # w okolicy skoku adaptacyjne wygladzenie powinno szybciej "nadazyc"
    # za nowym poziomem niz stale okno medianowe
    err_fixed = abs(fixed[103] - s[103])
    err_adapt = abs(adaptive[103] - s[103])
    assert err_adapt <= err_fixed + 1e-9


def test_trm_savgol_dziala_i_wyglasza(core):
    rng = np.random.default_rng(5)
    n = 101
    t = np.arange(n) * 0.01
    clean = np.sin(2 * np.pi * 1.0 * t)
    s = clean + rng.normal(0, 0.2, n)
    smooth = core.trm(t, s, method="savgol", window_length=11, polyorder=3)
    assert len(smooth) == n
    assert np.std(smooth - clean) < np.std(s - clean)


def test_trm_savgol_zle_okno_rzuca_wyjatek(core):
    with pytest.raises(ValueError):
        core.trm(np.arange(20) * 0.01, np.ones(20), method="savgol",
                  window_length=2, polyorder=3)


# ============================================================
# classify_anomalies — punkt 5
# ============================================================

def test_classify_anomalies_brak_dropout_na_zwyklym_szumie(core):
    # normalny szum tla, bez zadnego biegu identycznych wartosci - w
    # odroznieniu od sygnalu stale=0, ktory SAM W SOBIE jest dropoutem
    # (patrz test ponizej). Pojedyncze przypadkowe przekroczenie 3*MAD na
    # czystym szumie jest oczekiwane (factor=3.0 nie daje zera false
    # positives na skonczonej probce) - to nie jest to, co tu sprawdzamy.
    rng = np.random.default_rng(99)
    t = np.arange(50) * 0.01
    s = rng.normal(0, 0.02, 50)
    events = core.classify_anomalies(t, s)
    assert all(e["type"] != "dropout" for e in events)


def test_classify_anomalies_stala_wartosc_to_dropout(core):
    """Sygnal idealnie stale=0 przez wiele probek jest sam w sobie
    podejrzany (prawdziwy kanal sejsmiczny prawie zawsze ma jakis szum
    tla) - powinien zostac zaraportowany jako dropout, nie zignorowany."""
    t = np.arange(50) * 0.01
    s = np.zeros(50)
    events = core.classify_anomalies(t, s)
    assert len(events) == 1 and events[0]["type"] == "dropout"


def test_classify_anomalies_wykrywa_impuls(core):
    rng = np.random.default_rng(10)
    n = 200
    t = np.arange(n) * 0.01
    s = rng.normal(0, 0.02, n)
    s[100] += 3.0  # pojedyncza probka, natychmiast wraca do tla
    events = core.classify_anomalies(t, s)
    assert any(e["type"] == "impuls" and e["start"] <= 100 <= e["end"] for e in events)


def test_classify_anomalies_wykrywa_spike(core):
    rng = np.random.default_rng(11)
    n = 200
    t = np.arange(n) * 0.01
    s = rng.normal(0, 0.02, n)
    s[100:103] += 3.0  # krotki wybuch (3 probki), potem wraca do tla
    events = core.classify_anomalies(t, s)
    assert any(e["type"] == "spike" for e in events)


def test_classify_anomalies_wykrywa_step(core):
    rng = np.random.default_rng(12)
    n = 200
    t = np.arange(n) * 0.01
    s = rng.normal(0, 0.02, n)
    s[100:] += 5.0  # trwaly, natychmiastowy skok poziomu - nie wraca
    events = core.classify_anomalies(t, s)
    assert any(e["type"] == "step" for e in events)


def test_classify_anomalies_wykrywa_drift(core):
    rng = np.random.default_rng(13)
    n = 300
    t = np.arange(n) * 0.01
    s = rng.normal(0, 0.02, n)
    s[150:200] += np.linspace(0, 8.0, 50)  # stopniowo narastajacy wstrzas
    s[200:] += 8.0  # zostaje na nowym poziomie
    events = core.classify_anomalies(t, s)
    assert any(e["type"] == "drift" for e in events)


def test_classify_anomalies_wykrywa_dropout(core):
    rng = np.random.default_rng(14)
    n = 200
    t = np.arange(n) * 0.01
    s = rng.normal(0, 0.5, n)
    s[100:115] = 5.0  # czujnik "utkniety" na stalej, odstajacej wartosci
    events = core.classify_anomalies(t, s)
    assert any(e["type"] == "dropout" for e in events)


# ============================================================
# hybrid_trigger — punkt 7
# ============================================================

def test_hybrid_trigger_potwierdza_prawdziwy_wstrzas(core):
    rng = np.random.default_rng(20)
    n = 500
    t = np.arange(n) * 0.01
    s = rng.normal(0, 0.05, n)
    s[300:340] += np.sin(np.linspace(0, 6 * np.pi, 40)) * 6.0  # wyrazny wstrzas
    confirmed, rejected = core.hybrid_trigger(t, s, nsta=5, nlta=50)
    assert len(confirmed) > 0
    # nie wymagamy dokladnego startu=300 (STA/LTA/twist maja swoja bezwladnosc),
    # tylko ze potwierdzone okno POKRYWA SIE z prawdziwym wstrzasem 300-340
    assert any(c[0] <= 340 and c[1] >= 300 for c in confirmed)


def test_hybrid_trigger_odrzuca_czysty_szum_energetyczny_bez_twistu_i_anomalii(core):
    """Wzrost energii bez charakterystyki twistu ani statystycznej anomalii
    (np. szerokopasmowy szum o wiekszej amplitudzie, ale bez odstajacych
    probek wzgledem lokalnego tla) nie powinien byc potwierdzony."""
    rng = np.random.default_rng(21)
    n = 500
    s = rng.normal(0, 0.01, n)
    s[300:340] = rng.normal(0, 0.011, 40)  # ledwo wiekszy szum, nie realne zdarzenie
    t = np.arange(n) * 0.01
    confirmed, rejected = core.hybrid_trigger(t, s, nsta=5, nlta=50,
                                                anomaly_factor=5.0, twist_threshold=2.0)
    assert len(confirmed) == 0


def test_hybrid_trigger_zwraca_powod_odrzucenia(core):
    rng = np.random.default_rng(22)
    n = 300
    s = rng.normal(0, 0.01, n)
    t = np.arange(n) * 0.01
    confirmed, rejected = core.hybrid_trigger(t, s, nsta=5, nlta=30,
                                                anomaly_factor=5.0, twist_threshold=2.0)
    for r in rejected:
        assert "missing_twist" in r and "missing_anomaly" in r


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
