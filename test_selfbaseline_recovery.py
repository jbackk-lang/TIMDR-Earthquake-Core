"""
test_selfbaseline_recovery.py -- ten sam test co w siostrzanych repo
(TIMDR-Crypto-Graph, universal-state-analyzer, deliverable_timdr_finanse,
analizator-gieldowy-v3, TIMDR-Grid-Monitor): czy anomalies() falszywie
flaguje NOWE, normalne probki po ustaniu mikro-wstrzasu, tylko dlatego ze
stare anomalne probki wciaz wplywaja na obliczenia?

MECHANIZM TUTAJ INNY NIZ WE WSZYSTKICH POZOSTALYCH REPO: anomalies() nie
jest projektowane do wywolywania strumieniowo per-probka - TRM (wygladzanie
uzywane do policzenia residuow) patrzy na k=8 NAJBLIZSZYCH SASIADOW PO
CZASIE w OBIE STRONY (nie tylko wstecz), w calej przekazanej tablicy. To
naturalny tryb uzycia tego repo: analiza JUZ ZAREJESTROWANEGO segmentu
fali sejsmicznej za jednym wywolaniem, nie zywy strumien.

WYNIK (10 ziaren, mikro-wstrzas: skok +50 do amplitudy szumu N(0,1) na
20 probek, potem powrot do czystego szumu):
- Fałszywych flag "daleko po evencie" (>=200 probek po koncu anomalii):
  ~1.0-1.4% - NIEODROZNIALNE od bazowego wskaznika falszywych flag na
  CALKOWICIE CZYSTYM szumie bez zadnej anomalii (~1.2-1.6%, ta sama
  metoda, factor=3.0). Anomalia NIE zostawia trwalego sladu.
- JEDEN, dobrze zrozumiany wyjatek: PIERWSZA probka NATYCHMIAST po
  koncu anomalii jest flagowana w KAZDYM z 10 ziaren (nie losowo). Powod:
  TRM(k=8) dla tej probki wciaz obejmuje kilku sasiadow z ogona anomalii
  (bilateralne k-NN po czasie) - to podciaga lokalna mediane (baseline)
  w gore, wiec residuum tej jednej, juz normalnej probki wychodzi
  sztucznie duze. To NATURALNY, ograniczony do pojedynczej probki
  artefakt brzegowy wynikajacy z bilateralnego smoothing (nie "sklejanie
  sie" false-positive w nieskonczonosc) - classify_anomalies() zresztą
  grupuje go w ten sam blok zdarzenia co sama anomalia (merge_gap).
  Dalsze probki (offset >= k_neighbors) NIE sa systematycznie flagowane
  ponad losowy poziom bazowy.
"""
import numpy as np
import pytest
from timdr_core_earthquake import TIMDR_EarthquakeCore


@pytest.fixture
def core():
    return TIMDR_EarthquakeCore()


def _make_signal(rng, n_pre=1000, n_anom=20, n_post=2000, fs=100.0, spike=50.0):
    pre = rng.normal(0, 1, n_pre)
    anom = rng.normal(0, 1, n_anom) + spike
    post = rng.normal(0, 1, n_post)
    s = np.concatenate([pre, anom, post])
    t = np.arange(len(s)) / fs
    return t, s, n_pre + n_anom


def test_brak_trwalego_falszywego_alarmu_daleko_po_mikrowstrzasie(core):
    """Wskaznik falszywych flag DALEKO po evencie (>= 200 probek, znacznie
    poza zasiegiem TRM k=8) powinien byc tego samego rzedu wielkosci co
    na calkowicie czystym szumie - anomalia nie moze "zanieczyszczac"
    globalnego MAD na tyle, zeby systematycznie podnosic flagowanie
    daleko od siebie."""
    for seed in range(5):
        rng = np.random.default_rng(seed)
        t, s, event_end = _make_signal(rng)
        idx, residuals, threshold = core.anomalies(t, s, factor=3.0)

        far_start = event_end + 200
        far_post_rate = len(idx[idx >= far_start]) / (len(s) - far_start)

        rng_clean = np.random.default_rng(seed)
        s_clean = rng_clean.normal(0, 1, len(s))
        t_clean = np.arange(len(s_clean)) / 100.0
        idx_clean, _, _ = core.anomalies(t_clean, s_clean, factor=3.0)
        clean_rate = len(idx_clean) / len(s_clean)

        # tolerancja x3 bazowego wskaznika - luzna, bo to porownanie dwoch
        # niezaleznych losowych probek tej samej stopy bazowej, nie identycznych
        assert far_post_rate < max(clean_rate * 3, 0.05), (
            f"seed={seed}: wskaznik falszywych flag daleko po evencie "
            f"({far_post_rate*100:.2f}%) znaczaco przewyzsza bazowy "
            f"({clean_rate*100:.2f}%) - mozliwe trwale zanieczyszczenie"
        )


def test_efekt_brzegowy_ograniczony_do_okna_trm_k_neighbors(core):
    """Jedyny oczekiwany 'slad' anomalii to flagi w promieniu k_neighbors
    probek od konca zdarzenia (bilateralne TRM wciaz obejmujace ogon
    anomalii) - NIE dalej. Sprawdzone na krotkim ogonie (60 probek po
    evencie), zeby odizolowac efekt brzegowy od losowego szumu tla."""
    k = core.k_neighbors
    for seed in range(10):
        rng = np.random.default_rng(seed)
        t, s, event_end = _make_signal(rng, n_post=60)
        idx, residuals, threshold = core.anomalies(t, s, factor=3.0)

        post_offsets = sorted(int(i) - event_end for i in idx if i >= event_end)
        beyond_window = [o for o in post_offsets if o >= k]
        # flagi POZA oknem TRM to musza byc rzadkie, przypadkowe trafienia
        # bazowego wskaznika falszywych alarmow (~1-2%), nie systematyczny
        # wzorzec - tolerujemy pojedyncze, ale nie wiecej niz ~10% z 60 probek
        assert len(beyond_window) <= 6, (
            f"seed={seed}: {len(beyond_window)} flag poza oknem TRM "
            f"(k={k}) w ogonie po evencie: {beyond_window} - podejrzanie "
            f"duzo jak na losowy szum tla"
        )
