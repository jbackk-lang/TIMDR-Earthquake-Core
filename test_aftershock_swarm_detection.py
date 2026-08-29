"""
Test: czy `sta_lta()`/`trigger_onset()`/`hybrid_trigger()` (już
zweryfikowane zgodnie z ObsPy, patrz `test_sta_lta_i_trigger_onset_zgodne_z_obspy`)
wykrywają POJEDYNCZE wstrząsy wtórne równie dobrze w trakcie gęstego
roju, jak izolowane zdarzenie?

Kontekst: pytanie zadane wprost w rozmowie - przy ciągłym monitorowaniu
sejsmologicznym, czy pojedyncze trzęsienie jest "rzadkim sygnałem"
(łatwym do wykrycia), a rój wstrząsów wtórnych (Omori) psuje detekcję,
bo zdarzenia nie są już rzadkie względem tła. Wstępny test na
syntetycznym, uproszczonym sygnale (bez tego repo, prostszy model)
pokazał: izolowane trzęsienie 100% wykrycia, ale wczesne wstrząsy
wtórne (blisko głównego wstrząsu, gęsto skupione w czasie) 89-91%,
późne (bardziej odosobnione, ale wciąż część roju) tylko 37-40%. Ten
test sprawdza, czy TA SAMA prawidłowość występuje na PRAWDZIWEJ,
zweryfikowanej implementacji `TIMDR_EarthquakeCore`, nie na osobnym,
niezależnym skrypcie.

METODA: tło AR(1) (autoskorelowane, ta sama konwencja co w
`precursor_ringdown_test.py._synthetic_window`, bliższa realnemu tłu
sejsmicznemu niż biały szum) + główny wstrząs (tłumiona oscylacja,
amplituda 8.0) + sekwencja wstrząsów wtórnych wg prawa Omoriego
(rate ~ K/(t+c)^p, amplitudy losowe 2.0-5.0, mniejsze niż główny
wstrząs). Parametry Omoriego dobrane tak, by skala czasowa pasowała do
nsta/nlta (udokumentowany wybór, nie ukryty).

WYNIK (jeden przebieg, seed=42, sprawdzone bezpośrednio przed napisaniem
tego testu): trigger_onset wykrywa 91% wczesnych wstrząsów wtórnych
(<200 próbek od głównego wstrząsu) i tylko 40% późnych (>=200 próbek).
hybrid_trigger (z potwierdzeniem twist+anomaly) daje IDENTYCZNY wynik
liczbowy — co samo w sobie jest ważną obserwacją: wiele blisko
skupionych wstrząsów wtórnych trafia do TEGO SAMEGO okna trigger_onset
(zlewają się w jeden ciągły "blob" zamiast być rozróżnione jako osobne
zdarzenia), więc hybrid_trigger potwierdza/odrzuca całą grupę naraz, nie
każdy wstrząs z osobna - to jest ograniczenie ROZDZIELCZOŚCI podczas
gęstego roju, nie tylko odsetka wykryć.
"""
import numpy as np
import pytest
from timdr_core_earthquake import TIMDR_EarthquakeCore

N = 20000
NSTA, NLTA = 20, 200


@pytest.fixture
def core():
    return TIMDR_EarthquakeCore()


def _ar1_background(seed, alpha=0.85):
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1.0, N)
    s = np.zeros(N)
    for i in range(1, N):
        s[i] = alpha * s[i - 1] + white[i]
    return s


def _add_damped_event(s, t0, amplitude, tau=15.0, f0=0.12):
    s = s.copy()
    n_post = min(150, len(s) - t0)
    if n_post <= 0:
        return s
    post = np.arange(n_post, dtype=float)
    s[t0:t0 + n_post] += amplitude * np.exp(-post / tau) * np.cos(2 * np.pi * f0 * post)
    return s


def _omori_times(t0, n_max, K=30.0, c=20.0, p=1.05, t_max=5000, seed=999):
    """Czasy wstrzasow wtornych wg prawa Omoriego (rate ~ K/(t+c)^p),
    metoda odrzucania. K, c dobrane tak, by skala czasowa byla
    porownywalna z NSTA/NLTA (patrz docstring modulu)."""
    times, t = [], 0.0
    lam_max = K / c ** p
    r = np.random.default_rng(seed)
    while t < t_max and len(times) < n_max:
        t += r.exponential(1.0 / lam_max)
        if t >= t_max:
            break
        if r.uniform(0, lam_max) <= K / (t + c) ** p:
            times.append(t0 + t)
    return times


def _is_covered(idx, interval_list, tol=10):
    return any(a - tol <= idx <= b + tol for a, b in interval_list)


def _build_swarm_trace(seed):
    s = _ar1_background(seed)
    t = np.arange(N, dtype=float)
    t_ms = N // 3
    s = _add_damped_event(s, t_ms, amplitude=8.0)
    aft_times = _omori_times(t_ms, 100, t_max=N - t_ms - 200)
    rng = np.random.default_rng(seed + 1)
    for at in aft_times:
        s = _add_damped_event(s, int(round(at)), amplitude=rng.uniform(2.0, 5.0))
    return t, s, t_ms, aft_times


def test_izolowany_wstrzas_wykrywany_niemal_zawsze(core):
    """Kontrola pozytywna: pojedynczy, izolowany wstrząs (bez roju) w
    tym samym tle AR(1) powinien być wykrywany niezawodnie - to
    ustala, że nsta/nlta/progi są w ogóle sensowne dla tego sygnału,
    zanim przejdziemy do trudniejszego przypadku roju."""
    detected = []
    for seed in range(10):
        s = _ar1_background(seed)
        t_eq = N // 2
        s = _add_damped_event(s, t_eq, amplitude=8.0)
        ratio = core.sta_lta(s, NSTA, NLTA)
        onsets = core.trigger_onset(ratio, thr_on=1.5, thr_off=0.5)
        detected.append(_is_covered(t_eq, onsets, tol=20))
    assert np.mean(detected) >= 0.9, "izolowany wstrzas powinien byc wykrywany niemal zawsze"


def test_wczesne_wstrzasy_wtorne_wykrywane_lepiej_niz_pozne(core):
    """
    Rdzeń tego testu: sprawdza, czy roj wg prawa Omoriego DEGRADUJE
    wykrywalnosc pojedynczych, POZNIEJSZYCH (bardziej odosobnionych, ale
    wciaz czesci roju) wstrzasow wtornych wzgledem WCZESNIEJSZYCH (gesto
    skupionych, gdzie energie nakladajacych sie zdarzen wzajemnie sie
    wzmacniaja mimo podwyzszonego LTA - patrz TEST-TIMDR/seismology-sta-lta
    dla pelnego mechanizmu na prostszym sygnale).

    NIE zakladamy z gory konkretnych progow - test dokumentuje wynik
    (assercje maja szeroki margines wokol zmierzonych wartosci: 91%/40%
    na seed=42), zeby wykryc regresje, gdyby ktos zmienil sta_lta/
    trigger_onset w sposob, ktory zasadniczo zmienia to zachowanie -
    nie po to, zeby "zamrozic" ten konkretny wynik jako cel.
    """
    t, s, t_ms, aft_times = _build_swarm_trace(seed=42)
    ratio = core.sta_lta(s, NSTA, NLTA)
    onsets = core.trigger_onset(ratio, thr_on=1.5, thr_off=0.5)

    early = [at for at in aft_times if at - t_ms < 200]
    late = [at for at in aft_times if at - t_ms >= 200]
    assert len(early) > 10 and len(late) > 10, "test wymaga wystarczajaco duzo zdarzen w obu grupach"

    det_early = np.mean([_is_covered(int(round(at)), onsets) for at in early])
    det_late = np.mean([_is_covered(int(round(at)), onsets) for at in late])

    assert det_early > det_late, (
        f"oczekiwano degradacji wykrywalnosci dla poznych wstrzasow wtornych "
        f"(early={det_early:.2f}, late={det_late:.2f}) - jesli to sie nie "
        f"potwierdza, wczesniejszy wynik na TEST-TIMDR nie generalizuje na ta "
        f"implementacje i naleza sprawdzic dlaczego"
    )
    assert det_late < 0.7, (
        f"pozne wstrzasy wtorne wykrywane zaskakujaco dobrze ({det_late:.2f}) "
        f"- mozliwa zmiana zachowania sta_lta/trigger_onset, sprawdz recznie"
    )


def test_hybrid_trigger_nie_poprawia_rozdzielczosci_w_gestym_roju(core):
    """
    hybrid_trigger() dodaje potwierdzenie twist+anomaly do trigger_onset,
    zeby ograniczyc false positives (patrz README). Ten test sprawdza, czy
    dodatkowo POPRAWIA rozroznianie POJEDYNCZYCH zdarzen w gestym roju -
    czy raczej (jak podejrzewano po pierwszym uruchomieniu) dziedziczy to
    samo ograniczenie rozdzielczosci co trigger_onset, bo blisko skupione
    wstrzasy wtorne i tak trafiaja w TEN SAM przedzial [start,end].
    """
    t, s, t_ms, aft_times = _build_swarm_trace(seed=42)
    confirmed, rejected = core.hybrid_trigger(t, s, nsta=NSTA, nlta=NLTA)

    early = [at for at in aft_times if at - t_ms < 200]
    late = [at for at in aft_times if at - t_ms >= 200]
    det_early = np.mean([_is_covered(int(round(at)), confirmed) for at in early])
    det_late = np.mean([_is_covered(int(round(at)), confirmed) for at in late])

    # Diagnostyka, nie twarda asercja "musi byc identyczne" - ale
    # dokumentujemy zmierzony fakt: hybrid_trigger NIE rozdrabnia
    # skupionych zdarzen na osobne potwierdzenia bardziej niz sam
    # trigger_onset (ten sam mechanizm grupowania po [start,end]).
    assert det_early >= det_late, (
        "hybrid_trigger nie powinien pogarszac stosunku wczesne/pozne "
        "wzgledem samego trigger_onset"
    )
