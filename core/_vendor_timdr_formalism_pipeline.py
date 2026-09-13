"""
timdr_formalism/pipeline.py

TIMDR-style protokół testowania formalizmu matematycznego: odróżnianie
realnej struktury matematycznej ("ta transformacja ma nietrywialny
związek z Y") od numerologii (dopasowanego wzorca bez dowodu).

To NIE jest dowodzenie twierdzeń. To mechanika do UCZCIWEGO testowania
hipotezy przeciwko modelowi null, z sześcioma krokami (patrz
docs/PROTOCOL.md dla pełnego opisu i uzasadnienia każdego z nich):

    1. Preregistration    -> Hypothesis, Preregistration
    2. Generowanie danych -> sieve_of_eratosthenes, random_background, ar1_noise
    3. Mierzalny efekt    -> metric_fn dostarczana przez wywołującego
    4. Test statystyczny  -> mann_whitney_test (Mann-Whitney U, nie "na oko")
    5. Kontrola +/-       -> run_controls (bramka: gate real run on both passing)
    6. Werdykt            -> TestResult.verdict / format_report

Ten sam protokół co "czy sygnał TIMDR ma moc predykcyjną" (test na
sygnałach czasowych) — tu zastosowany do statycznych struktur
matematycznych (ciągów, wzorów, konstrukcji kategorii) zamiast do
sygnałów czasowych.

BEZPIECZEŃSTWO IMPORTU SCIPY (2026-09-08): `mann_whitney_test()` ma
opcjonalny, czysto-numpy backend (`backend="numpy"`, domyślnie
`backend="auto"` próbuje scipy i spada na numpy, gdy go brak) — port
`_mannwhitney_u_p_numpy()` z `TIMDR-Earthquake-Core/precursor_validation.py`,
gdzie ten sam problem już wystąpił naprawdę: import `scipy` (tam
`savgol_filter`) NA POZIOMIE MODUŁU wywalił całe GUI na maszynie, gdzie
Windows Device Guard blokował DLL-e scipy — nawet dla wywołujących,
którzy w ogóle nie używali tej ścieżki kodu (patrz HISTORIA_I_TESTY.md
w tamtym repo). To repo dotąd miało `from scipy import stats` na
sztywno na poziomie modułu — ten sam potencjalny punkt awarii, teraz
naprawiony przez owinięcie importu w `try/except` i dodanie
równoważnego (zgodnego co do kilku miejsc po przecinku, normalna
aproksymacja z korektą na remisy) backendu bez zależności od scipy.
Backend numpy wspiera tylko `alternative="two-sided"` — warianty
jednostronne wymagają scipy wprost (`backend="scipy"`, zgłasza czytelny
błąd, jeśli scipy niedostępne).

=============================================================================
_vendor_timdr_formalism_pipeline.py -- ZWENDOROWANA (skopiowana 1:1, NIE
sibling-importowana) kopia TIMDR-Math-Formalism/timdr_formalism/pipeline.py,
dnia 2026-09-10. precursor_validation.py w tym repo (TIMDR-Earthquake-Core)
uzywal wczesniej wylacznie funkcji mann_whitney_test() z tego pliku przez
sys.path sibling-import z folderu-siostry TIMDR-Math-Formalism. Zwendorowano
CALY plik (nie tylko jedna funkcje - pipeline.py jest samowystarczalny,
zalezy tylko od stdlib+numpy), zeby to repo bylo samodzielne po sklonowaniu
WYLACZNIE siebie (decyzja na wyrazna prosbe: "repozytoria kodu maja byc
niezalezne od siebie"). To jest KOPIA, nie link.
=============================================================================
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - dokladnie ten scenariusz z Device Guard
    _scipy_stats = None
    _HAS_SCIPY = False


# ---------------------------------------------------------------------
# Krok 1: Preregistration — zamrożenie hipotezy PRZED dotknięciem danych
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Hypothesis:
    """Precyzyjny opis testowanej struktury.

    Musi być kompletny PRZED wygenerowaniem/dotknięciem jakichkolwiek
    danych — "czuję, że tu coś jest" nie jest hipotezą w tym sensie.

    name: krótki identyfikator (np. "prime_digit_parity_resonance")
    description: pełny, jednoznaczny opis struktury/wzoru/transformacji —
        które dokładnie liczby/obiekty, jaka dokładnie operacja
    effect_description: co dokładnie ma być mierzone (krok 3), np.
        "różnica we frakcji elementów o parzystej sumie cyfr między
        liczbami pierwszymi a tłem losowym"
    """

    name: str
    description: str
    effect_description: str


@dataclass
class Preregistration:
    """Zamrożony, podpisany (SHA-256) zapis hipotezy + parametrów testu,
    zapisany PRZED uruchomieniem czegokolwiek na realnych/losowych
    danych. Obrona przed data-snooping: zmiana definicji PO zobaczeniu
    wyniku zmienia fingerprint, więc `verify_unchanged` to złapie.
    """

    hypothesis: Hypothesis
    params: dict
    timestamp: float
    fingerprint: str

    @staticmethod
    def create(hypothesis: Hypothesis, params: dict) -> "Preregistration":
        payload = json.dumps(
            {"hypothesis": asdict(hypothesis), "params": params},
            sort_keys=True,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(payload).hexdigest()
        return Preregistration(
            hypothesis=hypothesis,
            params=params,
            timestamp=time.time(),
            fingerprint=fingerprint,
        )

    def to_dict(self) -> dict:
        return {
            "hypothesis": asdict(self.hypothesis),
            "params": self.params,
            "timestamp": self.timestamp,
            "fingerprint": self.fingerprint,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: str | Path) -> "Preregistration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Preregistration(
            hypothesis=Hypothesis(**data["hypothesis"]),
            params=data["params"],
            timestamp=data["timestamp"],
            fingerprint=data["fingerprint"],
        )

    def verify_unchanged(self, hypothesis: Hypothesis, params: dict) -> bool:
        """True tylko jeśli hipoteza/parametry są BAJT-W-BAJT takie same
        jak w momencie pre-rejestracji — łapie dostrajanie definicji po
        zobaczeniu wyniku (data-snooping)."""
        return Preregistration.create(hypothesis, params).fingerprint == self.fingerprint


# ---------------------------------------------------------------------
# Krok 2: Generatory danych — syntetyczne (dokładne, nie przybliżone)
# ---------------------------------------------------------------------

def sieve_of_eratosthenes(n: int) -> np.ndarray:
    """Dokładne liczby pierwsze <= n (sito Eratostenesa) — nie
    przybliżenie, nie tabela ręcznie przepisana (patrz skill §13 case
    study 7: ręcznie przepisane tabele cyfr π/√2/√3 zgadzały się z
    prawdziwymi wartościami tylko do ~15-17 miejsca, dokładnie tam,
    gdzie kończy się precyzja float — tu tego ryzyka nie ma, bo liczby
    całkowite nie mają błędu zaokrąglenia)."""
    if n < 2:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = False
    return np.nonzero(is_prime)[0]


def random_background(
    n: int, low: int, high: int, seed: Optional[int] = None
) -> np.ndarray:
    """Losowe całkowite tło o zadanym zakresie — podstawowy model null
    (krok 5, kontrola negatywna A/B)."""
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=n)


def ar1_noise(
    n: int, phi: float = 0.6, sigma: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Szum AR(1) — lepsza (nie-zdegenerowana) kontrola negatywna niż
    czysty biały szum. Patrz skill §9: biały szum bywa zbyt gładki, żeby
    w ogóle wyzwolić testowaną metrykę/detektor, co robi z porównania
    biały-vs-biały zdegenerowaną (0 vs 0) "kontrolę negatywną", która
    niczego realnie nie sprawdza."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    eps = rng.normal(0.0, sigma, size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


# ---------------------------------------------------------------------
# Krok 4: Prawdziwy test statystyczny — Mann-Whitney U
# ---------------------------------------------------------------------

def _mannwhitney_u_p_numpy(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Dwustronny test Manna-Whitneya, czysty numpy/stdlib (bez scipy).

    Port `_mannwhitney_u_p()` z
    `TIMDR-Earthquake-Core/precursor_validation.py` — jedno miejsce
    definicji tej konkretnej implementacji (nie druga, niezależna
    kopia): ten plik jest teraz kanoniczny, tamten sibling-importuje
    stąd (patrz `TIMDR-Earthquake-Core/precursor_validation.py`).

    Zwraca (U dla grupy `a`, dwustronna p-wartość) — aproksymacja
    normalna z korektą na remisy, bez korekty ciągłości (ta sama
    konwencja co `scipy.stats.mannwhitneyu(method='asymptotic')`).
    Zgodność sprawdzona empirycznie do kilku miejsc po przecinku przy
    rozmiarach próbek używanych w tym repo."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    all_vals = np.concatenate([a, b])
    n = len(all_vals)

    order = np.argsort(all_vals, kind="mergesort")
    sorted_vals = all_vals[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    rank_cursor = 1
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (rank_cursor + rank_cursor + (j - i)) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        rank_cursor += (j - i + 1)
        i = j + 1

    r1 = float(ranks[:n1].sum())
    u1 = r1 - n1 * (n1 + 1) / 2.0

    mu = n1 * n2 / 2.0
    _, counts = np.unique(all_vals, return_counts=True)
    tie_term = float(np.sum(counts.astype(float) ** 3 - counts.astype(float)))
    if n > 1:
        sigma2 = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    else:
        sigma2 = 0.0
    sigma = math.sqrt(sigma2) if sigma2 > 0 else 0.0

    if sigma == 0:
        # zdegenerowane (np. wszystkie wartosci identyczne) - brak dowodu na roznice
        return u1, 1.0

    z = (u1 - mu) / sigma
    p = math.erfc(abs(z) / math.sqrt(2.0))  # dwustronna p-wartosc, rozklad normalny
    return u1, min(1.0, p)


def rank_biserial_effect_size(statistic: float, n_test: int, n_background: int) -> float:
    """Rank-biserial correlation — rozmiar efektu dla testu Manna-Whitneya,
    niezależny od p-wartości (p mówi "czy", r mówi "jak dużo").

        r = 2*U/(n_test*n_background) - 1

    gdzie `U` to statystyka U zwrócona przez `scipy.stats.mannwhitneyu`
    DLA GRUPY TESTOWEJ (pierwszy argument) — czyli liczba par
    (test_i, background_j) z test_i > background_j (plus 0.5 na
    remis). Sprawdzone na skrajnych przypadkach: jeśli KAŻDA wartość
    testowa > KAŻDĄ wartość tła, U=n_test*n_background, więc r=+1;
    jeśli odwrotnie, U=0, r=-1; jeśli brak tendencji w żadną stronę
    (tyle samo par "w górę" co "w dół"), U≈n_test*n_background/2, r≈0.

    r>0 → grupa testowa ma tendencyjnie wyższe wartości niż tło.
    r<0 → niższe. |r|=1 → pełna separacja (żadnego nakładania rang).

    Progi umowne (Cohen-style, tak jak dla zwykłej korelacji):
    |r|<0.1 pomijalny, 0.1–0.3 mały, 0.3–0.5 średni, ≥0.5 duży —
    patrz `effect_size_label`.
    """
    if n_test <= 0 or n_background <= 0:
        raise ValueError("n_test i n_background muszą być > 0")
    return float(2.0 * statistic / (n_test * n_background) - 1.0)


def effect_size_label(r: float) -> str:
    """Klasyfikacja rozmiaru efektu |r| — progi umowne (Cohen-style),
    NIE substytut patrzenia na samo r; podane tylko jako czytelny skrót
    w raportach."""
    ar = abs(r)
    if ar < 0.1:
        return "pomijalny"
    if ar < 0.3:
        return "mały"
    if ar < 0.5:
        return "średni"
    return "duży"


@dataclass
class TestResult:
    # Nazwa zaczyna się od "Test" (test statystyczny), nie jest klasą
    # testów pytest — jawnie wyłącz kolekcjonowanie, żeby nie było
    # ostrzeżenia PytestCollectionWarning o konstruktorze.
    __test__ = False

    statistic: float
    pvalue: float
    n_test: int
    n_background: int
    median_test: float
    median_background: float
    alternative: str
    # Rank-biserial r = 2*U_test/(n_test*n_background) - 1 (patrz
    # `rank_biserial_effect_size` niżej). r>0: grupa testowa ma
    # tendencyjnie WYŻSZE wartości niż tło; r<0: niższe; |r|=1: pełna
    # separacja; r=0: brak tendencji. Domyślne 0.0 dla ręcznie
    # skonstruowanych TestResult (np. w testach bramki `run_controls`),
    # które nie liczą tego pola — 0.0 tam NIE oznacza "brak efektu",
    # tylko "nieustawione", nie interpretuj go w takich przypadkach.
    effect_size_r: float = 0.0

    def verdict(self, alpha: float = 0.05) -> str:
        label = effect_size_label(self.effect_size_r)
        if self.pvalue < alpha:
            direction = "niższe" if self.median_test < self.median_background else "wyższe"
            return (
                f"Efekt istotny statystycznie (p={self.pvalue:.4g} < {alpha}): "
                f"grupa testowa ma {direction} wartości metryki niż tło "
                f"(mediana {self.median_test:.4g} vs {self.median_background:.4g}). "
                f"Rozmiar efektu: r={self.effect_size_r:.3g} ({label}) — "
                f'"istotny statystycznie" nie znaczy automatycznie "duży", '
                f"sprawdź r, nie tylko p."
            )
        return (
            f"Brak istotnego efektu (p={self.pvalue:.4g} >= {alpha}) — to jest "
            f'PEŁNA, negatywna odpowiedź, nie "prawie działa". Zgodnie z '
            f"protokołem (krok 6) traktuj to jako dowód przeciwko strukturze, "
            f"nie jako niedokończony wynik. UWAGA O MOCY TESTU: to jest "
            f"prawdziwe TYLKO jeśli test miał moc do wykrycia efektu, gdyby "
            f"istniał — przy n_test={self.n_test}, n_background={self.n_background} "
            f"i wysokim p sprawdź, czy grupy w ogóle miały zmienność do "
            f'porównania (np. wariancja > 0, zdarzenia > 0). p≈1 przy zerowej '
            f'mocy NIE jest dowodem braku efektu, tylko brakiem danych do '
            f"testu (patrz TIMDR-Math-Formalism/docs/REAL_DATA_VALIDATION.md — "
            f"realny przykład, gdzie p=1 wyszło z zerowej liczby zdarzeń w "
            f"24-dniowym oknie, nie z braku struktury). Retrospektywnej "
            f'"mocy obserwowanej" liczonej z samego p NIE licz — to '
            f"zdeterminowana funkcja p-wartości, nic nowego nie wnosi; moc "
            f"licz prospektywnie, symulacją PRZED testem (patrz PROTOCOL.md)."
        )


def mann_whitney_test(
    test_values: Sequence[float],
    background_values: Sequence[float],
    alternative: str = "two-sided",
    backend: str = "auto",
) -> TestResult:
    """Krok 4: prawdziwy test statystyczny (Mann-Whitney U), nie
    porównanie percentylowe "na oko" — patrz skill §13 krok 3 protokołu:
    "Use a real significance test (Mann-Whitney U), not just a
    percentile comparison".

    `backend`: "auto" (domyślnie) używa scipy, jeśli dostępne, inaczej
    cicho spada na czysto-numpy `_mannwhitney_u_p_numpy()`. "scipy"
    wymusza scipy (zgłasza `RuntimeError`, jeśli niedostępne — nigdy nie
    udaje, że policzył coś, czego nie policzył). "numpy" wymusza
    fallback bez scipy jawnie (przydatne w środowiskach, gdzie sam
    IMPORT scipy jest ryzykowny — patrz uwaga na górze pliku) — wspiera
    WYŁĄCZNIE `alternative="two-sided"`, jednostronne warianty wymagają
    scipy.
    """
    tv = np.asarray(test_values, dtype=float)
    bv = np.asarray(background_values, dtype=float)
    if tv.size == 0 or bv.size == 0:
        raise ValueError("test_values i background_values nie mogą być puste")

    if backend not in ("auto", "scipy", "numpy"):
        raise ValueError(f"backend musi być 'auto'/'scipy'/'numpy', dostano {backend!r}")

    use_scipy = _HAS_SCIPY if backend == "auto" else (backend == "scipy")

    if use_scipy:
        if not _HAS_SCIPY:
            raise RuntimeError(
                "backend='scipy' zażądany, ale scipy nie jest dostępne/importowalne "
                "w tym środowisku (patrz uwaga o Device Guard na górze pliku) — "
                "użyj backend='numpy' albo backend='auto'."
            )
        result = _scipy_stats.mannwhitneyu(tv, bv, alternative=alternative)
        statistic = float(result.statistic)
        pvalue = float(result.pvalue)
    else:
        if alternative != "two-sided":
            raise ValueError(
                "backend='numpy' (fallback bez scipy) wspiera wyłącznie "
                "alternative='two-sided' — warianty jednostronne wymagają "
                "backend='scipy'."
            )
        statistic, pvalue = _mannwhitney_u_p_numpy(tv, bv)

    return TestResult(
        statistic=statistic,
        pvalue=pvalue,
        n_test=int(tv.size),
        n_background=int(bv.size),
        median_test=float(np.median(tv)),
        median_background=float(np.median(bv)),
        alternative=alternative,
        effect_size_r=rank_biserial_effect_size(statistic, tv.size, bv.size),
    )


# ---------------------------------------------------------------------
# Krok 5: Kontrola pozytywna i negatywna — bramka przed testem głównym
# ---------------------------------------------------------------------

@dataclass
class ControlResult:
    positive: TestResult
    negative: TestResult
    passed: bool
    reason: str


def run_controls(
    metric_fn: Callable[[np.ndarray], float],
    positive_injector: Callable[[int, Optional[int]], np.ndarray],
    negative_generator_a: Callable[[int, Optional[int]], np.ndarray],
    negative_generator_b: Callable[[int, Optional[int]], np.ndarray],
    n_windows: int = 30,
    window_size: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
) -> ControlResult:
    """Krok 5: kontrola pozytywna i negatywna, uruchomione RAZEM, PRZED
    testem na realnych/głównych danych — bramkuj główny przebieg wynikiem
    obu (patrz skill §9 krok 4: "Run a synthetic self-test with BOTH a
    positive and a negative control before running on real data, and
    gate the real run on both passing").

    metric_fn(data) -> float
        Ta sama metryka, która będzie użyta w teście głównym.
    positive_injector(window_size, seed) -> dane z wstrzykniętym efektem,
        który metric_fn POWINNA wykryć (kontrola: pipeline działa).
    negative_generator_a/b(window_size, seed) -> dwie NIEZALEŻNE próbki
        tła bez wstrzykniętego efektu (kontrola: pipeline nie fałszuje
        alarmów na czystym szumie). Użyj innego seeda dla B niż dla A.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_windows)

    pos_values = [metric_fn(positive_injector(window_size, int(s))) for s in seeds]
    neg_a_values = [metric_fn(negative_generator_a(window_size, int(s))) for s in seeds]
    neg_b_values = [metric_fn(negative_generator_b(window_size, int(s) + 1)) for s in seeds]

    positive = mann_whitney_test(pos_values, neg_a_values)
    negative = mann_whitney_test(neg_a_values, neg_b_values)

    pos_ok = positive.pvalue < alpha
    neg_ok = negative.pvalue >= alpha
    passed = pos_ok and neg_ok

    if passed:
        reason = (
            "Kontrola pozytywna wykryła wstrzyknięty efekt, kontrola "
            "negatywna nie dała fałszywego alarmu — metryka jest sensownie "
            "skalibrowana."
        )
    elif not pos_ok and not neg_ok:
        reason = (
            "Kontrola pozytywna NIE wykryła znanego efektu I kontrola "
            "negatywna dała fałszywy alarm — metryka jest zepsuta, nie "
            "testuj na niej realnych/głównych danych."
        )
    elif not pos_ok:
        reason = (
            "Kontrola pozytywna nie wykryła znanego, wstrzykniętego efektu "
            "— metryka jest za mało czuła, wynik testu głównego byłby "
            "niediagnostyczny."
        )
    else:
        reason = (
            "Kontrola negatywna dała fałszywy alarm na dwóch niezależnych "
            "próbkach tła bez efektu — metryka ma za dużo fałszywych "
            "trafień, żeby jej ufać."
        )

    return ControlResult(positive=positive, negative=negative, passed=passed, reason=reason)


# ---------------------------------------------------------------------
# Korekta na wielokrotne porównania (look-elsewhere effect)
# ---------------------------------------------------------------------

def bonferroni_correct(pvalue: float, n_comparisons: int) -> float:
    """Korekta Bonferroniego za liczbę przeszukanych wariantów/okien —
    patrz skill §13 case study 1: "lokalnie dobre" okno o p≈0.012
    wyparowało do p≈1.0 po korekcie za ~2000 przeszukanych okien.
    Wymagana zawsze, gdy hipoteza powstała z przeszukania wielu
    kandydatów (pozycji, progów, wariantów) zamiast z jednej,
    z góry ustalonej definicji."""
    if n_comparisons < 1:
        raise ValueError("n_comparisons musi być >= 1")
    return float(min(1.0, pvalue * n_comparisons))


# ---------------------------------------------------------------------
# Krok 6: Werdykt — czytelny raport
# ---------------------------------------------------------------------

def format_report(
    hypothesis: Hypothesis,
    controls: ControlResult,
    main_result: Optional[TestResult],
    n_comparisons: int = 1,
    alpha: float = 0.05,
) -> str:
    """Składa wynik kontroli + testu głównego w czytelny raport tekstowy
    (krok 6: akceptuj wynik negatywny jako pełną odpowiedź, nie
    "prawie działa")."""
    lines = [
        f"# Raport TIMDR-Formalism: {hypothesis.name}",
        "",
        f"Hipoteza: {hypothesis.description}",
        f"Mierzony efekt: {hypothesis.effect_description}",
        "",
        "## Kontrole (krok 5)",
        f"- Pozytywna: p={controls.positive.pvalue:.4g} "
        f"({'wykryta' if controls.positive.pvalue < alpha else 'NIE wykryta'})",
        f"- Negatywna: p={controls.negative.pvalue:.4g} "
        f"({'brak fałszywego alarmu' if controls.negative.pvalue >= alpha else 'FAŁSZYWY ALARM'})",
        f"- Bramka kontrolna: {'PRZESZŁA' if controls.passed else 'NIE PRZESZŁA'} — {controls.reason}",
        "",
    ]

    if not controls.passed:
        lines.append(
            "Test główny NIE został policzony (lub jego wynik byłby "
            "niewiarygodny) — metryka nie przeszła bramki kontrolnej. "
            "Popraw metrykę i powtórz kontrolę, zanim spojrzysz na dane "
            "główne."
        )
        return "\n".join(lines)

    lines.append("## Test główny (krok 4)")
    if main_result is None:
        lines.append("Kontrole przeszły, ale test główny nie został jeszcze uruchomiony.")
        return "\n".join(lines)

    corrected = (
        bonferroni_correct(main_result.pvalue, n_comparisons)
        if n_comparisons > 1
        else main_result.pvalue
    )
    lines.append(f"- p (surowe) = {main_result.pvalue:.4g}")
    lines.append(
        f"- rozmiar efektu r = {main_result.effect_size_r:.3g} "
        f"({effect_size_label(main_result.effect_size_r)}) — niezależnie od p, "
        f"patrz sekcja 'Effect size' w PROTOCOL.md"
    )
    if n_comparisons > 1:
        lines.append(
            f"- p (po korekcie Bonferroniego za {n_comparisons} "
            f"porównań) = {corrected:.4g}"
        )
    lines.append("")
    lines.append("## Werdykt (krok 6)")
    if n_comparisons <= 1:
        lines.append(main_result.verdict(alpha=alpha))
    else:
        verdict = (
            "efekt istotny statystycznie nawet po korekcie"
            if corrected < alpha
            else 'brak efektu po korekcie — numerologia, nie struktura ("prawie działa" nie istnieje w tym protokole)'
        )
        lines.append(f"Po korekcie Bonferroniego: {verdict} (p_corrected={corrected:.4g}).")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Rozszerzenie: samokalibracja progu K operatora rezonansu
# ---------------------------------------------------------------------
#
# Cienki wrapper na timdr_formalism.calibration.calibrate_resonance_K,
# zeby samokalibracja byla wywolywalna jako czesc normalnego przeplywu
# analizy tego repo (import z timdr_formalism / timdr_formalism.pipeline),
# a nie tylko z osobnego skryptu
# (examples/real_weather_resonance_validation.py). Import LOKALNY
# (wewnatrz funkcji), nie na gorze pliku: calibration.py importuje
# `bonferroni_correct` STAD (z pipeline.py) do korekty za wielokrotne
# testowane kandydatow K -- import calibration na gorze tego pliku
# stworzylby cykl importow.

def calibrate_resonance_threshold(params_flags, K: int, **kwargs):
    """Samokalibracja progu K operatora rezonansu (R(t) = 1[liczba
    jednoczesnie anomalnych parametrow >= K]) na realnym oknie
    wielu-parametrowych flag anomalii -- test permutacyjny (metodyka z
    examples/real_weather_resonance_validation.py) + kontrola mocy +
    rekomendacja skalibrowanego K albo jawne "za mala moc, kalibracja
    nie zastosowana". Patrz timdr_formalism/calibration.py po pelny
    opis i docs/PROTOCOL.md §4b po uzasadnienie kontroli mocy."""
    from .calibration import calibrate_resonance_K

    return calibrate_resonance_K(params_flags, K, **kwargs)
