"""
residual_offset.py — czy odchylenie od linii bazowej PO zdarzeniu ZOSTAJE
(nie wraca do zera) zamiast zanikać oscylacyjnie
================================================================================
KONTEKST (dlaczego ten plik istnieje obok ringdown.py, nie zamiast niego):
`ringdown_resonance()` (ringdown.py) klasyfikuje powrót do linii bazowej
jako oscylacyjny/nieoscylacyjny - z definicji szuka sygnału, który W KOŃCU
wraca w okolice zera (`is_oscillatory` wymaga >=2 przejść przez próg w obie
strony, `damping_ratio` mierzy jak szybko oscylacja ZANIKA z powrotem do
zera). To dobre narzędzie do pytania "czy coś dzwoni", ale złe do pytania
"czy zostawiło trwały ślad" - to są przeciwne kierunki tej samej osi.

Test predykcyjności `ringdown_resonance()` jako prekursora sejsmicznego
(`precursor_ringdown_test.py`) dał wynik NEGATYWNY (p=0.997, patrz
HISTORIA_I_TESTY.md) - ale to jest odrzucenie KONKRETNEJ cechy
(`frac_oscillatory`), nie odrzucenie hipotezy "przed dużym wstrząsem coś
się zmienia w micro-wstrząsach" w ogóle. Ta sesja (rozmowa o tym, czym jest
"informacja" w sensie sygnałowym w GIA-TIMDR/docs/theory - "informacja to
ta część delty, która NIE wraca do zera", ugruntowana konkretnie w
`theta_bifurcation.py`'s `S_up`, który nasyca się i zostawia trwały,
niezerowy odcisk) wskazała, że `frac_oscillatory` nigdy nie mierzyło TEJ
własności - mierzyło coś przeciwnego (czy sygnał WRACA oscylacyjnie).

Ten moduł operacjonalizuje "nieodwracalny odcisk" wprost: czy średnie
bezwzględne odchylenie od linii bazowej, liczone w OSTATNIM fragmencie
(tail) okna po zdarzeniu, jest nadal poza szumem tła - czyli sygnał NIE
zdążył wrócić do poziomu odniesienia zanim skończyło się okno obserwacji.

PRE-REJESTRACJA (zamrożone PRZED jakimkolwiek uruchomieniem na realnych
danych - patrz precursor_residual_offset_test.py --mode synthetic, które
MUSI przejść przed --mode real, dokładnie ta sama dyscyplina co w
ringdown.py/precursor_ringdown_test.py):

  - `baseline`/`noise_floor`: DOKŁADNIE ta sama definicja co w
    `ringdown_resonance()` (średnia z `pre_event_window` próbek przed
    zdarzeniem; `noise_floor = noise_floor_factor * std(tych próbek)`) -
    żeby porównanie było uczciwe: obie cechy widzą to samo "co jest
    szumem" na starcie, różnią się TYLKO w tym, co robią z oknem PO
    zdarzeniu.
  - `tail_fraction=0.2`: cecha liczona z OSTATNICH 20% próbek okna
    lookahead - wystarczająco dużo próbek na stabilną średnią, na tyle
    mało, żeby wczesny, przejściowy fragment (który i tak zanikłby w
    ringdown) nie zdominował wyniku. Wybór a priori (typowy "ogon
    rozkładu"), nie dostrojony do żadnych danych.
  - `persistence_threshold=1.0`: `is_persistent=True` tylko jeśli średnie
    |odchylenie| w ogonie >= 1.0 * noise_floor - czyli wciąż na zewnątrz
    pasma szumu sprzed zdarzenia. Naturalna granica (dokładnie próg
    szumu), nie dostrojona.
  - Wymaga >=5 próbek w ogonie do policzenia stabilnej średniej - inaczej
    `is_persistent=False, insufficient_tail_samples=True` (fail-closed,
    nigdy cichy False bez powodu).

CZEGO TEN MODUŁ NIE ROBI: nie twierdzi z góry, że "nieodwracalny odcisk"
JEST lepszym prekursorem niż oscylacyjny ringdown - toja też trzeba
sprawdzić testem na realnych danych (precursor_residual_offset_test.py),
z tą samą dyscypliną uczciwego raportowania wyniku negatywnego jak
wszędzie indziej w tym repo.
"""
from __future__ import annotations

import numpy as np


def residual_offset(
    t,
    s,
    event_idx: int,
    baseline: float | None = None,
    pre_event_window: int = 10,
    max_lookahead: int | None = None,
    noise_floor_factor: float = 3.0,
    tail_fraction: float = 0.2,
    persistence_threshold: float = 1.0,
    min_tail_samples: int = 5,
) -> dict:
    """Analizuje, czy `s` NIE wraca do linii bazowej po `event_idx` -
    przeciwieństwo pytania zadawanego przez `ringdown_resonance()`
    (ringdown.py). baseline/noise_floor liczone identycznie jak tam, dla
    uczciwego porównania na tych samych danych.

    Zwraca dict: baseline, noise_floor, n_tail_samples,
    mean_abs_tail_offset, offset_ratio (mean_abs_tail_offset/noise_floor,
    None jeśli noise_floor==0), is_persistent, insufficient_tail_samples.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(s)
    if n == 0 or not (0 <= event_idx < n):
        raise ValueError(f"event_idx={event_idx} poza zakresem serii o długości {n}")

    pre_start = max(0, event_idx - pre_event_window)
    pre_samples = s[pre_start:event_idx]

    if baseline is None:
        baseline = float(np.mean(pre_samples)) if len(pre_samples) else float(s[event_idx])

    noise_std = float(np.std(pre_samples)) if len(pre_samples) >= 2 else 0.0
    noise_floor = noise_floor_factor * noise_std

    end = n if max_lookahead is None else min(n, event_idx + max_lookahead)
    d = s[event_idx:end] - baseline

    result: dict = {
        "baseline": float(baseline),
        "noise_floor": float(noise_floor),
        "n_tail_samples": 0,
        "mean_abs_tail_offset": None,
        "offset_ratio": None,
        "is_persistent": False,
        "insufficient_tail_samples": True,
    }

    if len(d) < min_tail_samples:
        return result

    tail_n = max(min_tail_samples, int(round(len(d) * tail_fraction)))
    tail_n = min(tail_n, len(d))
    tail = d[-tail_n:]

    result["n_tail_samples"] = int(tail_n)
    result["insufficient_tail_samples"] = tail_n < min_tail_samples
    if result["insufficient_tail_samples"]:
        return result

    mean_abs_tail = float(np.mean(np.abs(tail)))
    result["mean_abs_tail_offset"] = mean_abs_tail

    if noise_floor > 0:
        offset_ratio = mean_abs_tail / noise_floor
        result["offset_ratio"] = offset_ratio
        result["is_persistent"] = bool(offset_ratio >= persistence_threshold)
    else:
        # Zdegenerowany przypadek: brak zmienności PRZED zdarzeniem (np.
        # stała próbka wejściowa) - noise_floor=0 sprawia, że KAŻDE
        # niezerowe odchylenie dzielone przez zero byłoby nieskończone;
        # zamiast tego jawnie: "persistent" tylko jeśli w ogóle jest
        # jakiekolwiek mierzalne odchylenie (fail-closed w drugą stronę -
        # nie twierdzimy fałszywej pewności z dzielenia przez zero).
        result["is_persistent"] = bool(mean_abs_tail > 0)

    return result
