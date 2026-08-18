# TIMDR-Earthquake-Core

Rdzeń analizy sejsmicznej (`timdr_core_earthquake.py`): lokalny gradient
amplitudy (`flow`), nagłe zmiany kierunku / początek wstrząsu (`twist`),
wygładzenie szumu (`trm`), mikro-wstrząsy (`anomalies`), punkty
rozpoczęcia wstrząsu (`fronts`) i klasyczny picker STA/LTA (`sta_lta` /
`trigger_onset`, zweryfikowany zgodnie z ObsPy — patrz testy).

## ⚠️ Topology(t) / "Rezonans TIMDR" — test predykcyjności: WYNIK NEGATYWNY

Osobny wątek dyskusji zaproponował `R(t) = |Flow|·|Twist|·|Defect|·Topology`
jako "punkt zapłonu zdarzenia" z zapowiadaną predykcją ("Predykcja: tak").
`Topology(t)` zostało zoperacjonalizowane uczciwie — `topology_features.py`,
sliding-window (delay) embedding + homologia uporczywa Vietoris-Rips
(`gudhi`), metoda SW1PerS (Perea & Harer 2015), nie ad hoc metryka.
Parametry embeddingu zamrożone na syntetycznym sanity-checku
(sinus/szum/trend) PRZED policzeniem czegokolwiek na tym repo, żeby
uniknąć data snoopingu.

Test w `analyze_topology_resonance_seismic.py`: czy `R(t)`/`Topology(t)`,
liczone WYŁĄCZNIE z danych sprzed etykietowanego początku zdarzenia,
podnosi się w oknie tuż PRZED startem wstrząsu bardziej niż w losowych
oknach z tła (jedyne pytanie, które faktycznie znaczyłoby "predykcja" w
tym kontekście — `fronts()`/`hybrid_trigger()` już wykrywają zdarzenie,
które się zaczęło, to nie jest to samo).

**Wynik: nie.** `R(t)` jest matematycznie zdegenerowany (iloczyn czterech
już rzadkich wskaźników koliduje w ~0 niemal wszędzie). Sama `Topology(t)`
nie jest podwyższona przed startem ani na nagłym wstrząsie (41.5 percentyl
względem tła), ani na stopniowym dryfie (34.3 percentyl) — w obu
przypadkach poniżej mediany tła, nie powyżej. Kontrola negatywna (czysty
szum, brak zdarzenia) nie dała fałszywego alarmu. Pełny wynik:
`topology_resonance_seismic_output.txt`. Ten sam test na realnych danych
BTC/złota (`timdr-finanse/analyze_topology_resonance.py`) dał ten sam
werdykt — `R(t)` zdegenerowany, `Topology(t)` bez trwałego sygnału
out-of-sample (na złocie znak korelacji odwraca się między treningiem a
testem — klasyczny sygnał przeuczenia, nie realnej struktury).

## Propozycja (7 punktów) i co z niej weszło

Do repo trafiła propozycja siedmiu usprawnień, w tym modułu
"interpretacji fizycznej" mapującego flow/twist/rhythm na
EARTHQUAKE/BLAST/NOISE/MINING/TECTONIC oraz klasyfikacji fal na
P-wave/S-wave. Te trzy punkty (1, 3, 6) **celowo nie powstały** —
pojedynczy kanał amplitudy `s(t)` nie niesie informacji potrzebnej do
takiego rozróżnienia: P vs S wymaga polaryzacji ruchu / 3 składowych
sejsmometru, a trzęsienie vs wybuch klasycznie rozróżnia się widmowo
(np. stosunek Pn/Lg) plus głębokością ogniska i czasem trwania kody —
nazwanie kategorii prawdziwymi terminami sejsmologicznymi nie tworzy
między nimi fizycznego mostu. Zaimplementowane zostały cztery
pozostałe, które da się uczciwie policzyć z samego `s(t)`:

- **Szybszy `flow()`/`trm()`** (punkt 2) — KDTree per-punkt zastąpiony
  helperem `_nearest_k_bounds()`: skoro `t` jest ściśle rosnące, k
  najbliższych sąsiadów po czasie zawsze tworzy ciągły przedział
  indeksów wokół `i`, więc wystarczy dwuwskaźnikowe rozszerzanie okna
  zamiast budowy drzewa. Zweryfikowano identyczność z KDTree (0
  rozbieżności na 80 próbkach z przerwą w rejestracji) i przyspieszenie
  ~2.5x na n=20000. Uwaga: to NIE jest sztywne okno po indeksie
  `[i-k:i+k]` (taka wersja zepsułaby się dokładnie tam, gdzie `twist()`
  musi radzić sobie z przerwą w telemetrii) — sąsiedztwo liczone jest
  po realnej odległości w `t`.
- **`trm(..., method="adaptive"/"savgol")`** (punkt 4) — obok
  domyślnej mediany k-NN: `"adaptive"` skaluje rozmiar okna odwrotnie
  do lokalnej zmienności (mniejsze okno tam, gdzie dzieje się coś
  realnego, żeby mediana nie "usztywniała" skoku; większe w spokojnym
  tle, dla mocniejszego wygładzenia szumu); `"savgol"` to filtr
  Savitzky-Golay jako alternatywa lepiej zachowująca kształt zbocza
  (zakłada w przybliżeniu równomierne próbkowanie — nie używać przy
  danych z lukami czasowymi).
- **`classify_anomalies(t, s)`** (punkt 5) — grupuje sąsiednie punkty
  z `anomalies()` w zdarzenia i opisuje ich KSZTAŁT (nie typ
  fizyczny): `impuls` (pojedyncza próbka, wraca), `spike` (krótki
  wybuch, wraca), `step` (trwały skok poziomu), `drift` (poziom
  narasta stopniowo, nie skokiem), `dropout` (długi bieg praktycznie
  identycznych wartości — typowe dla utkniętego czujnika, wykrywane
  niezależnie od progu MAD, bo środek długiego płaskiego biegu ma
  lokalną medianę równą sobie samemu i `anomalies()` go nie łapie).
- **`hybrid_trigger(t, s, nsta, nlta)`** (punkt 7) — zdarzenie z
  `trigger_onset(sta_lta(...))` jest potwierdzone tylko, gdy w jego
  sąsiedztwie występuje też silny `twist` ORAZ punkt z `anomalies()`;
  bez tego trafia do listy `rejected` z podanym powodem
  (`missing_twist`/`missing_anomaly`). To ogranicza false positives
  samego STA/LTA (reaguje na każdy wzrost energii), ale **nie jest to
  zwalidowane względem katalogu prawdziwych zdarzeń** — czy faktycznie
  poprawia precision/recall, a nie tylko obcina też prawdziwe wykrycia,
  wymaga testu na danych z etykietami, tak jak każdy inny próg w tym
  projekcie.

Domyślne zachowanie istniejących metod (`flow`, `twist`, `trm()` bez
`method=`, `anomalies`, `fronts`, `sta_lta`, `trigger_onset`) się nie
zmieniło — wszystkie dotychczasowe testy przechodzą bez modyfikacji.

## Przykład użycia

```python
from timdr_core_earthquake import TIMDR_EarthquakeCore

core = TIMDR_EarthquakeCore()
flow_grad = core.flow(t, s)
twist_pts, twist_strength = core.twist(flow_grad, t)
smooth = core.trm(t, s)
anomaly_pts, residuals, th = core.anomalies(t, s)
fronts, _, _ = core.fronts(t, s)

# nowe:
smooth_adapt = core.trm(t, s, method="adaptive")
smooth_sg = core.trm(t, s, method="savgol", window_length=11, polyorder=3)
events = core.classify_anomalies(t, s)          # [{'start','end','duration','type','level_shift'}, ...]
confirmed, rejected = core.hybrid_trigger(t, s, nsta=50, nlta=500)
```

## Testy

`pytest -q` — 35 przechodzi + 1 pomijany bez ObsPy
(`test_sta_lta_i_trigger_onset_zgodne_z_obspy`).
