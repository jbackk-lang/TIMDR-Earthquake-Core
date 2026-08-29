# Historia techniczna i pełne wyniki testów

Ten dokument zawiera pełny, chronologiczny zapis audytu, poprawek i
testów tego repo — dla kogoś, kto chce zweryfikować konkretne liczby,
metodologię albo historię decyzji. Zwięzłe, zorientowane na
użytkownika README jest w [`README.md`](README.md).

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

## `ringdown.py` — powrót do równowagi PO zdarzeniu — i test predykcyjności (`precursor_ringdown_test.py`)

Port 1:1 matematyki `ringdown_resonance()` z universal-state-analyzer /
TIMDR-Grid-Monitor / analizator-gieldowy-v3 / deliverable_timdr_finanse
(5. port tej samej, zweryfikowanej funkcji). Analizuje, czy powrót
sygnału do poziomu odniesienia PO indeksie `event_idx` jest oscylacyjny
(tłumione „dzwonienie”) czy monotoniczny — to jest narzędzie **opisowe
(post-event)**, nie predykcyjne samo w sobie.

Test na prawdziwym śladzie `obspy_BW_RJOB_example.csv` (autentyczny
lokalny wstrząs, tutorial ObsPy): dla głównego wstrząsu w tym śladzie
wynik `is_oscillatory` jest **wrażliwy na próg szumu** — `False` przy
`noise_floor_factor>=2.0`, `True` (okres ~4-6s) przy poluzowaniu do
1.0-1.5. Bez niezależnego pomiaru prawdziwej częstotliwości tego
wstrząsu nie da się stwierdzić, która odpowiedź jest poprawna — to
udokumentowane ograniczenie metody zero-crossing na realnych,
wielomodowych sejsmogramach (patrz `test_ringdown.py`).

### Czy to ma jakąkolwiek moc predykcyjną? (`precursor_ringdown_test.py`)

Osobne pytanie od powyższego, tym samym protokołem co `Topology(t)`
wyżej w tym README (wynik tamtego testu: **negatywny**): czy cecha
zbudowana z `ringdown_resonance()` na kandydatach z `fronts()`, policzona
WYŁĄCZNIE z danych sprzed prawdziwego dużego wstrząsu, jest podwyższona
względem tych samych cech policzonych w losowych oknach tła (test
Manna-Whitneya U, nie tylko porównanie percentyli).

- `--mode synthetic` (domyślny, bez sieci): sanity-check metodologii na
  danych syntetycznych — kalibracja klasyfikatora `ringdown_resonance()`
  bezpośrednio na tle AR(1) (autoskorelowanym, bliższym realnemu tłu
  sejsmicznemu niż biały szum, na którym `fronts()` prawie nigdy nic nie
  znajduje). **Wynik: pipeline zweryfikowany** — wstrzyknięty sygnał
  wykryty (p≈5×10⁻⁷), brak fałszywego alarmu szum-vs-szum (p≈0.55,
  fałszywy odsetek klasyfikacji „oscylacyjne” na czystym tle ≈9.6%,
  vs ≈32% gdy sygnał faktycznie jest).
- `--mode real`: pełny test na realnych danych (katalog USGS + fale
  sejsmiczne EarthScope/IRIS, 8 stacji GSN, wykluczenie okien tła w
  promieniu 1000km od stacji od jakiegokolwiek M≥4.5). Sandbox, w którym
  to repo powstało, ma zablokowany dostęp sieciowy do wymaganych
  serwerów, więc test na realnych danych uruchomił użytkownik na
  własnej maszynie (z prawdziwym dostępem do sieci) — po drodze
  ujawniło to i pozwoliło naprawić trzy realne błędy w skrypcie, żaden
  z nich nie był blokadą sieci: (1) zapytanie o katalog wykluczający
  M≥4.5 pobierało cały globalny katalog na raz i przekraczało limit
  USGS FDSN 20000 wyników/zapytanie (`HTTP 400`) — naprawiono wąskimi,
  osobnymi zapytaniami per kandydat; (2) wykluczanie okien tła
  sprawdzało sejsmiczność na CAŁYM globie zamiast w pobliżu konkretnej
  stacji, więc odrzucało prawie każdego kandydata (M≥4.5 zdarza się
  gdzieś na świecie kilka razy dziennie) — naprawiono ograniczeniem do
  promienia 1000km od stacji; (3) krok liczenia okien tła nie miał
  żadnego limitu czasu, tylko limit liczby prób, więc czas działania był
  nieprzewidywalny — naprawiono twardym budżetem czasu (domyślnie 240s).

  **WYNIK (uruchomienie: 40 okien pre-event z realnego katalogu USGS
  M≥6.5 z ostatnich 5 lat, 60 okien tła, 8 stacji GSN):**

  | | średnia `frac_oscillatory` |
  |---|---|
  | pre-event (40 okien) | 0.0683 |
  | tło (60 okien) | 0.0601 |

  Test Manna-Whitneya U: **p = 0.997**. **Brak statystycznie istotnej
  różnicy — wynik negatywny.** Cecha zbudowana z `ringdown_resonance()`
  na kandydatach z `fronts()` NIE jest podwyższona w oknach sprzed
  prawdziwych dużych wstrząsów względem losowego tła. Zgodne z
  wcześniejszym testem `Topology(t)` w tym samym repo (też negatywny) —
  dwie zupełnie różne matematyki, ta sama odpowiedź. Pełny wynik z
  metadanymi każdego okna: `precursor_ringdown_test_output.json`.

  Odpalenie: `pip install obspy requests scipy && python
  precursor_ringdown_test.py --mode real`.

**Uczciwe podsumowanie**: USGS oficjalnie stwierdza, że nikt nigdy nie
przewidział trwale i wiarygodnie dużego trzęsienia ziemi i nie oczekuje
takiej metody w dającej się przewidzieć przyszłości. Ten wynik (p=0.997,
negatywny) jest z tym zgodny. Gdyby wyszedł pozornie pozytywny na tym
jednym przebiegu (40 zdarzeń, 5 lat, 8 stacji), byłaby to wstępna
przesłanka wymagająca replikacji na niezależnym zbiorze zdarzeń, nie
dowód — dokładnie tak jak „trop” z BTC w `deliverable_timdr_finanse` nie
przetrwał replikacji na
złocie.

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

## Rój wstrząsów wtórnych psuje rozdzielczość detekcji (`test_aftershock_swarm_detection.py`)

Pytanie z rozmowy: przy ciągłym monitorowaniu, czy pojedyncze
trzęsienie jest "rzadkim sygnałem" (łatwym do wykrycia), a rój
wstrząsów wtórnych psuje to założenie? Wstępny test na prostszym,
niezależnym syntetycznym sygnale (`TEST-TIMDR/seismology-sta-lta`,
osobne repo eksperymentalne) pokazał: izolowane trzęsienie 100%
wykrycia, wczesne wstrząsy wtórne (gęsto skupione tuż po głównym
wstrząsie) 89-91%, późne (bardziej odosobnione, ale wciąż część roju)
tylko 37-40%. Ten test sprawdza, czy TO SAMO zjawisko występuje na
prawdziwej, zweryfikowanej zgodnie z ObsPy implementacji `sta_lta()`/
`trigger_onset()`/`hybrid_trigger()` w tym repo — **potwierdzone**:

Tło AR(1) + główny wstrząs + sekwencja wstrząsów wtórnych wg prawa
Omoriego (`nsta=20, nlta=200`, seed=42):

| | Wykrycie |
|---|---|
| Izolowany wstrząs (bez roju) | ~100% (10/10 ziaren) |
| Wczesne wstrząsy wtórne (<200 próbek od głównego) | 91% |
| Późne wstrząsy wtórne (≥200 próbek) | **40%** |

`hybrid_trigger()` (potwierdzenie twist+anomaly ponad `trigger_onset`)
daje TEN SAM wynik liczbowy co samo `trigger_onset` — co jest osobną,
ważną obserwacją: blisko skupione wstrząsy wtórne trafiają w TEN SAM
przedział `[start,end]` z `trigger_onset` (zlewają się w jeden ciągły
"blob" zamiast być rozróżnione jako osobne zdarzenia), więc
`hybrid_trigger` potwierdza/odrzuca całą grupę naraz — to jest
ograniczenie ROZDZIELCZOŚCI podczas gęstego roju, którego dodanie
twist/anomaly nie naprawia, bo działa na tym samym pogrupowaniu
czasowym co bazowy `trigger_onset`.

**Uczciwe zastrzeżenie**: to test na sygnale syntetycznym (AR(1) +
tłumione oscylacje), nie na prawdziwym katalogu sekwencji wstrząsów
wtórnych z nakładającymi się falami P/S/kodą wielu zdarzeń jednocześnie
— realny problem interferencji fal podczas rojów jest bogatszy niż
tu zamodelowano (patrz ograniczenia w `TEST-TIMDR/seismology-sta-lta/README.md`).
Ale kierunek wyniku (degradacja rozdzielczości, nie tylko odsetka
wykryć) jest spójny między dwiema niezależnymi implementacjami i dwoma
niezależnymi modelami sygnału.

## To samo zjawisko na REALNYM katalogu Ridgecrest 2019 (`stai_real_ridgecrest_test.py`)

Powyższe zastrzeżenie ("nie na prawdziwym katalogu") zamknięte
częściowo: zamiast wymyślonej sekwencji Omoriego, czasy i magnitudy
zdarzeń są w 100% realne — pobrane z katalogu USGS FDSN
(`earthquake.usgs.gov/fdsnws/event`) dla prawdziwej sekwencji
Ridgecrest, Kalifornia, lipiec 2019:

- **Gęste okno**: 2019-07-06 03:15-06:15 UTC, pierwsze ~3h po
  prawdziwym trzęsieniu M7.1 — 283 realne zdarzenia (M≥2,0).
- **Okno izolowane**: 2019-08-01 (miesiąc później, sekwencja już
  wygasła) — 7 realnych zdarzeń przy tym samym nominalnym progu M≥2,5
  (próg podniesiony z M≥2,0 po tym, jak sondowanie pojedynczego
  zdarzenia pokazało, że poniżej ~M2,5 kontrola pozytywna sama w sobie
  zawodzi niezależnie od gęstości — patrz kalibracja w docstringu
  skryptu).

Te realne czasy/magnitudy napędzają syntetyczną falkę (bo prawdziwego
ciągłego zapisu sejsmometru nie udało się pozyskać w tym środowisku —
`service.iris.edu` jest wycofane ("NGF: Service Unavailable"),
`service.earthscope.org` i `raw.githubusercontent.com` nieosiągalne),
przepuszczoną przez DOKŁADNIE TĘ SAMĄ, niezmienioną implementację
`TIMDR_EarthquakeCore.sta_lta()`/`trigger_onset()` co powyżej:

| | Recall |
|---|---|
| Izolowane realne zdarzenia (kontrola pozytywna) | **100%** (7/7) |
| Gęste okno po M7.1 (283 realne zdarzenia) | **58,7%** (166/283) |

Ważny szczegół mechanistyczny: recall w gęstym oknie jest PŁASKI w
całym zakresie M2,5-4,0 (54-57% w każdym przedziale co 0,5), mimo że
KAŻDA z tych magnitud osobno, w izolacji, przekracza próg detekcji z
dużym zapasem (M2,5 samo: stosunek STA/LTA 4,13 przy progu 3,5; M3,5
samo: 9,48). Płaski, nie-zależny-od-SNR spadek recall w zakresie, gdzie
każde pojedyncze zdarzenie jest łatwo wykrywalne, to sygnatura
nakładania się kody (STAI), a nie brak czułości detektora.

Niezależne, bezdetektorowe potwierdzenie tego samego zjawiska — sam
oficjalny katalog USGS: najmniejsza skatalogowana magnituda w gęstym
oknie to 2,69, a w oknie izolowanym 1,50 — realna niekompletność
katalogu podczas gęstego okresu, widoczna jeszcze zanim jakikolwiek
nasz detektor dotknie danych.

Dane źródłowe: `data/ridgecrest_2019/ridgecrest_raw_dense.txt` (283
zdarzenia) i `ridgecrest_raw_isolated.txt` (46 zdarzeń, filtrowane do
M≥2,5 w skrypcie). Pełna pre-rejestracja parametrów (kształt falki,
prawo amplituda-magnituda, progi STA/LTA, reguła dopasowania trigger↔
zdarzenie) i obie korekty kalibracyjne opisane wprost w docstringu
`stai_real_ridgecrest_test.py`. Ten sam realny katalog (pierwsze 90s po
M7.1) jest też teraz dostępny jako piąty scenariusz w GUI (patrz
`README_gui.md`).

**Co NADAL pozostaje otwarte**: to realne czasy/magnitudy + syntetyczna
fala, NIE prawdziwy zarejestrowany sejsmogram — walidacja na
rzeczywistym ciągłym zapisie fal pozostaje zablokowana dostępnością
danych w tym środowisku, nie metodą.

## Testy

`pytest -q` — 68 testów przechodzi (35 istniejące + `test_ringdown.py` +
`test_selfbaseline_recovery.py` — patrz sekcja "Powrót do normy po
mikro-wstrząsie" niżej + `test_aftershock_swarm_detection.py` — patrz
sekcja "Rój wstrząsów wtórnych" wyżej), w tym test na prawdziwym
śladzie `obspy_BW_RJOB_example.csv`; ObsPy jest zainstalowane w
środowisku, w którym testy były ostatnio uruchamiane, więc
`test_sta_lta_i_trigger_onset_zgodne_z_obspy` też się wykonuje, nie jest
pomijany).

### Powrót do normy po mikro-wstrząsie (`test_selfbaseline_recovery.py`)

Ten sam test co w siostrzanych repo (TIMDR-Crypto-Graph,
universal-state-analyzer, deliverable_timdr_finanse,
analizator-gieldowy-v3, TIMDR-Grid-Monitor): czy `anomalies()` fałszywie
flaguje NOWE, normalne próbki po ustaniu mikro-wstrząsu?

Mechanizm tu jest inny niż wszędzie indziej — `anomalies()` nie jest
projektowane pod wywołania strumieniowe; TRM (wygładzanie do policzenia
residuów) patrzy na k=8 najbliższych sąsiadów po czasie **w obie strony**
w całej przekazanej tablicy — naturalny tryb użycia to analiza już
zarejestrowanego segmentu za jednym wywołaniem, nie żywy strumień.

Sprawdzone (10 ziaren, mikro-wstrząs: skok +50 do szumu N(0,1) na 20
próbek, potem powrót do czystego szumu):

- Fałszywe flagi **daleko** po zdarzeniu (≥200 próbek): ~1.0–1.4%,
  nieodróżnialne od bazowego wskaźnika na całkowicie czystym szumie
  (~1.2–1.6%). Anomalia nie zostawia trwałego śladu.
- Jeden, dobrze zrozumiany wyjątek: pierwsza próbka natychmiast po końcu
  anomalii jest flagowana w każdym z 10 ziaren — bo TRM(k=8) dla niej
  wciąż obejmuje kilku sąsiadów z ogona anomalii (bilateralne k-NN po
  czasie), co sztucznie podciąga lokalny baseline. To naturalny artefakt
  brzegowy ograniczony do pojedynczej próbki (nie "sklejanie się"
  fałszywych alarmów) — `classify_anomalies()` i tak grupuje go w ten sam
  blok zdarzenia co samą anomalię.
