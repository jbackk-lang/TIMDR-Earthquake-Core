# TIMDR-Earthquake-Core

Lekki, w pełni audytowalny zestaw narzędzi do analizy jednokanałowego
sygnału sejsmicznego: detekcja zdarzeń (STA/LTA zweryfikowane co do
bitu z ObsPy), opisowa klasyfikacja kształtu anomalii i heurystyka
redukcji fałszywych alarmów. Zero zależności od infrastruktury sieci
stacji — działa na jednym śladzie `s(t)`.

## Instalacja

```bash
pip install -e .            # rdzeń (numpy, scipy)
pip install -e ".[gui]"     # + interfejs graficzny (matplotlib)
pip install -e ".[dev]"     # + pytest, do uruchamiania testów
```

## Szybki start

```bash
# CLI - bez pisania kodu
python cli.py obspy_BW_RJOB_example.csv --out raport.json

# jako biblioteka
python -c "
from seismic_loader import SeismicLoader
from timdr_core_earthquake import TIMDR_EarthquakeCore

t, s = SeismicLoader().load_csv('obspy_BW_RJOB_example.csv')
core = TIMDR_EarthquakeCore()
confirmed, rejected = core.hybrid_trigger(t, s, nsta=25, nlta=100)
print(confirmed)
"

python gui_app.py            # interfejs graficzny (run.bat na Windows)
pytest -q                    # 68 testów
```

## Czym to jest w porównaniu z innymi narzędziami

To NIE jest zamiennik SeisComP/Earthworm (systemy sieciowe czasu
rzeczywistego z lokalizacją hipocentrum z wielu stacji) ani PhaseNet/
EQTransformer (modele głębokiego uczenia, istotnie lepsze od STA/LTA
przy gęstych sekwencjach — patrz `HISTORIA_I_TESTY.md`, sekcja "Rój
wstrząsów wtórnych"). Nie liczy magnitud, nie rozróżnia fal P/S, nie
lokalizuje źródła — to wymaga wielu stacji i celowo nie zostało tu
udawane.

To, co ten projekt faktycznie daje, a czego wymienione narzędzia nie
dają w tej formie:

- **Cały kod czytelny i zweryfikowany co do bitu względem referencji**
  (`sta_lta()`/`trigger_onset()` vs `obspy.signal.trigger`) — jeden
  plik `.py`, nie duża baza C++/Fortran. Dobre do nauki jak STA/LTA
  faktycznie działa albo do audytu przed użyciem gdzie indziej.
- **Opisowa klasyfikacja KSZTAŁTU anomalii** (`classify_anomalies`):
  impuls / spike / step / drift / dropout — bez danych treningowych,
  bez sieci neuronowej. Odróżnia np. realny, trwały skok poziomu od
  utkniętego czujnika (dropout) — praktyczna diagnostyka jakości
  danych, nie tylko detekcja trzęsień.
- **`hybrid_trigger`**: potwierdzenie zdarzenia STA/LTA przez dwa
  dodatkowe, niezależne sygnały (`twist`, `anomalies`) z TEGO SAMEGO
  kanału — próba redukcji false-positive tam, gdzie nie ma budżetu na
  drugą stację. Nie zwalidowane względem katalogu z etykietami (patrz
  ograniczenia niżej) — traktuj jako filtr przesiewający, nie
  certyfikowany detektor.
- **Publikowane wyniki negatywne** z pełną metodologią i danymi: dwie
  niezależne, uczciwie przetestowane hipotezy o przewidywaniu trzęsień
  (analiza topologiczna, ringdown) obie dały wynik negatywny na
  realnych danych USGS/IRIS. Rzadkość w tej przestrzeni — większość
  narzędzi nie publikuje "sprawdziliśmy, nie działa".

## Kierunki zastosowań

1. **Szybki triage na pojedynczym czujniku** — tam, gdzie sieć
   wielostanowiskowa nie wchodzi w grę (budżet, jeden czujnik testowy,
   monitoring niesejsmiczny o podobnym charakterze sygnału).
2. **Diagnostyka jakości danych/czujnika** — `classify_anomalies`
   wykrywa dropout/step/drift niezależnie od tego, czy interesuje Cię
   sejsmologia; działa na dowolnym sygnale czasowym o podobnym
   charakterze (wibracje, telemetria).
3. **Materiał edukacyjny/referencyjny** — zweryfikowana, czytelna
   implementacja STA/LTA do nauki lub jako baza porównawcza przy
   budowie własnego detektora.
4. **Prototypowanie przed inwestycją w pełną sieć** — sprawdzenie
   pomysłu na jednym kanale, zanim wdroży się SeisComP/Earthworm.
5. **Kontrolowane środowisko do testowania nowych hipotez detekcyjnych**
   — jak pokazuje `HISTORIA_I_TESTY.md`, ten projekt ma wbudowaną
   kulturę pre-rejestracji i kontroli negatywnej; dobra baza do
   sprawdzania kolejnych pomysłów, zanim trafią na prawdziwe dane.

## Integracja z TIMDR-META-DYNAMICS (eksperymentalna)

`meta_adapter.py` okienkuje ciągły ślad `s(t)` (domyślnie 5s, nienakładające
się okna) i mapuje wynik `flow()`/`twist()`/`anomalies()`/`trm()` na
`MetaState(Λ,τ,ρ,J)` z repozytorium-siostry `TIMDR-META-DYNAMICS` — trzecia
realna integracja tego formalizmu (pierwsza: finansowa w
`analizator-gieldowy-v3`, druga: pogodowa w `Synoptyk-v3`).

**Mapowanie** (Λ=high-frequency fraction widma amplitudy w oknie [0,1],
τ=średni `|flow_grad|`/próg globalny, ρ=fraction próbek-anomalii wg progu
globalnego, J=fraction próbek-"skrętu" wg progu globalnego) zaprojektowane
od razu jako bezwymiarowe — pełne wzory i dwie odrzucone/zaakceptowane
wersje w docstringu pliku.

**Test end-to-end na PRAWDZIWYM śladzie** (Ridgecrest 2019, stacja CLC,
360s, znany niezależnie czas mainshocku t≈60.0s — `HISTORIA_I_TESTY.md`):
z progami kalibrowanymi na **całym** śladzie, klasyfikacja faz poprawnie
wychwytuje mainshock (pierwsza faza `"krytyczna"` w kroku obejmującym
t≈60s) i pokazuje realny zanik aktywności w ciągu ~90s po zdarzeniu.

**Znaleziony i naprawiony błąd (V1→V2, patrz docstring)**: pierwsza wersja
normalizowała każde okno względem samego siebie (mediana+MAD *tego samego*
okna) — algebraicznie niezmiennicze na jednorodne przeskalowanie, więc
τ wychodził niemal identyczny (~0.3) na całym śladzie, *w tym w oknie z
mainshockiem* — zero mocy dyskryminującej. Naprawione przez policzenie
progów **raz, globalnie**, przed podziałem na okna.

**Zbadana (nie porzucona) kwestia obiegowości kalibracji**: czy liczenie
progu z całego śladu (włącznie z samym zdarzeniem) nie jest ukrytym
"przejściem między dwoma reżimami" w jednej statystyce — dokładnie ten sam
problem, jaki ten projekt już raz znalazł gdzie indziej (patrz niżej,
"kalibracja nie może być na oknie, które już zawiera rój"). Odpowiedź:
tak, pre-event i post-event to naprawdę drastycznie różne reżimy
(zweryfikowane na surowych danych — odch. std. amplitudy skacze z ~9 400
do ~2-5 mln i zostaje ~40× podwyższone nawet 3-5 min później — to realna
fizyka silnego wstrząsu + sekwencji wstrząsów wtórnych, nie artefakt
przetwarzania). `build_meta_series_from_waveform(calibration_end=...)`
udostępnia oba warianty (kalibracja z całego śladu vs. wyłącznie z okresu
przed zdarzeniem) — wariant przyczynowy na tym śladzie daje wynik niemal
binarny (>50% kroków "krytyczna" przez całą resztę zapisu), bo cała
sekwencja wstrząsów wtórnych jest ekstremalna względem spokojnego tła.
Oba warianty zostają w kodzie, żaden nie zastępuje drugiego — odpowiadają
na różne pytania ("jak ewoluuje aktywność w obrębie zdarzenia" vs. "czy
jestem już w reżimie po-katastroficznym").

Test: `test_meta_adapter.py` (7 testów: kontrole syntetyczne
pozytywna/negatywna, dowód algebraiczny błędu V1, walidacja wejścia, oba
warianty end-to-end na realnym śladzie).

## Ograniczenia (zwięźle — pełne dowody w `HISTORIA_I_TESTY.md`)

- Jeden kanał `s(t)` — brak lokalizacji, magnitud, rozróżnienia P/S.
- `hybrid_trigger` niezwalidowany względem katalogu z etykietami.
- Podczas gęstych rojów wstrząsów wtórnych wykrywalność pojedynczych
  zdarzeń spada (91%→40% w teście na sygnale syntetycznym; **potwierdzone
  też na realnym katalogu USGS sekwencji Ridgecrest 2019** — recall
  58,7% w gęstym oknie vs 100% kontrola pozytywna na izolowanych
  zdarzeniach, patrz `HISTORIA_I_TESTY.md`) — znane w literaturze jako
  Short-Term Aftershock Incompleteness; to repo tego nie rozwiązuje,
  tylko to mierzy.
- Dwie testowane hipotezy o przewidywaniu trzęsień dały wynik
  negatywny na realnych danych.
- `anomalies()`/`fronts()` nie są projektowane pod strumień na żywo —
  analizują już zarejestrowany segment za jednym wywołaniem.

  Dodatkowe zastrzeżenia ujawnione w audycie na realnym katalogu Ridgecrest 2019
EV / jump detection — liczby błędne w pierwszej analizie, kierunek wniosku poprawny.  
Realne dane M≥2.0 w kroczącym oknie 30 minut dają X_prev=5 (2.80, 2.15, 2.22, 4.97, 4.14) i X_now=125 — pierwsza wersja użyła tylko ostatnich ~5 minut zamiast pełnych 30, oraz tylko 7 zdarzeń po mainshocku zamiast 125. Realny rolling p10/p90 w tym samym oknie to 0 i 118, więc próg 0.3*(118−0)=35.4, nie 0.9.
EV=TRUE przetrwało nawet poprawiony próg, ale tylko dlatego, że rój jest ekstremalny.
Uwaga metodologiczna: próg nie może być kalibrowany na oknie, które już zawiera rój — to kołowe. Kalibracja musi być wykonana na spokojnym okresie sprzed sekwencji.

Bias correction — demonstracja arytmetyki, nie test na Ridgecrest.  
Przykład z §4 skilla używa wymyślonych par (prediction, ground truth) różniących się zawsze o dokładnie 1, co trywialnie daje bias=-1, MAE=1 niezależnie od realności danych.
To nie jest walidacja jakiegokolwiek modelu predykcyjnego na Ridgecrest, tylko pokaz mechaniki logowania i grupowania po lead time. Materiał źródłowy sam przyznaje: „nie mamy modelu predykcyjnego” — więc wniosek „działa poprawnie, zgodne z protokołem” dotyczy wyłącznie arytmetyki, nie jakości prognoz.

ringdown_resonance() — błąd kategorii: katalog magnitud ≠ sygnał amplitudy.  
Sekwencja malejących magnitud kolejnych zdarzeń (M7.1 → 4.8 → 4.3 → …) nie jest ringdownem jednej fali, tylko efektem prawa Bátha i rozkładu Gutenberga–Richtera (duże aftershocks przychodzą pierwsze).
Wniosek „monotonic decay, brak oscylacji” wyciągnięty z listy magnitud nie jest wynikiem ringdown_resonance(), bo funkcja nigdy nie została uruchomiona — wymaga ciągłego przebiegu amplitudy s(t), baseline’u, noise bandu i crossingów, których katalog zdarzeń nie zawiera.
Realny test ringdown wymaga ciągłego sejsmogramu (np. EarthScope/IRIS), nie katalogu.

## Pełna historia, metodologia i liczby

Zobacz [`HISTORIA_I_TESTY.md`](HISTORIA_I_TESTY.md) — chronologiczny
zapis audytu, znalezionych błędów, testów predykcyjności i pełnych
wyników statystycznych.
