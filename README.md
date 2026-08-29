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
