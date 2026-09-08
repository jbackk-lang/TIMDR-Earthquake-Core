# GUI + run.bat — TIMDR-Earthquake-Core

## Uruchomienie (Windows)

Dwuklik na **`run.bat`**. Skrypt sprawdza Pythona w PATH (z modułem
`tkinter`), instaluje/aktualizuje `numpy`/`scipy`/`matplotlib` i
uruchamia `gui_app.py`. Jeśli Python nie jest zainstalowany:
https://www.python.org/downloads/ (zaznacz **"Add python.exe to PATH"**).

## Interfejs jest po angielsku

Cały interfejs (etykiety, przyciski, komunikaty, panel wyników,
podpisy na wykresach) jest teraz w języku angielskim.

## Co robi GUI

- **Load CSV...** — dowolny plik z kolumnami czasu i amplitudy (nazwy
  kolumn konfigurowalne, domyślnie `t`/`s`) przez `SeismicLoader`.
- **Generate demo** — wybór z rozwijanej listy jednego z pięciu
  scenariuszy (patrz niżej), żeby wypróbować narzędzie bez własnych
  danych.
- Checkboxy preprocessingu: detrend / despike / normalizacja amplitudy.
- Parametry detekcji: `k_neighbors`, próg `twist`, `factor` anomalii,
  oraz **TRM preview** — podgląd wygładzenia metodą `median` /
  `adaptive` / `savgol` jako dodatkowa (przerywana) linia na wykresie
  sygnału. To tylko podgląd porównawczy — `anomalies()`/`fronts()`
  wewnątrz zawsze używają domyślnego wygładzenia medianowego,
  niezależnie od tego wyboru.
- STA/LTA: `nsta`/`nlta` (w **próbkach**, nie sekundach), próg włącz/
  wyłącz, oraz checkbox **Hybrid trigger** — gdy zaznaczony, wyzwolenia
  STA/LTA są pokazywane jako potwierdzone (zielone) tylko gdy w ich
  sąsiedztwie występuje też silny `twist` i punkt `anomalies()`;
  odrzucone kandydatury pokazane są w tle na szaro, z liczbą i powodem
  odrzucenia w panelu wyników.
- **Run analysis** — pełny pipeline: flow → twist → anomalie →
  `classify_anomalies` (kształt: impuls/spike/step/drift/dropout,
  kolorowane osobno na wykresie residuum) → fronty → STA/LTA (zwykły
  lub hybrydowy) → meta-dynamika (patrz niżej). Wynik jako 6-panelowy
  wykres i panel tekstowy z liczbami, w tym rozkładem typów anomalii i
  (przy hybrydzie) liczbą potwierdzeń/odrzuceń.
- **Meta-dynamics (Λ/τ/ρ/J)** — 6. panel wykresu, korzysta wprost z
  `meta_adapter.py` (ten sam moduł, którego `build_meta_series_from_waveform`
  jest reużywane przez sibling-import w TIMDR-Grid-Monitor/
  TIMDR-Industrial-Predict/Synoptyk-v3 — patrz ich odpowiednie
  `meta_adapter.py`). Dwa pola: **Window (s)** (domyślnie `5.0`, jak
  moduł) i **Rolling hist. (s)** (puste = progi liczone z CAŁEGO śladu,
  niecałkowicie przyczynowe, ale pokazujące pełne narastanie-i-zanik;
  wartość = tryb przyczynowy, aktualizujący się w czasie — patrz
  docstring `build_meta_series_from_waveform`). Fazy (stabilna/
  przejściowa/krytyczna) rysowane jako kolorowe pasy na osi czasu.

  **Uczciwe ograniczenie zweryfikowane przy budowie tego panelu**:
  cztery krótkie scenariusze syntetyczne (~4-5s przy domyślnym oknie
  5.0s) są ZA KRÓTKIE na choćby jedno pełne okno — panel pokazuje
  wtedy czytelny komunikat "trace too short for chosen window" zamiast
  wykresu, i **nie** przerywa reszty analizy (pozostałe 5 paneli
  liczą się normalnie, błąd złapany osobno). Zmniejszenie okna, żeby
  "zmieściło się" w te 4-5s, przywróciłoby dokładnie ten sam problem
  udokumentowany w `TIMDR-Grid-Monitor/meta_adapter.py` (PRÓBA 1): na
  bardzo krótkich oknach próbka Λ (frakcja FFT wysokich częstotliwości)
  ma tak dużą wariancję, że fałszywie klasyfikuje czysty szum jako
  "przejściowa"/"krytyczna" — świadomie NIE zrobiono tego tutaj.
  Scenariusz **Ridgecrest 2019** (90s, fs=100Hz) jest wystarczająco
  długi i daje sensowny, sprawdzony wynik: pierwsze ~20s po głównym
  wstrząsie (M7.1, t=0) wychodzi "krytyczna" (4 kolejne okna), potem
  faza opada do "przejściowa"/"stabilna" w miarę zanikania sekwencji
  wstrząsów wtórnych w oknie kalibracji z całego śladu — fizycznie
  sensowny kierunek (krytyczna tuż po dużym zdarzeniu, potem uspokojenie),
  choć progi klasyfikacji (0.1/1.0) same w sobie NIE są skalibrowane na
  danych sejsmicznych (patrz zastrzeżenia w nagłówku `meta_adapter.py`).

## Pięć scenariuszy demo

- **Earthquake + sensor glitch** — tło + narastająco-opadający wstrząs
  + pojedynczy izolowany glitch czujnika.
- **Stuck sensor (dropout)** — czujnik "zawiesza się" na stałej
  wartości na jakiś czas, potem wraca do normy.
- **Gradual drift (no sudden onset)** — poziom narasta stopniowo (nie
  skokiem) i zostaje na nowym poziomie.
- **Background noise only (no event)** — czysty szum tła, zero
  zdarzeń — sprawdza, że detektor nie generuje fałszywych alarmów.
- **Ridgecrest 2019 - real event times/mags (synthetic waveform)** —
  pierwsze 90s po prawdziwym trzęsieniu M7.1 Ridgecrest (2019-07-06),
  z trzema kolejnymi realnymi wstrząsami wtórnymi. **Uczciwa
  proweniencja danych**: czasy i magnitudy zdarzeń są w 100% realne
  (katalog USGS FDSN, `data/ridgecrest_2019/ridgecrest_raw_dense.txt`,
  283 realnych zdarzeń łącznie) — ten scenariusz bierze pierwsze 90s z
  tego pliku. Same PRÓBKI FALI pozostają syntetyczne (stała, z góry
  ustalona falka umieszczona w każdym realnym czasie zdarzenia,
  amplituda skalowana do realnej magnitudy) — prawdziwego ciągłego
  zapisu sejsmometru nie udało się pozyskać w tym środowisku
  (`service.iris.edu` jest wycofane, inne serwery z falami były
  nieosiągalne). Pełny opis metody i wyników (recall 58,7% w gęstym
  oknie vs 100% w izolacji) w `HISTORIA_I_TESTY.md` i
  `stai_real_ridgecrest_test.py`.

## Ograniczenie zakresu — to NIE jest predykcja trzęsień

Wszystko w tym narzędziu wykrywa i klasyfikuje cechy JUŻ obecne w
sygnale (front, który już się zaczął; anomalię, która już zaszła;
wyzwolenie pickera na energii, która już jest w danych). Nic tu nie
prognozuje trzęsienia PRZED jego wystąpieniem — krótkoterminowa
predykcja sejsmiczna (wiarygodny sygnał prekursorowy, uruchamiający
się z sensownym wyprzedzeniem przed pęknięciem) to otwarty,
nierozwiązany problem sejsmologii, którego to narzędzie się nie
podejmuje. GUI ma o tym stały, widoczny komunikat pod nagłówkiem.

## ⚠️ Uwaga o domyślnym progu `twist`

Domyślny próg w `TIMDR_EarthquakeCore.twist()` (`0.4`) nie jest dobrany
pod konkretną skalę danych — dla znormalizowanego sygnału demo mediana
`|twist|` wynosi ~10, więc próg `0.4` flagowałby niemal wszystko. GUI
ustawia domyślnie `20` (czytelne dla demo), ale to nadal nie jest
zwalidowana norma — przy własnych danych dopasuj próg patrząc na
wartości w panelu `|twist|`.

## 🐛 Poprawka: panel wykresu ucinany zamiast skalowany na węższych oknach

Zgłoszone (ze zrzutem ekranu): na części ekranów prawy panel (wykresy)
był ucięty, a nie przeskalowany. Przyczyna: `matplotlib.Figure` ma
STAŁY rozmiar w pikselach (`figsize*dpi` = 750×850px), którego Tk canvas
nie skaluje automatycznie przy zmianie rozmiaru okna — po prostu
przycina to, co się nie mieści. Przy oknie węższym niż domyślne ~1220px
(typowe przy skalowaniu DPI Windows >100% albo mniejszym ekranie) prawa
kolumna wypadała ucięta. Naprawione: `<Configure>` na widgecie wykresu
teraz dynamicznie woła `fig.set_size_inches()` + `canvas.draw_idle()`,
więc wykres zawsze wypełnia dostępną przestrzeń. Zweryfikowano w
zakresie od minimalnego rozmiaru okna (920×620) do powiększonego
(1400×900) — rozmiar figury poprawnie nadąża za oknem w każdym
przypadku.

## Testowanie

GUI przetestowane automatycznie (Xvfb + zrzuty ekranu) na wszystkich 4
oryginalnych scenariuszach demo, ze wszystkimi 3 metodami TRM preview i
z hybrid triggerem włączonym/wyłączonym — bez wyjątków, oraz na 4
różnych rozmiarach okna po poprawce resize. **Nieuzupełnione**: 5. scenariusz
("Ridgecrest 2019") dodano później i przetestowano tylko logikę
generowania danych (`demo_ridgecrest_real()` w izolacji, bez tkinter —
środowisko developerskie tej sesji nie ma `tkinter`) — sama ścieżka
GUI (rysowanie, panele wyników) dla tego scenariusza NIE przeszła
automatycznego testu Xvfb opisanego wyżej.

**Dodatkowo (na prawdziwym pliku CSV od użytkownika, nie na demo)**:
`tkinter` nadal nie jest dostępny w tym środowisku, więc rzeczywisty
render GUI (Xvfb) na prawdziwych danych CI.CLC/CI.RIO nie został
wykonany — ale dokładna sekwencja wywołań `on_load_csv()` →
`on_analyze()` (te same funkcje/argumenty, z domyślnymi wartościami
wszystkich widgetów) została odtworzona bezpośrednio w Pythonie na
prawdziwym pliku `CLC_HHZ.csv` (36001 próbek, surowe zliczenia rzędu
1e7). Wynik: przy tym najpierw ujawnił się i został naprawiony realny
bug (`SeismicLoader.load_csv()` odrzucał ten plik z powodu braku
nagłówka — patrz `HISTORIA_I_TESTY.md`), a po naprawie cała ścieżka z
domyślnymi progami (`twist_thr=20`, `MAD factor=3.0`, STA/LTA 25/100,
thr_on/off=3.0/1.0) poprawnie złapała prawdziwy mainshock na pierwszym
wyzwoleniu STA/LTA (t=61.1s wobec realnego t≈60.0s), choć detektory
twist/anomalia zostają, jak już zaznaczono wyżej, przeczulone na
realnej skali danych (patrz `HISTORIA_I_TESTY.md` po pełne liczby).
