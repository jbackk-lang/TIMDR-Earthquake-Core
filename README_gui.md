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
- **Generate demo** — wybór z rozwijanej listy jednego z czterech
  syntetycznych scenariuszy (patrz niżej), żeby wypróbować narzędzie
  bez własnych danych.
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
  lub hybrydowy). Wynik jako 5-panelowy wykres i panel tekstowy z
  liczbami, w tym rozkładem typów anomalii i (przy hybrydzie) liczbą
  potwierdzeń/odrzuceń.

## Cztery scenariusze demo

- **Earthquake + sensor glitch** — tło + narastająco-opadający wstrząs
  + pojedynczy izolowany glitch czujnika.
- **Stuck sensor (dropout)** — czujnik "zawiesza się" na stałej
  wartości na jakiś czas, potem wraca do normy.
- **Gradual drift (no sudden onset)** — poziom narasta stopniowo (nie
  skokiem) i zostaje na nowym poziomie.
- **Background noise only (no event)** — czysty szum tła, zero
  zdarzeń — sprawdza, że detektor nie generuje fałszywych alarmów.

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

## Testowanie

GUI przetestowane automatycznie (Xvfb + zrzuty ekranu) na wszystkich 4
scenariuszach demo, ze wszystkimi 3 metodami TRM preview i z hybrid
triggerem włączonym/wyłączonym — bez wyjątków.
