# GUI + run.bat — TIMDR-Earthquake-Core

## Uruchomienie (Windows)

Dwuklik na **`run.bat`**. Skrypt:
1. sprawdza, czy Python jest w PATH (i czy ma moduł `tkinter` —
   standardowo dołączony w instalatorze z python.org, ale bywa
   pominięty w niektórych dystrybucjach/minimalnych instalacjach),
2. instaluje/aktualizuje `numpy`, `scipy`, `matplotlib`,
3. uruchamia `gui_app.py`.

Jeśli Python nie jest zainstalowany: https://www.python.org/downloads/
(przy instalacji zaznacz **"Add python.exe to PATH"**).

## Co robi GUI

- **Wczytaj CSV** — dowolny plik z kolumnami czasu i amplitudy (nazwy
  kolumn konfigurowalne, domyślnie `t`/`s`) przez `SeismicLoader`.
- **Wygeneruj sygnał demo** — syntetyczny sygnał (szum tła + narastający
  wstrząs + izolowany glitch czujnika) do wypróbowania bez własnych
  danych.
- Checkboxy do włączania/wyłączania kroków preprocessingu
  (detrend/despike/normalizacja).
- Pola parametrów: `k_neighbors`, próg `twist`, `factor` anomalii.
- **Uruchom analizę** — pełny pipeline `TIMDR_EarthquakeCore`
  (flow → twist → anomalie → fronty), wynik jako 4-panelowy wykres
  (sygnał / flow / |twist| z zaznaczonymi punktami / residuum z
  zaznaczonymi anomaliami i frontami) plus panel tekstowy z liczbami.

## ⚠️ Uwaga o domyślnym progu `twist`

Domyślny próg w `TIMDR_EarthquakeCore.twist()` (`0.4`) był dobrany bez
odniesienia do konkretnej skali danych. Zweryfikowano na sygnale demo
(znormalizowana amplituda, próbkowanie 100Hz): mediana `|twist|`
wynosiła tam ~10, więc próg `0.4` flagował **>90% próbek** jako "twist" —
bezużyteczne. GUI ustawia domyślnie `20` (dobrane tak, by demo dawało
czytelny wynik), ale **to nadal nie jest zwalidowana norma** — przy
własnych danych dostosuj próg, patrząc na wartości w panelu `|twist|`
wykresu (linia przerywana pokazuje aktualny próg na tle rzeczywistego
rozkładu wartości).

## Ograniczenie tego środowiska

GUI zbudowane jest na `tkinter` (standardowa biblioteka Pythona) +
`matplotlib` — nie testowałem renderowania okna bezpośrednio w tym
środowisku (piaskownica bez `tkinter`/wyświetlacza), tylko: składnię
(`py_compile`), statyczną analizę (`pyflakes` — brak błędów) i osobno
całą logikę przetwarzania danych używaną przez GUI (identyczne wywołania
`SeismicLoader`/`TIMDR_EarthquakeCore`, już pokryte 23 testami
jednostkowymi w tym repo). Jeśli po uruchomieniu `run.bat` coś nie
zadziała tak jak powinno, daj znać z treścią błędu z konsoli.
