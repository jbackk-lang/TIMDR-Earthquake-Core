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
- Pola STA/LTA: `nsta`/`nlta` (długości okien w **próbkach**, nie
  sekundach — przelicz sam: `nsta = int(sekundy * fs)`), próg włącz
  (`thr_on`) i próg wyłącz (`thr_off`).
- **Uruchom analizę** — pełny pipeline `TIMDR_EarthquakeCore`
  (flow → twist → anomalie → fronty → STA/LTA picker), wynik jako
  5-panelowy wykres (sygnał / flow / |twist| z zaznaczonymi punktami /
  residuum z zaznaczonymi anomaliami i frontami / stosunek STA/LTA z
  progami włącz-wyłącz i zacieniowanymi przedziałami wyzwolenia) plus
  panel tekstowy z liczbami, teraz obejmujący też liczbę i zakres
  czasowy wyzwoleń STA/LTA.

## 🆕 STA/LTA w GUI

GUI zostało uzupełnione o `sta_lta()`/`trigger_onset()` — te same,
zweryfikowane bit-po-bicie wobec ObsPy metody, które są już opisane w
głównym `README.md`. Domyślne wartości (`nsta=25`, `nlta=100`,
`thr_on=3.0`, `thr_off=1.0`) odpowiadają oknom 0.25s/1.0s przy 100Hz
(jak w `demo.py`) i dają na sygnale demo jedno czytelne wyzwolenie w
oknie narastającego wstrząsu — sprawdzone bezpośrednio (bez GUI, przez
wywołanie tej samej logiki), nie tylko założone. Jak przy `twist`: to
punkty startowe, nie zwalidowane normy — wymagają kalibracji pod
realny szum tła i sensor.

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
całą logikę przetwarzania danych używaną przez GUI, włącznie z
`sta_lta()`/`trigger_onset()` na dokładnie tym samym sygnale demo co
generuje GUI (identyczne wywołania `SeismicLoader`/
`TIMDR_EarthquakeCore`, już pokryte 30 testami jednostkowymi w tym
repo). Jeśli po uruchomieniu `run.bat` coś nie zadziała tak jak
powinno, daj znać z treścią błędu z konsoli.
