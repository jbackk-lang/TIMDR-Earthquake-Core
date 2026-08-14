# TIMDR-Earthquake-Core

Rdzeń analizy sejsmicznej (`timdr_core_earthquake.py`): lokalny gradient
amplitudy (flow), nagłe zmiany kierunku / początek wstrząsu (twist),
wygładzenie szumu (TRM), mikro-wstrząsy (anomalies), punkty rozpoczęcia
wstrząsu (fronts) i klasyczny picker STA/LTA (`sta_lta` /
`trigger_onset`).

## Status

Kod ze zgłoszenia uruchomiony i przetestowany (`test_timdr_core_earthquake.py`,
30/30 testów). Potwierdzone: nie crashuje na n=0/1/2 (zgodnie z opisem
zgłoszenia). Znalezione i naprawione: 2 błędy, oba realnie wpływające na
dokładność detekcji na prawdziwych danych sejsmicznych.

## 🆕 STA/LTA — klasyczny picker, zweryfikowany zgodnie z ObsPy

![Własna implementacja STA/LTA vs ObsPy](screenshot_stalta_vs_obspy.png)

Na pytanie "czy da się zaimplementować u nas to, co ma ObsPy" —
odpowiedź: tak, i to bez dokładania ObsPy jako zależności. `sta_lta()`
i `trigger_onset()` to własna implementacja klasycznego algorytmu
STA/LTA napisana od podstaw wg powszechnie znanego wzoru (stosunek
krótko- do długoterminowej średniej energii sygnału), **nie** skopiowana
z kodu ObsPy.

Zweryfikowano najsurowszym możliwym testem: bezpośrednie porównanie
liczba-po-liczbie z `obspy.signal.trigger.classic_sta_lta` i
`trigger_onset` na prawdziwych danych sejsmicznych (przykładowy strumień
dołączony do samego ObsPy, stacja BW.RJOB) — wynik identyczny do ~1e-14
(precyzja zmiennoprzecinkowa) na całej długości sygnału, dla kilku
różnych kombinacji okien i progów, włącznie z przypadkiem dwóch
osobnych zdarzeń. Test `test_sta_lta_i_trigger_onset_zgodne_z_obspy`
pomija się automatycznie, jeśli ObsPy nie jest zainstalowane — to
opcjonalna weryfikacja, nie twarda zależność repo (do samego
`sta_lta()`/`trigger_onset()` potrzeba tylko numpy).

Po drodze złapałem i poprawiłem dwa subtelne błędy off-by-one względem
naiwnej pierwszej wersji: (1) pierwsze `nlta-1` próbek (niepełne okno
LTA) muszą zwracać 0, nie stosunek z okna "rozpędzającego się" — dla
pierwszej próbki taki stosunek zawsze wychodzi dokładnie 1.0, bez
sensu fizycznego; (2) `trigger_onset` musi zapisywać jako koniec
zdarzenia OSTATNI indeks jeszcze powyżej progu wyłączenia, nie pierwszy
indeks poniżej niego.

## 🐛 Błąd 1: `twist()` liczył gradient po indeksie próbki, nie po czasie

![Bug na przerwie w rejestracji](screenshot_twist_gap_fix.png)

```python
dg = np.gradient(flow_grad)  # bez t
```

`flow()` poprawnie liczy lokalny gradient względem **rzeczywistego
czasu** (LSQ na `t`), ale `twist()` różniczkował wynik `flow()` względem
**indeksu próbki**, tracąc tę informację. Dla ciągłego, równomiernie
próbkowanego sygnału to nie robi różnicy (stały mnożnik). Ale realne
dane sejsmiczne rzadko są tak czyste — **przerwy w rejestracji**
(dropout telemetrii, scalanie segmentów z różnych stacji, luki po
restarcie sprzętu) są normą, nie wyjątkiem.

Zweryfikowano na czystej fali sinusoidalnej z 3-sekundową przerwą
pośrodku (fizycznie: to samo gładkie zjawisko, tylko nie zarejestrowane
przez chwilę): oryginalny kod dawał na granicy przerwy szczyt "siły
twistu" **3.8× większy** niż typowa wartość w reszcie sygnału — fałszywy
alarm wywołany wyłącznie strukturą przerwy w danych, nie żadną
prawdziwą zmianą fizyczną. Po poprawce (`np.gradient(flow_grad, t)`)
granica przerwy wypada **0.0×** typowej wartości — poprawnie rozpoznana
jako nic nadzwyczajnego.

**Zmiana API:** `twist()` teraz przyjmuje `t` jako argument
(`twist(flow_grad, t, threshold=0.4)`) — bez tego nie da się poprawnie
liczyć gradientu względem czasu. `fronts()` już to uwzględnia.

## 🐛 Błąd 2: `anomalies()` — próg zerowy na skwantowanym sygnale

Gdy sygnał jest silnie skwantowany / ma dużo powtórzonych wartości
(typowe dla przetworników o ograniczonej rozdzielczości albo
skompresowanych formatów danych sejsmicznych), mediana bezwzględnych
reszt (MAD) może wyjść dokładnie **0**. Próg detekcji (`factor * mad`)
wychodzi wtedy też 0, i **każda niezerowa reszta** — łącznie z czystym
szumem numerycznym — zostaje sklasyfikowana jako "anomalia".

Zweryfikowano: na gładkim, tylko-zaokrąglonym sygnale sinusoidalnym (bez
żadnej realnej anomalii) dawało to **7 z 30 punktów (23%)** fałszywie
sklasyfikowanych jako mikro-wstrząsy. Naprawiono: gdy MAD wychodzi ~0,
próg spada na odchylenie standardowe reszt (a jeśli i to jest ~0 —
sygnał faktycznie stały — używana jest mała stała zamiast dosłownego
zera).

## ✅ Co faktycznie działa dobrze (potwierdzone testami)

- `flow()` poprawnie liczy lokalny gradient względem rzeczywistego
  czasu metodą regresji LSQ — dobry wybór, w przeciwieństwie do
  wcześniejszych modułów w tej serii (Radar-TIMDR, Echosonda-3D), gdzie
  to właśnie brakowało.
- Odporność na krótkie sygnały (n=0,1,2) — potwierdzona testem, zgodnie
  z zapowiedzią w opisie zgłoszenia.
- `trm()` (mediana k-NN w czasie) poprawnie wygładza szum.
- Pełny pipeline `fronts()` poprawnie wykrywa początek syntetycznego,
  narastającego wstrząsu i **nie generuje fałszywego frontu** na gładkiej
  fali z przerwą w rejestracji (patrz test
  `test_fronts_brak_falszywego_frontu_na_gladkiej_fali_z_przerwa`).

![Pełny pipeline na syntetycznym wstrząsie](screenshot_earthquake_pipeline.png)

## 🎯 Zastosowania i warunki

- **Monitoring ciągły z realną telemetrią**: błąd 1 był tu szczególnie
  ważny — bez poprawki każda przerwa w danych ryzykowała fałszywy alarm
  detekcji.
- **Krótkie, wyzwalane nagrania** (triggered recording, cała próbka to
  głównie wstrząs): błąd 2 (MAD=0) jest tu bardziej prawdopodobny przy
  danych o niskiej rozdzielczości - warto sprawdzić próg `th` zwracany
  przez `anomalies()` i upewnić się, że nie jest podejrzanie mały.
- **`threshold` w `twist()` i `factor` w `anomalies()`** to punkty
  startowe (odpowiednio 0.4 i 3.0), nie zwalidowane normy — zależą od
  czułości sensora, poziomu szumu tła i tego, co uznajesz za "istotną"
  zmianę. Wymagają kalibracji na Twoich danych.
- Metoda nie jest przyczynowa w pełnym sensie (k-NN w czasie może
  sięgać w przyszłość względem próbki), więc do detekcji w czasie
  rzeczywistym nadaje się z niewielkim opóźnieniem, nie "na bieżąco" bez
  żadnego lagu.

## Przykład użycia

```python
from timdr_core_earthquake import TIMDR_EarthquakeCore

core = TIMDR_EarthquakeCore()
flow_grad = core.flow(t, s)
twist_pts, twist_strength = core.twist(flow_grad, t)   # teraz wymaga t
smooth = core.trm(t, s)
anomaly_pts, residuals, th = core.anomalies(t, s)
fronts, _, _ = core.fronts(t, s)
```

Uruchomienie: `python demo.py` / testy: `pytest -q`.
