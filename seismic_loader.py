"""
seismic_loader.py — Seismic Loader for TIMDR Earthquake Detector
====================================================================
Wczytuje dane sejsmiczne (CSV, JSON API, waveform) i przygotowuje je
(detrend, odszumianie odosobnionych skoków, normalizacja) do
TIMDR_EarthquakeCore.
"""

import numpy as np
import json
import csv
import warnings


class SeismicLoader:
    def __init__(self, normalize=True, detrend=True, clip_outliers=True,
                 despike_k=7, despike_factor=6.0):
        self.normalize = normalize
        self.detrend = detrend
        self.clip_outliers = clip_outliers
        self.despike_k = despike_k
        self.despike_factor = despike_factor

    # -----------------------------------------
    # CSV loader
    # -----------------------------------------
    def load_csv(self, path, t_col="t", s_col="s"):
        t, s = [], []
        n_rows_total = 0
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                n_rows_total += 1
                try:
                    t.append(float(row[t_col]))
                    s.append(float(row[s_col]))
                except Exception:
                    continue

        # POPRAWKA (bug cichego pustego wyniku): jesli podana kolumna nie
        # istnieje w CSV (literowka, inna konwencja nazw), oryginalny kod
        # po cichu zwracal puste t, s bez zadnego bledu/ostrzezenia -
        # zweryfikowano: CSV z kolumnami ["time","amplitude"] wczytany
        # domyslnymi t_col="t"/s_col="s" dawal t=[], s=[] bez zadnej
        # informacji, co poszlo nie tak.
        if n_rows_total > 0 and len(t) == 0:
            # POPRAWKA 2 (znaleziona na realnym pliku uzytkownika, eksport
            # ObsPy "tr.times()/tr.data" do CSV bez naglowka - pierwszy
            # wiersz danych to np. "0.0,18754"): DictReader bez naglowka
            # bierze PIERWSZY WIERSZ DANYCH jako nazwy kolumn (tu:
            # fieldnames=['0.0','18754']), wiec KAZDY wiersz - wlacznie z
            # tym pierwszym - nie pasuje do t_col/s_col i caly plik ladowal
            # sie jako pusty, mimo ze dane sa poprawne, tylko bez naglowka.
            # Fallback: jesli oba "fieldnames" parsuja sie jako liczby
            # (czyli to nie sa prawdziwe nazwy kolumn, tylko dane), wczytaj
            # plik ponownie zakladajac 2 kolumny w kolejnosci (t, s) BEZ
            # naglowka, z tym pierwszym wierszem wlaczonym z powrotem.
            if len(fieldnames) == 2:
                try:
                    first_t, first_s = float(fieldnames[0]), float(fieldnames[1])
                    headerless_ok = True
                except (TypeError, ValueError):
                    headerless_ok = False
                if headerless_ok:
                    t2, s2 = [first_t], [first_s]
                    with open(path, "r", newline="") as f2:
                        plain_reader = csv.reader(f2)
                        next(plain_reader, None)  # ten sam pierwszy wiersz, juz dodany wyzej
                        for row2 in plain_reader:
                            if len(row2) < 2:
                                continue
                            try:
                                t2.append(float(row2[0]))
                                s2.append(float(row2[1]))
                            except ValueError:
                                continue
                    warnings.warn(
                        f"Plik CSV nie ma naglowka (pierwszy wiersz to dane: "
                        f"{fieldnames}) - wczytano jako 2 kolumny bez naglowka "
                        f"w kolejnosci (t, s), zamiast szukac kolumn "
                        f"t_col='{t_col}'/s_col='{s_col}'."
                    )
                    return self._postprocess(np.asarray(t2), np.asarray(s2))
            raise ValueError(
                f"Nie udalo sie odczytac zadnego wiersza z kolumn "
                f"t_col='{t_col}', s_col='{s_col}'. Kolumny w pliku: "
                f"{fieldnames}. Podaj poprawne t_col/s_col."
            )
        if len(t) < n_rows_total:
            warnings.warn(
                f"Pominieto {n_rows_total - len(t)} z {n_rows_total} wierszy "
                f"(bledne/brakujace wartosci w t_col/s_col)."
            )

        return self._postprocess(np.asarray(t), np.asarray(s))

    # -----------------------------------------
    # JSON API loader
    # Format: { "data": [ {"t":..., "s":...}, ... ] }
    # -----------------------------------------
    def load_json_api(self, json_string):
        obj = json.loads(json_string)
        raw = obj.get("data", [])
        t, s = [], []
        for item in raw:
            try:
                t.append(float(item["t"]))
                s.append(float(item["s"]))
            except Exception:
                continue

        if len(raw) > 0 and len(t) == 0:
            raise ValueError(
                f"Nie udalo sie odczytac zadnego punktu z {len(raw)} "
                f"wpisow 'data' - sprawdz, czy kazdy element ma klucze "
                f"'t' i 's'."
            )
        if len(t) < len(raw):
            warnings.warn(f"Pominieto {len(raw) - len(t)} z {len(raw)} wpisow (brak/bledne t lub s).")

        return self._postprocess(np.asarray(t), np.asarray(s))

    # -----------------------------------------
    # Waveform loader (list, numpy array)
    # t może być None → generujemy indeksy
    # -----------------------------------------
    def load_waveform(self, s, t=None):
        s = np.asarray(s, dtype=float)
        if t is None:
            # UWAGA: bez prawdziwego t generujemy indeks probki jako "czas"
            # (dt=1). TIMDR_EarthquakeCore liczy gradienty i progi w
            # jednostkach amplituda/czas - jesli Twoje realne probkowanie
            # nie jest dokladnie 1 jednostka/probke, progi (threshold,
            # factor) beda dotyczyc innej skali niz myslisz. Podaj
            # prawdziwe t w sekundach, jesli to mozliwe.
            t = np.arange(len(s), dtype=float)
        else:
            t = np.asarray(t, dtype=float)
        return self._postprocess(t, s)

    # -----------------------------------------
    # Postprocessing: sortowanie, detrend, despiking, normalizacja
    # -----------------------------------------
    def _postprocess(self, t, s):
        if len(s) == 0:
            return t, s

        # Sortowanie po czasie + odrzucenie duplikatow t - TIMDR_EarthquakeCore
        # wymaga scisle rosnacego t (patrz timdr_core_earthquake.py._validate),
        # a dane z CSV/API nie zawsze przychodza posortowane.
        order = np.argsort(t, kind="stable")
        t, s = t[order], s[order]
        keep = np.concatenate([[True], np.diff(t) > 0])
        if not np.all(keep):
            warnings.warn(f"Usunieto {np.sum(~keep)} probek z duplikatem/nierosnacym znacznikiem czasu.")
            t, s = t[keep], s[keep]
        if len(s) == 0:
            return t, s

        # Detrend (usuniecie dryfu)
        if self.detrend and len(s) > 2:
            # POPRAWKA: oryginalny kod odejmowal tylko `slope * t`,
            # zostawiajac wyraz wolny (przesuniecie DC) nietkniety.
            # Zweryfikowano: czysty trend liniowy s=5+2*t (bez szumu)
            # po oryginalnym "detrend" dawal STALA WARTOSC 5.0 wszedzie
            # zamiast ~0 - "usuniecie dryfu" usuwalo tylko nachylenie,
            # nie caly trend. Naprawiono: odejmujemy pelne dopasowanie
            # liniowe (nachylenie I wyraz wolny).
            slope, intercept = np.polyfit(t, s, 1)
            s = s - (slope * t + intercept)

        # Odszumianie odosobnionych skokow (despiking)
        if self.clip_outliers:
            # POPRAWKA (bug powazny): oryginalny kod liczyl `mean`/`std`
            # na CALYM sygnale i obcinal wszystko poza mean +/- 5*std.
            # Zweryfikowano: pojedynczy, wyrazny impuls wstrzasu (5.0
            # przy tle o std=0.05, czyli 100x poziom szumu) w 300-probkowym
            # nagraniu zostal obciety do 1.48 - **70% redukcji prawdziwego
            # sygnalu**, ktory ten detektor ma za zadanie znajdowac.
            # Dodatkowy problem: zachowanie zalezy od tego, jaki procent
            # CALEGO pliku zajmuje wstrzas (przypadkowa wlasciwosc
            # segmentacji nagrania, nie fizyki) - dluzszy/wiekszy wstrzas
            # moze "samo-znormalizowac" globalne std i przetrwac, krotszy
            # zostanie obciety, niezaleznie od tego, ktory jest ważniejszy.
            #
            # Naprawiono: lokalne odszumianie (Hampel-style) - kazda
            # probka porownywana do mediany/MAD SASIADOW W CZASIE
            # (bez niej samej), nie do statystyk calego nagrania. Dzieki
            # temu wynik nie zalezy od dlugosci/proporcji pliku, a
            # PRAWDZIWY wstrzas trwajacy wiele kolejnych probek NIE jest
            # obcinany (bo lokalne sasiedztwo kazdej jego probki tez jest
            # podniesione, wiec odchylenie od lokalnej mediany jest male) -
            # obcinane sa tylko naprawde ODOSOBNIONE (1-2 probki) skoki,
            # typowe dla usterek czujnika/przetwornika.
            #
            # Uczciwe zastrzeżenie: to NIE rozwiazuje fundamentalnej
            # niejednoznacznosci "pojedyncza probka poczatku wstrzasu"
            # vs "pojedyncza probka usterki czujnika" - z samej amplitudy
            # nie da sie ich matematycznie odroznic. Jesli Twoje dane maja
            # bardzo ostre, jednopróbkowe onsety P-wave, rozważ wylaczenie
            # `clip_outliers` albo podniesienie `despike_factor`.
            s = self._local_despike(t, s)

        # Normalizacja amplitudy
        if self.normalize:
            maxv = np.max(np.abs(s))
            if maxv > 0:
                s = s / maxv

        return t, s

    def _local_despike(self, t, s):
        n = len(s)
        out = s.copy()
        half = self.despike_k // 2
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            window = np.concatenate([s[lo:i], s[i + 1:hi]])
            if len(window) == 0:
                continue
            med = np.median(window)
            mad = np.median(np.abs(window - med)) * 1.4826
            mad = mad if mad > 1e-9 else 1e-9
            if abs(s[i] - med) > self.despike_factor * mad:
                out[i] = np.clip(s[i], med - self.despike_factor * mad, med + self.despike_factor * mad)
        return out
