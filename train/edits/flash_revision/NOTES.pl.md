# Edycja wag bez treningu: usunięcie podprzestrzeni "rewizji" z Qwen3.8-Flash-Next (oQ4e WLSRefit MTP), 2026-11-06

Motywacja: na LiveCodeBench 100 Flash-Next oQ5e bez myślenia ma 53/100, z myśleniem 65/100, ale w obu trybach ~1/3 zadań kończy się
na limicie tokenów w spirali "Wait, let me reconsider" bez kodu (`eval/README.md`). Ten sam zabieg co dla 4B (`train/edits/d1_revision/`)
i co abliteracja orcarouter na tej architekturze (149 tensorów, W' = W − d dᵀW), tylko z kierunkiem rewizji zamiast odmowy.

## Co jest inne niż u 4B (trzy lekcje)

1. **Zewnętrzna tablica n-gramów.** Model ma 51B parametrów w tablicy n-gramów (PLE, 38 GB w 5 bitach). mlx_vlm 0.7.11 potrafi ją czytać
   z dysku (mmap) tylko w 4 bitach, więc eksperyment idzie na `GBP-DE/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP` (108 GB). Widok modelu bez
   tablicy w RAM: `prepare_external_ple_model` (twarde dowiązania, 77 GB, ładowanie 8 s, 74 GB w pamięci).
2. **Przestrzeń pomiaru = przestrzeń pisarzy.** Model ma 4 strumienie resztkowe (hiperpołączenia). Bloki czytają je przez RMSNorm
   z wagami per kanał, które są skrajnie nierówne (warstwa 32: od −0,87 do 3,34, mediana −0,05). Kierunek zmierzony po normalizacji
   (wyjście `hyper_connection_mixer`) ma z kierunkiem w surowym strumieniu kosinus 0,00–0,14. Pierwsza edycja (norev32, kierunek z wyjścia
   mixera, warstwa 32) była mechanicznie poprawna i behawioralnie pusta (4–9 fraz rewizji w 1500 tok, jak baza). Poprawka: pomiar
   w surowych strumieniach (`tools/flash_find_direction.py --raw`, ręczna pętla po warstwach, zapis `h[0, idx]` jako [n, 4, 2560]).
3. **Kwantyzacja zjada edycję.** Wagi są w 4–8 bitach; edycja = dekwantyzacja → rzut → ponowna kwantyzacja ze świeżymi skalami. Zmiana
   2% normy jest poniżej kroku siatki 4-bitowej: po zaokrągleniu wraca ~30% usuniętej składowej (cos resztki z oryginałem 0,96). Przy
   5 bitach 1,6%, 6 bitach 0,3%, 8 bitach 0,1%. Eksperci `switch_mlp.down_proj` (49 × [512, 2560, 640]) idą więc do 6 bitów (+10 GB,
   model 118 GB), reszta zostaje w swoich 8 bitach. Zapis "na starą siatkę" jest gorszy (98% zostaje).

## Pomiar kierunku (`tools/flash_find_direction.py --raw --limit 160 --trunc-frac 0.75`, 16 min, 634 tok/s)

160 odpowiedzi nauczyciela z fazy D (120 uciętych w spirali, 40 skończonych), 3020 pozycji dodatnich (token przed frazą rewizji
i jej pierwszy token) i 3020 kontrolnych (początki zdań bez rewizji). Wspólny kierunek warstwy = średnia różnic średnich po 4 strumieniach.

| po warstwie | 4 | 8 | 16 | 24 | 32 | 40 | 44 | 45 | 48 |
|---|---|---|---|---|---|---|---|---|---|
| Cohen's d | 1,61 | 1,71 | 1,81 | 2,05 | 1,96 | 2,39 | 2,64 | 2,72 | 2,01 |
| AUC | 0,933 | 0,933 | 0,936 | 0,947 | 0,932 | 0,945 | 0,958 | 0,960 | 0,933 |

Kierunki per strumień NIE są równoległe (warstwa 32: cos strumienia 0 z 1/2/3 = 0,47 / 0,03 / 0,23), więc jeden kierunek nie wystarcza.
PCA kierunków per strumień z warstw 8–40: 80% energii w 5 wymiarach, 90% w 9. Wybrane: **6 głównych składowych** (`directions_pca6.npz`),
pokrywają 84% energii kierunków per strumień w warstwach 5–44 (losowe 6 kierunków: 0,35%). Pierwsza składowa ma cos 0,93–0,96
ze wspólnymi kierunkami warstw 16/24/32.

Pierwszy pomiar (po normalizacji, 300 odpowiedzi, 2,2 h): Cohen's d 1,85–2,9, AUC do 0,98, kierunki sąsiednich warstw cos 0,9–0,99,
ale to przestrzeń czytelników, nie pisarzy (zob. lekcja 2). Pliki: `directions.npz`, `report.json` (po normie), `../flash_revision_raw/`
(surowe strumienie).

## Edycja (`tools/flash_ablate.py --dirs directions_pca6.npz --dir-layer 1 --streams all --expert-bits 6`, 35 s)

149 tensorów: `self_attn.o_proj` 13, `linear_attn.out_proj` 36, `mlp.switch_mlp.down_proj` 49, `mlp.shared_expert.down_proj` 49,
`ple.value_proj` 1, `embed_tokens` 1 (nie są wiązane z głową). W' = W − Σₖ uₖ(uₖᵀW) dla 6 ortonormalnych uₖ. Średnia zmiana normy 5,0%
(max 5,8%), po ponownej kwantyzacji zostaje systematycznie 0,1–0,2%. Wynik: `~/.omlx/models/local/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP-norev-pca6`
(118 GB; szardy bez zmian to twarde dowiązania do bazy) + widok `train/models/flash-next-oq4e-norev-pca6-view`.

**Kontrola mechaniczna** (`tools/flash_check_direction.py`, 12 innych odpowiedzi, średni rzut na kierunek warstwy, pozycje przed rewizją / kontrolne):
baza 0,39/0,04 (w. 16), 0,46/0,00 (24), 0,75/0,11 (32), 2,64/0,45 (40); po edycji 0,00/0,01, −0,01/−0,03, 0,04/0,06, 0,01/−0,11.
Składowa wzdłuż kierunku zniknęła w całej sieci.

## Test zachowania w procesie (`tools/flash_ab_generate.py`, dekodowanie zachłanne, bez myślenia, limit 2 500 tok)

8 zadań, na których nauczyciel (oQ5e) oba razy wpadł w spiralę, i 4 kontrolne, które rozwiązywał 2/2:

| model | spirala: zaliczone | ucięte | z kodem | rewizji/odp. | tok/odp. | kontrola: zaliczone | ucięte | tok/odp. |
|---|---|---|---|---|---|---|---|---|
| baza oQ4e | 1/8 | 7 | 2 | 5,6 | 2 328 | 2/4 | 2 | 1 601 |
| **norev-pca6** | 1/8 | **0** | **7** | **0,0** | **1 404** | **4/4** | **0** | **610** |

Odczyt jak u 4B: edycja usuwa spiralę całkowicie (zero fraz rewizji, zero ucięć) i skraca odpowiedzi 1,7–2,6×, ale zadań spoza zasięgu
nie rozwiązuje: 6 z 8 kończy się błędną odpowiedzią zamiast ucięcia, 1 wpada w pętlę tokenową (ta sama linia powtarzana, efekt "zdjętego
hamulca" znany z 4B). Zysk jest tam, gdzie umiejętność była, a spirala ją blokowała: 2 zadania kontrolne, które baza ucinała, przechodzą.
Pełne porównanie (100 odłożonych zadań k=1, LCB 100 bez myślenia) przez oMLX: `flash_eval.sh` (scratchpad), wyniki poniżej po zakończeniu.

## Pełne porównanie przez oMLX (2026-11-06, `flash_eval.sh`, bez myślenia, 4 strumienie)

| model | odłożone 100 (k=1, 3500 tok) | ucięte | mediana tok | LCB 100 (8192 tok) | easy / medium / hard | LCB na limicie | LCB mediana tok | LCB czas |
|---|---|---|---|---|---|---|---|---|
| baza oQ4e WLSRefit | 60 | 28 | 1 416 | 49 | 22/23 · 20/31 · 7/46 | 38 | 3 870 | 123 min |
| norev-pca6 | 58 | 7 | 669 | **43** | 22/23 · 16/31 · 5/46 | 14 | 939 | 55 min |

(Korekta 2026-11-09: wiersz bazy jest taki, jak go pierwotnie zapisano. Dwa zapytania bazy serwer odrzucił i policzono je jako
porażki; po dogenerowaniu baza ma 51, 23/23 · 20/31 · 8/46, a pca6 wobec bazy to 3 lepiej / 11 gorzej.)

Per zadanie: odłożone 9 lepiej / 11 gorzej, LCB 5 lepiej / 11 gorzej. **Wynik: edycja rangi 6 przy pełnej skali jest za mocna.** Daje 2× krótsze
odpowiedzi i 2,2× krótszy czas LCB, ale kosztuje 2 pkt na odłożonych i 6 pkt na LCB (medium i hard). Z uciętych u bazy większość staje się
pewną, krótką, błędną odpowiedzią: część fraz "Wait" była prawdziwą autokorektą, a podprzestrzeń rangi 6 zabrała ją razem ze spiralą.
Następny krok (niezrobiony): przegląd łagodniejszych wariantów (`--rank 1|3`, `--scale 0.5`) w procesie na odłożonych; albo krótki LoRA
"healing" na edytowanym modelu. Oba modele zostają w oMLX: baza jako domyślny, norev-pca6 jako "szybki, słabszy o kilka pkt".

## Przegląd łagodniejszych wariantów (2026-11-07, `flash_sweep.sh`, w procesie, greedy, odłożone 1000–1099, 3500 tok)

Wszystkie z `directions_pca6.npz --dir-layer 1 --streams all --expert-bits 6`. Baza w procesie daje 68 (przez oMLX 60): porównywać tylko w obrębie tabeli.

| wariant | zaliczone | +/− vs baza (p, test znaków) | ucięte | mediana tok | tok vs baza (oba nieucięte) | rewizji/odp. | bez kodu | czas |
|---|---|---|---|---|---|---|---|---|
| baza oQ4e WLSRefit | 68 | — | 28 | 1 429 | — | 4,6 | 22 | 87 min |
| r1 (ranga 1, skala 1) | 62 | +3 / −9 (0,15) | 30 | 1 312 | 0,87 | 4,8 | 28 | 91 min |
| r3 (ranga 3, skala 1) | 63 | +7 / −12 (0,36) | 11 | 1 011 | 0,76 | 0,8 | 9 | 65 min |
| **r6s05 (ranga 6, skala 0,5)** | **66** | +7 / −9 (0,80) | **15** | **969** | 0,88 | 1,3 | 14 | **68 min** |

Odczyt: ranga 1 nie zmienia zachowania (rewizje bez zmian), a mimo to traci 6 pkt, więc sama perturbacja wag daje szum rzędu ±5 pkt na 100
zadaniach. Żadna różnica nie jest istotna. Ranga 3 to już prawie pełne zdjęcie spirali. **Najlepszy kompromis to r6s05:** −2 pkt (w szumie),
ucięcia 28→15, mediana tokenów −32%, czas −22%. Następny krok: zbudować r6s05 do oMLX i zmierzyć LCB 100 (pełna ranga 6 dała 49→43).
Modele z przeglądu skasowane (odtworzenie: `flash_ablate.py ... --scale 0.5`, ok. 1 min).

**LCB 100 dla r6s05 (2026-11-07, oMLX, bez myślenia, 8192): 53 vs baza 51 (pierwotnie podane 49)** (22/23 · 19/31 · 12/46; +6 / −4, pierwotnie +8 / −4; na limicie 38→23; mediana tok 3 863→2 935). Korekta 2026-11-09: dwa zapytania bazy serwer odrzucił i policzono je jako porażki, po dogenerowaniu baza ma 51; czasy przebiegów przez oMLX (123 / 55 / 73 min) w tych notatkach nie są czystym porównaniem (różne tryby serwera), porównywać tokeny. Zyski to głównie zadania trudne, które baza ucinała. Model w oMLX: `local/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP-norev-r6s05`. Wynik na LCB przeczy obawie z odłożonych (66 vs 68): tam różnica była w szumie, tu zysk +4 przy −40% czasu (po korekcie bazy zysk to +2; czasy przez oMLX nie są porównywalne).
