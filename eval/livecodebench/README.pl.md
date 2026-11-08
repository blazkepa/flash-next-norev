# LiveCodeBench (code_generation_lite) jako drugi strażnik regresji

Po co: HumanEval jest nasycony i skażony (baza 4B 79,9%, Qwen3.6-35B 95,1%), więc słabo wykrywa regresję po treningu.
LiveCodeBench to zadania konkursowe (AtCoder, LeetCode) z testami ukrytymi, trudniejsze i z datą publikacji.

## Dane (`data/`, nie modyfikować)

| Co | Wartość |
|---|---|
| Zestaw | `livecodebench/code_generation_lite` na Hugging Face, wersja `release_v6` (najnowsza; ostatnia aktualizacja repo 2025-06-05) |
| Rewizja repo | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` |
| Pobrany plik | `test6.jsonl` (przyrost v6: 175 zadań, 2025-01-04..2025-04-06, 134 MB). Pełne `release_v6` = `test.jsonl`..`test6.jsonl`, 1055 zadań; reszty nie pobieramy, bo okno obejmuje tylko najnowsze zadania |
| sha256 `test6.jsonl` | `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5` |
| Podzbiór | `data/subset.json`, 100 zadań, generuje `make_subset.py` (deterministycznie) |

Testy prywatne są w pliku jako `base64(zlib(pickle(str JSON)))`. Plik jest pobrany, więc niezaufany: odkodowujemy go unpicklerem,
który odrzuca każdy obiekt poza łańcuchem (`_SafeUnpickler`), a skrypty uruchamiamy przez `python3 -I`.

## Okno zadań i ryzyko skażenia

Reguła z planu: zadania opublikowane po dacie odcięcia danych Qwen3.5, maks. 150; jeśli mniej niż 60, najnowsze 100.
Karta `Qwen/Qwen3.5-4B` nie podaje daty odcięcia. Pośrednio: raportuje wyniki na HMMT **Nov 25** i LiveCodeBench v6, model wydano ok. 2026-02-24,
więc dane sięgają co najmniej listopada 2025. Najnowsze zadanie w LCB v6 jest z 2025-04-06, czyli **0 zadań po odcięciu** → wariant awaryjny:
**100 najnowszych** (2025-02-15..2025-04-06; AtCoder 61, LeetCode 39; easy 23, medium 31, hard 46), posortowane rosnąco po dacie.

Konsekwencja: zestaw nie jest wolny od skażenia i nie mierzy absolutnej umiejętności. Służy do porównania *przed/po* treningu na tych samych
zadaniach (regresja), w obrębie tego samego `subset.json`. Zmiana podzbioru = nowa linia bazowa.

## Uruchamianie

```bash
LLM_BASE_URL=http://127.0.0.1:8081/v1 LLM_API_KEY=x python3 -I eval/livecodebench/run_livecodebench.py --model mlx-community/Qwen3.5-4B-bf16 --no-think --max-tokens 2048 --tag base-qwen35-4b-bf16-nothink
```

Opcje jak w HumanEval (`--model --limit --workers --no-think --tag --max-tokens`) plus `--timeout` (s na test, domyślnie 10) i `--mem-mb` (domyślnie 1024).
Prompt: szablon "generic" z oficjalnego `lcb_runner` (system + `### Question / ### Format / ### Answer`, starter code dla LeetCode, stdin/stdout dla AtCoder).
Ekstrakcja: ostatni blok ```python, także niezamknięty. Ocena: każdy test publiczny i prywatny w osobnym podprocesie `python3 -I` w pustym katalogu tymczasowym,
minimalne środowisko, limit czasu i limit RSS (strażnik `ps`, bo `setrlimit` na macOS nie działa). stdout porównywany wierszami po `strip()`, a przy różnicy
jako liczby `Decimal` (jak w oficjalnym checkerze); wynik funkcji przez JSON (krotka → lista). Zaliczenie = wszystkie testy przechodzą.
Po timeout, przekroczeniu pamięci albo SyntaxError przerywamy resztę testów zadania (`stopped_early`), więc `failed_test_count` to porażki zaobserwowane, nie pełna liczba.

Wynik `results/<tag>.json`: model, no_think, max_tokens, dataset (wersja, sha256 podzbioru, okno), n, passed, pass_at_1, by_difficulty, wall_seconds,
completion_tokens, tok_per_s_per_stream, per_problem (id, platform, difficulty, passed, failed_test_count, tests_run, tests_total, error, tokens, truncated, code).
