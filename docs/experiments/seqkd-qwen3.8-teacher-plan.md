# Plan: Qwen3.8-27B als echter Teacher (Sequence-Level-KD)

Status: umgesetzt auf Branch `feature/seqkd-qwen-teacher` (2026-10-03), nicht committet.
Ergebnis und Abweichungen vom Plan: siehe Abschnitt „Umsetzung" am Ende.

## Context

`distill`/`opd` sind Logit-KD (TRL `GKDTrainer`, `online_training.py:588-609`) und verlangen einen
Teacher mit identischem Tokenizer (`online_training.py:483-496`). Da Boldt nur 1B-Modelle hat, gibt
es keinen brauchbaren Teacher; der Config-Eintrag (`secure-current.json:59`) ist der Seed selbst und
scheitert im echten Lauf (`loop.py:118-124`), während der Dry-Run den Teacher nicht prüft
(`cli.py:140-151`). Ein externer Teacher geht nur als **Sequence-Level-KD**: Teacher erzeugt deutsche
Antworten → Student wird per SFT darauf trainiert. Der vorhandene `pt data synthesize` hängt an der
Legacy-Pipeline, lädt ungeprüft in fp32, schreibt einen fest auf „clean" gesetzten Leakage-Report und
sein Output ist im sicheren Pfad nicht trainierbar – er wird nicht wiederverwendet.

Entscheidungen (User): ganzes Repo auf transformers 5.x upgraden; Teacher
`Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` (Apache-2.0, 27.8 GB); Feature-Branch,
Guard für die Umsetzung deaktiviert.

Harte Randbedingungen:
- Qwen3.8 (`Qwen3_5ForConditionalGeneration`, hybride Linear-Attention, multimodal, vocab 248k) braucht
  transformers ≥ 5.8; vLLM 0.19.1 pinnt `torch==2.13.0` (= Repo-Pin) und `transformers>=5.10.4`.
- A6000 = Ampere, kein natives FP8 → vLLM nutzt Marlin W8A16; laut vLLM-Recipe auf Ampere
  **nicht verifiziert** → Go/No-Go-Smoke-Test vor jeder weiteren Arbeit.
- Thinking ist default an → `enable_thinking: false`; Non-thinking-Sampling laut Model Card:
  T=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5.
- Generierung von ~10k Antworten dauert voraussichtlich länger als das 90-min-Loop-Budget →
  Generierung ist ein **eigener Vorbereitungsschritt** (wie `data prepare`); der Lever konsumiert eine
  gepinnte Generierungs-Run-ID.

## Schritte

### 0. Branch
`git checkout -b feature/seqkd-qwen-teacher`.

### 1. Dependency-Upgrade (gesamtes Repo) — `pyproject.toml`, `uv.lock`
- `transformers` ≥ 5.10.4, `vllm==0.19.1` (neues Extra `teacher`), `torch==2.13.0` bleibt;
  `trl`, `peft`, `accelerate`, `datasets`, `lm-eval`, `mergekit`, `liger-kernel` auf mit
  transformers 5 kompatible Versionen (exakt per `uv lock` auflösen und pinnen).
- `huggingface-hub` 1.x (von transformers 5 verlangt) → `hf-transfer` entfernen,
  `HF_HUB_ENABLE_HF_TRANSFER`-`setdefault` in `src/boldt_posttrain/__init__.py:13-14` ersetzen (hf-xet).
- API-Brüche fixen: `torch_dtype`→`dtype`, Tokenizer-/Chat-Template-API,
  SFTConfig/DPOConfig/GRPOConfig-Felder, Fundort von `GKDTrainer` (ggf. `trl.experimental`);
  betroffen u. a. `secure_compat/training.py`, `secure_compat/evaluation.py`, `online_training.py`,
  `resolver.py`.
- Gate: `uv run --locked pytest`, `ruff`, `pt doctor --mode all`, `pt policy validate`, plus die
  vorhandenen Real-CUDA-Tests (`tests/test_online_training.py:320-370`).

### 2. GPU-Smoke-Test (Go/No-Go)
Wegwerf-Skript (Scratchpad): vLLM `LLM(model="Qwen/Qwen3.8-27B-FP8", revision="017b9c7…",
max_model_len=16384, limit_mm_per_prompt={"image": 0}, gpu_memory_utilization=0.90)`,
20 deutsche Prompts aus dem SFT-Manifest, `chat_template_kwargs={"enable_thinking": False}`.
Messen: lädt es, VRAM, Tokens/s, kein `<think>` im Output, Deutsch-Qualität stichprobenartig.
**Scheitert das, wird gestoppt** und die Alternativen (W4A16-Quant, anderer Teacher) werden vorgelegt.

### 3. Policy (human-owned) — `configs/posttrain/policy.json`, `src/boldt_posttrain/policy.py`
- Neuer Top-Level-Block `teachers`: Liste exakter Einträge
  `{repo_id, revision, license, purpose: "seqkd"}` mit Qwen3.8-27B-FP8; Validierung in `policy.py`
  (`_require_keys`, 40-hex-Revision via `HUB_REVISION_RE`, Lizenz ∈ `allowed_licenses`).
- `training.allowed_methods` += `"seqkd"`.
- Folge: neuer `policy_sha256` → bestehendes Daten-Manifest ist stale → `data prepare` neu.
- Der Diff wird gesondert vorgelegt und vom Owner freigegeben.

### 4. Neuer Lever `seqkd` (nicht `distill` umdeuten – das bricht OPD-Semantik und Tests)

**Experiment-Schema** (`secure_compat/config.py:_SCHEMA`): neuer Pflichtblock `seqkd`
`{teacher, generation_run, max_prompts, max_new_tokens, temperature, top_p, top_k,
presence_penalty, seed}` (keine Forbidden-Fragmente aus `config.py:18-29`). Bestehende Configs
bekommen den Block.

**Stufe 1 – `pt seqkd generate --real --allow-gpu --config …`** (neues Modul
`src/boldt_posttrain/seqkd.py`, Ausführung als Subprozess, damit vLLM den GPU-Speicher freigibt):
- Teacher muss exakt einem `policy.teachers`-Eintrag entsprechen; Lizenz zusätzlich via
  `distillation._teacher_license` (Hub-Card). Fingerprint über `resolver.resolve_hub_model`
  (falls Qwen kein `chat_template.jinja` ausliefert: Fingerprint der vorhandenen Dateien ergänzen).
- Prompts: User-Turns aus dem verifizierten SFT-Manifest (`verify_data_manifest`,
  `secure_compat/training.load_manifest_rows`), deterministisch per Seed gesampelt, `max_prompts`.
- Generierung mit vLLM wie im Smoke-Test, Sampling aus der Config.
- Filter: Verwerfen bei `<think>`, leer, `finish_reason == "length"`; danach dieselben Gates wie
  `data prepare` – `normalize_row`, `LanguageIdentifier` (de ≥ 0.8), Dedup, `leakage_filter`
  (Treffer = harter Abbruch). Dafür die Per-Row-Logik aus `secure_compat/data_pipeline.prepare`
  (`:611ff`) in eine wiederverwendbare Funktion ziehen statt duplizieren.
- Output `outputs/posttrain/seqkd/<run_id>/`: Shard `train_sft-…jsonl` (Rolle `sft_shard`), Manifest
  im sicheren Schema (eval_suite_hash, policy_sha256, Leakage-Stats) + Teacher-Ref/Revision/Lizenz,
  Sampling-Parameter, Prompt-Quell-Manifest-SHA, Zähler/Verwürfe; Run-Card `run_type`
  `seqkd_generate` (in `artifacts.RUN_TYPES`), Event + atomare Publikation wie
  `data_pipeline.py:860-867`.

**Stufe 2 – Lever `seqkd` im Loop** (`loop._execute_lever`, `_MANUAL_LEVERS`, CLI `train seqkd`):
- Verifiziert die in `seqkd.generation_run` gepinnte Generierung (Hashes, Teacher = Config-Teacher,
  Policy-SHA aktuell), dann `secure_compat/training.train_adapter` mit diesem Manifest,
  `input_artifacts` = Shard, `parent_run_ids` = Generierungs-Run, `lineage`; `run_type`
  `train_seqkd` (Parameter statt Hardcode bei `training.py:296-298`). Rückgabe `student_run_id`
  (`loop.py:305`).
- Dry-Run-Preflight-Zweig in `cli.py:140-151`: Policy-Teacher, Generierungs-Run, Shards.
- Eval/Score/Promote unverändert.

### 5. Tests (ohne GPU via injizierbarem Generator, analog `rows_provider`)
Policy-`teachers`-Validierung (gültig, falsche Lizenz, bewegliche Revision); Schema `seqkd`;
Filter (`<think>`, Trunkierung, Sprache, Leakage → Abbruch); Manifest-Tamper; Loop-Dispatch und
Dry-Run für `seqkd`; Teacher ≠ Policy-Eintrag → Abbruch. Ein markierter Real-GPU-Test für Stufe 1
mit 5 Prompts.

### 6. Doku
`AGENTS.md` (Lever-Liste, `pt seqkd generate`), `README.md`, `.claude/commands/pt-seqkd.md`.
`AUTORESEARCH_POSTTRAIN.md`/`CLAUDE.md` sind protected → Textvorschlag an den Owner.

## Wichtige Nebenwirkung
Der immutable Baseline wurde mit dem alten Stack (transformers 4.57.6) erzeugt. Nach dem Upgrade
vergleicht der Scorer Kandidaten (neuer Stack) mit einem Baseline aus dem alten Stack → Verzerrung.
`outputs/posttrain/baseline/**` ist protected: der Baseline muss nach dem Upgrade vom Owner neu erzeugt
werden (`baseline run`); alt vs. neu wird verglichen und die Drift berichtet.

## Nicht im Scope
- `pt data synthesize` (Legacy) bleibt unverändert.
- `distillation.teacher` = Seed in `secure-current.json` bleibt vorerst; `opd`/`distill` sind damit
  weiterhin nicht lauffähig. Eintrag wird in einem separaten Schritt bereinigt.

## Verification (mit echten Run-IDs + Exit-Codes, keine Behauptung ohne Beleg)
1. `uv run --locked pytest`, `ruff check`, `pt policy validate`, `pt doctor --mode all` → Exit 0.
2. Smoke-Test-Ausgabe (VRAM, tok/s, Beispielantworten).
3. `pt data prepare --real` → neue Manifest-Run-ID; `pt baseline run --real --allow-gpu` (Owner).
4. `pt seqkd generate` mit `max_prompts=200` → Laufzeit hochrechnen, Verwurfquoten, Stichprobe;
   danach Volllauf → Run-ID + Manifest-SHA.
5. `pt loop run --real …` mit `lever: seqkd` → Run-ID, Run-Card, Checkpoint-Hash,
   Data-Manifest-Hash, Suite-Hash, Score-ID, Event-Sequenz; Promotion nur wenn die Gates passen.

## Umsetzung (2026-10-03)

### Abweichungen vom Plan
- **Versionen:** vLLM 0.19.1 pinnt `torch==2.10.0`. Gelockt sind deshalb vLLM 0.27.1 (passt zu
  `torch==2.13.0`), transformers 5.17.0, TRL 1.14.1, PEFT 0.21.2, accelerate 1.15.0 und
  huggingface-hub 1.16.1.
- **mergekit:** 0.1.4 ist mit pydantic ≥ 2.11 und transformers 5 inkompatibel; Overrides
  scheiterten. mergekit ist deshalb auf den Upstream-Commit `9eeb539892c6753b032ea164e818b44a8c7a401b`
  (0.1.5-dev) gepinnt.
- **`seqkd`-Block ist optional statt Pflicht:** Er ist strikt validiert, aber nur bei vorhandenem
  Block oder `lever: seqkd` nötig. Bestehende Experiment-Dateien bleiben unverändert gültig.
- **Generierung ohne extra Subprozess:** `pt seqkd generate` ist selbst ein eigener Prozess, der
  danach endet.
- **vLLM-Einstellungen für die A6000:**
  - `language_model_only=True` (kein Vision-Tower)
  - `max_num_seqs=128` (Mamba-Cache: ca. 221 Blöcke)
  - `VLLM_USE_FLASHINFER_SAMPLER=0` (kein nvcc auf dem Host)
- **Zusatz:** Der Dry-Run von `opd`/`distill` lehnt den Seed als Teacher jetzt ab; vorher fiel
  das erst im echten Lauf auf.
- **Migration transformers 5 / TRL 1.x:**
  - `warmup_ratio` → `warmup_steps` (mit [0,1]-Prüfung)
  - `max_prompt_length` aus den Preference/GRPO-Configs entfernt; die repo-eigenen
    Längen-Gates bleiben
  - GKD/ORPO nach `trl.experimental`
  - OPD-Lossnormierung (`num_items_in_batch=None`)
  - explizite `dtype=float32`-Loads
  - `return_dict=False` in der Eval-Längenprüfung

### Verifikation
- **Tests (neuer Stack):** 303 passed, 7 failed. Die 7 scheitern nur, weil `.claude` nach
  `.claude_tmp` umbenannt ist. In einer Kopie mit `.claude` laufen `test_commands`,
  `test_environment` und `test_docs_commands` grün (11 passed).
- **Exit 0:** `pt policy validate`, `pt doctor --mode all`.
- **Smoke-Test (20 Prompts):**
  - Qwen3.8-27B-FP8 lädt mit 27,92 GiB (Marlin W8A16); KV-Cache 156.330 Tokens.
  - 20/20 `stop`, kein `<think>`, sauberes Deutsch.
- **Echter E2E-Lauf** in einem isolierten Outputs-Root `outputs/e2e-seqkd/posttrain`
  (2.000 Datenzeilen, 200 Prompts, 10 Trainingsschritte):
  - `data-prepare-20261003T160740.209400Z-edd91e9f533d2557`
  - `seqkd-generate-20261003T160808.754968Z-8847e8ac00f71ef4`, Manifest-SHA
    `31ee89ae7eae406bb98c6ed5fe08b236f164ee227abea02223fa9a5510811b24`:
    - 197/200 trainierbar (2 trunkiert, 1 nicht Deutsch), Leakage clean
    - 77.546 Tokens in 225 s inklusive Laden, also ≥ 345 Tokens/s
  - `train-seqkd-20261003T161216.374667Z-e0158176d832e31b`: `succeeded`, event-verankert,
    Checkpoint-SHA `9ae4c8897c03e4a0…`, Loss 2,109, Peak-VRAM 8,8 GB.
  - Die Kandidaten-Auflösung über `resolve_candidate` setzt das Layout `outputs/posttrain`
    voraus und lief deshalb im isolierten Root nicht. Run-Card und Checkpoint wurden direkt
    verifiziert.

### Offen (Owner)
1. Diff von `policy.json` (`teachers`, `seqkd`) reviewen. Danach sind Daten-Manifest und Baseline
   unter `outputs/posttrain` stale (Policy-Hash):
   - `data prepare --real` neu laufen lassen
   - `baseline run --real --allow-gpu` neu laufen lassen (neuer Stack, neue Policy)
2. Den IPv6-Pfad des Hosts zum Hub reparieren. IPv6 läuft in den Timeout, und Python
   (httpx/socket) bleibt dann hängen, während curl auf IPv4 ausweicht.
3. `seqkd generate` über die echte Config (10.000 Prompts, voraussichtlich 2–3 h) und die
   Run-ID als `seqkd.generation_run` pinnen.
4. `AUTORESEARCH_POSTTRAIN.md`/`CLAUDE.md` (protected) um den Lever `seqkd` ergänzen.
5. `.claude_tmp` nach `.claude` zurückbenennen.

