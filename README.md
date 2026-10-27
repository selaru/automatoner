# Automatoner

Train tiny, single-purpose models from a browser. Pick a base model, bring or generate a dataset, train, and export for the place it will run (AWS Lambda, Docker, a phone, a browser, a Cactus device).

Covers classic machine learning, text classifiers, speech, tool calling and small language models. Runs on Windows, Linux and macOS.

## What tiny models are good for

A tiny model does one narrow job well, runs in milliseconds on a CPU, and costs almost nothing to run. Ideas, grouped by where they run. The category in brackets is what you pick in Automatoner.

**Infrastructure and operations**
- Score how unusual this minute is for a service, given its daily rhythm [tabular anomaly].
- Judge whether a log file's error mix is normal for that service [tabular anomaly].
- Flag a cloud audit event as normal or strange [tabular classification].
- Decide whether a detached disk, unused IP or old snapshot is really idle [tabular classification].
- Tell bots from busy shoppers per IP, to set rate limits from real traffic [tabular classification].
- Read a one-line config or policy change and say whether it widened or narrowed access [text classification].
- Route alerts or support tickets to the right team [text classification].
- Group similar log lines or incidents [embeddings, planned].

**Apps**
- Turn a spoken or typed note into a structured card: title, summary, steps, in the user's language [text to JSON].
- Pull fields out of messy text such as invoices, bookings or emails [extraction].
- Classify sentiment, intent or topic of short messages on the device, so the text never leaves it [text classification].
- Offline voice commands that become app actions [speech plus tool calling].
- Add a language or your own product names to speech recognition [speech].

**Smart services and devices**
- Map a request to a function call: "dim the living room to 30" [tool calling].
- Wake-word-free command handling on a phone, watch, robot or Raspberry Pi [speech plus tool calling].
- Predictive checks on sensor readings from machines or buildings: vibration, temperature, power draw [tabular anomaly].
- Fraud or abuse flags on sign-ups and payments, scored in a few milliseconds [tabular classification].

**Privacy and cost**
- Replace a call to a large cloud model for a task that never needed one, such as tagging or moderation [any category].
- Keep sensitive text, audio or records on the user's machine or inside your own account [any category].
- Run thousands of small checks per second without a GPU or a per-token bill.

Tiny models are a poor fit for open-ended chat, long reasoning, or tasks with no clear right answer. They also need examples of the job you want, which is why Automatoner can generate a starting dataset for you.

## Install and run

Install [uv](https://docs.astral.sh/uv/) once (`curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS and Linux, `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows). uv fetches the right Python itself.

```
git clone <this repo> automatoner && cd automatoner
uv sync --extra tabular        # base install plus classic ML
uv run automatoner doctor      # shows what is installed and what is missing
uv run automatoner serve       # opens http://127.0.0.1:8765
```

### Installing extras: read this first

`uv sync` is exact. It makes the environment match the extras you list and **removes every other extra**. Running `uv sync --extra tabular` after you installed PyTorch uninstalls PyTorch, and can leave a package half-installed (for example the `needle` command missing).

Pick one way and stay with it:

- **Recommended:** install extras from the Doctor page, or with `uv run automatoner doctor --install text`. This only adds packages and keeps what you have.
- **Or** list everything you want every time you sync:

```
uv sync --extra tabular --extra text --extra mlx --extra speech --extra coreml
```

Add `--extra dev` if you want to run the tests. A plain `uv run <command>` also re-syncs to the default extras, so if packages disappear after one, this is why. After installing an extra, restart `automatoner serve`.

| Extra | Adds | Needed for |
|---|---|---|
| `tabular` | scikit-learn, LightGBM, ONNX | anomaly detection, tabular classification, Lambda exports |
| `text` | PyTorch, Transformers, PEFT | text classifiers, small-LLM LoRA, the autoencoder |
| `mlx` | mlx-lm (Apple Silicon only) | fastest LoRA on a Mac |
| `speech` | cactus-needle | Needle 3 and Whistle |
| `coreml` | coremltools (macOS) | Core ML export |
| `mcp` | MCP SDK | letting AI agents use Automatoner (see below) |

On macOS, LightGBM also needs OpenMP: `brew install libomp`. The Doctor checks for it.

### Reading the Doctor

`ok` means ready. `!` is a warning and is usually optional (Ollama not running, Docker daemon off, no token). `+` is an optional part that is not installed, with a fix. `-` is something you have not configured and may not need. Only a red problem blocks you.

### Keys and tokens

Open **Settings** in the web UI (after `automatoner serve`) and paste the keys there. Press Save, then Check (Hugging Face) or Test (LLM providers).

- **Hugging Face token:** only needed for gated models such as Gemma. Create a read token at huggingface.co/settings/tokens, and accept the model's licence on its page. A token from `hf auth login` or the `HF_TOKEN` variable is also picked up.
- **Anthropic, Gemini, OpenAI, OpenRouter:** only needed to generate synthetic data with that provider. Ollama needs no key.
- If a test says "model not found", type a current model name from that provider into the card. Model names change often.

Keys are stored on this computer only, in the data folder, readable by your user alone, and are never sent back to the browser.

## How it works

1. **Models.** Nine categories (tabular anomaly, tabular classification, text classification, extraction, tool calling, text to JSON, speech, embeddings, vision). Nothing downloads until you press Download. Models go to the normal Hugging Face cache, so other tools reuse them. Add a Hugging Face token in Settings for gated models.
2. **Datasets.** Upload CSV, Parquet, JSON or JSONL (or a zip with audio), pull a Hugging Face dataset, or generate one. Generators: a tabular simulator (no LLM needed), and LLM writers for text classification, tool calls and text-to-JSON using Ollama, Claude, Gemini, OpenAI or OpenRouter. Every dataset is validated, de-duplicated and split.
3. **Train.** One subprocess per job, live log and loss curve, cancel kills the whole process tree.
4. **Export.** Only targets that fit the model and this computer are offered, with the reason when one is not.
5. **Recipes.** One-click pipelines for common small jobs: rhythm watcher, ALB error mix, IAM whisperer, drift sniffer, cost hunter, WAF classifier, and voice command to tool call.

## Command line

Everything the web interface does is also a command, for scripts, CI and remote machines. Add `--json` to any list or show command for machine-readable output. Exit code 0 is success, 1 a failure, 2 a usage mistake.

```
automatoner doctor                         # what is installed, what is missing
automatoner categories                     # what you can train for
automatoner models --category tabular_anomaly
automatoner download minilm-l6             # models download only when you ask

automatoner datasets generate --category tabular_anomaly --name "checkout" --spec spec.json
automatoner datasets add data.csv --category tabular_classification --name "events" --label-column label
automatoner datasets list
automatoner datasets show ds-20261027-101500-ab12cd

automatoner train --model isolation_forest --dataset ds-... --config '{"n_estimators": 200}'
automatoner train --model isolation_forest --dataset ds-... --no-wait   # prints the job id and returns
automatoner jobs list
automatoner jobs show job-... --log-lines 20
automatoner jobs log job-... --follow
automatoner jobs cancel job-...

automatoner export --job job-... --list                      # exports that fit this job, and why others do not
automatoner export --job job-... --exporter lambda_zip --options '{"function_name": "scorer"}'

automatoner recipes list
automatoner recipes run rhythm-watcher     # data, model, training and export in one go
```

`--spec` takes the same JSON the Datasets page uses (a simulator spec for tabular data, `{"labels": [...], "per_label": 100}` for text classification, `{"tools": [...], "total": 200}` for tool calling). For tabular data you can instead write `--describe "Lambda behind an ALB for a web shop ..."` and let an LLM design the simulator (needs a provider key in Settings, or Ollama). Training and generation progress prints to the terminal's error stream, so `--json` output stays clean on stdout. A job keeps running if you close the terminal; check it later with `jobs show`.

## Use it from AI agents (MCP)

Automatoner can act as an [MCP](https://modelcontextprotocol.io) server, so an agent in Claude Code, Claude Desktop, Cursor or another MCP host can run the whole loop: check the setup, pick a model, make a dataset, train, read the scores and export.

Install the extra, then register the server. Point the host at the `automatoner` program inside this project's virtual environment. Do not use `uv run` here: it re-syncs the environment and can remove extras (see "Installing extras").

```
automatoner doctor --install mcp

# Claude Code (macOS and Linux)
claude mcp add automatoner -- /path/to/automatoner/.venv/bin/automatoner mcp
# Windows: C:\path\to\automatoner\.venv\Scripts\automatoner.exe
```

For hosts configured with a JSON file (Claude Desktop, Cursor and similar):

```json
{
  "mcpServers": {
    "automatoner": {
      "command": "/path/to/automatoner/.venv/bin/automatoner",
      "args": ["mcp", "--allow-dir", "/path/to/your/data"]
    }
  }
}
```

**Tools.** Read: `get_limits`, `doctor`, `list_categories`, `list_models`, `list_datasets`, `get_dataset`, `list_recipes`, `list_jobs`, `get_job`, `list_exports`, `get_task`, `get_export`. Act: `download_model`, `add_dataset_from_file`, `generate_dataset`, `start_training`, `cancel_job`, `start_export`, `run_recipe`. Downloads, data generation and exports return a task id to poll with `get_task`; training returns a job id to poll with `get_job`. Errors come back as plain messages the agent can act on.

**Safe by default.** An agent can spend your time, disk and money, so the server restricts it:

| Rule | Default | Change with |
|---|---|---|
| Write tools | on | `--read-only` removes them entirely |
| Importing files | only from `~/.automatoner/inbox` (symlinks and `..` are resolved and checked) | `--allow-dir FOLDER`, repeatable |
| Data written by paid LLM providers | off (the tabular simulator needs no LLM and is always allowed) | `--allow-llm` |
| Training jobs per session | 5 | `--max-jobs N` |
| Jobs running at once | 2 | `--max-running N` |
| Largest synthetic dataset | 5000 examples | `--max-examples N` |
| Largest model download | 5000 MB | `--max-download-mb N` |

Stored API keys and tokens are never returned by any tool, and settings cannot be read or changed through MCP. Anything the agent runs still uses your computer and any provider key you have saved, so review the limits before enabling `--allow-llm`. A good first prompt: "Use automatoner to train an anomaly detector on the CSV in my inbox and export it for AWS Lambda."

## What works with what

| Trainer | Models | Exports |
|---|---|---|
| scikit-learn, LightGBM | Isolation Forest, One-Class SVM, Random Forest, Logistic Regression, LightGBM | Lambda zip and container, Docker, browser, joblib, ONNX |
| PyTorch | tiny autoencoder | the above plus Core ML |
| Transformers + PEFT | MiniLM, bge-small, ModernBERT (classification); SmolLM2, Qwen3, LFM2, Gemma (LoRA to JSON or tool calls) | text Lambda, ONNX, Hugging Face folder, GGUF |
| MLX LM (Apple Silicon) | same small LLMs | Hugging Face folder, GGUF, Apple Core AI (Qwen3 0.6B) |
| Cactus Needle / Whistle | Needle 3, Whistle | `.cact` plus the engine for 13 platforms |

Listed in the catalogue but not trainable yet: GLiNER, Moonshine, Whisper, Parakeet (download only), SmolVLM, embeddings.

## Good to know

- Needle fine-tuning runs on CPU unless you install Cactus's `metal` or `gpu` extra. On native Windows JAX is CPU only; use WSL2 for an NVIDIA GPU.
- Local Needle training tunes the full 20 layers, so exporting fewer layers loses accuracy. Check the test score the job reports.
- Core ML covers the autoencoder only: coremltools lags current scikit-learn and PyTorch.
- The Apple Core AI export is a port of `taskllm/scripts/export_coreai.sh` and has not been run end to end.
- The server binds to 127.0.0.1 and rejects requests with a foreign Host or Origin. Only use `--host 0.0.0.0` on a network you trust.
- Data lives in `~/.automatoner` (`%LOCALAPPDATA%\automatoner` on Windows). Override with `AUTOMATONER_HOME`.

## Development

```
cd web && npm install && npm run dev      # UI with hot reload, proxies to the server on 8765
cd web && npm run build                   # writes automatoner/web/dist, which is committed
uv run --extra mcp pytest                 # ~45 s: tabular, CLI and MCP
AUTOMATONER_SLOW=1 uv run pytest tests/test_slow_real_models.py -s   # real downloads and training, ~3 min
```

Adding a model is an entry in `automatoner/catalogue/catalogue.yaml`. Adding a trainer or exporter is a module in `training/plugins` or `export/plugins` plus a line in the registry.

## License

MIT, see `LICENSE`. Copyright (c) 2026 Cristian Selaru. You may use and change it freely as long as the copyright notice stays with the code.

This covers Automatoner's code only. Models you download have their own licenses, shown on each model card in the catalogue (for example Gemma, LFM and CC-BY-4.0 terms).

