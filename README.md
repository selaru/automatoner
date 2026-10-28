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

### Installing the command

`automatoner` is a normal command line program. After the install above you can run it three ways. Pick one.

**1. From the project folder, with uv (nothing extra to install).** Put `uv run` in front of it. Remember that `uv run` re-syncs the environment and can remove extras (see "Installing extras"), so ways 2 and 3 are safer once you have installed PyTorch or other extras:

```
cd automatoner
uv run automatoner doctor
```

**2. Install it once, then run it from any folder.** This puts `automatoner` on your PATH:

```
uv tool install --editable ".[tabular,mcp]"     # run inside the cloned folder; list the extras you want
uv tool update-shell                            # first time only, then open a new terminal
automatoner doctor
```

To add an extra later, run the install again with the longer list and `--force`, for example `uv tool install --force --editable ".[tabular,text,mcp]"`. Remove it with `uv tool uninstall automatoner`. Your datasets, models and settings are stored in `~/.automatoner` (`%LOCALAPPDATA%\\automatoner` on Windows), so they are the same whichever way you run it.

**3. Activate the virtual environment.** After `uv sync`, uv creates `.venv` in the project:

```
source .venv/bin/activate          # macOS and Linux
.venv\Scripts\activate             # Windows
automatoner doctor
```

Check that it works with `automatoner --help`. `atn` is a shorter name for the same command, so `atn doctor` and `atn train ...` do the same as the long form. The examples below leave out `uv run`; add it if you use way 1.

### Using it

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

automatoner token create                   # access token for sharing the server (see "Deploying in a company")
automatoner audit --limit 20               # who did what
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
| Data written by paid LLM providers | off (the tabular simulator needs no LLM and is always allowed). Local-only mode blocks it even with the flag | `--allow-llm` |
| Training jobs per session | 5 | `--max-jobs N` |
| Jobs running at once | 2 | `--max-running N` |
| Largest synthetic dataset | 5000 examples | `--max-examples N` |
| Largest model download | 5000 MB | `--max-download-mb N` |

Stored API keys and tokens are never returned by any tool, and settings cannot be read or changed through MCP. Anything the agent runs still uses your computer and any provider key you have saved, so review the limits before enabling `--allow-llm`. A good first prompt: "Use automatoner to train an anomaly detector on the CSV in my inbox and export it for AWS Lambda."

## Deploying in a company

Automatoner is built for one person on one computer. These notes cover running it for a team, and what protects you in each setup.

| Setup | What to do |
|---|---|
| **One person, one computer** | Nothing. The server listens on `127.0.0.1` only and rejects requests with a foreign Host or Origin, so a web page you visit cannot talk to it. |
| **A shared server** (GPU box, cloud VM) | Create an access token, put the server behind TLS (a VPN, or a reverse proxy such as nginx, Caddy or oauth2-proxy), and start it with `--host`. |
| **AI agents** | Use the MCP server over stdio. It runs as the user who starts it, with the limits listed above. |
| **Scripts and CI** | Use the command line. It runs as the calling user and needs no token. |

### Access token

On a non-loopback address (`--host 0.0.0.0` or any address other than `127.0.0.1`), Automatoner **refuses to start without an access token**.

```
automatoner token create          # prints the token once and saves it to ~/.automatoner/token
automatoner serve --host 0.0.0.0  # the web interface now asks for the token
```

Or set `AUTOMATONER_TOKEN` to a random value of at least 20 characters, which suits containers and service managers. The environment variable wins over the file.

- **Browser:** sign in once with the token. The browser keeps an HttpOnly, SameSite=Strict cookie derived from the token, never the token itself. Failed sign-ins are slowed down, limited to 10 a minute per address, and logged.
- **Scripts calling the API:** send `Authorization: Bearer <token>`.
- **Rotate or remove:** `automatoner token rotate` and `automatoner token remove`, then restart the server. `automatoner token status` shows where the token comes from without printing it.
- **If a token exists, it is enforced everywhere**, including on `127.0.0.1`.

The token travels in clear text over plain HTTP, so use TLS in front of the server. There are no user accounts or roles: everyone with the token can do everything, including using the API keys saved on the server. If you need per-person sign-in, put the server behind your company's single sign-on proxy and keep the token as a second layer. If you run it behind a proxy that terminates TLS, forward `X-Forwarded-Proto: https` so the cookie is marked Secure.

### Local-only mode

Turn on **Settings, Privacy, Local-only mode** to guarantee that company data is never sent to a cloud LLM. Only Ollama may write synthetic data, and only when its URL is on this computer or a private network (loopback, private address, or a name like `gpu-box` or `ollama.corp.internal`). Claude, Gemini, OpenAI and OpenRouter are refused everywhere: the web interface, the command line, recipes and MCP agents. Blocked attempts are written to the audit log.

An administrator can pin the policy so nobody can switch it off from the web interface:

```
AUTOMATONER_LOCAL_ONLY=1 automatoner serve --host 0.0.0.0
```

Model and dataset downloads from Hugging Face still work, because they fetch files and send no data. Block that at your firewall if you need to.

### Audit log

Every start of a training job, dataset import or generation, model download, export (and its download), install, settings change, token change, sign-in (successful, failed or denied), recipe, MCP tool call and LLM request is recorded with the time and who did it (`web:<address>`, `cli:<user>` or `mcp:<user>`).

```
automatoner audit                  # the last 50 entries
automatoner audit --action llm     # only LLM calls and blocked attempts
automatoner audit --json
```

The same list is on the Settings page. The log never contains API keys, tokens, prompts, dataset rows or model output. LLM requests are recorded by provider, model and size only. It is a plain JSON-lines file, `~/.automatoner/audit.log`, rotated at 10 MB (five files kept). Set `AUTOMATONER_AUDIT_FILE` to write it elsewhere. It is a record for review, not tamper-proof: forward it to your log system if you need that.

### Other things to know

- On macOS and Linux the token, settings and audit files are readable by their owner only. Windows has no such file modes, so those files are protected by the folder's permissions in your user profile (`%LOCALAPPDATA%\automatoner`). On a shared Windows server, run Automatoner under its own account.
- Saved provider keys stay in `~/.automatoner/settings.json` as plain text, readable only by the user who runs the server. On a shared server, run it as a dedicated user, or skip cloud keys altogether and turn on local-only mode.
- Downloaded models and datasets come from Hugging Face. Use models you trust, and review dataset sources before importing them.
- Training and exports run as the server's user, with no sandbox. Use a container or a dedicated account if the data or the model files are not yours to trust.

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
- The server binds to 127.0.0.1 and rejects requests with a foreign Host or Origin. To share it, read "Deploying in a company": it needs an access token and TLS.
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

