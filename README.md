# servecheck

[![CI](https://github.com/SiluPanda/servecheck/actions/workflows/ci.yml/badge.svg)](https://github.com/SiluPanda/servecheck/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/SiluPanda/servecheck)](https://github.com/SiluPanda/servecheck/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**15-second conformance probe & offline GGUF/Jinja2 doctor for local and self-hosted OpenAI-compatible LLM servers (`vLLM`, `llama-server`, `SGLang`, `Ollama`, `mlx-lm`).**

`servecheck` catches the silent server-side tool-calling, `<think>`-tag, streaming, and KV-cache bugs that break coding agents — and prints **exact copy-paste server CLI flag fixes**.

Single static Go binary (`CGO_ENABLED=0`). Zero Python. Read-only diagnostic CLI.

---

## Why `servecheck`?

Local LLM servers fail coding agents in subtle ways that return **HTTP 200 OK**:

- Nested JSON schemas silently drop required fields on ~8% of tool calls without `--tool-strict-level parameter`
- `<tool_call>` decoy text discussed inside `<think>` reasoning misfires as a live tool call
- Reasoning models swallow tool calls inside `<think>` / `reasoning_content` and return `finish_reason: "stop"` with empty `tool_calls`
- Streaming tool parsers swallow leading `<p>` / `<` tokens in HTML prose (`vllm#65934`) or emit negative `tool_index: -1` deltas under MTP/speculative decoding
- `.reasoning` vs `.reasoning_content` wire divergence breaks downstream agent SDKs
- Embedded GGUF Jinja2 chat templates retroactively strip Turn 1 `<think>` blocks or inject live `strftime_now()` timestamps — silently collapsing Turn 2 KV-cache hits from ~96% to ~4%

`servecheck` diagnoses all of these in ~15 seconds against a live server — or in **<40ms completely offline with zero GPU** against any `.gguf` binary header or `tokenizer_config.json`.

---

## Example Output

### Live Server Conformance Probe

```text
$ servecheck http://localhost:8000/v1 --model Qwen/Qwen3-32B
servecheck v0.1.0 http://localhost:8000/v1 (Qwen/Qwen3-32B)
Detected engine: vLLM 0.28.1 [header:x-vllm-version] | Elapsed: 9.82s

  [PASS] models_endpoint              model "Qwen/Qwen3-32B" advertised (1 model(s))
  [PASS] basic_chat                   OK (TTFT 42ms, 86.4 tok/s)
  [PASS] single_tool_call             valid JSON args, non-empty id, finish_reason="tool_calls"
  [FAIL] strict_nested_schema         [SCHEMA_BUG] 1/3 calls failed nested schema validation (call 2: dropped required field "target.path")
  [PASS] parallel_tool_calls          2 parallel tool calls with distinct IDs (Tokyo & Paris)
  [PASS] tool_role_roundtrip          multi-turn assistant(tool_calls) -> tool -> assistant roundtrip succeeded
  [FAIL] think_decoy_isolation        [SERVER_PARSER_BUG] decoy <tool_call> text in prose/reasoning misfired as live tool call (delete_all)
  [PASS] tool_swallowed_in_think      tool call add(a=2, b=3) extracted cleanly alongside reasoning
  [WARN] reasoning_field_compat       [SERVER_PARSER_BUG] emits .reasoning without .reasoning_content (SGLang/ADK/Cursor clients expect .reasoning_content)
  [PASS] stream_tool_delta            reassembled valid delta.tool_calls across 8 SSE chunks + [DONE]
  [FAIL] stream_angle_bracket         [SERVER_PARSER_BUG] streaming tool parser swallowed leading angle-bracket tag (got "ok</p> <div>hello</div>")
  [PASS] json_schema_response_format  strict json_schema response_format returned valid conforming JSON
  [FAIL] multi_tool_kv_prefix         [KV_CACHE_BREAK] Turn 2 KV prefix cache hit collapsed to 4% (38/960 tok via usage.prompt_tokens_details.cached_tokens)

Recommended server fixes (vLLM / qwen3):
  + --enable-auto-tool-choice --tool-strict-level parameter --tool-call-parser qwen3_coder
      Enables Qwen3-aware tool parsing, strict parameter grammar enforcement (fixes ~8% nested schema drops), and angle-bracket streaming isolation. (fixes: strict_nested_schema, think_decoy_isolation, stream_angle_bracket)
  + --reasoning-parser qwen3
      Splits <think> blocks cleanly into reasoning fields and prevents tool parser collisions inside reasoning. (fixes: think_decoy_isolation, reasoning_field_compat)
  + --enable-prefix-caching --chat-template /path/to/prefix_stable_template.jinja
      Enables Automatic Prefix Caching (APC); ensure your chat template does not strip prior-turn <think> blocks, inject live timestamps, or reorder tool-arg JSON keys. (fixes: multi_tool_kv_prefix)

Summary: 8 PASS, 1 WARN, 4 FAIL (exit 1)
```

### Offline GGUF & Jinja2 Template Doctor (`<40ms`, Zero GPU)

```text
$ servecheck gguf ./Qwen3-35B-A3B-Q4_K_M.gguf
servecheck gguf v0.1.0 ./Qwen3-35B-A3B-Q4_K_M.gguf (gguf-v3)
Metadata: model=Qwen3-35B-A3B-Q4_K_M | arch=qwen3 | tensors=256 | kv=32 | bos="<|im_start|>" | eos="<|im_end|>" | elapsed=14ms

  [PASS] template_present             extracted chat_template (2140 bytes)
  [PASS] template_syntax_parse        Jinja2 chat_template parsed cleanly
  [PASS] template_tool_render         rendered tool definitions + multi-tool calls + role="tool" cleanly (1482 bytes)
  [FAIL] template_prefix_stability    [KV_CACHE_BREAK] Turn 2 mutated Turn 1 prefix at byte 742 (retroactively stripped Turn 1 <think> block) — busts llama.cpp/vLLM KV cache!
  [PASS] template_dynamic_time        no dynamic date/time injection detected (prompt prefix is time-invariant)
  [WARN] template_think_tool_hygiene  [KV_CACHE_BREAK] template contains last_query_index + </think> logic (risk of retroactive thinking strip on multi-turn)

Recommended server fixes (llama-server / qwen3):
  + --jinja --cache-reuse 256 --chat-template-file ./fixed_prefix_stable.jinja
      Override the embedded GGUF template with a prefix-stable Jinja template (do not strip prior-turn <think> blocks or inject strftime_now) and enable KV cache slot reuse. (fixes: template_prefix_stability, template_think_tool_hygiene)

Summary: 4 PASS, 1 WARN, 1 FAIL (exit 1)
```

---

## Install

### Pre-built static binary (macOS & Linux, SHA-256 verified)

```bash
curl -fsSL https://raw.githubusercontent.com/SiluPanda/servecheck/main/install.sh | sh
```

Or download tarballs + `checksums.txt` directly from [GitHub Releases](https://github.com/SiluPanda/servecheck/releases).

### Homebrew formula

```bash
brew install --formula https://raw.githubusercontent.com/SiluPanda/servecheck/main/packaging/homebrew/servecheck.rb
```

### `go install` (Go 1.25+, zero CGo)

```bash
CGO_ENABLED=0 go install github.com/SiluPanda/servecheck/cmd/servecheck@latest
```

### Build from source

```bash
git clone https://github.com/SiluPanda/servecheck.git
cd servecheck
make build
./bin/servecheck version
```

---

## Usage

### 1. Probe a Live OpenAI-Compatible Server

```bash
servecheck <endpoint-url> --model <id> [flags]
```

Flags:
- `--model <id>`: Model ID to probe (auto-selected from `/v1/models` if omitted)
- `--api-key <key>`: Bearer token (falls back to `$OPENAI_API_KEY`)
- `--engine <engine>`: `auto` (default) | `vllm` | `llama-server` | `sglang` | `ollama` | `mlx`
- `--timeout <duration>`: Per-probe HTTP timeout (default `20s`)
- `--format <fmt>`: `table` (default) | `json` | `markdown`
- `--no-color`: Disable ANSI colors (also respects `NO_COLOR` and non-TTY stdout)

### 2. Inspect a Local `.gguf` File or `tokenizer_config.json` Offline

```bash
servecheck gguf ./Qwen3-35B-Q4_K_M.gguf
servecheck gguf ./tokenizer_config.json --engine vllm --format markdown
```

Reads only the GGUF v2/v3 header and metadata KV pairs (never touches multi-GB tensor weights) and executes the embedded Jinja2 chat template in pure Go via `gonja/v2` with HuggingFace helpers (`tojson`, `namespace`, `strftime_now`, `now`, `raise_exception`, string filters).

### 3. Browse Built-In Server Fix Recipes

```bash
servecheck recipes
servecheck recipes --engine vllm --model-family qwen3
servecheck recipes --engine llama-server --format markdown
```

### Exit Codes (CI-Friendly)

| Exit Code | Meaning |
|---:|---|
| `0` | All probes/checks passed (or `WARN` only, 0 `FAIL`) |
| `1` | One or more probes/checks returned `FAIL` |
| `2` | CLI usage error, file error, HTTP 401/403 auth error, or network connection error |

---

## The 13 Live Conformance Probes

All probes run deterministically at `temperature=0` with tight `max_tokens`:

| # | Probe ID | What It Catches | Primary Cause Tag |
|---:|---|---|---|
| 1 | `models_endpoint` | `/v1/models` availability, requested model presence, engine/version auto-detection | `UNSUPPORTED` |
| 2 | `basic_chat` | Completion sanity + streaming TTFT and generation tok/s | `SERVER_PARSER_BUG` |
| 3 | `single_tool_call` | `finish_reason=="tool_calls"`, valid JSON args, non-empty `tool_call.id`, leaked markup | `SERVER_PARSER_BUG` / `MODEL_FLOOR_FAIL` |
| 4 | `strict_nested_schema` | 3 deterministic calls against nested required-fields schema (`create_patch`); catches ~8% drops | `SCHEMA_BUG` |
| 5 | `parallel_tool_calls` | Two parallel calls (Tokyo & Paris) in one turn with distinct non-empty IDs | `SERVER_PARSER_BUG` / `MODEL_FLOOR_FAIL` |
| 6 | `tool_role_roundtrip` | Multi-turn `user -> assistant(tool_calls) -> tool(tool_call_id)` catching HTTP 400/500 role errors | `SERVER_PARSER_BUG` |
| 7 | `think_decoy_isolation` | Prose/reasoning discussing `<tool_call>{"name":"delete_all"}</tool_call>` misfiring live tools | `SERVER_PARSER_BUG` |
| 8 | `tool_swallowed_in_think` | HTTP 200 `finish_reason="stop"` with empty `tool_calls` while tool syntax leaked in `<think>`/`reasoning_content` | `SERVER_PARSER_BUG` |
| 9 | `reasoning_field_compat` | Wire audit of `.reasoning` vs `.reasoning_content` vs unstripped inline `<think>` in `.content` | `SERVER_PARSER_BUG` |
| 10 | `stream_tool_delta` | SSE `delta.tool_calls` JSON reassembly, negative `index: -1` (vLLM MTP bug), and `[DONE]` sentinel | `SERVER_PARSER_BUG` |
| 11 | `stream_angle_bracket` | Tool-enabled SSE echoing `<p>ok</p>` and `<div>hello</div>`; catches `<`/`<p>` swallowing (`vllm#65934`) | `SERVER_PARSER_BUG` |
| 12 | `json_schema_response_format` | `response_format: {type: "json_schema"}` conformance, markdown-fence leaks, and XGrammar errors | `SCHEMA_BUG` / `UNSUPPORTED` |
| 13 | `multi_tool_kv_prefix` | Turn 1 (large system prompt + 2 tool calls) -> Turn 2 cache hit via `cached_tokens`, `timings.cache_n`, or eval latency | `KV_CACHE_BREAK` |

## The 6 Offline GGUF / Jinja2 Doctor Checks

| # | Check ID | What It Verifies |
|---:|---|---|
| 1 | `template_present` | `tokenizer.chat_template` is embedded in GGUF / `tokenizer_config.json` |
| 2 | `template_syntax_parse` | Template compiles cleanly as valid Jinja2 |
| 3 | `template_tool_render` | Renders tool definitions, multi-tool calls, and `role="tool"` without `raise_exception` or silent drops |
| 4 | `template_prefix_stability` | Strict byte-for-byte Turn 1 -> Turn 2 prefix stability (catches retroactive `<think>` stripping & tool-arg mutation) |
| 5 | `template_dynamic_time` | Detects `strftime_now` / `now()` timestamps that invalidate KV caches per second/minute or across midnight |
| 6 | `template_think_tool_hygiene` | Checks balanced `<think>`/`<tool_call>` tags and detects `<think><think>` double-wrapping |

---

## Architecture

```text
cmd/servecheck/     CLI entrypoint, flag parsing, exit codes (0/1/2)
internal/client/    OpenAI-compatible HTTP + SSE client, engine/version detection, timings/cache telemetry
internal/probe/     13 deterministic live server probes and root-cause taxonomy
internal/gguf/      Pure-Go GGUF v2/v3 binary header reader, tokenizer_config.json loader, offline gonja/v2 Jinja2 doctor
internal/recipes/   Embedded recipes.yaml mapping (engine, model_family, failed_probe) -> exact CLI flags
internal/report/    ANSI terminal table (NO_COLOR-aware), GitHub Markdown, and JSON formatters
```

---

## License

MIT — see [LICENSE](LICENSE).
