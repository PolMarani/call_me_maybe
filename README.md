*This project has been created as part of the 42 curriculum by pmarani.*

# 📞 call me maybe

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-de4c36?logo=uv&logoColor=white)
![Qwen3](https://img.shields.io/badge/model-Qwen3--0.6B-6a5acd)
![JSON](https://img.shields.io/badge/output-100%25%20valid%20JSON-2ea44f)
![42](https://img.shields.io/badge/42-project-000000)

## 📑 Table of Contents

- [📖 Description](#-description)
- [🚀 Instructions](#-instructions)
  - [📋 Requirements](#-requirements)
  - [⚙️ Installation](#-installation)
  - [▶️ Running](#-running)
  - [🛠️ Other Makefile Targets](#-other-makefile-targets)
- [📚 Resources](#-resources)
  - [🤖 How AI Was Used](#-how-ai-was-used)
- [🧠 Algorithm Explanation](#-algorithm-explanation)
- [🏗️ Design Decisions](#-design-decisions)
- [📊 Performance Analysis](#-performance-analysis)
- [🐛 Challenges Faced](#-challenges-faced)
- [✅ Testing Strategy](#-testing-strategy)
- [🎁 Bonus Features](#-bonus-features)
- [💡 Example Usage](#-example-usage)

## 📖 Description

`call me maybe` is a function-calling tool that translates natural language prompts into structured function calls. Given a prompt like *"What is the sum of 2 and 3?"*, the program does not answer the question directly — instead, it outputs a JSON object identifying which function should be called and with which arguments:

```json
{
  "name": "fn_add_numbers",
  "parameters": { "a": 2, "b": 3 }
}
```

The core challenge is that the underlying model, **Qwen/Qwen3-0.6B**, is a small (0.6B parameter) language model that is not reliable at producing structured output on its own. This project solves that problem with **constrained decoding**: a finite state machine (FSM) that filters the model's output logits at every generation step, guaranteeing that only tokens compatible with valid JSON — and with the expected function-calling schema — can ever be selected. As a result, the program produces 100% syntactically valid, schema-compliant JSON, regardless of how well the model "understands" the task.

## 🚀 Instructions

### 📋 Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### ⚙️ Installation

```bash
make install
```

This runs `uv sync`, which creates a virtual environment and installs all dependencies (`numpy`, `pydantic`, and the local `llm_sdk` package).

> **Disk space note (42 environment):** the virtual environment, `uv` cache, and downloaded model weights combined are several GB — more than a standard 42 home directory quota allows. Before running `make install` or `make run`, point `UV_CACHE_DIR` and `HF_HOME` at `/sgoinfre/<login>` (or any location with sufficient space):
>
> ```bash
> export UV_CACHE_DIR=/sgoinfre/<login>/.cache/uv
> export HF_HOME=/sgoinfre/<login>/.cache/huggingface
> ```
>
> Add these two lines to `~/.bashrc` (replacing `<login>` with your own) so they persist across sessions and apply automatically for the reviewer as well.
>
> If you hit HuggingFace Hub rate limits or slow downloads while model weights are being fetched, you can additionally set `HF_TOKEN` in your environment.

### ▶️ Running

```bash
make run
```

This is equivalent to:

```bash
uv run python -m src
```

By default, the program reads `data/input/functions_definition.json` and `data/input/function_calling_tests.json`, and writes the result to `data/output/function_calling_results.json`.

You can override the model, input, and output paths:

```bash
uv run python -m src --model Qwen/Qwen2.5-0.5B --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

Or, using the Makefile's `MODEL` variable:

```bash
make run MODEL=Qwen/Qwen2.5-0.5B
```

### 🛠️ Other Makefile Targets

- 🐞 `make debug` — runs the program under Python's built-in debugger (`pdb`)
- 🧹 `make clean` — removes `__pycache__` and `.mypy_cache` directories
- 🔍 `make lint` — runs `flake8` and `mypy` with the required flags

## 📚 Resources

- [Qwen3 technical documentation](https://huggingface.co/Qwen/Qwen3-0.6B) — model card and usage for the default model
- [Qwen2.5 technical report](https://arxiv.org/pdf/2412.15115) — used when testing the multi-model bonus
- [Hugging Face — Causal Language Modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) — background on how `AutoModelForCausalLM` generation works
- [Pydantic v2 documentation](https://docs.pydantic.dev/) — used for `model_post_init` and validated data classes
- General references on constrained decoding / grammar-constrained generation as a technique for guaranteeing structured LLM output (the specific technique implemented here — logit masking based on a finite state machine over token IDs — was designed from scratch for this project, following the explanation and constraints given in the subject)

### 🤖 How AI Was Used

Claude (Anthropic) was used throughout this project as a Socratic tutoring "navigator": for the core implementation it asked guiding questions and gave minimal hints when stuck, but wrote no code — every line of the finite state machine, the token pre-computation, the error handling, and the `Makefile`/lint configuration in `src/constrained_decoder.py`, `src/file_handler.py`, and `src/__main__.py` was written by me. Claude's role there was to help me:

- Understand *why* constrained decoding is needed before writing *how* to implement it
- Design the finite state machine (states, transitions, which parts of the JSON are fixed vs. model-generated)
- Debug logic errors (e.g. tensor shapes from `encode()`, Pydantic attribute declarations, off-by-one errors in prefix matching)
- Diagnose and fix a performance bug (the model was being queried for logits even in fixed states that never used them)
- Diagnose and fix a correctness bug (greedy decoding causing the model to loop on repeated regex-like patterns)
- Design (through testing against the actual failing outputs) the two targeted `fix_regex_pattern` and `fix_repeated_replacement` corrections, which I then implemented myself
- Configure `flake8`/`mypy` to exclude the third-party `llm_sdk` package without hiding it from type resolution

By explicit prior agreement, three parts of the project were excluded from that "navigator only" rule and were written directly by Claude: this **README**, the **NumPy-style docstrings** on the classes and functions, and the **colored terminal visualization** of the generation process (the ANSI color scheme and print formatting in `generate_function_call`).

## 🧠 Algorithm Explanation

The generation loop in `ConstrainedDecoder.generate_function_call` is driven by a finite state machine with the following states:

| State | Type | Description |
|---|---|---|
| `START` | fixed | Emits `{` |
| `NAME_KEY` | fixed | Emits `"name": "` |
| `FUNCTION_NAME` | generated | Model picks tokens; only tokens matching a valid, still-compatible function name are allowed (prefix matching against pre-tokenized function names) |
| `PARAMS_OPEN` | fixed | Emits `, "parameters": {"` |
| `PARAM_NAME` | fixed | Emits the pre-tokenized name of the current parameter (looked up from the chosen function's schema) |
| `VALUE_DEFINITION_STRING` / `VALUE_DEFINITION_NUMBER` | fixed | Emits `": "` or `": ` depending on the parameter's declared type |
| `PARAM_VALUE_STRING` | generated | Model generates the string content, token by token, until it emits a closing `"` |
| `PARAM_VALUE_NUMBER` | generated | Model generates digits/`.`/`-` until it emits `,` or `}` |
| `MULTIPLE_PARAM` | fixed | Emits the separator before the next parameter (`, "` for values coming after a string, `"` for values coming after a number, since the model already emitted the comma itself) |
| `END_STRING` / `END_NUMBER` | fixed | Emits the closing `}}` or `}`, depending on whether the last value was a string (already closed by its own `"`) or a number (which still needs the final `}` after the parameters object) |

At every generation step:

1. The model's raw logits (one score per vocabulary token, ~150k entries) are requested **only** in states where the model actually needs to choose a token — fixed states skip this call entirely, since they already know their next tokens.
2. A `valid_tokens` list is built for the current state (e.g. the set of token IDs that are digits/`.`/`-`/`,`/`}` for `PARAM_VALUE_NUMBER`, or every token that does not contain a `"` character for `PARAM_VALUE_STRING`).
3. All logits outside `valid_tokens` are set to `-inf` using NumPy array masking; `argmax` is then guaranteed to pick a token from the valid set.
4. The chosen token is appended to the running `input_ids`, and the state machine transitions accordingly.

Which tokens count as "valid" for numbers and strings is computed once, in `model_post_init`, by reading the model's own vocabulary file (`get_path_to_vocab_file()`) and filtering it by character content — this makes the whole approach tokenizer-agnostic and is what allows the same code to work unmodified across different models (see Bonus section below).

Function selection works through **prefix matching**: at `model_post_init` time, every function name in `functions_definition.json` is pre-tokenized. During `FUNCTION_NAME` generation, the set of "still-compatible" functions shrinks token by token as the model's choices narrow down which function it is spelling out, until only one remains and the model is forced to close the name with `"`.

### Post-generation correction

After the JSON is fully generated and parsed into a Python `dict`, if the result contains a `regex` parameter, `ConstrainedDecoder.fix_regex_pattern` applies two narrow, evidence-tested corrections directly on the Python string (no re-tokenization, no effect on generation speed):

1. A dangling trailing `|` (empty alternation) is stripped — in Python's `re` module this matches at every position and silently breaks any substitution built on it.
2. A trailing `.*` appended directly to a single plain word (e.g. `cat.*`) is stripped down to the word alone — the unclipped pattern greedily matches everything after the word instead of just the word itself.

Both corrections were verified against the actual failure cases produced by the model before being adopted (see Challenges Faced).

## 🏗️ Design Decisions

- **Pydantic for all state**: `ConstrainedDecoder` is a Pydantic `BaseModel`; all token pre-computation (fixed JSON fragments, function name token sequences, per-function parameter name tokens, the number/string token sets) happens once in `model_post_init`, not on every call to `generate_function_call`. This keeps per-prompt generation fast.
- **Splitting fixed vs. generated states explicitly**: rather than always calling the model and then discarding unused logits, the loop only calls `get_logits_from_input_ids` inside the three states that actually need a model decision (`FUNCTION_NAME`, `PARAM_VALUE_STRING`, `PARAM_VALUE_NUMBER`). This was a deliberate late-stage optimization after profiling showed most iterations were fixed-structure states paying for an unused forward pass.
- **Greedy decoding (`argmax`)** was chosen over sampling, to keep generation deterministic and reproducible — important both for testing and for the constrained-decoding guarantee itself.
- **Repetition guard**: greedy decoding on a small model tends to get stuck repeating a just-generated block (e.g. a regex group) indefinitely. Rather than a single fixed-length repetition check, the code checks multiple window sizes (`K` in `[2..7]`) after each token in `PARAM_VALUE_STRING`, comparing the last `K` tokens against the `K` before them; on a match, the duplicated tokens are dropped and the string is force-closed. This was chosen over prompt-engineering fixes (which were tried first and made both output quality and speed worse — see Challenges below) because it acts directly on the failure mode without inflating the prompt.
- **Hard length caps** (`value_cont >= 30` for strings, `>= 10` for numbers) act as a safety net beneath the repetition guard, so that even a repetition the guard doesn't catch (mismatched period) cannot run away and blow the 5-minute time budget.
- **Post-generation correction over prompt engineering**: after prompt-engineering attempts to improve `regex` argument quality backfired (see Challenges Faced), the fix was moved to a much simpler, lower-risk layer — plain Python string corrections applied to the already-generated `regex` value in the final result `dict`, before it is returned. This was deliberately kept narrow (two specific, evidence-tested patterns) rather than a general "regex normalizer", since parsing regex with more regex is itself notoriously fragile; each correction was verified with `re.sub` against the actual failing test cases before being adopted, and neither rule fires on patterns that don't match its exact shape.

## 📊 Performance Analysis

- ⏱️ **Speed**: on the default model (Qwen3-0.6B), all 11 provided test prompts complete in well under 5 minutes (typically ~4 minutes 10 seconds on the reference hardware used during development), satisfying the subject's requirement.
- 🔒 **Reliability**: output JSON is 100% valid and schema-compliant across every run — this is a structural guarantee of the FSM, not a statistical outcome, since invalid tokens are mathematically excluded (`-inf` logits) rather than merely discouraged.
- 🎯 **Accuracy**: with the default model, **10 of 11 test prompts (90.9%)** are semantically correct end-to-end (correct function, correct argument values), meeting the subject's 90%+ requirement. This was achieved by combining the repetition guard (below) with a small, targeted post-generation correction step (`fix_regex_pattern`) that fixes two specific malformed-regex shapes the model tends to produce (see Algorithm Explanation and Challenges Faced). The one remaining imperfect case is a `regex` argument that is structurally valid JSON but functionally too literal (matches the specific input rather than the general pattern) — a genuine model-capability limitation rather than a decoding bug. Earlier development used a larger model (`Qwen2.5-1.5B`) to reach the same 10/11 accuracy, but that pushed runtime to roughly 8 minutes, over budget; the post-generation correction approach reaches the same accuracy on the *default* model at no extra time cost, since it operates on the already-generated Python dict rather than adding tokens to the generation loop.

## 🐛 Challenges Faced

- **Tensor shapes from `encode()`**: `Small_LLM_Model.encode()` returns a 2D tensor (`[[token_ids]]`); every pre-computed token sequence needed an extra `.tolist()[0]` unwrap that was easy to forget and caused silent structural bugs (e.g. `numpy` trying to index with a list containing a nested list).
- **A redundant forward pass per fixed state**: originally, `get_logits_from_input_ids` was called once per loop iteration unconditionally, even in states that immediately `continue` without using the logits at all. Since a fixed-structure JSON call touches roughly 8 fixed-state iterations before it ever needs the model, this wasted the single most expensive operation in the whole program repeatedly. Moving the call inside only the three model-driven states cut total runtime by over 25%.
- **Greedy decoding loops**: on `PARAM_VALUE_STRING`, the model would sometimes get stuck re-emitting the same short token sequence indefinitely (e.g. repeating a regex group `([aeiou])|` six times in a row), because greedy decoding always has the just-generated tokens fresh in context and highly probable to repeat. A single fixed-window repetition check (`K=3`) only caught periods that were multiples of 3; the fix was to check several window sizes per token instead of one.
- **Prompt engineering made things worse, not better**: an attempt to fix the regex-quality issue by adding a worked example to the system instructions backfired — the extra regex-like symbols in the example gave the model *more* material to imitate/repeat, and also slowed every single forward pass (the instructions are re-processed as part of the context on every generation step), pushing total runtime over the 5-minute limit. This was reverted; the regex-quality problem was ultimately solved at a different layer entirely — a small, targeted post-generation correction (`fix_regex_pattern` and `fix_repeated_replacement`) applied to the already-parsed result, which fixed 2 of the 3 observed failure cases with zero effect on generation speed and pushed overall accuracy from 8/11 to 10/11 (90.9%) on the default model.
- **`mypy` and a dict with mixed value types**: `result: dict = {}` (no type annotation) was inferred by `mypy` as `dict[str, str]` from its first assignment (`result["prompt"] = prompt`), causing false "invalid index" errors once `result["parameters"]` (itself a `dict`) was accessed. Explicitly annotating `result: dict[str, Any]` resolved it.
- **Crash-safety on missing/malformed inputs**: the moulinette evaluation checklist explicitly tests missing and malformed input files. An unguarded `compatible_function[0]` access would raise an `IndexError` if `functions_definition.json` was missing or empty (since `function_name_tokens` would then also be empty) — `load_json` already returned `[]` gracefully for a missing/malformed file, but nothing downstream checked for that empty case before proceeding. `main()` now checks both `function_definition` and `test_prompts` immediately after loading them and exits with a clear message (`sys.exit(1)`) before any model is loaded, rather than failing deep inside the generation loop after several minutes of setup.
- **Cross-model tokenizer differences**: the multi-model bonus initially failed on `TinyLlama-1.1B-Chat`, because its vocabulary file is not a simple UTF-8 JSON like Qwen's BPE vocabulary. Rather than special-casing tokenizer formats, the project scope was kept to models sharing Qwen's BPE-with-JSON-vocab tokenizer family, which is compatible with the code unmodified.
- **`flake8`/`mypy` on third-party code**: linting the repository as a whole initially failed inside the provided `llm_sdk` package (not code I own or should modify) and even crashed `pyflakes` outright. `llm_sdk` is excluded from `flake8` via a dedicated `.flake8` config, and from `mypy` error-reporting (while still being resolvable for type-checking imports elsewhere) via a `[[tool.mypy.overrides]]` block with `ignore_errors = true` in `pyproject.toml`.

## ✅ Testing Strategy

The project was validated iteratively against the 11 prompts in `data/input/function_calling_tests.json`, covering every function in `functions_definition.json` at least once, including a function with three string parameters (`fn_substitute_string_with_regex`) to stress-test the repetition guard and multi-parameter transitions.

For each run, the output JSON was checked at two levels:

1. **Structural validity** — every result parses as JSON, contains exactly `prompt`, `name`, and `parameters`, and every parameter type matches the schema (verified by `json.loads` succeeding and manual inspection of types).
2. **Semantic correctness** — for each prompt, whether the selected function and generated argument values actually correspond to what the prompt asked for (e.g. does `fn_add_numbers(a=2, b=3)` match "the sum of 2 and 3?").

Runtime was measured end-to-end via the terminal timing of `make run` to confirm compliance with the 5-minute requirement.

Robustness was verified against the specific failure modes the moulinette evaluation checklist calls out: a missing or empty `functions_definition.json`, a missing or empty `function_calling_tests.json`, and malformed JSON in either file (confirmed to be caught by `load_json`'s `except (FileNotFoundError, json.JSONDecodeError)` and then reported with a clear message and a clean exit, rather than an unhandled traceback). A prompt with no clear matching function was also tested; since the FSM's `valid_tokens` in `FUNCTION_NAME` never include an "abstain" option, the model is structurally forced to pick some function, so this case does not crash — it can produce a semantically wrong pick, which is a model-capability limit rather than a stability bug.

The multi-model bonus was validated by re-running the same 11 prompts against `Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-1.5B` without any code changes, confirming both that the FSM/tokenizer logic generalizes across models and that model size trades off accuracy against runtime.

## 🎁 Bonus Features

- **Support for multiple LLM models beyond Qwen/Qwen3-0.6B**: the model is selectable via `--model` (or `make run MODEL=...`), and every token the code needs — fixed JSON fragments, function/parameter names, the valid-number and valid-string token sets — is computed fresh from the chosen model's own tokenizer and vocabulary file in `model_post_init`, with no hardcoded token IDs anywhere.

  Models tried during development:

  | Model | Status | Notes |
  |---|---|---|
  | `Qwen/Qwen3-0.6B` (default) | ✅ Works | 10/11 accuracy (90.9%), ~4m10s total runtime — meets both the 90%+ accuracy and <5 minute requirements at once |
  | `Qwen/Qwen2.5-0.5B` | ✅ Works | Same BPE/JSON-vocab tokenizer family as the default; runs unmodified |
  | `Qwen/Qwen2.5-1.5B` | ✅ Works | Also reaches 10/11 accuracy, but total runtime rises to ~8 minutes — more capable on the harder regex cases, at a real time cost that exceeds the subject's budget |
  | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ❌ Not compatible | Uses a SentencePiece vocabulary file that isn't plain UTF-8 JSON, which breaks the number/string token filtering in `model_post_init` (see Challenges Faced) |

  Example:
  ```bash
  make run MODEL=Qwen/Qwen2.5-1.5B
  ```
- **Visualization of the generation process**: running the program prints a colored, live trace of the constrained-decoding FSM as it runs — a bordered header per prompt, the selected function name, each parameter being generated, every individual token the model picks with its state and decoded text, and the final result pretty-printed in green — making the effect of the logit masking visible step by step rather than only showing the final output.

## 💡 Example Usage

```bash
$ make install
$ make run
```

Example input (`data/input/function_calling_tests.json`):

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" }
]
```

Example output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2, "b": 3 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  }
]
```

Running with a different model:

```bash
$ make run MODEL=Qwen/Qwen2.5-0.5B
```