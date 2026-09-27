*This activity has been created as part of the 42 curriculum by yafranco*

# Call Me Maybe - Function Calling via Constrained Decoding

## Description

**Call Me Maybe** is a lightweight Python tool designed to bridge natural language prompts with machine-executable code. Using a 0.6B parameter model (`Qwen/Qwen3-0.6B`), the project translates natural language requests (e.g., *"What is the sum of 2 and 3?"*) into structured, type-safe JSON function calls (e.g., `{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}`).

---

## The problem & The Solution

Small language models (SLMs) frequently hallucinate syntax errors, unclosed brackets, or malformed parameter keys when generating structured data. Rather than relying on fragile prompt engineering or slow retry heuristics, **Call Me Maybe** enforces **token-level constrained decoding**.

By combining a **Finite State Machine (FSM)** with **vectorized logit masking**, invalid token transitions are mathematically eliminated at every forward step ($P = 0$), guaranteeing deterministic adherence to JSON schemas.

### Example

**Natural Language Input (`function_calling_tests.json`)**:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  }
]
```

### Guaranteed Output (function_calling_results.json)

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2,
      "b": 3
    }
  }
]
```

## Architecture & Algorithm

```text
                    ┌─────────────────────────┐
                    │      Model Logits       │
                    └────────────┬────────────┘
                                 │
                                 ▼
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────────┐
│  FSM State &     ├───►│  Logit Masking  ├───►│ Validated Token Step │
│  Schema Rules    │    │ (Invalid = -inf)│    │      (Argmax)        │
└──────────────────┘    └─────────────────┘    └──────────┬───────────┘
                                                          │
                                 ◄────────────────────────┘
                              Update State
```

1. **Vocabulary Inversion & Token Cache**: Token IDs from the model vocabulary are pre-decoded and mapped to exact string representations via llm.decode().
2. **Deterministic State Tracking**: A typed GenerationState model monitors structural JSON depth (EXPECT_START, EXPECT_PROMPT, EXPECT_NAME_KEY, EXPECT_NAME_VAL, EXPECT_PARAMS_KEY, EXPECT_PARAM_OBJECT, EXPECT_END).
3. **Vectorized Logit Masking**: At each generation step, get_allowed_tokens() evaluates allowable prefixes:
   - Structural Literals: Forces exact delimiters ("prompt":, "name":, ",").
   - Schema Matching: Constrains function names and parameter keys to valid prefixes defined in functions_definition.json.
   - Primitive Types: Enforces strict numeric patterns (integers/floats via NUMERIC_PREFIX_PATTERN), booleans, and string termination.
   - Logit Penalty: Disallowed tokens are set to -\infty, reducing their probability to 0 during selection.
4. **State Transition**: update_state() advances the FSM step and synchronizes parameter tracking buffers.


## Engineering Challenges

- **BPE Token Overlaps Across JSON Boundaries**: Byte-Pair Encoding (BPE) tokenizers often emit single tokens spanning multiple structural roles (e.g., ,"key" merges a comma, a space, a quote, and part of an identifier). Solved via dynamic current_value_buffer preservation to handle post-delimiter remnants without state desynchronization.
- **Deterministic Numeric Termination**: Differentiating between continuing digits/decimals and transitions to closing delimiters required specialized regex boundary checks without introducing parsing latency.

## Quickstart

**Prerequisites**:
- Python 3.10+
- uv package manager

### Installation

```bash
make install
```

### Execution

Run the default pipeline:

```bash
make run
```

Or run directly with custom inputs via uv:
```bash
uv run python -m src \
--functions_definition data/input/functions_definition.json \
--input data/input/function_calling_tests.json \
--output data/output/function_calling_results.json
```

### Development & Quality Checks

```bash
make lint     # Runs strict static typing and style audits (mypy --strict, flake8)
make debug    # Launches the application in interactive pdb mode
make clean    # Clears compilation artifacts, pytest and mypy caches
```

## Design Choices & Quality Standards
- **Strict Type Safety**: Full input validation powered by Pydantic V2 models (FunctionDef, TestPrompt).
- **Zero-Dependency Core Logic**: FSM and logit masking rely strictly on standard library data structures and vectorized NumPy arrays for low latency overhead.
- **Production-Grade Tooling**: Zero-error compliance with mypy --strict and flake8

### Performance Analysis

- **Reliability & Accuracy**: Achieves 100% syntactically valid JSON outputs. Function selection and argument extraction accuracy exceed 90% even on a 0.6B parameter model due to hard token masking.

- **Speed**: Fast token selection using vectorized numpy logit masking, processing test prompt suites well under the 5-minute threshold.

### Testing Strategy

The implementation was validated against multiple test suites:

- **Happy Path**: Standard natural language prompts matching single and multi-parameter functions.

- **Schema Validation**: Tests with empty parameter definitions ({}), invalid identifiers (e.g., starting with numbers), missing files, and malformed input JSONs to verify graceful non-crashing exits.

- **Static Analysis**: Enforced zero errors with flake8 and mypy --strict.
