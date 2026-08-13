*This activity has been created as part of the 42 curriculum by yafranco*

# Call Me Maybe - Function Calling via Constrained Decoding

## Description

**Call Me Maybe** is a lightweight Python tool designed to bridge natural language prompts with machine-executable code. Using a 0.6B parameter model (`Qwen/Qwen3-0.6B`), the project translates natural language requests (e.g., *"What is the sum of 2 and 3?"*) into structured, type-safe JSON function calls (e.g., `{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}`).

Small language models traditionally struggle to consistently produce syntactically valid JSON. This project implements **token-level constrained decoding** using a Finite State Machine (FSM) and logit masking, guaranteeing 100% valid JSON output and strict adherence to function schemas without relying on heuristic post-processing or prompt luck.

## Instructions

### Installation

Ensure you have `uv` installed. Set up the virtual environment and install all necessary dependencies by running:

```bash
make install
```

### Execution

Run the main script using the mandatory command structure:

```bash
make run
```

or execute it directly via uv:
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

### Additionnal Makefile Rules

- make debug: Executes the application in debug mode using Python's pdb.

- make clean: Removes temporary cache directories (__pycache__, .mypy_cache, .pytest_cache) and output files.

- make lint: Performs static analysis and strict style checks using flake8 and mypy.

### Storage Quota Issues (42 Cluster)

Installing heavy ML packages like `torch` can exceed your session's home directory disk quota (`No space left on device`). If you encounter this during evaluation, set up the virtual environment and cache in `/goinfre`:

```bash
# Clean UV cache to free up space
uv cache clean

# Create environment folder in /goinfre and link it
mkdir -p /goinfre/$USER/Call_Me_Maybe_venv
rm -rf .venv
ln -s /goinfre/$USER/Call_Me_Maybe_venv .venv

# Run install with cache redirected to /goinfre
export UV_CACHE_DIR=/goinfre/$USER/.cache/uv
make install
```

## Resources

- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers/index
- Pydantic V2 Documentation: https://www.google.com/search?q=https://docs.pydantic.dev/latest/
- PEP 8 -- Style Guide for Python Code: https://peps.python.org/pep-0008/

## AI Usage

AI tools were utilized as an adaptive collaborator to assist in:

1. Brainstorming edge-case scenarios for JSON schema validation.
2. Reviewing BPE tokenizer behavior regarding multi-character token overlaps.
3. Structuring and refining documentation.

## Algorithm Explanation

The core engine relies on Constrained Decoding driven by a Finite State Machine (FSM):

1. Vocabulary Inversion & Decoding: Token IDs from the LLM vocabulary are decoded into exact string representations using llm.decode([token_id]).

2. FSM State Tracking: A GenerationState model tracks the structural context of the output (EXPECT_START, EXPECT_PROMPT, EXPECT_NAME_KEY, EXPECT_NAME_VAL, EXPECT_PARAMS_KEY, EXPECT_PARAM_OBJECT, EXPECT_END).

3. Logit Masking: Before each token generation step, get_allowed_tokens() evaluates every token in the vocabulary against the current FSM state and target schema:

    - Keys & Literals: Restricts tokens to exact structural strings (e.g., "prompt":, "name":, "parameters":).

    - Function Names: Restricts tokens to valid prefixes of defined functions in functions_definition.json.

    - Typed Parameter Values: Restricts tokens to valid type representations (integers/floats for number, booleans, and string continuations).

    - Invalid token logits are set to -inf, ensuring 0% probability during argmax selection.

4. State Transition: update_state() updates the FSM step and parameter tracking buffers based on newly appended tokens.

## Design Decisions

- Pydantic Data Models: Used across input loading (FunctionDef, TestPrompt) to enforce strict type constraints, argument identifier formats, and graceful error reporting upon initialization.

- SDK-Native Decoding: Used Small_LLM_Model.decode() instead of manual character substitutions (like replacing BPE symbols Ġ or Ċ) to maintain compatibility with different BPE tokenizers.

- Buffer-Preserving State Updates: When token boundaries cross separators (e.g., a single BPE token containing , "key"), the state update preserves post-separator remnants in current_value_buffer to prevent state desynchronization.

## Performance Analysis

- Reliability & Accuracy: Achieves 100% syntactically valid JSON outputs. Function selection and argument extraction accuracy exceed 90% even on a 0.6B parameter model due to hard token masking.

- Speed: Fast token selection using vectorized numpy logit masking, processing test prompt suites well under the 5-minute threshold.

## Challenges Faced

- Token Overlap Across JSON Structural Boundaries: BPE tokenizers frequently produce tokens that bridge two logical JSON components (e.g., closing a value quote, writing a comma, and opening a new key quote in a single token). This was resolved by maintaining a dynamic current_value_buffer that extracts post-delimiter substrings during FSM updates.

- Numeric Value Termination: Distinguishing between valid numeric continuations and transitions to closing delimiters required dedicated regex matching (NUMERIC_PREFIX_PATTERN) and transition validation.

## Testing Strategy

The implementation was validated against multiple test suites:

- Happy Path: Standard natural language prompts matching single and multi-parameter functions.

- Schema Validation: Tests with empty parameter definitions ({}), invalid identifiers (e.g., starting with numbers), missing files, and malformed input JSONs to verify graceful non-crashing exits.

- Static Analysis: Enforced zero errors with flake8 and mypy --strict.

## Example Usage

Input Prompt (function_calling_tests.json):
```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  }
]
```

Generated Output (function_calling_results.json):
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

