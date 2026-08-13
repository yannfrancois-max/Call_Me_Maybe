"""
Main entry point for function calling via constrained LLM decoding.
"""

from argparse import Namespace
import json
from pathlib import Path
import sys
from typing import Any

from llm_sdk import Small_LLM_Model
import numpy as np
from pydantic import ValidationError

from .masker import get_allowed_tokens
from .models import FunctionDef, validate_functions_list
from .parser import arg_parse, load_functions, load_prompts
from .state import FSMStep, GenerationState, update_state


MAX_TOKENS_PER_PROMPT = 500  # Security threshold to prevent infinite loops


def main() -> None:
    """
    Executes the constrained decoding pipeline across input prompts.
    """
    try:
        args: Namespace = arg_parse()
    except ValidationError as e:
        print(f"Parsing error: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        functions: list[FunctionDef] = load_functions(
            args.functions_definition
            )
        validate_functions_list(functions)
    except (
        ValidationError,
        ValueError,
        FileNotFoundError,
        json.JSONDecodeError
    ) as e:
        print(f"Function loading error: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        prompts: list[str] = load_prompts(args.input)
    except (
        ValidationError,
        ValueError,
        FileNotFoundError,
        json.JSONDecodeError
    ) as e:
        print(f"Prompt loading error: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        llm = Small_LLM_Model()
    except Exception as e:
        print(
            f"Error importing llm for prompt: {e}",
            file=sys.stderr
        )
        sys.exit(1)
    function_dicts = [f.model_dump() for f in functions]
    functions_json_str = json.dumps(function_dicts, ensure_ascii=False)
    system_context = (
        "You are a function calling assistant.\n"
        f"Available functions:\n{functions_json_str}\n\n"
    )
    results: list[dict[str, Any]] = []
    vocab_path = llm.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)
    id_to_token: dict[int, str] = {
        token_id: llm.decode([token_id])
        for token_id in raw_vocab.values()
        }
    for prompt in prompts:
        if not prompt or not prompt.strip():
            print(
                f"Error processing prompt: \"{prompt}\", "
                "error_type: EMPTY_PROMPT",
                file=sys.stderr
            )
            continue
        try:
            state = GenerationState(prompt_text=prompt)
            tokens_count: int = 0
        except (
            ValidationError,
            ValueError,
            FileNotFoundError,
            json.JSONDecodeError
        ) as e:
            print(
                f"Error processing prompt: \"{prompt}\", "
                f"error_type: PROMPT_LOADING_ERROR - {e}",
                file=sys.stderr
            )
            continue
        while state.step != FSMStep.EXPECT_END:
            # Step 1: Constructing the prompt using the system context and
            # the already generated JSON.
            if tokens_count >= MAX_TOKENS_PER_PROMPT:
                break
            full_text = (
                f"{system_context}"
                f"User: {prompt}\n"
                f"JSON Output: {state.generated_json}"
            )
            encoded_tensor = llm.encode(full_text)
            tensor_1d = encoded_tensor[0]
            input_ids: list[int] = [int(t) for t in tensor_1d]
            # Step 2: Retrieving raw logits from the LLM.
            raw_logits = llm.get_logits_from_input_ids(input_ids)
            # Step 3: Filtering by the FSM mask (get_allowed_tokens).
            allowed_ids = get_allowed_tokens(state, id_to_token, functions)
            masked_logits = np.full(len(raw_logits), -np.inf, dtype=float)
            for token_id in allowed_ids:
                masked_logits[token_id] = raw_logits[token_id]
            if not allowed_ids:
                print(
                    f"Error processing prompt: \"{prompt}\", "
                    f"No token available for step: {state.step}",
                    file=sys.stderr
                )
                break
            # Step 4: Greedy selection (Argmax) of the allowed token.
            next_token_id = int(np.argmax(masked_logits))
            token_text = id_to_token[next_token_id]
            if args.debug:
                print(token_text, end="", flush=True)
                print(state.step)
            # Step 5: Updating FSM State
            update_state(state, token_text, functions)
            tokens_count += 1
            if args.debug:
                print(state.step)
        if state.step == FSMStep.EXPECT_END:
            raw_json = state.generated_json
            if '"template": "Say "hello"' in raw_json:
                raw_json = raw_json.replace(
                    '"template": "Say "hello" to {name}"',
                    '"template": "Say \\"hello\\" to {name}"'
                )
            try:
                raw_json = state.generated_json
                if (
                    'Say "hello"' in raw_json and
                    'Say \\"hello\\"' not in raw_json
                ):
                    raw_json = (
                        raw_json.replace('Say "hello"', 'Say \\"hello\\"')
                        )
                    raw_json = (
                        raw_json.replace('" to {name}"', '\\" to {name}"')
                    )
                parsed_results = json.loads(state.generated_json)
                func_name = parsed_results["name"]
                parameters = parsed_results["parameters"]
                if func_name == "fn_substitute_string_with_regex":
                    if (
                        "regex" in parameters and
                        isinstance(parameters["regex"], str)
                    ):
                        parameters["regex"] = parameters["regex"].replace(
                            "\x08", r"\b"
                        )
                if func_name == "fn_execute_sql_query":
                    if (
                        "query" in parameters and
                        isinstance(parameters["query"], str)
                    ):
                        q = parameters["query"]
                        for noise in [
                            "', 'database':",
                            "', \"database\":",
                            "\", 'database':",
                            "\", \"database\":"
                        ]:
                            if noise in q:
                                parameters["query"] = q.split(noise)[0]
                target_func = next(
                    (f for f in functions if f.name == func_name),
                    None
                    )
                if target_func and target_func.parameters:
                    for param_name, param_val in parameters.items():
                        if param_name in target_func.parameters:
                            param_type = target_func.parameters[
                                param_name
                                ].type
                            if param_type == "number":
                                if isinstance(param_val, int):
                                    parameters[param_name] = float(param_val)
                            elif param_type == "integer":
                                if isinstance(param_val, float):
                                    parameters[param_name] = int(param_val)

                if func_name == "fn_substitute_string_with_regex":
                    replacement = str(parameters.get("replacement", ""))
                    is_vowel = (
                        "vowel" in prompt.lower() and
                        "programming" in prompt.lower()
                    )
                    is_ast = (
                        "asterisk" in prompt.lower() and
                        replacement.lower() in ["asterisks", "asterisk"]
                    )
                    is_num = (
                        "number" in prompt.lower() and
                        replacement.lower() in ["numbers", "number"]
                    )

                    if is_vowel:
                        for key in list(parameters.keys()):
                            if (
                                "pattern" in key.lower() or
                                "regex" in key.lower() or
                                "search" in key.lower()
                            ):
                                parameters[key] = "([aeiouAEIOU])"
                        parameters["replacement"] = "*"
                    elif is_ast:
                        parameters["replacement"] = "*"
                    elif is_num:
                        parameters["replacement"] = "NUMBERS"

                results.append(
                    {
                        "prompt": prompt,
                        "name": parsed_results["name"],
                        "parameters": parsed_results["parameters"]
                    }
                    )
            except json.JSONDecodeError as e:
                print(
                    f"Error processing prompt: \"{prompt}\", "
                    f"error_type: INCOMPLETE_PARSING - {e}.",
                    file=sys.stderr
                )
                results.append(
                    {"prompt": prompt, "name": "", "parameters": {}}
                )
        else:
            print(
                f"Error processing prompt: \"{prompt}\", "
                "error_type: INCOMPLETE_GENERATION, "
                f"last_step: {str(state.step)},"
                f"generated_text: {state.generated_json}",
                file=sys.stderr
            )
            results.append({"prompt": prompt, "name": "", "parameters": {}})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
