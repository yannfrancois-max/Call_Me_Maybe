"""
Logit masking engine enforcing token-level JSON schema constraints.
"""

import json
import re

from .models import FunctionDef
from .state import FSMStep, GenerationState


NUMERIC_PREFIX_PATTERN = re.compile(r'^-?\d*\.?\d*$')
VALID_JSON_ESCAPE_PATTERN = re.compile(r'\\(?:["\\/bfnrtu]|$)', re.DOTALL)


def _is_transition_allowed(raw_token: str, state: GenerationState) -> bool:
    """
    Verifies whether a transition token after a value is valid.
    Args:
        raw_token: Decoded token text.
        state: Current generation state.

    Returns:
        True if the token represents a valid transition, False otherwise.
    """
    if raw_token.strip() in ("}", "}\n"):
        return len(state.remaining_params) <= 1

    if (
        raw_token.lstrip().startswith(",") and
        len(state.remaining_params) > 1
    ):
        if '"' in raw_token:
            token_after_quote = raw_token.split('"')[-1]
            return (
                not token_after_quote or
                state.remaining_params[1].startswith(token_after_quote)
            )
        return True
    return False


def _is_valid_json_string_continuation(val_so_far: str, token: str) -> bool:
    """
    Checks if appending a token forms a valid JSON string escape sequence.
    Args:
        val_so_far: Accumulated string value content.
        token: Candidate token to append.

    Returns:
        True if continuation is valid, False otherwise.
    """
    candidate = val_so_far + token
    slash_car = "\\"
    # The handling of escape sequences (\n, \", \\) aims to prevent a token
    # from breaking the syntax of a JSON string currently being generated.
    if '"' in token:
        temp = val_so_far + token
        i = len(val_so_far)
        while i < len(temp):
            if temp[i] == '"':
                bs_count = 0
                j = i - 1
                while j >= 0 and temp[j] == '\\':
                    bs_count += 1
                    j -= 1
                if bs_count % 2 == 0:
                    rest = temp[i + 1:]
                    if rest and not re.match(r'^\s*[,}\n]', rest):
                        return False
            i += 1

    if slash_car not in token and not val_so_far.endswith('\\'):
        return True
    for match in re.finditer(r'\\.', candidate + ' '):
        sub = match.group(0)
        if not (
            sub[1] in '"\\/bfnrtu' or
            (match.start() == len(candidate) - 1)
        ):
            return False
    return True


def get_allowed_tokens(
    state: GenerationState,
    id_to_token: dict[int, str],
    functions_defs: list[FunctionDef]
) -> set[int]:
    """
    Computes the set of valid token IDs allowed for the current FSM step.
    Args:
        state: Current generation state.
        id_to_token: Dictionary mapping token IDs to decoded strings.
        functions_defs: List of valid function definitions.

    Returns:
        A set of integer token IDs permitted for generation.
    """
    allowed_ids: set[int] = set()
    valid_func_names = [f.name for f in functions_defs]
    target_segment: str | None = None

    remainder = ""
    existing_prefix = ""
    val_already_written: str | None = None
    # Suffix search in order to get the BPE tokenizers to
    # arbitrarily cut words.
    # Token might only contain part of the JSON key.
    # Longest suffix search allow to exactly know which part of
    # the strings needs to be generated.
    if state.step == FSMStep.EXPECT_PROMPT:
        target_segment = f'"prompt": {json.dumps(state.prompt_text)}, '

    elif state.step == FSMStep.EXPECT_NAME_KEY:
        target_segment = '"name": "'

    elif state.step == FSMStep.EXPECT_PARAMS_KEY:
        target_segment = ', "parameters": {'

    elif state.step == FSMStep.EXPECT_NAME_VAL:
        existing_prefix = state.current_value_buffer

    elif state.step == FSMStep.EXPECT_PARAM_OBJECT:
        pos = state.generated_json.rfind('"parameters":')
        params_json = (
            state.generated_json[pos:]
            if pos != -1 else state.generated_json
        )
        match = re.search(r'[\{,]\s*"([^"]*)$', state.generated_json)
        existing_prefix = match.group(1) if match else ""
        if state.current_param_name:
            pattern = rf'"{re.escape(state.current_param_name)}"\s*:\s*(.*)$'
            val_match = re.search(pattern, params_json, re.DOTALL)
            if val_match:
                val_already_written = val_match.group(1)
    if target_segment is not None:
        for i in range(len(target_segment), -1, -1):
            prefix = target_segment[:i]
            if state.generated_json.endswith(prefix):
                remainder = target_segment[i:]
                break
    json_so_far = state.generated_json.rstrip()
    is_starting_new_key = (
        not state.in_string_quotes and
        (
            json_so_far.endswith('{') or
            json_so_far.endswith(',')
            )
        )
    for token_id, raw_token in id_to_token.items():

        # 1. Waiting for JSON object opening '{'
        if state.step == FSMStep.EXPECT_START:
            if raw_token.lstrip().startswith("{"):
                allowed_ids.add(token_id)

        # 2. Entering the "prompt" key or the prompt text
        elif state.step == FSMStep.EXPECT_PROMPT:
            if remainder.startswith(raw_token):
                allowed_ids.add(token_id)

        # 3. Ensures json_str ends with '"name": "'
        elif state.step == FSMStep.EXPECT_NAME_KEY:
            if remainder.startswith(raw_token):
                allowed_ids.add(token_id)

        # 4. Choosing the function name
        elif state.step == FSMStep.EXPECT_NAME_VAL:
            candidate = existing_prefix + raw_token
            if (
                any(func_name.startswith(candidate)
                    for func_name in valid_func_names)
                or (existing_prefix in valid_func_names and
                    raw_token.startswith('"'))
            ):
                allowed_ids.add(token_id)

        # 5. Waiting for the "parameters" key: {
        elif state.step == FSMStep.EXPECT_PARAMS_KEY:
            if remainder.startswith(raw_token):
                allowed_ids.add(token_id)

        # 6. Entering arguments in the "parameters" object
        elif state.step == FSMStep.EXPECT_PARAM_OBJECT:
            if is_starting_new_key and state.remaining_params:
                if not raw_token.lstrip().startswith('"'):
                    continue
            if val_already_written is not None:
                if state.current_param_type == "number":
                    if all(c in "0123456789.- " for c in raw_token):
                        candidate_val = (
                            val_already_written +
                            raw_token
                            ).strip()
                        if len(candidate_val) >= 12:
                            continue
                        if (candidate_val and
                                NUMERIC_PREFIX_PATTERN.match(candidate_val)):
                            allowed_ids.add(token_id)
                    else:
                        try:
                            float(val_already_written)
                            if _is_transition_allowed(raw_token, state):
                                allowed_ids.add(token_id)
                        except ValueError:
                            pass
                elif state.current_param_type == "boolean":
                    candidate_bool = (
                        val_already_written + raw_token
                        ).strip()
                    if (
                        "true".startswith(candidate_bool) or
                        "false".startswith(candidate_bool)
                    ):
                        allowed_ids.add(token_id)
                    elif val_already_written in ("true", "false"):
                        if _is_transition_allowed(raw_token, state):
                            allowed_ids.add(token_id)
                elif state.current_param_type == "string":
                    if any(ord(c) < 32 for c in raw_token):
                        continue
                    if not val_already_written.startswith('"'):
                        if raw_token.lstrip().startswith('"'):
                            allowed_ids.add(token_id)
                    else:
                        is_closed = (
                            val_already_written.endswith('"') and
                            len(val_already_written) >= 2 and
                            (len(val_already_written) == 2 or
                             val_already_written[-2] != '\\')
                        )
                        if is_closed:
                            if _is_transition_allowed(raw_token, state):
                                allowed_ids.add(token_id)
                        else:
                            if _is_valid_json_string_continuation(
                                val_already_written,
                                raw_token
                            ):
                                allowed_ids.add(token_id)
                elif state.current_param_type == "integer":
                    if all(c in "0123456789- " for c in raw_token):
                        candidate_val = (
                            val_already_written +
                            raw_token
                        ).strip()
                        if (
                            candidate_val and
                            re.match(r'^-?\d*$', candidate_val)
                        ):
                            allowed_ids.add(token_id)
                    else:
                        try:
                            int(val_already_written)
                            if _is_transition_allowed(raw_token, state):
                                allowed_ids.add(token_id)
                        except ValueError:
                            pass
            else:
                if not state.remaining_params:
                    # Case 1: Everything has been generated -> Only the
                    # closing of the JSON is valid.
                    if (
                        raw_token.strip() == "}" or
                        raw_token.strip() == "}\n"
                    ):
                        allowed_ids.add(token_id)
                else:
                    # Case 2: Parameters remain -> Validate key tokens
                    if (
                        not existing_prefix and
                        not state.generated_json.rstrip().endswith('"') and
                        not raw_token.lstrip().startswith('"')
                    ):
                        continue
                    target_param = state.remaining_params[0]
                    if (
                        not existing_prefix and
                        not state.generated_json.rstrip().endswith('"')
                    ):
                        if raw_token.lstrip().startswith('"'):
                            token_after_quote = raw_token.lstrip()[1:]
                            clean_key = token_after_quote.rstrip('": ')
                            if target_param.startswith(clean_key):
                                allowed_ids.add(token_id)
                    else:
                        if len(existing_prefix) < len(target_param):
                            candidate = existing_prefix + raw_token
                            if (
                                target_param.startswith(candidate) and
                                len(candidate) > len(existing_prefix)
                            ):
                                allowed_ids.add(token_id)
                        elif existing_prefix == target_param:
                            if (
                                raw_token.lstrip().startswith('"') or
                                raw_token.lstrip().startswith(':')
                            ):
                                allowed_ids.add(token_id)

        # 7. JSON closing
        elif state.step == FSMStep.EXPECT_END:
            if state.generated_json.count("}") < 2:
                if "}" in raw_token:
                    allowed_ids.add(token_id)
            else:
                if not raw_token.strip():
                    allowed_ids.add(token_id)

    return allowed_ids
