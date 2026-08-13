"""
Finite State Machine (FSM) state definition and tracking for JSON generation.
"""

from enum import Enum
import json
import re
from pydantic import BaseModel, Field

from .models import FunctionDef


class FSMStep(str, Enum):
    """
    Major execution steps of the FSM for JSON construction.
    """
    EXPECT_START = "EXPECT_START"
    EXPECT_PROMPT = "EXPECT_PROMPT"
    EXPECT_NAME_KEY = "EXPECT_NAME_KEY"
    EXPECT_NAME_VAL = "EXPECT_NAME_VAL"
    EXPECT_PARAMS_KEY = "EXPECT_PARAMS_KEY"
    EXPECT_PARAM_OBJECT = "EXPECT_PARAM_OBJECT"
    EXPECT_END = "EXPECT_END"


class GenerationState(BaseModel):
    """
    Maintains state tracking information during constrained decoding.
    Attributes:
        step: Current FSM step.
        selected_function: Name of the function currently selected by the LLM.
        remaining_params: Argument parameter names list remaining to generate.
        current_param_name: Parameter name currently being generated.
        current_param_type: Expected JSON data type for the current parameter.
        in_string_quotes: Indicates whether decoding is inside an open string.
        generated_json: Cumulative JSON text generated so far.
        prompt_text: Original natural language prompt string.
        current_value_buffer: Working buffer storing partial token segments.
    """
    step: FSMStep = FSMStep.EXPECT_START
    selected_function: str | None = None
    remaining_params: list[str] = Field(default_factory=list)
    current_param_name: str | None = None
    current_param_type: str | None = None
    in_string_quotes: bool = False
    generated_json: str = ""
    prompt_text: str = ""
    current_value_buffer: str = ""


def _is_in_string(json_str: str) -> bool:
    """
    Determines whether the cursor is currently inside a quoted string.
    Args:
        json_str: Accumulated JSON string.
    Returns:
        True if inside an unescaped string quote, False otherwise.
    """
    in_str = False
    escaped = False
    for char in json_str:
        if escaped:
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '"':
            in_str = not in_str
    return in_str


def update_state(
    state: GenerationState,
    token_text: str,
    functions_defs: list[FunctionDef]
) -> None:
    """
    Updates FSM state, parameter tracking, and buffers based on token input.
    Args:
        state: Current generation state object.
        token_text: Decoded string representation of the latest token.
        functions_defs: List of available function definitions.
    """
    state.generated_json += token_text
    state.current_value_buffer += token_text
    json_str = state.generated_json

    # Update string flag open or closed
    state.in_string_quotes = _is_in_string(json_str)

    # 1. Step EXPECT_START -> Detection of first '{'
    if state.step == FSMStep.EXPECT_START:
        if "{" in token_text:
            state.step = FSMStep.EXPECT_PROMPT
            state.current_value_buffer = ""

    # 2. Key/Value of the prompt (ex: "prompt": "...")
    elif state.step == FSMStep.EXPECT_PROMPT:
        target = f'"prompt": {json.dumps(state.prompt_text)}, '
        if json_str.endswith(target):
            state.step = FSMStep.EXPECT_NAME_KEY
            state.current_value_buffer = ""

    # 3. Ensures json_str ends with '"name": "'
    elif state.step == FSMStep.EXPECT_NAME_KEY:
        if json_str.endswith('"name": "'):
            state.step = FSMStep.EXPECT_NAME_VAL
            state.current_value_buffer = ""

    # 4. Function name extraction
    elif state.step == FSMStep.EXPECT_NAME_VAL:
        if '"' in token_text and not state.in_string_quotes:
            func_name = (
                re.sub(r'[^a-zA-Z0-9_]', '', state.current_value_buffer)
                )
            state.selected_function = func_name

            # Definition and parameters recovery
            target_func = next(
                (f for f in functions_defs if f.name == func_name),
                None,
            )

            state.remaining_params = []

            if target_func and target_func.parameters:
                params_dict = target_func.parameters
                state.remaining_params = list(params_dict.keys())

            state.step = FSMStep.EXPECT_PARAMS_KEY
            state.current_value_buffer = ""

    # 5. Key "parameters": {
    elif state.step == FSMStep.EXPECT_PARAMS_KEY:
        if json_str.endswith(', "parameters": {'):
            state.step = FSMStep.EXPECT_PARAM_OBJECT
            state.current_value_buffer = ""

    # 6. Parameters treatment and end detection
    elif state.step == FSMStep.EXPECT_PARAM_OBJECT:
        if "," in token_text and not state.in_string_quotes:
            if state.remaining_params:
                state.remaining_params.pop(0)
            state.current_value_buffer = token_text.split(",")[-1]
        if "}" in token_text and not state.in_string_quotes:
            state.remaining_params = []
            clean_json = (
                state.generated_json.replace(" ", "").replace("\n", "")
                )
            if clean_json.endswith("}}"):
                state.step = FSMStep.EXPECT_END
            state.current_value_buffer = ""
        target_func = next(
            (
                f for f in functions_defs
                if f.name == state.selected_function
            ),
            None,
        )
        if target_func is not None and state.remaining_params:
            state.current_param_name = state.remaining_params[0]
            state.current_param_type = (
                target_func.parameters[state.current_param_name].type
                )
        else:
            state.current_param_name = None
            state.current_param_type = None
