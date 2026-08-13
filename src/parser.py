"""
Input parsing utilities for command-line arguments, functions, and prompts.
"""

from argparse import ArgumentParser, Namespace
import json
from typing import Any, cast

from .models import FunctionDef, TestPrompt


def load_functions(filepath: str) -> list[FunctionDef]:
    """
    Loads and validates function definitions from a JSON file.
    Args:
        filepath: Path to the JSON file containing function definitions.
    Returns:
        A list of validated FunctionDef objects.
    """
    functions_list: list[FunctionDef] = []  # Typed empty list creation
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            func_item = cast(list[object], data)
            for raw_func in func_item:
                if isinstance(raw_func, dict):
                    valid_func = cast(dict[str, Any], raw_func)
                    functions_list.append(FunctionDef(**valid_func))
    return functions_list


def load_prompts(filepath: str) -> list[str]:
    """
    Loads natural language test prompts from a JSON file.
    Args:
        filepath: Path to the JSON file containing test prompts.
    Returns:
        A list of prompt strings.
    """
    prompts_list: list[str] = []  # Typed empty list creation
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            prompt_item = cast(list[object], data)
            for raw_prompt in prompt_item:
                if isinstance(raw_prompt, dict):
                    valid_prompt = cast(dict[str, Any], raw_prompt)
                    validated_item = TestPrompt(**valid_prompt)
                    prompts_list.append(validated_item.prompt)
    return prompts_list


def arg_parse() -> Namespace:
    """
    Parses command-line arguments for input, output, and debug settings.
    Returns:
        A Namespace object containing parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to functions definitions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to output prompts JSON"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    return parser.parse_args()
