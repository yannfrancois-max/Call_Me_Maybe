"""
Data models and validation schemas for functions, prompts, and outputs.
"""

import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator


IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class PropertyDef(BaseModel):
    """
    Definition of a single property type inside a function schema.
    Attributes:
        type: The JSON data type expected
        ('string', 'number', 'integer', 'boolean').
    """
    type: Literal["string", "number", "integer", "boolean"]


class FunctionDef(BaseModel):
    """
    Definition of an executable function schema.
    Attributes:
        name: Unique identifier of the function.
        description: Natural language explanation of the function's purpose.
        parameters: Dictionary mapping argument names to their property
        definitions.
        returns: Property definition of the return value.
    """
    name: str
    description: str
    parameters: dict[str, PropertyDef] = Field(default_factory=dict)
    returns: PropertyDef

    @field_validator("name")
    @classmethod
    def validate_function_name(cls, value: str) -> str:
        """Validates that the function name is a valid non-empty identifier.
        Args:
            value: The function name to check.
        Returns:
            The cleaned function name string.
        Raises:
            ValueError: If the name is empty or contains invalid characters.
        """
        clean_val = value.strip()
        if not clean_val:
            raise ValueError("Function name must not be empty")
        if not IDENTIFIER_PATTERN.match(clean_val):
            raise ValueError(
                f"Invalid function name '{value}'."
                "It must comply with the identifier format"
                "(ex: 'fn_add_numbers')."
            )
        return clean_val

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        """
        Ensures the description is not empty or whitespace-only.
        Args:
            value: Description string.
        Returns:
            The validated description string.
        Raises:
            ValueError: If the description is empty.
        """
        if not value.strip():
            raise ValueError("Description must not be empty")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_param_names(
        cls, params: dict[str, PropertyDef]
    ) -> dict[str, PropertyDef]:
        """
        Validates that all parameter keys follow identifier naming rules.
        Args:
            params: Dictionary of parameter definitions.
        Returns:
            The validated dictionary of parameters.
        Raises:
            ValueError: If any parameter name is an invalid identifier.
        """
        for param_name in params.keys():
            if not IDENTIFIER_PATTERN.match(param_name):
                raise ValueError(
                    f"Invalid parameter name '{param_name}'."
                    "It must be a valid identifier."
                )
        return params


def validate_functions_list(functions: list[FunctionDef]) -> list[FunctionDef]:
    """
    Ensures no duplicate function names exist in the loaded definitions.
    Args:
        functions: List of function definitions to validate.
    Returns:
        The validated list of function definitions.
    Raises:
        ValueError: If no functions are provided or duplicate names exist.
    """
    if not functions:
        raise ValueError("At least one function definition is required.")
    seen_names: set[str] = set()
    for func in functions:
        if func.name in seen_names:
            raise ValueError(
                f"Duplicate function name detected: '{func.name}'"
                )
        seen_names.add(func.name)
    return functions


class TestPrompt(BaseModel):
    """
    Model wrapping an input natural language prompt.
    Attributes:
        prompt: The input prompt string.
    """
    prompt: str


class FunctionCallResult(BaseModel):
    """
    Schema representing the expected structured output item.
    Attributes:
        prompt: Original user input prompt.
        name: Name of the selected function.
        parameters: Key-value map of extracted parameter arguments.
    """
    prompt: str
    name: str
    parameters: dict[str, str | float | int | bool]
