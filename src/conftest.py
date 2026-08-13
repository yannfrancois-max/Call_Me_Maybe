import pytest
from .models import FunctionDef, PropertyDef


@pytest.fixture
def dummy_vocab() -> dict[int, str]:
    """Vocabulaire réduit pour les tests unitaires."""
    return {
        1: "{",
        2: '"prompt": "',
        3: "Hello",
        4: '", ',
        5: '"name": "',
        6: "fn_test",
        7: "fn_",
        8: "test",
        9: '"parameters": {',
        10: '"city": "',
        11: "Paris",
        12: "42",
        13: " true",
        14: "}",
    }


@pytest.fixture
def dummy_functions() -> list[FunctionDef]:
    """Définition d'une fonction valide pour le FSM."""
    return [
        FunctionDef(
            name="fn_test",
            description="Petite test maison",
            parameters={"city": PropertyDef(type="string")},
            returns=PropertyDef(type="string")
        )
    ]
