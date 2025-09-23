# tests/conftest.py

import pytest
from app.calculation import CalculationFactory
from app.calculation import (
    AddCalculation,
    SubtractCalculation,
    MultiplyCalculation,
    DivideCalculation,
    PowerCalculation,
    ModulusCalculation,
)

@pytest.fixture(autouse=True)
def reset_calculation_factory():
    """
    Fixture to reset CalculationFactory's registered calculations before each test.
    """
    # Clear existing registrations
    CalculationFactory._calculations.clear()

    # Re-register the default calculations
    CalculationFactory.register_calculation('add')(AddCalculation)
    CalculationFactory.register_calculation('subtract')(SubtractCalculation)
    CalculationFactory.register_calculation('multiply')(MultiplyCalculation)
    CalculationFactory.register_calculation('divide')(DivideCalculation)
    CalculationFactory.register_calculation('power')(PowerCalculation)
    CalculationFactory.register_calculation('modulus')(ModulusCalculation)
