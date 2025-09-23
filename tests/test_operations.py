import pytest
from typing import Union  # Import Union for type hinting multiple possible types
from app.operations import Operations 

Number = Union[int, float]

@pytest.mark.parametrize(
    "a, b, expected", 
    [
        (2, 3, 5),
        (0, 0, 0),
        (-1, 1, 0),
        (-2,-3, -5),
    ],
    ids=[
        "add_two_positive_numbers", 
        "add_two_zeroes", 
        "add_negative_and_positive_numbers",
        "add_two_negative_numbers",
        ]
)   
def test_addition(a: Number, b: Number, expected: Number) -> None:
    result = Operations.addition(a, b)
    assert result == expected, f"Expected addition({a}, {b}) to be {expected}, but got {result}"

@pytest.mark.parametrize(
    "a, b, expected", 
    [
        (5, 3, 2),
        (0, 0, 0),
        (10, 5, 5),
        (-5,-3,-2),
    ],
    ids=[
        "subtract_two_positive_numbers", 
        "subtract_two_zeroes", 
        "subtract_two_positive_numbers",
        "subtract_two_negative_numbers",    
        ]           
)
def test_subtraction(a: Number, b: Number, expected: Number) -> None:
    result = Operations.subtraction(a, b)
    assert result == expected, f"Expected subtraction({a}, {b}) to be {expected}, but got {result}"

@pytest.mark.parametrize(
    "a, b, expected",   
    [
        (2, 3, 6),
        (0, 10, 0),
        (-2, 3, -6),
        (-2,-3, 6),
    ],  
    ids=[
        "multiply_two_positive_numbers", 
        "multiply_by_zero", 
        "multiply_negative_and_positive_numbers",
        "multiply_two_negative_numbers",
        ]
)
def test_multiplication(a: Number, b: Number, expected: Number) -> None:
    result = Operations.multiplication(a, b)
    assert result == expected, f"Expected multiplication({a}, {b}) to be {expected}, but got {result}"

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (6, 3, 2),
        (-6, -3, 2),
        (-6, 3, -2),
    ],
    ids=[
        "divide_two_positive_numbers",
        "divide_two_negative_numbers",
        "divide_negative_and_positive_numbers", 
    ]
)   
def test_division(a: Number, b: Number, expected: Number) -> None:
    result = Operations.division(a, b)
    assert result == expected, f"Expected division({a}, {b}) to be {expected}, but got {result}"

@pytest.mark.parametrize(
    "a, b",
    [
        (1, 0),
        (-1, 0),
        (0, 0),
    ],
    ids=[
        "divide_positive_by_zero",
        "divide_negative_by_zero",
        "divide_zero_by_zero",
    ]
)
def test_division_by_zero(a: Number, b: Number) -> None:    
    with pytest.raises(ValueError, match="Division by zero is not allowed.") as excinfo:
        Operations.division(a, b)
    assert "Division by zero is not allowed." in str(excinfo.value), \
        f"Expected error message 'Division by zero is not allowed.', but got '{excinfo.value}'"       
    

#finish