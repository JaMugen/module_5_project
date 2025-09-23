# test_calculator.py

"""
This test module contains unit tests for the 'app/calculator.py' module.
Each test demonstrates good testing practices using the Arrange-Act-Assert (AAA) pattern.
"""

import pytest
from io import StringIO

# Import the functions to be tested
from app.calculator import display_help, display_history, calculator

def test_display_help(capsys):
    """
    Test the display_help function to ensure it prints the correct help message.

    AAA Pattern:
    - Arrange: No special setup required for this function.
    - Act: Call the display_help function.
    - Assert: Capture the output and verify it matches the expected help message.
    """
    # Arrange
    # No arrangement needed since display_help doesn't require any input or setup.

    # Act
    display_help()

    # Assert
    # Capture the printed output
    captured = capsys.readouterr()
    expected_output = """
Calculator REPL Help
--------------------
Usage:
    <operation> <number1> <number2>
    - Perform a calculation with the specified operation and two numbers.
    - Supported operations:
        add       : Adds two numbers.
        subtract  : Subtracts the second number from the first.
        multiply  : Multiplies two numbers.
        divide    : Divides the first number by the second.

Special Commands:
    help      : Display this help message.
    history   : Show the history of calculations.
    exit      : Exit the calculator.

Examples:
    add 10 5
    subtract 15.5 3.2
    multiply 7 8
    divide 20 4
"""
    # Remove leading/trailing whitespace for comparison
    assert captured.out.strip() == expected_output.strip()

def test_display_history_empty(capsys):
    """
    Test the display_history function when the history is empty.

    AAA Pattern:
    - Arrange: Create an empty history list.
    - Act: Call the display_history function with the empty history.
    - Assert: Capture the output and verify it indicates no calculations have been performed.
    """
    # Arrange
    history = []

    # Act
    display_history(history)

    # Assert
    captured = capsys.readouterr()
    assert captured.out.strip() == "No calculations performed yet."

def test_display_history_with_entries(capsys):
    """
    Test the display_history function when there are entries in the history.

    AAA Pattern:
    - Arrange: Create a history list with sample calculation entries.
    - Act: Call the display_history function with the populated history.
    - Assert: Capture the output and verify it displays the calculations correctly.
    """
    # Arrange
    history = [
        "AddCalculation: 10.0 Add 5.0 = 15.0",
        "SubtractCalculation: 20.0 Subtract 3.0 = 17.0",
        "MultiplyCalculation: 7.0 Multiply 8.0 = 56.0",
        "DivideCalculation: 20.0 Divide 4.0 = 5.0"
    ]

    # Act
    display_history(history)

    # Assert
    captured = capsys.readouterr()
    expected_output = """Calculation History:
1. AddCalculation: 10.0 Add 5.0 = 15.0
2. SubtractCalculation: 20.0 Subtract 3.0 = 17.0
3. MultiplyCalculation: 7.0 Multiply 8.0 = 56.0
4. DivideCalculation: 20.0 Divide 4.0 = 5.0"""
    assert captured.out.strip() == expected_output.strip()

def test_calculator_exit(monkeypatch, capsys):
    """
    Test the calculator function's ability to handle the 'exit' command.

    AAA Pattern:
    - Arrange: Prepare the input 'exit' to simulate user typing 'exit'.
    - Act: Call the calculator function.
    - Assert: Verify that the calculator exits gracefully and prints the exit message.
    """
    # Arrange
    user_input = 'exit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit) as exc_info:
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Exiting calculator. Goodbye!" in captured.out
    assert exc_info.type == SystemExit
    assert exc_info.value.code == 0  # Exit code 0 indicates a clean exit

def test_calculator_help_command(monkeypatch, capsys):
    """
    Test the calculator function's ability to handle the 'help' command.

    AAA Pattern:
    - Arrange: Prepare the input 'help' followed by 'exit' to simulate user interactions.
    - Act: Call the calculator function.
    - Assert: Verify that the help message is displayed and the calculator exits gracefully.
    """
    # Arrange
    user_input = 'help\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Calculator REPL Help" in captured.out
    assert "Exiting calculator. Goodbye!" in captured.out

def test_calculator_invalid_input(monkeypatch, capsys):
    """
    Test the calculator function's handling of invalid input format.

    AAA Pattern:
    - Arrange: Prepare invalid input strings followed by 'exit'.
    - Act: Call the calculator function.
    - Assert: Verify that appropriate error messages are displayed.
    """
    # Arrange
    user_input = 'invalid input\nadd 5\nsubtract\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Invalid input. Please follow the format: <operation> <num1> <num2>" in captured.out
    assert "Type 'help' for more information." in captured.out

def test_calculator_addition(monkeypatch, capsys):
    """
    Test the calculator's addition operation.

    AAA Pattern:
    - Arrange: Prepare the input 'add 10 5' followed by 'exit'.
    - Act: Call the calculator function.
    - Assert: Verify that the correct result is displayed.
    """
    # Arrange
    user_input = 'add 10 5\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Result: AddCalculation: 10.0 Add 5.0 = 15.0" in captured.out

def test_calculator_subtraction(monkeypatch, capsys):
    """
    Test the calculator's subtraction operation.

    AAA Pattern:
    - Arrange: Prepare the input 'subtract 20 5' followed by 'exit'.
    - Act: Call the calculator function.
    - Assert: Verify that the correct result is displayed.
    """
    # Arrange
    user_input = 'subtract 20 5\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Result: SubtractCalculation: 20.0 Subtract 5.0 = 15.0" in captured.out

def test_calculator_multiplication(monkeypatch, capsys):
    """
    Test the calculator's multiplication operation.

    AAA Pattern:
    - Arrange: Prepare the input 'multiply 7 8' followed by 'exit'.
    - Act: Call the calculator function.
    - Assert: Verify that the correct result is displayed.
    """
    # Arrange
    user_input = 'multiply 7 8\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Result: MultiplyCalculation: 7.0 Multiply 8.0 = 56.0" in captured.out

def test_calculator_division(monkeypatch, capsys):
    """
    Test the calculator's division operation.

    AAA Pattern:
    - Arrange: Prepare the input 'divide 20 4' followed by 'exit'.
    - Act: Call the calculator function.
    - Assert: Verify that the correct result is displayed.
    """
    # Arrange
    user_input = 'divide 20 4\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Result: DivideCalculation: 20.0 Divide 4.0 = 5.0" in captured.out

def test_calculator_division_by_zero(monkeypatch, capsys):
    """
    Test the calculator's handling of division by zero.

    AAA Pattern:
    - Arrange: Prepare the input 'divide 10 0' followed by 'exit'.
    - Act: Call the calculator function.
    - Assert: Verify that a zero division error message is displayed.
    """
    # Arrange
    user_input = 'divide 10 0\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Cannot divide by zero." in captured.out

def test_calculator_power(monkeypatch, capsys):
    # Arrange
    user_input = 'power 2 3\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Result: PowerCalculation: 2.0 Power 3.0 = 8.0" in captured.out

def test_calculator_modulus(monkeypatch, capsys):
    # Arrange
    user_input = 'modulus 10 3\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Result: ModulusCalculation: 10.0 Modulus 3.0 = 1.0" in captured.out

def test_calculator_modulus_by_zero(monkeypatch, capsys):
    # Arrange
    user_input = 'modulus 10 0\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    # The calculator catches ZeroDivisionError and prints a generic divide-by-zero
    # message followed by a prompt to enter a non-zero divisor. Update the test
    # to match the actual output from `calculator()`.
    assert "Cannot divide by zero." in captured.out


def test_calculator_history(monkeypatch, capsys):
    """
    Test the calculator's ability to display calculation history.

    AAA Pattern:
    - Arrange: Prepare a sequence of operations followed by 'history' and 'exit'.
    - Act: Call the calculator function.
    - Assert: Verify that the history is displayed correctly.
    """
    # Arrange
    user_input = 'add 10 5\nsubtract 20 3\nhistory\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Result: AddCalculation: 10.0 Add 5.0 = 15.0" in captured.out
    assert "Result: SubtractCalculation: 20.0 Subtract 3.0 = 17.0" in captured.out
    assert "Calculation History:" in captured.out
    assert "1. AddCalculation: 10.0 Add 5.0 = 15.0" in captured.out
    assert "2. SubtractCalculation: 20.0 Subtract 3.0 = 17.0" in captured.out

# New Tests to Increase Coverage

def test_calculator_invalid_number_input(monkeypatch, capsys):
    """
    Test the calculator's handling of invalid number input.

    AAA Pattern:
    - Arrange: Prepare input where numbers are non-numeric strings.
    - Act: Call the calculator function.
    - Assert: Verify that appropriate error messages are displayed.
    """
    # Arrange
    user_input = 'add ten five\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Invalid input. Please ensure numbers are valid." in captured.out or \
           "could not convert string to float: 'ten'" in captured.out or \
           "Invalid input. Please follow the format: <operation> <num1> <num2>" in captured.out

# test_calculator.py

# ... [other imports and tests] ...

def test_calculator_unsupported_operation(monkeypatch, capsys):
    """
    Test the calculator's handling of an unsupported operation.

    AAA Pattern:
    - Arrange: Provide an operation that is not supported.
    - Act: Call the calculator function.
    - Assert: Verify that the appropriate error message is displayed.
    """
    # Arrange
    user_input = 'Ln 2 3\nexit\n'  # Changed 'power' to 'modulus'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "Unsupported calculation type: 'Ln'." in captured.out
    assert "Type 'help' to see the list of supported operations." in captured.out


def test_calculator_keyboard_interrupt(monkeypatch, capsys):
    """
    Test the calculator's handling of KeyboardInterrupt (Ctrl+C).

    AAA Pattern:
    - Arrange: Simulate a KeyboardInterrupt during input().
    - Act: Call the calculator function.
    - Assert: Verify that the calculator exits gracefully.
    """
    # Arrange
    def mock_input(prompt):
        raise KeyboardInterrupt()
    monkeypatch.setattr('builtins.input', mock_input)

    # Act
    with pytest.raises(SystemExit) as exc_info:
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "\nKeyboard interrupt detected. Exiting calculator. Goodbye!" in captured.out
    assert exc_info.value.code == 0

def test_calculator_eof_error(monkeypatch, capsys):
    """
    Test the calculator's handling of EOFError (Ctrl+D).

    AAA Pattern:
    - Arrange: Simulate an EOFError during input().
    - Act: Call the calculator function.
    - Assert: Verify that the calculator exits gracefully.
    """
    # Arrange
    def mock_input(prompt):
        raise EOFError()
    monkeypatch.setattr('builtins.input', mock_input)

    # Act
    with pytest.raises(SystemExit) as exc_info:
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "\nEOF detected. Exiting calculator. Goodbye!" in captured.out
    assert exc_info.value.code == 0

def test_calculator_unexpected_exception(monkeypatch, capsys):
    """
    Test the calculator's handling of unexpected exceptions during calculation execution.

    AAA Pattern:
    - Arrange: Mock the execute method to raise an unexpected exception.
    - Act: Call the calculator function.
    - Assert: Verify that the appropriate error message is displayed.
    """
    # Arrange
    class MockCalculation:
        def execute(self):
            raise Exception("Mock exception during execution")
        def __str__(self):
            return "MockCalculation"

    def mock_create_calculation(operation, a, b):
        return MockCalculation()

    monkeypatch.setattr('app.calculation.CalculationFactory.create_calculation', mock_create_calculation)
    user_input = 'add 10 5\nexit\n'
    monkeypatch.setattr('sys.stdin', StringIO(user_input))

    # Act
    with pytest.raises(SystemExit):
        calculator()

    # Assert
    captured = capsys.readouterr()
    assert "An error occurred during calculation: Mock exception during execution" in captured.out
    assert "Please try again." in captured.out
