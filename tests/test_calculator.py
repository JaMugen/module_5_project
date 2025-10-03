from ast import ExceptHandler
import datetime
from pathlib import Path
from sqlite3 import Time
import pandas as pd
import pytest
from unittest.mock import Mock, patch, PropertyMock
from decimal import Decimal
from tempfile import TemporaryDirectory
import app
from app.calculator import Calculator
from app.calculation import Calculation
from app.calculator_repl import calculator_repl
from app.calculator_config import CalculatorConfig
from app.exceptions import OperationError, ValidationError
from app.history import LoggingObserver, AutoSaveObserver
from app.operations import OperationFactory
from tempfile import TemporaryDirectory
from pathlib import Path
from app.calculator_config import CalculatorConfig

# Fixture to initialize Calculator with a temporary directory for file paths
@pytest.fixture
def calculator():
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = CalculatorConfig(base_dir=temp_path)

        # Patch properties to use the temporary directory paths
        with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
             patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file, \
             patch.object(CalculatorConfig, 'history_dir', new_callable=PropertyMock) as mock_history_dir, \
             patch.object(CalculatorConfig, 'history_file', new_callable=PropertyMock) as mock_history_file:
            
            # Set return values to use paths within the temporary directory
            mock_log_dir.return_value = temp_path / "logs"
            mock_log_file.return_value = temp_path / "logs/calculator.log"
            mock_history_dir.return_value = temp_path / "history"
            mock_history_file.return_value = temp_path / "history/calculator_history.csv"
            
            # Return an instance of Calculator with the mocked config
            yield Calculator(config=config)

# Test Calculator Initialization

def test_calculator_initialization(calculator):
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []
    assert calculator.operation_strategy is None

# Test Logging Setup

@patch('app.calculator.logging.info')
def test_logging_setup(logging_info_mock):
    with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
         patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file:
        mock_log_dir.return_value = Path('/tmp/logs')
        mock_log_file.return_value = Path('/tmp/logs/calculator.log')
        
        # Instantiate calculator to trigger logging
        calculator = Calculator(CalculatorConfig())
        logging_info_mock.assert_any_call("Calculator initialized with configuration")


@patch('app.calculator.logging.basicConfig', side_effect=Exception("boom"))
@patch('builtins.print')
def test_setup_logging_failure_prints_and_raises(mock_print, mock_basicConfig):
    """
    Ensure that if logging.basicConfig raises during _setup_logging, the
    Calculator prints the error message and re-raises the exception.
    This exercises the except block that prints "Error setting up logging: {e}".
    """
    

    with TemporaryDirectory() as td:
        cfg = CalculatorConfig(base_dir=Path(td))
        with pytest.raises(Exception, match="boom"):
            Calculator(config=cfg)

    mock_print.assert_any_call("Error setting up logging: boom")

# Test Adding and Removing Observers

def test_add_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    assert observer in calculator.observers

def test_remove_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    calculator.remove_observer(observer)
    assert observer not in calculator.observers

# Test Setting Operations

def test_set_operation(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    assert calculator.operation_strategy == operation

# Test Performing Operations

def test_perform_operation_addition(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    result = calculator.perform_operation(2, 3)
    assert result == Decimal('5')

def test_perform_operation_validation_error(calculator):
    calculator.set_operation(OperationFactory.create_operation('add'))
    with pytest.raises(ValidationError):
        calculator.perform_operation('invalid', 3)

def test_perform_operation_operation_error(calculator):
    with pytest.raises(OperationError, match="No operation set"):
        calculator.perform_operation(2, 3)

def test_perform_operation_max_history(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    for i in range(calculator.config.max_history_size + 5):
        calculator.perform_operation(i, i)
    assert len(calculator.history) == calculator.config.max_history_size
    assert calculator.history[0].operand1 == Decimal('5')

def test_perform_operation_exception_handling(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    with patch.object(Calculation, 'calculate', side_effect=Exception("calculation error")):
        with pytest.raises(OperationError, match="calculation error"):
            calculator.perform_operation(2, 3)
                                                  
                                                  
# Test Undo/Redo Functionality

def test_undo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    assert calculator.history == []

def test_undo_nothing_to_undo(calculator):
    result = calculator.undo()
    assert result is False

def test_redo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    calculator.redo()
    assert len(calculator.history) == 1

def test_redo_nothing_to_redo(calculator):
    result = calculator.redo()
    assert result is False


# Test History Management

@patch('app.calculator.pd.DataFrame.to_csv')
def test_save_history(mock_to_csv, calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.save_history()
    mock_to_csv.assert_called_once()

@patch('app.calculator.pd.DataFrame.to_csv')
def test_save_history_empty(mock_to_csv, calculator):   
    # Test saving history when history is empty
    # Ensure history is empty
    calculator.history = []
    calculator.save_history()
    assert mock_to_csv.called

@patch('app.calculator.pd.DataFrame.to_csv', side_effect=Exception("disk full"))
def test_save_history_exception(mock_to_csv, calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    with pytest.raises(Exception, match="disk full"):
        calculator.save_history()
    mock_to_csv.assert_called_once()


@patch('app.calculator.pd.read_csv')
@patch('app.calculator.Path.exists', return_value=True)
def test_load_history(mock_exists, mock_read_csv, calculator):
    # Mock CSV data to match the expected format in from_dict
    mock_read_csv.return_value = pd.DataFrame({
        'operation': ['Addition'],
        'operand1': ['2'],
        'operand2': ['3'],
        'result': ['5'],
        'timestamp': [datetime.datetime.now().isoformat()]
    })
    
    # Test the load_history functionality
    try:
        calculator.load_history()
        # Verify history length after loading
        assert len(calculator.history) == 1
        # Verify the loaded values
        assert calculator.history[0].operation == "Addition"
        assert calculator.history[0].operand1 == Decimal("2")
        assert calculator.history[0].operand2 == Decimal("3")
        assert calculator.history[0].result == Decimal("5")
    except OperationError:
        pytest.fail("Loading history failed due to OperationError")
        
@patch('app.calculator.pd.read_csv')
@patch('app.calculator.Path.exists', return_value=True)
def test_load_history_empty(mock_exists, mock_read_csv, calculator):    
    # Mock read_csv to return an empty DataFrame
    mock_read_csv.return_value = pd.DataFrame(columns=['operation', 'operand1', 'operand2', 'result', 'timestamp'])
    
    # Test loading history when the CSV is empty
    try:
        calculator.load_history()
        assert len(calculator.history) == 0  # History should remain empty
    except OperationError:
        pytest.fail("Loading history failed due to OperationError")

@patch('app.calculator.pd.read_csv', side_effect=Exception("file not found"))
@patch('app.calculator.Path.exists', return_value=True)
def test_load_history_exception(mock_exists, mock_read_csv, calculator):
    with pytest.raises(OperationError, match="file not found"):
        calculator.load_history()
    mock_read_csv.assert_called_once()

def test_get_history_dataframe(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    df = calculator.get_history_dataframe()
    assert not df.empty
    assert list(df.columns) == ['operation', 'operand1', 'operand2', 'result', 'timestamp']
    assert df.iloc[0]['operation'] == 'Addition'
    assert df.iloc[0]['operand1'] == '2'
    assert df.iloc[0]['operand2'] == '3'
    assert df.iloc[0]['result'] == '5'
    assert isinstance(df.iloc[0]['timestamp'], datetime.datetime)
    
def test_show_history(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    history = calculator.show_history()
    assert len(history) == 1
    # show_history returns formatted strings like: "Addition(2, 3) = 5"
    assert isinstance(history[0], str)
    assert history[0] == "Addition(2, 3) = 5"

# Test Clearing History

def test_clear_history(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.clear_history()
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []

# Test REPL Commands (using patches for input/output handling)

@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
def test_calculator_repl_exit(mock_print, mock_input):
    with patch('app.calculator.Calculator.save_history') as mock_save_history:
        calculator_repl()
        mock_save_history.assert_called_once()
        mock_print.assert_any_call("History saved successfully.")
        mock_print.assert_any_call("Goodbye!")


@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
def test_calculator_repl_exit_save_failure_warns(mock_print, mock_input):
    # Simulate save_history raising an exception to exercise the warning branch
    with patch('app.calculator.Calculator.save_history', side_effect=Exception("disk full")):
        calculator_repl()
        mock_print.assert_any_call("Warning: Could not save history: disk full")
        mock_print.assert_any_call("Goodbye!")

@patch('builtins.input', side_effect=['history', 'exit'])
@patch('builtins.print')
def test_calculator_repl_history_empty_prints_no_history(mock_print, mock_input):
    # When show_history returns an empty list, REPL should print the 'No calculations in history' message
    with patch('app.calculator.Calculator.show_history', return_value=[]):
        with patch('app.calculator.Calculator.save_history'):
            calculator_repl()

    mock_print.assert_any_call("No calculations in history")


@patch('builtins.input', side_effect=['history', 'exit'])
@patch('builtins.print')
def test_calculator_repl_history_prints_entries(mock_print, mock_input):
  
    calc1 = Calculation(operation="Addition", operand1=Decimal('2'), operand2=Decimal('3'))
    calc2 = Calculation(operation="Multiplication", operand1=Decimal('4'), operand2=Decimal('5'))

    with patch('app.calculator.Calculator.show_history', return_value=[calc1, calc2]):
        # Patch save_history to avoid side effects on exit
        with patch('app.calculator.Calculator.save_history'):
            calculator_repl()

    # Header should be printed
    mock_print.assert_any_call("\nCalculation History:")
    # Entries should be printed with enumeration and Calculation.__str__ representation
    mock_print.assert_any_call("1. Addition(2, 3) = 5")
    mock_print.assert_any_call("2. Multiplication(4, 5) = 20")

@patch('builtins.input', side_effect=['clear', 'exit'])
@patch('builtins.print')
def test_calculator_repl_clear_history(mock_print, mock_input):
    with patch('app.calculator.Calculator.clear_history') as mock_clear_history:
        with patch('app.calculator.Calculator.save_history'):
            calculator_repl()
            mock_clear_history.assert_called_once()
            mock_print.assert_any_call("History cleared")

@patch('builtins.input', side_effect=['help', 'exit'])
@patch('builtins.print')
def test_calculator_repl_help(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nAvailable commands:")

@patch('builtins.input', side_effect=['redo', 'exit'])
@patch('builtins.print')
def test_calculator_repl_redo_true(mock_print, mock_input):
    with patch('app.calculator.Calculator.redo', return_value=True) as mock_redo:
        with patch('app.calculator.Calculator.save_history'):
            calculator_repl()
            mock_redo.assert_called_once()
            mock_print.assert_any_call("Operation redone")


@patch('builtins.input', side_effect=['undo', 'exit'])
@patch('builtins.print')
def test_calculator_repl_undo_true(mock_print, mock_input):
    with patch('app.calculator.Calculator.undo', return_value=True) as mock_undo:
        with patch('app.calculator.Calculator.save_history'):
            calculator_repl()
            mock_undo.assert_called_once()
            mock_print.assert_any_call("Operation undone")

@patch('builtins.input', side_effect=['undo', 'exit'])
@patch('builtins.print')
def test_calculator_repl_undo_false(mock_print, mock_input):
    with patch('app.calculator.Calculator.undo', return_value=False) as mock_undo:
        with patch('app.calculator.Calculator.save_history'):
            calculator_repl()
            mock_undo.assert_called_once()
            mock_print.assert_any_call("Nothing to undo")

@patch('builtins.input', side_effect=['redo', 'exit'])
@patch('builtins.print')
def test_calculator_repl_redo_false(mock_print, mock_input):
    with patch('app.calculator.Calculator.redo', return_value=False) as mock_redo:
        with patch('app.calculator.Calculator.save_history'):
            calculator_repl()
            mock_redo.assert_called_once()
            mock_print.assert_any_call("Nothing to redo")

@patch('builtins.input', side_effect=['save', 'exit'])
@patch('builtins.print')
def test_calculator_repl_save_success(mock_print, mock_input):
    # Patch save_history and assert it was called when 'save' command is used
    with patch('app.calculator.Calculator.save_history'):
        calculator_repl()
        mock_print.assert_any_call("History saved successfully.")

@patch('builtins.input', side_effect=['save', 'exit'])
@patch('builtins.print')
def test_calculator_repl_save_failure_warns(mock_print, mock_input):
    with patch('app.calculator.Calculator.save_history', side_effect=Exception("disk full")):
        calculator_repl()
        mock_print.assert_any_call("Error saving history: disk full")

@patch('builtins.input', side_effect=['load', 'exit'])
@patch('builtins.print')
def test_calculator_repl_load_success(mock_print, mock_input):
    with patch('app.calculator.Calculator.load_history'):
        with patch('app.calculator.Calculator.save_history'):
            calculator_repl()
            mock_print.assert_any_call("History loaded successfully")

@patch('builtins.input', side_effect=['load', 'exit'])
@patch('builtins.print')
def test_calculator_repl_load_failure_warns(mock_print, mock_input):
    with patch('app.calculator.Calculator.load_history', side_effect=Exception("file not found")):
        with patch('app.calculator.Calculator.save_history'):
            calculator_repl()
            mock_print.assert_any_call("Error loading history: file not found") 

@pytest.mark.parametrize("operation", [
    'add', 'subtract', 'multiply', 'divide', 'power', 'root'
])
def test_calculator_repl_first_input_cancel(monkeypatch, operation):
    """
    Verify that entering an operation then 'cancel' triggers the REPL to print
    "Operation cancelled" and not raise.
    """
    inputs = [operation, 'cancel', 'exit']
    # monkeypatch input to return successive values from inputs
    monkeypatch.setattr('builtins.input', lambda prompt='': inputs.pop(0))
    with patch('builtins.print') as mock_print:
        calculator_repl()
    mock_print.assert_any_call("Operation cancelled")


@pytest.mark.parametrize("operation", [
    'add', 'subtract', 'multiply', 'divide', 'power', 'root'
])
def test_calculator_repl_second_input_cancel(monkeypatch, operation):
    """
    Verify that entering an operation, a first operand, then 'cancel' triggers
    the REPL to print "Operation cancelled".
    """
    inputs = [operation, '2', 'cancel', 'exit']
    monkeypatch.setattr('builtins.input', lambda prompt='': inputs.pop(0))
    with patch('builtins.print') as mock_print:
        calculator_repl()
    mock_print.assert_any_call("Operation cancelled")

@patch('builtins.input', side_effect=['add', '2', '3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_addition(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nResult: 5")


@patch('builtins.input', side_effect=['unknown', 'exit'])
@patch('builtins.print')
def test_calculator_repl_unknown_command(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("Unknown command: 'unknown'. Type 'help' for available commands.")

                                                                                                                                                          
