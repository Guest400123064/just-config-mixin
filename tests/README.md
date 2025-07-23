# YACM Test Suite

This directory contains a comprehensive test suite for the YACM (Yet Another Config Manager) core functionality. The test suite is organized into modular components that test different aspects of the system.

## Test Organization

The test suite is organized into the following modules:

### Core Component Tests

- **`test_frozen_dict.py`** - Tests for the `FrozenDict` class
  - Initialization with various data types
  - Immutability enforcement
  - Read operations (get, keys, values, items)
  - Error handling for modification attempts
  - Edge cases with nested structures

- **`test_config_mixin.py`** - Tests for the `ConfigMixin` class
  - Configuration registration
  - Config property access and attribute shortcuts
  - Save and load functionality
  - JSON serialization with complex types
  - Class instantiation from config files
  - Error handling and validation

- **`test_register_decorator.py`** - Tests for the `@register_to_config` decorator
  - Automatic argument registration
  - Default value tracking
  - Handling of ignored parameters (`ignore_for_config`)
  - Handling of private parameters (starting with `_`)
  - Positional and keyword argument handling
  - Integration with ConfigMixin features

### Integration Tests

- **`test_integration.py`** - Integration tests for complete workflows
  - Full configuration lifecycle (create → save → load → use)
  - Multiple configuration classes working together
  - Configuration inheritance and extension
  - Error handling across integrated components
  - Performance and complexity scenarios
  - Type safety across the workflow

### Test Infrastructure

- **`conftest.py`** - Shared fixtures and utilities
  - Common test fixtures for temporary directories and sample data
  - Sample configuration classes for testing
  - Utility functions for config comparison and file operations
  - Test data generators
  - Performance testing utilities
  - Mock objects for isolated testing

## Running Tests

### Prerequisites

Make sure you have the required dependencies installed. From the project root:

```bash
poetry install  # or pip install -e .
```

### Running All Tests

From the project root directory:

```bash
# Using poetry (recommended)
poetry run pytest tests/

# Or using python directly
python -m pytest tests/
```

### Running Specific Test Modules

```bash
# Test only FrozenDict functionality
poetry run pytest tests/test_frozen_dict.py

# Test only ConfigMixin functionality
poetry run pytest tests/test_config_mixin.py

# Test only the register_to_config decorator
poetry run pytest tests/test_register_decorator.py

# Test only integration scenarios
poetry run pytest tests/test_integration.py
```

### Running Tests with Different Verbosity

```bash
# Verbose output showing test names
poetry run pytest tests/ -v

# Very verbose output showing detailed test execution
poetry run pytest tests/ -vv

# Show local variables in tracebacks on failures
poetry run pytest tests/ -l
```

### Running Tests by Markers

The test suite uses pytest markers to categorize tests:

```bash
# Run only unit tests
poetry run pytest tests/ -m unit

# Run only integration tests
poetry run pytest tests/ -m integration

# Run only slow tests
poetry run pytest tests/ -m slow

# Exclude slow tests
poetry run pytest tests/ -m "not slow"
```

### Test Coverage

To run tests with coverage reporting:

```bash
# Install coverage if not already installed
poetry add --group dev pytest-cov

# Run tests with coverage
poetry run pytest tests/ --cov=src/yacm --cov-report=html --cov-report=term

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Scenarios Covered

### FrozenDict Tests
- ✅ Empty and populated initialization
- ✅ Attribute access after initialization
- ✅ Immutability enforcement for all modification methods
- ✅ Read operations work correctly
- ✅ Order preservation (OrderedDict behavior)
- ✅ Complex nested data structures
- ✅ String representation

### ConfigMixin Tests
- ✅ Basic configuration registration
- ✅ Config property access and attribute shortcuts
- ✅ JSON serialization with type casting (Path → string, etc.)
- ✅ Save functionality with directory creation and overwrite protection
- ✅ Load functionality with validation and error handling
- ✅ Class instantiation from config with unused parameter tracking
- ✅ Error handling for missing files, wrong class names, missing parameters

### Register Decorator Tests
- ✅ Automatic registration of all tracked parameters
- ✅ Default value tracking with `_use_default_values`
- ✅ Positional, keyword, and mixed argument handling
- ✅ Ignored parameters (`ignore_for_config`) are excluded
- ✅ Private parameters (starting with `_`) are automatically excluded
- ✅ Complex data types (lists, dicts, optional types)
- ✅ Required vs optional parameter handling
- ✅ Error handling for non-ConfigMixin classes
- ✅ Function metadata preservation

### Integration Tests
- ✅ Complete configuration lifecycle workflows
- ✅ Multiple configuration classes in the same project
- ✅ Configuration inheritance and extension patterns
- ✅ Type safety across save/load cycles
- ✅ Large configuration handling (performance)
- ✅ Deeply nested data structures
- ✅ Error handling consistency across components

## Test Data and Fixtures

The test suite includes comprehensive fixtures in `conftest.py`:

- **Temporary directories** for file I/O tests
- **Sample configuration classes** with various complexity levels
- **Test data generators** for different scenarios
- **Utility functions** for config comparison and validation
- **Performance measurement utilities**
- **Mock objects** for isolated testing

## Best Practices for Adding Tests

### Test Organization
1. **Group related tests in classes** - Use descriptive class names like `TestFrozenDictImmutability`
2. **Use descriptive test names** - Test names should clearly describe what is being tested
3. **Follow the AAA pattern** - Arrange, Act, Assert in your test methods

### Test Data
1. **Use fixtures for common test data** - Leverage `conftest.py` fixtures
2. **Create isolated test environments** - Use temporary directories for file operations
3. **Test edge cases** - Include tests for None values, empty collections, etc.

### Assertions
1. **Test both positive and negative cases** - Verify expected behavior and error conditions
2. **Use specific assertions** - Prefer `assert x == y` over `assert x`
3. **Test error messages** - Verify exception messages for user-facing errors

### Example Test Structure

```python
class TestNewFeature:
    """Test suite for the new feature."""

    def test_basic_functionality(self):
        """Test basic functionality works as expected."""
        # Arrange
        config = SampleConfigClass(param1=42)

        # Act
        result = config.some_method()

        # Assert
        assert result == expected_value

    def test_error_conditions(self):
        """Test that error conditions are handled properly."""
        config = SampleConfigClass()

        with pytest.raises(ValueError) as exc_info:
            config.invalid_operation()

        assert "expected error message" in str(exc_info.value)
```

## Debugging Test Failures

### Running a Single Test
```bash
poetry run pytest tests/test_config_mixin.py::TestConfigRegistration::test_basic_registration -v
```

### Adding Debug Output
```python
def test_something(capfd):
    """Test with debug output."""
    config = SampleConfigClass(param1=42)
    print(f"Config: {config.config}")  # Will be captured

    # Your test logic here

    # Access captured output if needed
    captured = capfd.readouterr()
    assert "Config:" in captured.out
```

### Using Pytest Debugger
```bash
# Drop into debugger on failures
poetry run pytest tests/ --pdb

# Drop into debugger on first failure
poetry run pytest tests/ -x --pdb
```

## Contributing to Tests

When adding new functionality to YACM:

1. **Add corresponding tests** in the appropriate test module
2. **Update integration tests** if the change affects multiple components
3. **Add fixtures to `conftest.py`** if they would be useful for multiple test modules
4. **Update this README** if you add new test categories or change the organization

The test suite aims for high coverage and comprehensive testing of both normal and edge cases. All tests should pass before merging new changes.
