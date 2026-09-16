import pytest

@pytest.fixture
def dummy_error_event_count():
    def f(process_name):
        if process_name in {'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'}:
            return 1
        elif process_name == 'DEPOLARIZE1':
            return 3
        elif process_name == 'DEPOLARIZE2':
            return 15
        else:
            raise ValueError(f"Unknown error process: {process_name}")
    return f