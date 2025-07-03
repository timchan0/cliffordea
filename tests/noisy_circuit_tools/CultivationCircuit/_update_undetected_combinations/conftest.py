import pytest

@pytest.fixture
def dummy_fault_count():
    def f(source_name):
        if source_name in {'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'}:
            return 1
        elif source_name == 'DEPOLARIZE1':
            return 3
        elif source_name == 'DEPOLARIZE2':
            return 15
        else:
            raise ValueError(f"Unknown fault source: {source_name}")
    return f