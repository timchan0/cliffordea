from cliffordea.noisy_circuit_tools import (
    replace_noise_level,
    swap_t_and_s_gates,
)


def test_swap_t_and_s_gates() -> None:
    """All four T/S gate names swap without changing surrounding circuit text."""
    circuit = (
        'T 0 1\n'
        '\tT_DAG 2\n'
        'S 3\n'
        '  S_DAG 4 5\n'
        'TICK\n'
        'M 0\n'
        '# T 6 and S_DAG 7 stay in this comment\n'
    )

    assert swap_t_and_s_gates(circuit) == (
        'S 0 1\n'
        '\tS_DAG 2\n'
        'T 3\n'
        '  T_DAG 4 5\n'
        'TICK\n'
        'M 0\n'
        '# T 6 and S_DAG 7 stay in this comment\n'
    )


def test_swap_t_and_s_gates_is_an_involution() -> None:
    """Swapping an already swapped circuit restores its exact original text."""
    circuit = 'T_DAG 0\r\nS 1\r\nTICK\r\n'

    assert swap_t_and_s_gates(swap_t_and_s_gates(circuit)) == circuit


def test_replace_noise_level() -> None:
    """Every matching noise literal changes while other probabilities remain intact."""
    circuit = (
        'X_ERROR(0.001) 0\n'
        'DEPOLARIZE1(0.001) 1\n'
        'DEPOLARIZE2(0.002) 0 1\n'
        'MX(0.001) 0\n'
    )

    assert replace_noise_level(circuit, 0.001, 0.0005) == (
        'X_ERROR(0.0005) 0\n'
        'DEPOLARIZE1(0.0005) 1\n'
        'DEPOLARIZE2(0.002) 0 1\n'
        'MX(0.0005) 0\n'
    )


def test_compose_circuit_string_transformations() -> None:
    """Gate and noise transformations compose for the logical-X measurement fragment."""
    reference_circuit = (
        'DEPOLARIZE1(0.001) 0 3 7\n'
        'T_DAG 0 3 7\n'
        'MPP X0*X3*X7\n'
        'T 0 3 7\n'
        'MX(0.001) 1 2\n'
    )

    s_circuit = swap_t_and_s_gates(reference_circuit)
    updated_circuit = replace_noise_level(s_circuit, 0.001, 0.003)

    assert updated_circuit == (
        'DEPOLARIZE1(0.003) 0 3 7\n'
        'S_DAG 0 3 7\n'
        'MPP X0*X3*X7\n'
        'S 0 3 7\n'
        'MX(0.003) 1 2\n'
    )
